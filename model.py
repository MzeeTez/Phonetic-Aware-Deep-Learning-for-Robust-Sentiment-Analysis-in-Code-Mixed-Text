"""
Enhanced Dual-Channel LSTM with Cross-Modal Attention Fusion
for Hinglish Sentiment Analysis

Key improvements over baseline:
  - Language-tag embedding injected into word channel
  - 2-layer stacked BiLSTM with variational dropout
  - Multi-head self-attention (per channel)
  - Cross-modal attention: word queries over phoneme keys/values
  - Learnable gated fusion (replaces naive concat)
  - GELU activation + layer norm throughout
  - Label-smoothing-ready logit output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class VariationalDropout(nn.Module):
    """Apply the SAME dropout mask at every time step (locked dropout).
    Empirically superior to standard dropout for RNNs (Gal & Ghahramani 2016).
    """
    def __init__(self, p: float = 0.3):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        # x: (batch, seq, dim)
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)
        mask = mask / (1 - self.p)
        return x * mask


class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head attention (self)."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """x: (batch, seq, d_model)  →  same shape"""
        B, T, _ = x.shape
        residual = x

        def split_heads(t):
            return t.view(B, T, self.h, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(self.q(x)), split_heads(self.k(x)), split_heads(self.v(x))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = self.drop(F.softmax(scores, dim=-1))
        ctx = torch.matmul(weights, V)                       # (B, h, T, d_k)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, -1)
        return self.norm(residual + self.out(ctx)), weights


class CrossModalAttention(nn.Module):
    """Word sequence queries attend over phoneme sequence keys/values.
    Returns a single context vector summarising the cross-modal interaction.
    """

    def __init__(self, word_dim: int, phone_dim: int, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        # Project both channels to a common dimension
        self.d = word_dim
        self.h = num_heads
        self.d_k = word_dim // num_heads
        self.scale = math.sqrt(self.d_k)

        self.q_proj = nn.Linear(word_dim, word_dim)
        self.k_proj = nn.Linear(phone_dim, word_dim)
        self.v_proj = nn.Linear(phone_dim, word_dim)
        self.out = nn.Linear(word_dim, word_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(word_dim)

    def forward(self, word_seq: torch.Tensor, phone_seq: torch.Tensor):
        """
        word_seq:  (B, Tw, word_dim)
        phone_seq: (B, Tp, phone_dim)
        Returns:   (B, word_dim)  — mean-pooled cross-attended context
        """
        B, Tw, _ = word_seq.shape
        Tp = phone_seq.size(1)

        def sh_w(t):
            return t.view(B, Tw, self.h, self.d_k).transpose(1, 2)
        def sh_p(t):
            return t.view(B, Tp, self.h, self.d_k).transpose(1, 2)

        Q = sh_w(self.q_proj(word_seq))
        K = sh_p(self.k_proj(phone_seq))
        V = sh_p(self.v_proj(phone_seq))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B,h,Tw,Tp)
        weights = self.drop(F.softmax(scores, dim=-1))
        ctx = torch.matmul(weights, V)                               # (B,h,Tw,d_k)
        ctx = ctx.transpose(1, 2).contiguous().view(B, Tw, self.d)
        ctx = self.norm(word_seq + self.out(ctx))
        return ctx.mean(dim=1)                                       # (B, d)


class GatedFusion(nn.Module):
    """Element-wise learned gate: output = g ⊙ a + (1−g) ⊙ b."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([a, b], dim=-1))
        return self.norm(g * a + (1 - g) * b)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class EnhancedDualChannelLSTM(nn.Module):
    """
    Research-grade Hinglish sentiment model.

    Architecture:
        Word channel   → embed (word + lang tag) → 2-layer BiLSTM → MH self-attn
        Phoneme channel → embed → 2-layer BiLSTM → MH self-attn
        Cross-modal fusion: word attends over phoneme → cross ctx
        Gated fusion of [word pool, cross ctx, phone pool]
        MLP: Linear(→512) → GELU → Dropout → Linear(→3)
    """

    def __init__(
        self,
        word_vocab_size: int,
        phone_vocab_size: int,
        num_lang_tags: int = 4,       # eng / hin / rest / pad
        word_embed_dim: int = 256,
        phone_embed_dim: int = 128,
        lang_embed_dim: int = 32,
        word_hidden: int = 256,
        phone_hidden: int = 128,
        lstm_layers: int = 2,
        num_attn_heads: int = 4,
        num_classes: int = 3,
        dropout: float = 0.4,
        var_dropout: float = 0.3,
    ):
        super().__init__()
        self.word_hidden = word_hidden
        self.phone_hidden = phone_hidden

        # ── Embeddings ────────────────────────────────────────────────────
        self.word_embed = nn.Embedding(word_vocab_size, word_embed_dim, padding_idx=0)
        self.lang_embed = nn.Embedding(num_lang_tags, lang_embed_dim, padding_idx=0)
        self.phone_embed = nn.Embedding(phone_vocab_size, phone_embed_dim, padding_idx=0)

        # ── Word channel ──────────────────────────────────────────────────
        word_input_dim = word_embed_dim + lang_embed_dim
        self.word_lstm = nn.LSTM(
            word_input_dim, word_hidden,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=True, dropout=dropout if lstm_layers > 1 else 0
        )
        self.word_var_drop = VariationalDropout(var_dropout)
        self.word_attn = MultiHeadSelfAttention(word_hidden * 2, num_attn_heads, dropout)

        # ── Phoneme channel ───────────────────────────────────────────────
        self.phone_lstm = nn.LSTM(
            phone_embed_dim, phone_hidden,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=True, dropout=dropout if lstm_layers > 1 else 0
        )
        self.phone_var_drop = VariationalDropout(var_dropout)
        self.phone_attn = MultiHeadSelfAttention(phone_hidden * 2, num_attn_heads, dropout)

        # ── Cross-modal & fusion ──────────────────────────────────────────
        self.cross_attn = CrossModalAttention(
            word_hidden * 2, phone_hidden * 2, num_attn_heads, dropout
        )
        # Fuse: word pool + cross ctx → gated pair A
        self.gate_word_cross = GatedFusion(word_hidden * 2)
        # Then fuse with phone pool
        self.gate_final = GatedFusion(word_hidden * 2)
        # phone_hidden*2 may differ from word_hidden*2; project if needed
        phone_out = phone_hidden * 2
        word_out = word_hidden * 2
        self.phone_proj = (
            nn.Linear(phone_out, word_out) if phone_out != word_out else nn.Identity()
        )

        # ── Classifier ────────────────────────────────────────────────────
        fused_dim = word_out
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),
            nn.Linear(512, num_classes)
        )

        self._init_weights()

    # ── Weight init ───────────────────────────────────────────────────────
    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                # Set forget gate bias to 1 (helps long-range memory)
                if "bias_ih" in name or "bias_hh" in name:
                    n = p.size(0)
                    p.data[n // 4: n // 2].fill_(1.0)
        nn.init.normal_(self.word_embed.weight, 0, 0.01)
        nn.init.normal_(self.phone_embed.weight, 0, 0.01)
        nn.init.normal_(self.lang_embed.weight, 0, 0.01)

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(
        self,
        word_ids: torch.Tensor,       # (B, T)
        phone_ids: torch.Tensor,      # (B, T)
        lang_ids: torch.Tensor = None # (B, T)  — optional language tags
    ):
        # Word channel
        w_emb = self.word_embed(word_ids)                     # (B,T,word_dim)
        if lang_ids is not None:
            l_emb = self.lang_embed(lang_ids)
            w_emb = torch.cat([w_emb, l_emb], dim=-1)
        else:
            # Pad lang embedding with zeros when not provided
            B, T, _ = w_emb.shape
            pad = w_emb.new_zeros(B, T, self.lang_embed.embedding_dim)
            w_emb = torch.cat([w_emb, pad], dim=-1)

        w_out, _ = self.word_lstm(w_emb)                      # (B,T,word_h*2)
        w_out = self.word_var_drop(w_out)
        w_seq, w_weights = self.word_attn(w_out)              # (B,T,word_h*2)
        w_pool = w_seq.mean(dim=1)                            # (B,word_h*2)

        # Phoneme channel
        p_emb = self.phone_embed(phone_ids)
        p_out, _ = self.phone_lstm(p_emb)                     # (B,T,phone_h*2)
        p_out = self.phone_var_drop(p_out)
        p_seq, p_weights = self.phone_attn(p_out)             # (B,T,phone_h*2)
        p_pool = self.phone_proj(p_seq.mean(dim=1))           # (B,word_h*2)

        # Cross-modal: word queries over phoneme sequence
        cross_ctx = self.cross_attn(w_seq, p_seq)             # (B,word_h*2)

        # Gated fusion
        fused = self.gate_word_cross(w_pool, cross_ctx)
        fused = self.gate_final(fused, p_pool)

        logits = self.classifier(fused)                       # (B, num_classes)
        return logits, w_weights, p_weights
