"""
model.py  (v2 — full phoneme sequences)

Architecture change:
  The phoneme channel now receives phone_ids of shape (B, T, max_phones)
  instead of (B, T).  A new PhonemeEncoder sub-module handles the inner
  phoneme dimension:

    phone_ids (B, T, P)
        → embed each phoneme → (B, T, P, phone_embed_dim)
        → mean-pool over P  → (B, T, phone_embed_dim)
        → 2-layer BiLSTM    → (B, T, phone_hidden*2)

  This gives every token its full phoneme context before the cross-modal
  attention mechanism combines it with the word channel.

Everything else is identical to v1 (VariationalDropout, MultiHeadSelfAttention,
CrossModalAttention, GatedFusion, weight init, forward signature).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Helpers (unchanged from v1)
# ---------------------------------------------------------------------------

class VariationalDropout(nn.Module):
    """Locked dropout — same mask at every time step (Gal & Ghahramani 2016)."""
    def __init__(self, p: float = 0.3):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)
        mask = mask / (1 - self.p)
        return x * mask


class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.h   = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)
        self.q   = nn.Linear(d_model, d_model)
        self.k   = nn.Linear(d_model, d_model)
        self.v   = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        B, T, _ = x.shape
        residual = x

        def split_heads(t):
            return t.view(B, T, self.h, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(self.q(x)), split_heads(self.k(x)), split_heads(self.v(x))
        scores  = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = self.drop(F.softmax(scores, dim=-1))
        ctx     = torch.matmul(weights, V)
        ctx     = ctx.transpose(1, 2).contiguous().view(B, T, -1)
        return self.norm(residual + self.out(ctx)), weights


class CrossModalAttention(nn.Module):
    """Word queries attend over phoneme keys/values → (B, word_dim) context."""

    def __init__(self, word_dim: int, phone_dim: int, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.d   = word_dim
        self.h   = num_heads
        self.d_k = word_dim // num_heads
        self.scale = math.sqrt(self.d_k)
        self.q_proj = nn.Linear(word_dim,  word_dim)
        self.k_proj = nn.Linear(phone_dim, word_dim)
        self.v_proj = nn.Linear(phone_dim, word_dim)
        self.out  = nn.Linear(word_dim, word_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(word_dim)

    def forward(self, word_seq: torch.Tensor, phone_seq: torch.Tensor):
        B, Tw, _ = word_seq.shape
        Tp = phone_seq.size(1)

        def sh_w(t): return t.view(B, Tw, self.h, self.d_k).transpose(1, 2)
        def sh_p(t): return t.view(B, Tp, self.h, self.d_k).transpose(1, 2)

        Q = sh_w(self.q_proj(word_seq))
        K = sh_p(self.k_proj(phone_seq))
        V = sh_p(self.v_proj(phone_seq))

        scores  = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = self.drop(F.softmax(scores, dim=-1))
        ctx     = torch.matmul(weights, V)
        ctx     = ctx.transpose(1, 2).contiguous().view(B, Tw, self.d)
        ctx     = self.norm(word_seq + self.out(ctx))
        return ctx.mean(dim=1)


class GatedFusion(nn.Module):
    """Element-wise gate: output = g ⊙ a + (1−g) ⊙ b."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([a, b], dim=-1))
        return self.norm(g * a + (1 - g) * b)


# ---------------------------------------------------------------------------
# NEW: PhonemeEncoder
# ---------------------------------------------------------------------------

class PhonemeEncoder(nn.Module):
    """
    Encodes the full phoneme sequence of each token into a single vector.

    Input:  phone_ids  (B, T, max_phones)   — padded phoneme ID matrix
    Output: (B, T, phone_hidden * 2)         — per-token phoneme representations

    Steps:
      1. Embed each phoneme:       (B, T, P) → (B*T, P, phone_embed_dim)
      2. Mask padding positions
      3. Mean-pool over P dim:     (B*T, P, E) → (B*T, E)
      4. Reshape back:             (B*T, E) → (B, T, E)

    We deliberately keep this simple (embed + pool) rather than adding another
    LSTM over phonemes, because:
      a) phoneme sequences are short (avg ~3, max ~8)
      b) the cross-modal attention already provides contextual mixing
      c) adding another LSTM would roughly double the phoneme-side parameters
    """

    def __init__(self, phone_vocab_size: int, phone_embed_dim: int, pad_idx: int = 0):
        super().__init__()
        self.embed   = nn.Embedding(phone_vocab_size, phone_embed_dim, padding_idx=pad_idx)
        self.pad_idx = pad_idx

    def forward(self, phone_ids: torch.Tensor) -> torch.Tensor:
        """
        phone_ids : (B, T, P)
        returns   : (B, T, phone_embed_dim)
        """
        B, T, P = phone_ids.shape
        # Flatten to (B*T, P) for embedding lookup
        flat        = phone_ids.view(B * T, P)                   # (B*T, P)
        emb         = self.embed(flat)                            # (B*T, P, E)
        # Mask padding positions before pooling
        pad_mask    = (flat != self.pad_idx).float().unsqueeze(-1)  # (B*T, P, 1)
        masked      = emb * pad_mask
        counts      = pad_mask.sum(dim=1).clamp(min=1)           # (B*T, 1)
        pooled      = masked.sum(dim=1) / counts                  # (B*T, E)
        return pooled.view(B, T, -1)                              # (B, T, E)


# ---------------------------------------------------------------------------
# Main model (v2)
# ---------------------------------------------------------------------------

class EnhancedDualChannelLSTM(nn.Module):
    """
    Hinglish sentiment model with full phoneme sequences.

    Architecture:
      Word channel:
          embed(word + lang_tag) → 2-layer BiLSTM → MH self-attn → w_pool

      Phoneme channel  (NEW in v2):
          PhonemeEncoder: embed(phone_ids[B,T,P]) → mean-pool → (B,T,phone_embed)
          → 2-layer BiLSTM → MH self-attn → p_pool

      Fusion:
          CrossModalAttention(word queries, phone keys/values) → cross_ctx
          GatedFusion(w_pool, cross_ctx) → GatedFusion(·, p_pool) → fused

      Classifier:
          Linear(→512) → GELU → Dropout → LayerNorm → Linear(→3)
    """

    def __init__(
        self,
        word_vocab_size:  int,
        phone_vocab_size: int,
        num_lang_tags:    int   = 4,
        word_embed_dim:   int   = 256,
        phone_embed_dim:  int   = 128,
        lang_embed_dim:   int   = 32,
        word_hidden:      int   = 256,
        phone_hidden:     int   = 128,
        lstm_layers:      int   = 2,
        num_attn_heads:   int   = 4,
        num_classes:      int   = 3,
        dropout:          float = 0.4,
        var_dropout:      float = 0.3,
    ):
        super().__init__()
        self.word_hidden  = word_hidden
        self.phone_hidden = phone_hidden

        # ── Word channel embeddings ───────────────────────────────────────
        self.word_embed = nn.Embedding(word_vocab_size,  word_embed_dim,  padding_idx=0)
        self.lang_embed = nn.Embedding(num_lang_tags,    lang_embed_dim,  padding_idx=0)

        # ── Phoneme channel: full sequence encoder (NEW) ──────────────────
        self.phone_encoder = PhonemeEncoder(phone_vocab_size, phone_embed_dim, pad_idx=0)

        # ── Word BiLSTM ───────────────────────────────────────────────────
        word_input_dim = word_embed_dim + lang_embed_dim
        self.word_lstm = nn.LSTM(
            word_input_dim, word_hidden,
            num_layers=lstm_layers, batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.word_var_drop = VariationalDropout(var_dropout)
        self.word_attn     = MultiHeadSelfAttention(word_hidden * 2, num_attn_heads, dropout)

        # ── Phoneme BiLSTM ────────────────────────────────────────────────
        # Input is now phone_embed_dim (from PhonemeEncoder), not a raw ID
        self.phone_lstm = nn.LSTM(
            phone_embed_dim, phone_hidden,
            num_layers=lstm_layers, batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.phone_var_drop = VariationalDropout(var_dropout)
        self.phone_attn     = MultiHeadSelfAttention(phone_hidden * 2, num_attn_heads, dropout)

        # ── Cross-modal & fusion ──────────────────────────────────────────
        self.cross_attn = CrossModalAttention(
            word_hidden * 2, phone_hidden * 2, num_attn_heads, dropout
        )
        self.gate_word_cross = GatedFusion(word_hidden * 2)
        self.gate_final      = GatedFusion(word_hidden * 2)
        phone_out = phone_hidden * 2
        word_out  = word_hidden  * 2
        self.phone_proj = (
            nn.Linear(phone_out, word_out) if phone_out != word_out else nn.Identity()
        )

        # ── Classifier ────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(word_out, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                if "bias_ih" in name or "bias_hh" in name:
                    n = p.size(0)
                    p.data[n // 4: n // 2].fill_(1.0)
        nn.init.normal_(self.word_embed.weight,  0, 0.01)
        nn.init.normal_(self.lang_embed.weight,  0, 0.01)
        nn.init.normal_(self.phone_encoder.embed.weight, 0, 0.01)

    def forward(
        self,
        word_ids:  torch.Tensor,            # (B, T)
        phone_ids: torch.Tensor,            # (B, T, max_phones)   ← NEW shape
        lang_ids:  torch.Tensor = None,     # (B, T)
    ):
        B, T = word_ids.shape

        # ── Word channel ──────────────────────────────────────────────────
        w_emb = self.word_embed(word_ids)                         # (B, T, word_dim)
        if lang_ids is not None:
            w_emb = torch.cat([w_emb, self.lang_embed(lang_ids)], dim=-1)
        else:
            pad   = w_emb.new_zeros(B, T, self.lang_embed.embedding_dim)
            w_emb = torch.cat([w_emb, pad], dim=-1)

        w_out, _      = self.word_lstm(w_emb)
        w_out         = self.word_var_drop(w_out)
        w_seq, w_attn = self.word_attn(w_out)
        w_pool        = w_seq.mean(dim=1)                         # (B, word_h*2)

        # ── Phoneme channel ───────────────────────────────────────────────
        # PhonemeEncoder collapses (B, T, P) → (B, T, phone_embed_dim)
        p_emb         = self.phone_encoder(phone_ids)             # (B, T, phone_embed)
        p_out, _      = self.phone_lstm(p_emb)
        p_out         = self.phone_var_drop(p_out)
        p_seq, p_attn = self.phone_attn(p_out)
        p_pool        = self.phone_proj(p_seq.mean(dim=1))        # (B, word_h*2)

        # ── Cross-modal attention ─────────────────────────────────────────
        cross_ctx = self.cross_attn(w_seq, p_seq)                 # (B, word_h*2)

        # ── Gated fusion ──────────────────────────────────────────────────
        fused  = self.gate_word_cross(w_pool, cross_ctx)
        fused  = self.gate_final(fused, p_pool)

        logits = self.classifier(fused)                           # (B, num_classes)
        return logits, w_attn, p_attn
