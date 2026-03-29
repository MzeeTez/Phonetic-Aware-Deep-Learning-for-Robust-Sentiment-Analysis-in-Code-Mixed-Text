"""
ablation_model.py
Wraps EnhancedDualChannelLSTM with three ablation flags so you can
disable individual components and measure their contribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.insert(0, ".")          # so we can import from the project root

from model import (
    VariationalDropout, MultiHeadSelfAttention,
    CrossModalAttention, GatedFusion, PhonemeEncoder
)

class AblationDualChannelLSTM(nn.Module):
    def __init__(
        self,
        word_vocab_size: int,
        phone_vocab_size: int,
        num_lang_tags: int = 4,
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
        ablate_phoneme: bool    = False,
        ablate_cross_attn: bool = False,
        ablate_lang_tag: bool   = False,
    ):
        super().__init__()
        self.ablate_phoneme    = ablate_phoneme
        self.ablate_cross_attn = ablate_cross_attn
        self.ablate_lang_tag   = ablate_lang_tag
        self.word_hidden       = word_hidden
        self.phone_hidden      = phone_hidden

        self.word_embed  = nn.Embedding(word_vocab_size,  word_embed_dim,  padding_idx=0)
        self.lang_embed  = nn.Embedding(num_lang_tags,    lang_embed_dim,  padding_idx=0)
        self.phone_encoder = PhonemeEncoder(phone_vocab_size, phone_embed_dim, pad_idx=0)

        word_input_dim = word_embed_dim + lang_embed_dim
        self.word_lstm = nn.LSTM(
            word_input_dim, word_hidden,
            num_layers=lstm_layers, batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.word_var_drop = VariationalDropout(var_dropout)
        self.word_attn     = MultiHeadSelfAttention(word_hidden * 2, num_attn_heads, dropout)

        self.phone_lstm = nn.LSTM(
            phone_embed_dim, phone_hidden,
            num_layers=lstm_layers, batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.phone_var_drop = VariationalDropout(var_dropout)
        self.phone_attn     = MultiHeadSelfAttention(phone_hidden * 2, num_attn_heads, dropout)

        self.cross_attn = CrossModalAttention(
            word_hidden * 2, phone_hidden * 2, num_attn_heads, dropout
        )

        self.gate_word_cross = GatedFusion(word_hidden * 2)
        self.gate_final      = GatedFusion(word_hidden * 2)
        phone_out = phone_hidden * 2
        word_out  = word_hidden  * 2
        self.phone_proj = nn.Linear(phone_out, word_out) if phone_out != word_out else nn.Identity()

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

    def forward(self, word_ids, phone_ids, lang_ids=None):
        B, T = word_ids.shape

        w_emb = self.word_embed(word_ids)
        if self.ablate_lang_tag or lang_ids is None:
            pad   = w_emb.new_zeros(B, T, self.lang_embed.embedding_dim)
            w_emb = torch.cat([w_emb, pad], dim=-1)
        else:
            w_emb = torch.cat([w_emb, self.lang_embed(lang_ids)], dim=-1)

        w_out, _      = self.word_lstm(w_emb)
        w_out         = self.word_var_drop(w_out)
        w_seq, w_attn = self.word_attn(w_out)
        w_pool        = w_seq.mean(dim=1)

        p_emb = self.phone_encoder(phone_ids)
        p_out, _      = self.phone_lstm(p_emb)
        p_out         = self.phone_var_drop(p_out)
        p_seq, p_attn = self.phone_attn(p_out)

        if self.ablate_phoneme:
            p_pool    = w_pool.new_zeros(B, self.phone_proj(p_seq.mean(1)).shape[-1])
            p_seq_z   = torch.zeros_like(p_seq)
        else:
            p_pool  = self.phone_proj(p_seq.mean(dim=1))
            p_seq_z = p_seq

        if self.ablate_cross_attn or self.ablate_phoneme:
            fused = self.gate_word_cross(w_pool, w_pool) 
        else:
            cross_ctx = self.cross_attn(w_seq, p_seq_z)
            fused     = self.gate_word_cross(w_pool, cross_ctx)

        fused  = self.gate_final(fused, p_pool)
        logits = self.classifier(fused)
        return logits, w_attn, p_attn