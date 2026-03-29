"""
dataset.py  (v2 — full phoneme sequences)

Key changes over v1:
  - phone_ids is now a 2-D padded tensor (T × max_phones_per_token) instead
    of a 1-D tensor of single phoneme IDs.
    This lets the model's phoneme channel see the full phoneme sequence of
    each word — e.g. "love" → [L, AH, V] — rather than just the first phoneme.
  - max_phones_per_token (default 8) caps sub-word phoneme length.
    Words rarely have more than 7–8 phonemes; cap keeps memory bounded.
  - A new collate_fn (get_collate_fn) is exported so DataLoader can handle
    the 3-D batch tensor (B × T × max_phones_per_token) automatically.
  - Augmentation now also masks full phoneme rows, not just phone_ids[:,0].
  - Everything else (stratified_split, lang_ids, augmentation, label_map)
    is unchanged so train.py / evaluate.py require only minor edits.

Tensor shapes emitted per sample:
  word_ids   : (max_seq_len,)                long  — word token IDs
  phone_ids  : (max_seq_len, max_phones)     long  — phoneme ID matrix
  lang_ids   : (max_seq_len,)                long  — lang tag IDs
  label      : ()                            long  — 0 / 1 / 2

Model change required:
  The phoneme channel BiLSTM now receives (B, T, max_phones) and must
  first embed + pool the inner dimension before feeding into the token-level
  LSTM.  See model.py v2 for the PhonemeEncoder sub-module that handles this.
"""

import json
import random
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple

LANG_TAG_MAP = {"eng": 1, "hin": 2, "rest": 3}
LABEL_MAP    = {"negative": 0, "neutral": 1, "positive": 2}

MAX_PHONES_DEFAULT = 8   # max phonemes per token; covers 99 %+ of words


class CodeMixedDataset(Dataset):
    """
    Parameters
    ----------
    data_path        : path to phonetic_data.json  (produced by phonetic_encoder.py v2)
    vocab_path       : path to vocabs.json
    max_seq_len      : max tokens per sentence (hard truncation)
    max_phones       : max phonemes per token  (hard truncation, right-padded)
    augment          : token-masking + adjacent-swap augmentation (train only)
    mask_prob        : probability a token row is masked to <UNK>
    """

    def __init__(
        self,
        data_path:   str,
        vocab_path:  str,
        max_seq_len: int   = 64,
        max_phones:  int   = MAX_PHONES_DEFAULT,
        augment:     bool  = False,
        mask_prob:   float = 0.10,
    ):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        with open(vocab_path, "r", encoding="utf-8") as f:
            v = json.load(f)
            self.word_vocab:  Dict[str, int] = v["word_vocab"]
            self.phone_vocab: Dict[str, int] = v["phone_vocab"]
            self.label_map:   Dict[str, int] = v.get("label_map", LABEL_MAP)

        self.max_seq_len = max_seq_len
        self.max_phones  = max_phones
        self.augment     = augment
        self.mask_prob   = mask_prob
        self.unk_w       = self.word_vocab.get("<UNK>", 1)
        self.unk_p       = self.phone_vocab.get("<UNK>", 1)
        self.pad_p       = self.phone_vocab.get("<PAD>", 0)

    # ── helpers ───────────────────────────────────────────────────────────

    def _encode_phones(self, phonemes: List[str]) -> List[int]:
        """
        Map a phoneme list to IDs, truncate/pad to max_phones.
        e.g. ["L", "AH", "V"] → [23, 5, 17, 0, 0, 0, 0, 0]  (max_phones=8)
        """
        ids = [self.phone_vocab.get(p, self.unk_p) for p in phonemes[: self.max_phones]]
        ids += [self.pad_p] * (self.max_phones - len(ids))
        return ids

    def _encode_sample(self, item: Dict) -> Dict:
        tokens = item["phonetic_tokens"][: self.max_seq_len]
        T = len(tokens)

        word_ids:  List[int]       = []
        phone_ids: List[List[int]] = []   # T × max_phones
        lang_ids:  List[int]       = []

        for t in tokens:
            # Word ID
            word_ids.append(self.word_vocab.get(t["word"], self.unk_w))
            # Full phoneme sequence → padded row of length max_phones
            phone_ids.append(self._encode_phones(t.get("phonemes", [])))
            # Language tag
            lang_ids.append(LANG_TAG_MAP.get(t.get("tag", "rest").lower(), 3))

        # Augmentation (training only)
        if self.augment:
            word_ids, phone_ids = self._augment(word_ids, phone_ids)

        # Pad sequence dimension to max_seq_len
        pad_len = self.max_seq_len - T
        word_ids  += [0]                      * pad_len
        phone_ids += [[self.pad_p] * self.max_phones] * pad_len
        lang_ids  += [0]                      * pad_len

        return {
            "word_ids":  torch.tensor(word_ids,  dtype=torch.long),            # (T,)
            "phone_ids": torch.tensor(phone_ids, dtype=torch.long),            # (T, P)
            "lang_ids":  torch.tensor(lang_ids,  dtype=torch.long),            # (T,)
            "label":     torch.tensor(
                self.label_map.get(item.get("sentiment", "neutral"), 1),
                dtype=torch.long,
            ),
        }

    def _augment(
        self,
        word_ids:  List[int],
        phone_ids: List[List[int]],
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Token masking: replace whole token rows with <UNK>/<PAD>.
        Adjacent swap: swap two neighbouring tokens with 5 % chance.
        """
        w = word_ids[:]
        p = [row[:] for row in phone_ids]

        for i in range(len(w)):
            if random.random() < self.mask_prob:
                w[i] = self.unk_w
                p[i] = [self.unk_p] + [self.pad_p] * (self.max_phones - 1)

        if len(w) > 2 and random.random() < 0.05:
            i = random.randint(0, len(w) - 2)
            w[i], w[i + 1]   = w[i + 1], w[i]
            p[i], p[i + 1]   = p[i + 1], p[i]

        return w, p

    # ── Dataset API ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self._encode_sample(self.data[idx])


# ── Collate function ──────────────────────────────────────────────────────

def get_collate_fn():
    """
    Returns a collate_fn for DataLoader that stacks the 3-D phone_ids tensor
    (B, T, max_phones) correctly.  word_ids, lang_ids, label stack normally.
    """
    def collate_fn(batch):
        return {
            "word_ids":  torch.stack([b["word_ids"]  for b in batch]),   # (B, T)
            "phone_ids": torch.stack([b["phone_ids"] for b in batch]),   # (B, T, P)
            "lang_ids":  torch.stack([b["lang_ids"]  for b in batch]),   # (B, T)
            "label":     torch.stack([b["label"]     for b in batch]),   # (B,)
        }
    return collate_fn


# ── Stratified split ──────────────────────────────────────────────────────

def stratified_split(
    dataset: CodeMixedDataset,
    val_ratio:  float = 0.1,
    test_ratio: float = 0.1,
    seed:       int   = 42,
):
    """Return three index lists (train, val, test) with balanced class dist."""
    from collections import defaultdict
    rng = random.Random(seed)

    buckets = defaultdict(list)
    for i, item in enumerate(dataset.data):
        label = dataset.label_map.get(item.get("sentiment", "neutral"), 1)
        buckets[label].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for label, idxs in buckets.items():
        rng.shuffle(idxs)
        n       = len(idxs)
        n_test  = max(1, int(n * test_ratio))
        n_val   = max(1, int(n * val_ratio))
        test_idx.extend(idxs[:n_test])
        val_idx.extend(idxs[n_test: n_test + n_val])
        train_idx.extend(idxs[n_test + n_val:])

    return train_idx, val_idx, test_idx
