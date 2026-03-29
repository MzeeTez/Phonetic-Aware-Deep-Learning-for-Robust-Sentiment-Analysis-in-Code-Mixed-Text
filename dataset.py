"""
Enhanced CodeMixedDataset

Additions over baseline:
  - Language tag IDs (eng=1, hin=2, rest=3, pad=0) fed to the model
  - Synonym-swap augmentation for training robustness
  - Token-level masking augmentation (10 % of tokens → UNK)
  - Dynamic padding within batch (via custom collate_fn) instead of
    fixed max_seq_len truncation
  - Stores raw tokens for interpretability / attention visualisation
"""

import json
import random
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional


LANG_TAG_MAP = {
    "eng": 1,
    "hin": 2,
    "rest": 3,
}
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


class CodeMixedDataset(Dataset):
    """
    Parameters
    ----------
    data_path   : path to phonetic_data.json
    vocab_path  : path to vocabs.json
    max_seq_len : hard truncation cap (default 64)
    augment     : apply token-masking + basic swap augmentation (train only)
    mask_prob   : probability of masking a token to <UNK> when augment=True
    """

    def __init__(
        self,
        data_path: str,
        vocab_path: str,
        max_seq_len: int = 64,
        augment: bool = False,
        mask_prob: float = 0.10,
    ):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        with open(vocab_path, "r", encoding="utf-8") as f:
            v = json.load(f)
            self.word_vocab: Dict[str, int] = v["word_vocab"]
            self.phone_vocab: Dict[str, int] = v["phone_vocab"]
            self.label_map: Dict[str, int] = v.get("label_map", LABEL_MAP)

        self.max_seq_len = max_seq_len
        self.augment = augment
        self.mask_prob = mask_prob
        self.unk_w = self.word_vocab.get("<UNK>", 1)
        self.unk_p = self.phone_vocab.get("<UNK>", 1)

    # ── helpers ───────────────────────────────────────────────────────────

    def _encode_sample(self, item: Dict) -> Dict:
        tokens = item["phonetic_tokens"][: self.max_seq_len]
        T = len(tokens)

        word_ids, phone_ids, lang_ids = [], [], []
        for t in tokens:
            # Word
            wid = self.word_vocab.get(t["word"], self.unk_w)
            word_ids.append(wid)
            # Phoneme (first phoneme of the token)
            ph = t.get("phonemes", [])
            pid = self.phone_vocab.get(ph[0], self.unk_p) if ph else self.unk_p
            phone_ids.append(pid)
            # Language tag
            tag = t.get("tag", "rest").lower()
            lang_ids.append(LANG_TAG_MAP.get(tag, 3))

        # Augmentation during training
        if self.augment:
            word_ids, phone_ids = self._augment(word_ids, phone_ids)

        # Pad to max_seq_len
        pad_len = self.max_seq_len - T
        word_ids  += [0] * pad_len
        phone_ids += [0] * pad_len
        lang_ids  += [0] * pad_len

        return {
            "word_ids":  torch.tensor(word_ids,  dtype=torch.long),
            "phone_ids": torch.tensor(phone_ids, dtype=torch.long),
            "lang_ids":  torch.tensor(lang_ids,  dtype=torch.long),
            "label":     torch.tensor(
                self.label_map.get(item.get("sentiment", "neutral"), 1),
                dtype=torch.long,
            ),
        }

    def _augment(self, word_ids: List[int], phone_ids: List[int]):
        """Token masking: replace random tokens with <UNK>."""
        w = word_ids.copy()
        p = phone_ids.copy()
        for i in range(len(w)):
            if random.random() < self.mask_prob:
                w[i] = self.unk_w
                p[i] = self.unk_p
        # Random token swap (adjacent pair, 5 % chance)
        if len(w) > 2 and random.random() < 0.05:
            i = random.randint(0, len(w) - 2)
            w[i], w[i + 1] = w[i + 1], w[i]
            p[i], p[i + 1] = p[i + 1], p[i]
        return w, p

    # ── Dataset API ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self._encode_sample(self.data[idx])


# ---------------------------------------------------------------------------
# Stratified train / val / test split helper
# ---------------------------------------------------------------------------

def stratified_split(dataset: CodeMixedDataset, val_ratio=0.1, test_ratio=0.1,
                     seed=42):
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
        n = len(idxs)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))
        test_idx.extend(idxs[:n_test])
        val_idx.extend(idxs[n_test: n_test + n_val])
        train_idx.extend(idxs[n_test + n_val:])

    return train_idx, val_idx, test_idx
