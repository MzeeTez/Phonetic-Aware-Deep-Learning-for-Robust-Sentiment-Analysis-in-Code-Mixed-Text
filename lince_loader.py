"""
lince_loader.py
Loads the LinCE Hinglish Sentiment dataset into the same dict format
produced by data_loader.py (SentiMix), so all downstream preprocessing
(preprocess.py → phonetic_encoder.py → dataset.py) works unchanged.

LinCE file format (hinglish_sentiment/):
  Each split lives in a separate file: train.conll, dev.conll, test.conll
  Format is CoNLL-style:

    # sent_enum = 1
    # label = positive
    yaar	hin
    tu	hin
    toh	hin
    kamaal	hin
    hai	hin

    # sent_enum = 2
    # label = negative
    ...

  Differences from SentiMix:
    - Separator is TAB (same)
    - Sentence boundaries marked by blank lines + comment header
    - Labels are in comment lines, not in a "meta" prefix line
    - Language tags are lowercase already ("eng", "hin", "mixed")
    - "mixed" tag → we normalise to "rest" to match our LANG_TAG_MAP

Usage:
    from lince_loader import load_lince_data
    sentences = load_lince_data("path/to/test.conll")
    # Returns List[Dict] with keys: tokens, tags, sentiment
    # Identical schema to load_sentimix_data() output.

Download LinCE from: https://ritual.uh.edu/lince/
  → Benchmarks → Hinglish Sentiment → Download
"""

import re
from pathlib import Path
from typing import List, Dict


# LinCE uses "mixed" as a third tag for tokens that are a blend of both langs.
# We map it to "rest" to match our existing LANG_TAG_MAP.
_TAG_NORM = {"eng": "eng", "hin": "hin", "mixed": "rest",
             "rest": "rest", "other": "rest", "ne": "rest",
             "fw": "rest", "ambiguous": "rest"}

# LinCE sentiment labels map directly to our schema
_LABEL_NORM = {"positive": "positive", "negative": "negative",
               "neutral": "neutral", "mixed": "neutral"}


def load_lince_data(file_path: str) -> List[Dict]:
    """
    Parse a LinCE .conll file and return a list of sentence dicts.

    Each dict has:
        tokens    : List[str]   — surface word forms
        tags      : List[str]   — language tags (eng / hin / rest)
        sentiment : str         — "positive" / "negative" / "neutral"
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"LinCE file not found: {file_path}\n"
            f"Download from https://ritual.uh.edu/lince/ → Hinglish Sentiment"
        )

    sentences = []
    current   = {"tokens": [], "tags": [], "sentiment": None}

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            # Comment line — may carry the label
            if line.startswith("#"):
                m = re.search(r"label\s*=\s*(\w+)", line, re.IGNORECASE)
                if m:
                    raw_label = m.group(1).lower()
                    current["sentiment"] = _LABEL_NORM.get(raw_label, "neutral")
                continue

            # Blank line → sentence boundary
            if line.strip() == "":
                if current["tokens"]:
                    sentences.append(current)
                current = {"tokens": [], "tags": [], "sentiment": None}
                continue

            # Token line: "word\ttag"  (sometimes has a 3rd column — ignore it)
            parts = line.split("\t")
            if len(parts) >= 2:
                word = parts[0].strip()
                tag  = _TAG_NORM.get(parts[1].strip().lower(), "rest")
                if word:
                    current["tokens"].append(word)
                    current["tags"].append(tag)

    # Catch final sentence (file may not end with blank line)
    if current["tokens"]:
        sentences.append(current)

    # Fill missing labels with "neutral" (shouldn't happen but be safe)
    for s in sentences:
        if s["sentiment"] is None:
            s["sentiment"] = "neutral"

    _print_summary(sentences, file_path)
    return sentences


def _print_summary(sentences: List[Dict], path: str) -> None:
    from collections import Counter
    label_counts = Counter(s["sentiment"] for s in sentences)
    total_tokens = sum(len(s["tokens"]) for s in sentences)
    print(f"Loaded LinCE: {path}")
    print(f"  Sentences : {len(sentences):,}")
    print(f"  Tokens    : {total_tokens:,}")
    print(f"  Labels    : {dict(label_counts)}")


# ── Convenience: load all three splits at once ────────────────────────────

def load_lince_all(lince_dir: str) -> Dict[str, List[Dict]]:
    """
    Load train / dev / test from a LinCE directory.
    Returns {"train": [...], "dev": [...], "test": [...]}.
    Missing splits are skipped gracefully.
    """
    base   = Path(lince_dir)
    splits = {}
    for split, fname in [("train", "train.conll"),
                         ("dev",   "dev.conll"),
                         ("test",  "test.conll")]:
        fp = base / fname
        if fp.exists():
            splits[split] = load_lince_data(str(fp))
        else:
            print(f"  [skip] {fp} not found")
    return splits


if __name__ == "__main__":
    # Quick smoke-test — adjust path to your download location
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "lince/hinglish_sentiment/test.conll"
    sents = load_lince_data(path)
    print(f"\nFirst sentence: {sents[0]}")
