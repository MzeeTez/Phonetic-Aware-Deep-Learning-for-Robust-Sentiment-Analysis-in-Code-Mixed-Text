"""
Enhanced vocabulary builder

Additions:
  - Frequency threshold filtering (min_word_freq, min_phone_freq)
  - Reports OOV rate on the dataset for diagnostics
  - Writes vocab statistics to vocab_stats.json
  - Includes special tokens: <PAD>=0, <UNK>=1, <MASK>=2 (for future MLM)
"""

import json
from collections import Counter


def build_vocabs(
    input_file: str = "phonetic_data.json",
    output_file: str = "vocabs.json",
    max_word_vocab: int = 15000,   # increased from 10k
    max_phone_vocab: int = 3000,
    min_word_freq: int = 2,        # drop hapax legomena
    min_phone_freq: int = 1,
):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    word_counter  = Counter()
    phone_counter = Counter()
    label_map     = {"negative": 0, "neutral": 1, "positive": 2}

    for entry in data:
        for item in entry.get("phonetic_tokens", []):
            word_counter[item["word"]] += 1
            for p in item.get("phonemes", []):
                phone_counter[p] += 1

    # Filter by minimum frequency
    qualified_words  = [(w, c) for w, c in word_counter.items() if c >= min_word_freq]
    qualified_phones = [(p, c) for p, c in phone_counter.items() if c >= min_phone_freq]

    # Sort by frequency, take top-N
    qualified_words.sort(key=lambda x: -x[1])
    qualified_phones.sort(key=lambda x: -x[1])

    # Special tokens: PAD=0, UNK=1, MASK=2
    word_vocab  = {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2}
    phone_vocab = {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2}

    for i, (word, _) in enumerate(qualified_words[:max_word_vocab], start=3):
        word_vocab[word] = i

    for i, (phone, _) in enumerate(qualified_phones[:max_phone_vocab], start=3):
        phone_vocab[phone] = i

    vocabs = {
        "word_vocab":  word_vocab,
        "phone_vocab": phone_vocab,
        "label_map":   label_map,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(vocabs, f, indent=2, ensure_ascii=False)

    # ── OOV diagnostics ───────────────────────────────────────────────────
    total_tokens, oov_tokens = 0, 0
    for entry in data:
        for item in entry.get("phonetic_tokens", []):
            total_tokens += 1
            if item["word"] not in word_vocab:
                oov_tokens += 1

    oov_rate = oov_tokens / max(1, total_tokens) * 100

    stats = {
        "word_vocab_size":  len(word_vocab),
        "phone_vocab_size": len(phone_vocab),
        "total_word_types": len(word_counter),
        "total_phone_types": len(phone_counter),
        "min_word_freq":    min_word_freq,
        "oov_rate_percent": round(oov_rate, 3),
    }
    with open("vocab_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Word  vocab : {len(word_vocab):,} tokens  (OOV rate: {oov_rate:.2f}%)")
    print(f"Phone vocab : {len(phone_vocab):,} phonemes")
    print(f"Vocabs saved → {output_file}")
    print(f"Stats  saved → vocab_stats.json")


if __name__ == "__main__":
    build_vocabs()
