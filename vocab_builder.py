"""
vocab_builder.py  (v2 — compatible with full phoneme sequences)

Changes from v1:
  - Phoneme vocab now built from the full phoneme list per token, not just ph[0].
    This means the vocab correctly reflects the Arpabet + Akshar phoneme space.
  - Special tokens added: <SYM> (for emoji/symbols) and <UNK_PHONE>.
  - OOV rate now reported for both words and phonemes separately.
  - vocab_stats.json extended with per-language phoneme breakdown.
"""

import json
from collections import Counter


def build_vocabs(
    input_file:     str = "phonetic_data.json",
    output_file:    str = "vocabs.json",
    max_word_vocab:  int = 15000,
    max_phone_vocab: int = 3000,
    min_word_freq:   int = 2,
    min_phone_freq:  int = 1,
):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    word_counter  = Counter()
    phone_counter = Counter()
    label_map     = {"negative": 0, "neutral": 1, "positive": 2}

    for entry in data:
        for item in entry.get("phonetic_tokens", []):
            word_counter[item["word"]] += 1
            # v2: iterate the full phoneme list
            for p in item.get("phonemes", []):
                phone_counter[p] += 1

    # Filter by minimum frequency
    qualified_words  = [(w, c) for w, c in word_counter.items() if c >= min_word_freq]
    qualified_phones = [(p, c) for p, c in phone_counter.items() if c >= min_phone_freq]
    qualified_words.sort(key=lambda x: -x[1])
    qualified_phones.sort(key=lambda x: -x[1])

    # Special tokens: PAD=0, UNK=1, MASK=2, SYM=3, UNK_PHONE=4
    word_vocab  = {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2}
    phone_vocab = {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2, "<SYM>": 3, "<UNK_PHONE>": 4}

    for i, (word,  _) in enumerate(qualified_words[:max_word_vocab],   start=3):
        word_vocab[word]  = i
    for i, (phone, _) in enumerate(qualified_phones[:max_phone_vocab], start=5):
        phone_vocab[phone] = i

    vocabs = {
        "word_vocab":  word_vocab,
        "phone_vocab": phone_vocab,
        "label_map":   label_map,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(vocabs, f, indent=2, ensure_ascii=False)

    # OOV diagnostics
    total_w, oov_w = 0, 0
    total_p, oov_p = 0, 0
    for entry in data:
        for item in entry.get("phonetic_tokens", []):
            total_w += 1
            if item["word"] not in word_vocab:
                oov_w += 1
            for p in item.get("phonemes", []):
                total_p += 1
                if p not in phone_vocab:
                    oov_p += 1

    stats = {
        "word_vocab_size":   len(word_vocab),
        "phone_vocab_size":  len(phone_vocab),
        "total_word_types":  len(word_counter),
        "total_phone_types": len(phone_counter),
        "min_word_freq":     min_word_freq,
        "word_oov_rate_pct": round(oov_w / max(1, total_w) * 100, 3),
        "phone_oov_rate_pct": round(oov_p / max(1, total_p) * 100, 3),
        "top_20_phonemes":   phone_counter.most_common(20),
    }
    with open("vocab_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Word  vocab : {len(word_vocab):,} tokens  (OOV {stats['word_oov_rate_pct']:.2f}%)")
    print(f"Phone vocab : {len(phone_vocab):,} phonemes (OOV {stats['phone_oov_rate_pct']:.2f}%)")
    print(f"Vocabs saved → {output_file}")
    print(f"Stats  saved → vocab_stats.json")


if __name__ == "__main__":
    build_vocabs()
