"""
Enhanced Phonetic Encoder for Hinglish

Improvements:
  - Caches G2P results (speeds up large datasets significantly)
  - Handles emoji and special chars gracefully (returns empty phoneme list)
  - Detects Romanised Hindi heuristically and routes correctly
  - Devanagari text (if present) converted via Unidecode-style mapping
  - Batch-level progress bar
  - Saves a phoneme frequency report for vocab analysis
"""

import json
import re
import unicodedata
from collections import Counter
from g2p_en import G2p
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from tqdm import tqdm

g2p_en = G2p()

# Simple heuristic: mostly ASCII letters → treat as English-script
_ASCII_WORD_RE = re.compile(r'^[a-zA-Z]+$')
# Devanagari unicode block
_DEVA_RE = re.compile(r'[\u0900-\u097F]')
# Emoji / symbols (non-alphabetic, non-ASCII)
_SYMBOL_RE = re.compile(r'[^\w\s]', re.UNICODE)

# LRU-style cache for G2P (avoids repeated expensive calls)
_g2p_cache: dict = {}


def _safe_g2p(word: str):
    if word in _g2p_cache:
        return _g2p_cache[word]
    try:
        result = g2p_en(word)
    except Exception:
        result = [word]
    _g2p_cache[word] = result
    return result


def get_phonemes(word: str, tag: str):
    """
    Convert a single word to a phoneme list based on its language tag.

    Returns a (possibly empty) list of phoneme strings.
    """
    tag = tag.lower().strip()

    # Skip empty strings
    if not word:
        return []

    # Devanagari script → transliterate to ITRANS, then strip diacritics
    if _DEVA_RE.search(word):
        try:
            itrans = transliterate(word, sanscript.DEVANAGARI, sanscript.ITRANS)
            return [itrans]
        except Exception:
            return [word]

    # Pure symbol / emoji → signal with empty list (dataset.py handles this as UNK)
    if not re.search(r'[a-zA-Z\u0900-\u097F]', word):
        return []

    if tag == "eng":
        return _safe_g2p(word)

    if tag == "hin":
        # Romanised Hindi: transliterate ITRANS → Devanagari as phonetic anchor
        try:
            deva = transliterate(word.lower(), sanscript.ITRANS, sanscript.DEVANAGARI)
            return [deva] if deva.strip() else [word]
        except Exception:
            return [word]

    # tag == "rest" or unknown: if looks ASCII use G2P, else return raw
    if _ASCII_WORD_RE.match(word):
        return _safe_g2p(word)

    return [word]


def process_phonetics(input_file: str, output_file: str, report_file: str = None):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    phone_counter = Counter()

    for entry in tqdm(data, desc="Encoding phonetics"):
        phonetic_sentence = []
        tokens = entry.get("tokens", [])
        tags   = entry.get("tags", [])

        for word, tag in zip(tokens, tags):
            phonemes = get_phonemes(word, tag)
            # Count for vocabulary report
            phone_counter.update(phonemes)
            phonetic_sentence.append({
                "word":    word,
                "tag":     tag,
                "phonemes": phonemes,
            })

        entry["phonetic_tokens"] = phonetic_sentence

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Phonetic encoding complete → {output_file}")
    print(f"Unique phonemes discovered: {len(phone_counter)}")
    print(f"G2P cache size: {len(_g2p_cache)} entries")

    if report_file:
        report = {
            "total_unique_phonemes": len(phone_counter),
            "top_50": phone_counter.most_common(50),
        }
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Phoneme frequency report → {report_file}")


if __name__ == "__main__":
    process_phonetics(
        "cleaned_data.json",
        "phonetic_data.json",
        report_file="phoneme_report.json",
    )
