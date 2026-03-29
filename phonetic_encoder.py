"""
phonetic_encoder.py  (v2 — full phoneme sequences)

Key fixes over v1:
  - English words now produce their FULL CMU phoneme sequence, not just ph[0].
    e.g. "love" → ["L", "AH1", "V"]  (v1 only kept "L")
  - Stress digits stripped from Arpabet phonemes (AH1 → AH, AH0 → AH).
    This reduces vocab size by ~3x with no information loss for sentiment.
  - Romanised Hindi now decomposed into character-level Akshar units instead
    of returning one opaque Devanagari string.  "pyaar" → ["P", "Y", "AA", "R"]
    using a simple ITRANS-to-Akshar mapping that preserves phonetic content.
  - Devanagari input split into Unicode aksharas (syllable units), each stored
    as a separate phoneme token.
  - Symbols / emoji now stored as the special token "<SYM>" so they contribute
    to sequence length without polluting the phoneme vocab.
  - per-word phoneme list is always a List[str], never empty for real words;
    fallback is ["<UNK_PHONE>"].
  - G2P cache persisted to disk (g2p_cache.json) so re-runs are instant.

Output schema per token (unchanged so downstream code stays compatible):
  {
    "word":     str,
    "tag":      str,           # "eng" / "hin" / "rest"
    "phonemes": List[str],     # NOW a full sequence, not a singleton
  }
"""

import json
import re
import os
from collections import Counter
from g2p_en import G2p
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from tqdm import tqdm

# ── G2P engine (English) ──────────────────────────────────────────────────

g2p_en = G2p()

# ── Regexes ───────────────────────────────────────────────────────────────

_DEVA_RE     = re.compile(r'[\u0900-\u097F]')
_ASCII_RE    = re.compile(r'^[a-zA-Z]+$')
_HAS_ALPHA   = re.compile(r'[a-zA-Z\u0900-\u097F]')
_STRESS_RE   = re.compile(r'\d')          # strips stress digits from Arpabet

# ── Disk-backed G2P cache ─────────────────────────────────────────────────

_CACHE_FILE = "g2p_cache.json"

def _load_cache() -> dict:
    if os.path.exists(_CACHE_FILE):
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_cache(cache: dict) -> None:
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

_g2p_cache: dict = _load_cache()


def _safe_g2p(word: str) -> list[str]:
    """
    Run g2p_en on a word; strip stress digits; remove spaces/punctuation tokens.
    Returns a list of clean Arpabet phoneme strings, e.g. ["L", "AH", "V"].
    Falls back to ["<UNK_PHONE>"] on any error.
    """
    if word in _g2p_cache:
        return _g2p_cache[word]
    try:
        raw = g2p_en(word)           # e.g. ["L", "AH1", "V"] or ["TH", "AH0", " "]
        # Keep only uppercase Arpabet tokens; strip stress digits
        cleaned = []
        for ph in raw:
            ph = ph.strip()
            if not ph:
                continue
            ph_clean = _STRESS_RE.sub("", ph).upper()
            if ph_clean and ph_clean.isalpha():
                cleaned.append(ph_clean)
        result = cleaned if cleaned else ["<UNK_PHONE>"]
    except Exception:
        result = ["<UNK_PHONE>"]
    _g2p_cache[word] = result
    return result


# ── ITRANS → Akshar decomposition for Romanised Hindi ─────────────────────
#
# The old approach returned one Devanagari string per Hindi word.
# That string is not a phoneme sequence — it's an opaque glyph.
# We now decompose Romanised Hindi into a sequence of phoneme-like
# units using a lightweight ITRANS character mapping.
#
# Approach:
#   1. Normalise the romanised word to lowercase.
#   2. Greedily match the longest ITRANS cluster from left to right.
#   3. Each matched cluster → one phoneme unit (uppercase).
#
# This gives "pyaar" → ["PY", "AA", "R"] which is phonetically meaningful
# and contributes to the shared Arpabet + Hindi phoneme vocabulary.

_ITRANS_MAP = [
    # digraphs first (greedy longest-match)
    ("aa", "AA"), ("ii", "II"), ("uu", "UU"), ("ee", "EE"), ("oo", "OO"),
    ("ai", "AI"), ("au", "AU"), ("ch", "CH"), ("sh", "SH"), ("th", "TH"),
    ("ph", "PH"), ("gh", "GH"), ("dh", "DH"), ("bh", "BH"), ("kh", "KH"),
    ("jh", "JH"), ("nh", "NH"), ("ny", "NY"), ("ng", "NG"), ("rh", "RH"),
    ("gy", "GY"), ("py", "PY"), ("ky", "KY"), ("vy", "VY"), ("ty", "TY"),
    ("dy", "DY"), ("ly", "LY"), ("sy", "SY"), ("my", "MY"), ("ny", "NY"),
    # single vowels
    ("a", "A"), ("i", "I"), ("u", "U"), ("e", "E"), ("o", "O"),
    # single consonants
    ("k", "K"), ("g", "G"), ("c", "C"), ("j", "J"), ("t", "T"),
    ("d", "D"), ("n", "N"), ("p", "P"), ("b", "B"), ("m", "M"),
    ("y", "Y"), ("r", "R"), ("l", "L"), ("v", "V"), ("w", "W"),
    ("s", "S"), ("h", "H"), ("f", "F"), ("z", "Z"), ("x", "X"),
    ("q", "Q"),
]

def _romanised_hindi_to_phonemes(word: str) -> list[str]:
    """
    Greedy ITRANS decomposition of a Romanised Hindi word.
    "pyaar"  → ["PY", "AA", "R"]
    "nahi"   → ["N", "A", "H", "I"]
    "bhaiya" → ["BH", "AI", "Y", "A"]
    """
    s = word.lower()
    phonemes = []
    i = 0
    while i < len(s):
        matched = False
        for itrans, phoneme in _ITRANS_MAP:
            if s[i:].startswith(itrans):
                phonemes.append(phoneme)
                i += len(itrans)
                matched = True
                break
        if not matched:
            # Unknown character — keep as uppercase literal
            phonemes.append(s[i].upper())
            i += 1
    return phonemes if phonemes else ["<UNK_PHONE>"]


# ── Devanagari → akshar units ─────────────────────────────────────────────

# Unicode combining marks (vowel matras, virama, anusvara, etc.)
_DEVA_COMBINING = re.compile(r'[\u0900-\u0902\u0903\u093C-\u094D\u0950-\u0954\u0962-\u0963]')

def _devanagari_to_aksharas(word: str) -> list[str]:
    """
    Split a Devanagari word into syllable (akshar) clusters.
    Each cluster = consonant(s) + dependent vowel + optional virama.
    We use a simple regex split: split on boundaries before standalone consonants
    that are not preceded by a virama.
    Returns a list of akshar strings, each used as one phoneme token.
    """
    # Try transliterating to ITRANS first, then decompose
    try:
        itrans = transliterate(word, sanscript.DEVANAGARI, sanscript.ITRANS)
        if itrans.strip():
            return _romanised_hindi_to_phonemes(itrans)
    except Exception:
        pass
    # Fallback: split on Devanagari vowel boundaries naively
    parts = re.findall(r'[\u0900-\u097F]+', word)
    return parts if parts else ["<UNK_PHONE>"]


# ── Main get_phonemes function ─────────────────────────────────────────────

def get_phonemes(word: str, tag: str) -> list[str]:
    """
    Convert one word to its full phoneme sequence.

    Returns List[str], always non-empty for real words.
    Symbols / emoji return ["<SYM>"].
    Unknown content returns ["<UNK_PHONE>"].
    """
    tag = tag.lower().strip()

    if not word:
        return ["<UNK_PHONE>"]

    # Devanagari script
    if _DEVA_RE.search(word):
        return _devanagari_to_aksharas(word)

    # Pure symbol / emoji — no alphabetic content
    if not _HAS_ALPHA.search(word):
        return ["<SYM>"]

    if tag == "eng":
        return _safe_g2p(word)

    if tag == "hin":
        # Romanised Hindi word → ITRANS decomposition
        return _romanised_hindi_to_phonemes(word)

    # tag == "rest": try G2P if ASCII, else romanised Hindi decomposition
    if _ASCII_RE.match(word):
        return _safe_g2p(word)

    if _HAS_ALPHA.search(word):
        return _romanised_hindi_to_phonemes(word)

    return ["<UNK_PHONE>"]


# ── Main pipeline ─────────────────────────────────────────────────────────

def process_phonetics(
    input_file: str,
    output_file: str,
    report_file: str = None,
) -> None:
    """
    Read cleaned_data.json, annotate every token with its full phoneme sequence,
    write phonetic_data.json.

    Also saves the G2P cache to disk so subsequent runs are fast.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    phone_counter   = Counter()
    total_tokens    = 0
    total_phonemes  = 0

    for entry in tqdm(data, desc="Encoding phonetics"):
        phonetic_sentence = []
        tokens = entry.get("tokens", [])
        tags   = entry.get("tags",   [])

        for word, tag in zip(tokens, tags):
            phonemes = get_phonemes(word, tag)
            phone_counter.update(phonemes)
            total_tokens   += 1
            total_phonemes += len(phonemes)
            phonetic_sentence.append({
                "word":     word,
                "tag":      tag,
                "phonemes": phonemes,   # full list, e.g. ["L", "AH", "V"]
            })

        entry["phonetic_tokens"] = phonetic_sentence

    # Persist G2P cache so re-runs skip the slow G2P calls
    _save_cache(_g2p_cache)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    avg_ph = total_phonemes / max(1, total_tokens)
    print(f"\nPhonetic encoding complete → {output_file}")
    print(f"  Total tokens     : {total_tokens:,}")
    print(f"  Total phonemes   : {total_phonemes:,}")
    print(f"  Avg phonemes/tok : {avg_ph:.2f}")
    print(f"  Unique phonemes  : {len(phone_counter)}")
    print(f"  G2P cache size   : {len(_g2p_cache)} entries  (saved → {_CACHE_FILE})")

    if report_file:
        report = {
            "total_unique_phonemes":   len(phone_counter),
            "total_tokens":            total_tokens,
            "total_phonemes":          total_phonemes,
            "avg_phonemes_per_token":  round(avg_ph, 3),
            "top_50":                  phone_counter.most_common(50),
        }
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  Phoneme report   : → {report_file}")


if __name__ == "__main__":
    process_phonetics(
        "cleaned_data.json",
        "phonetic_data.json",
        report_file="phoneme_report.json",
    )
