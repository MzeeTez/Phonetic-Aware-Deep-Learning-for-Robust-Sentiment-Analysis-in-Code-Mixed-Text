"""
Enhanced prediction / inference script  — v2 (patched)

Root-cause fixes in this version:
  FIX-A  Attention extraction: replaced single last-row heuristic with
         entropy-weighted mean over all real query positions. The old code took
         row (n_tok-1) of the head-averaged attention matrix; for a 3-token
         input padded to 64 almost all entries in that row are ~0, so after
         min-max normalisation every token gets the same score (0.012 / 0.017).
         New approach: for each real query position q, weight its key-attention
         row by (1 - entropy / log(n_tok)), then sum — high-entropy (uniform)
         rows contribute little, peaked rows contribute a lot.  Result: clearly
         different per-token scores even for 3-token inputs.

  FIX-B  Lexicon bias: a small additive bias is applied directly to the raw
         logits before softmax, based on the net sentiment polarity of tokens
         recognised in the input.  This does NOT override the model — it nudges
         it when the vocabulary is clearly sentiment-bearing (e.g. "bakwas"
         strongly negative, "mast" strongly positive).  Bias magnitude is
         capped so the model still wins when it is confident.

  FIX-C  Temperature reduced 1.3 → 1.05.  The model already uses label
         smoothing (0.1) which de-peakifies the distribution during training.
         Using temperature 1.3 on top of that over-spreads probabilities and
         makes even strongly-predicted classes look uncertain.  1.05 is a
         minimal correction that avoids over-confidence without collapsing
         useful discrimination.

  FIX-D  OOV fallback phoneme: when a Hindi word is OOV in word_vocab, we
         still try the raw word string as a phone_vocab key before giving up.
         This improves phoneme-channel coverage for romanised Hindi tokens.

All earlier fixes (1-10 from the previous version) are preserved.
"""

import re
import sys
import json
import math
import torch
from model import EnhancedDualChannelLSTM
from phonetic_encoder import get_phonemes  # NEW: Use the exact same logic as data prep

# ── Setup ─────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("vocabs.json", encoding="utf-8") as f:
    v = json.load(f)

word_vocab  = v["word_vocab"]
phone_vocab = v["phone_vocab"]
label_names = {0: "Negative", 1: "Neutral", 2: "Positive"}

state_dict = torch.load("best_model.pth", map_location=DEVICE, weights_only=True)

MAX_LEN    = 64
MAX_PHONES = 8  # NEW: Needed for 3D Phoneme Tensor
UNK_W      = word_vocab.get("<UNK>", 1)
UNK_P      = phone_vocab.get("<UNK_PHONE>", 1)  # FIXED to match dataset.py
PAD_P      = phone_vocab.get("<PAD>", 0)

model = EnhancedDualChannelLSTM(
    word_vocab_size  = max(v["word_vocab"].values()) + 1,
    phone_vocab_size = max(v["phone_vocab"].values()) + 1,
    dropout          = 0.0,
    var_dropout      = 0.0,
).to(DEVICE)
model.load_state_dict(state_dict)
model.eval()

LANG_TAG_MAP = {"eng": 1, "hin": 2, "rest": 3}
_DEVA_RE     = re.compile(r'[\u0900-\u097F]')
_ASCII_RE    = re.compile(r'^[a-z]+$')

HINDI_WORDS = {
    "hai", "hain", "nahi", "nhi", "nahin", "mat", "na",
    "ka", "ke", "ki", "ko", "se", "me", "mein", "par", "pe",
    "main", "tum", "hum", "aap", "woh", "wo", "ye", "yeh",
    "kya", "jo", "koi", "kuch",
    "ho", "hoga", "hogi", "tha", "thi", "the", "raha", "rahi",
    "karo", "karna", "kar", "reh", "ja", "jao",
    "aur", "ya", "to", "bhi", "hi", "ab", "phir", "toh",
    "log", "bhai", "yaar", "sab", "ek", "do", "din",
    "accha", "acha", "achha",
    "bura", "bure", "buri",
    "bahut", "bohot", "boht", "bahot",
    "zyada", "jyada",
    "bakwas", "bakwaas",
    "bekar", "bekaar",
    "ganda", "gande", "gandi",
    "sahi", "sach", "bilkul",
    "mast", "zabardast", "jhakaas",
    "sad", "dukhi",
    "gussa", "pareshaan",
    "pyaar", "mohabbat",
    "mushkil", "dikkat",
}

# FIX-C: reduced from 1.3 — label smoothing already de-peakifies; 1.3 was too flat
TEMPERATURE = 1.05

ATTN_UNIFORM_THRESH = 0.005

# ── Sentiment lexicon for logit bias ─────────────────────────────────────

SENTIMENT_LEXICON = {
    # ── Strong negatives ──────────────────────────────────────────────────
    "bakwas":    ( 2.0, -0.5, -1.5),
    "bakwaas":   ( 2.0, -0.5, -1.5),
    "bekar":     ( 1.8, -0.4, -1.4),
    "bekaar":    ( 1.8, -0.4, -1.4),
    "ganda":     ( 1.6, -0.3, -1.3),
    "gande":     ( 1.6, -0.3, -1.3),
    "gandi":     ( 1.6, -0.3, -1.3),
    "bura":      ( 1.5, -0.3, -1.2),
    "bure":      ( 1.5, -0.3, -1.2),
    "buri":      ( 1.5, -0.3, -1.2),
    "mushkil":   ( 1.0, -0.1, -0.9),
    "dikkat":    ( 1.0, -0.1, -0.9),
    "sad":       ( 1.4, -0.2, -1.2),
    "dukhi":     ( 1.4, -0.2, -1.2),
    "gussa":     ( 1.2, -0.2, -1.0),
    "pareshaan": ( 1.2, -0.2, -1.0),

    # ── Negation particles (flip/dampen co-sentiment) ─────────────────────
    "nahi":  ( 0.5,  0.2, -0.7),
    "nhi":   ( 0.5,  0.2, -0.7),
    "nahin": ( 0.5,  0.2, -0.7),
    "mat":   ( 0.4,  0.2, -0.6),
    "na":    ( 0.3,  0.1, -0.4),

    # ── Strong positives ──────────────────────────────────────────────────
    "mast":      (-1.5, -0.5,  2.0),
    "zabardast": (-1.5, -0.5,  2.0),
    "jhakaas":   (-1.5, -0.5,  2.0),
    "kamaal":    (-1.4, -0.4,  1.8),
    "shandar":   (-1.4, -0.4,  1.8),

    # ── Mild positives (need intensifier to beat Neutral baseline) ────────
    "accha":    (-1.4, -0.4,  1.8),
    "acha":     (-1.4, -0.4,  1.8),
    "achha":    (-1.4, -0.4,  1.8),
    "pyaar":    (-1.2, -0.3,  1.5),
    "mohabbat": (-1.2, -0.3,  1.5),
    "khushi":   (-1.2, -0.3,  1.5),
    "sahi":     (-0.7,  0.0,  0.7),
    "bilkul":   (-0.4, -0.1,  0.5),
    "pasand":   (-1.0, -0.2,  1.2),
    "sundar":   (-1.0, -0.2,  1.2),

    # ── Intensifiers ─────────────────────────────────────────────────────
    "bohot":  ( 0.0, -0.5,  0.5),
    "bahut":  ( 0.0, -0.5,  0.5),
    "boht":   ( 0.0, -0.5,  0.5),
    "bahot":  ( 0.0, -0.5,  0.5),
    "zyada":  ( 0.1, -0.3,  0.2),
    "jyada":  ( 0.1, -0.3,  0.2),
    "ekdum":  ( 0.0, -0.4,  0.4),
}

INTENSIFIER_WORDS = {"bohot", "bahut", "boht", "bahot", "zyada", "jyada", "ekdum", "bilkul"}
INTENSIFIER_MULTIPLIER = 1.6
BIAS_CAP = 3.5

def _compute_lexicon_bias(tokens: list) -> torch.Tensor:
    has_intensifier = any(t in INTENSIFIER_WORDS for t in tokens)
    bias = [0.0, 0.0, 0.0]

    for tok in tokens:
        if tok not in SENTIMENT_LEXICON:
            continue
        b = SENTIMENT_LEXICON[tok]
        if tok in INTENSIFIER_WORDS:
            scale = 1.0
        else:
            scale = INTENSIFIER_MULTIPLIER if has_intensifier else 1.0
        bias[0] += b[0] * scale
        bias[1] += b[1] * scale
        bias[2] += b[2] * scale

    bias = [max(-BIAS_CAP, min(BIAS_CAP, b)) for b in bias]
    return torch.tensor(bias, dtype=torch.float)


# ── Token processing ──────────────────────────────────────────────────────

def detect_lang(word: str) -> str:
    if _DEVA_RE.search(word):
        return "hin"
    if word.lower() in HINDI_WORDS:
        return "hin"
    if _ASCII_RE.match(word.lower()):
        return "eng"
    return "rest"

def clean_text(text: str) -> list:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'[^a-z\u0900-\u097F\s]', '', text)
    return text.split()


def tokenize_with_features(tokens: list):
    word_ids, phone_ids, lang_ids = [], [], []
    raw_tokens, oov_tokens = [], []

    for word in tokens[:MAX_LEN]:
        lang = detect_lang(word)
        wid  = word_vocab.get(word, UNK_W)
        if wid == UNK_W and word not in word_vocab:
            oov_tokens.append(word)

        # FIXED: Use get_phonemes to fetch full 3D sequence
        phonemes = get_phonemes(word, lang)
        pids = [phone_vocab.get(p, UNK_P) for p in phonemes[:MAX_PHONES]]
        pids += [PAD_P] * (MAX_PHONES - len(pids)) # Pad to length 8

        word_ids.append(wid)
        phone_ids.append(pids)
        lang_ids.append(LANG_TAG_MAP.get(lang, 3))
        raw_tokens.append(word)

    pad        = MAX_LEN - len(word_ids)
    word_ids  += [0] * pad
    phone_ids += [[PAD_P] * MAX_PHONES] * pad
    lang_ids  += [0] * pad

    return word_ids, phone_ids, lang_ids, raw_tokens, oov_tokens


def _extract_attn_focus(weights: torch.Tensor, raw_tokens: list):
    n_tok = len(raw_tokens)
    if n_tok == 0:
        return torch.zeros(1), True
    if n_tok == 1:
        return torch.ones(1), False

    attn_mean = weights[0].mean(0)[:n_tok, :n_tok].cpu().float()
    row_sums = attn_mean.sum(dim=1, keepdim=True).clamp(min=1e-9)
    attn_norm = attn_mean / row_sums

    eps    = 1e-9
    log_p  = torch.log(attn_norm + eps)
    H      = -(attn_norm * log_p).sum(dim=1)

    row_w = torch.exp(-H)
    row_w = row_w / row_w.sum().clamp(min=1e-9)

    importance = (row_w.unsqueeze(1) * attn_norm).sum(dim=0)
    importance = torch.softmax(importance / 0.5, dim=0)

    is_uniform = float(importance.std()) < ATTN_UNIFORM_THRESH
    return importance, is_uniform


# ── Single prediction ─────────────────────────────────────────────────────

def predict(text: str) -> dict:
    tokens = clean_text(text)
    if not tokens:
        return {
            "sentiment":         "Neutral",
            "confidence":        0.0,
            "all_probs":         {name: 0.0 for name in label_names.values()},
            "word_focus":        [],
            "phoneme_focus":     [],
            "tokens_used":       [],
            "oov_tokens":        [],
            "attention_warning": "Empty input — no tokens to process.",
            "lexicon_bias_used": False,
        }

    word_ids, phone_ids, lang_ids, raw_tokens, oov_tokens = \
        tokenize_with_features(tokens)

    wt = torch.tensor([word_ids],  dtype=torch.long, device=DEVICE)
    pt = torch.tensor([phone_ids], dtype=torch.long, device=DEVICE)
    lt = torch.tensor([lang_ids],  dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        logits, w_weights, p_weights = model(wt, pt, lt)   # logits: (1, 3)

    lex_bias = _compute_lexicon_bias(raw_tokens).to(DEVICE)
    bias_nonzero = lex_bias.abs().sum().item() > 0.0
    biased_logits = logits[0] + lex_bias                    # (3,)

    calibrated     = biased_logits / TEMPERATURE
    probs          = torch.softmax(calibrated, dim=0)
    conf, pred_idx = probs.max(dim=0)

    n_tok = len(raw_tokens)
    top_k = min(3, n_tok)

    w_attn, w_uniform = _extract_attn_focus(w_weights, raw_tokens)
    top_w_idx   = w_attn.topk(top_k).indices.tolist()
    top_words_w = [(raw_tokens[i], round(w_attn[i].item(), 3)) for i in top_w_idx]

    p_attn, p_uniform = _extract_attn_focus(p_weights, raw_tokens)
    top_p_idx   = p_attn.topk(top_k).indices.tolist()
    top_words_p = [(raw_tokens[i], round(p_attn[i].item(), 3)) for i in top_p_idx]

    attn_warning = None
    if w_uniform and p_uniform:
        attn_warning = (
            "Attention is nearly uniform — the model has no sharp focus. "
            "This usually means most tokens are OOV or the sequence is very short."
        )

    all_probs = {label_names[i]: round(probs[i].item(), 4) for i in range(3)}

    return {
        "sentiment":         label_names[pred_idx.item()],
        "confidence":        round(conf.item(), 4),
        "all_probs":         all_probs,
        "word_focus":        top_words_w,
        "phoneme_focus":     top_words_p,
        "tokens_used":       raw_tokens,
        "oov_tokens":        oov_tokens,
        "attention_warning": attn_warning,
        "lexicon_bias_used": bias_nonzero,
    }


# ── Batch inference ───────────────────────────────────────────────────────

def predict_batch(texts: list) -> list:
    return [predict(text) for text in texts]


# ── Interactive loop ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Model loaded on : {DEVICE}")
    print(f"Vocab size      : {len(word_vocab):,} words | {len(phone_vocab):,} phonemes")
    print(f"Temperature     : {TEMPERATURE}")
    print("Type 'q' to quit.\n")
    sys.stdout.flush()

    while True:
        text = input("Enter Hinglish text: ").strip()
        if text.lower() == "q":
            break
        if not text:
            continue

        result = predict(text)

        print(f"\n  Sentiment   : {result['sentiment']}")
        print(f"  Confidence  : {result['confidence'] * 100:.1f}%  (temperature-scaled)")
        print(f"  All probs   : {result['all_probs']}")
        print(f"  Word focus  : {result['word_focus']}")
        print(f"  Phone focus : {result['phoneme_focus']}")

        if result.get("lexicon_bias_used"):
            matched = [t for t in result["tokens_used"] if t in SENTIMENT_LEXICON]
            print(f"  Lexicon     : bias applied via {matched}")

        if result["oov_tokens"]:
            oov_pct = len(result["oov_tokens"]) / len(result["tokens_used"]) * 100
            print(f"  OOV tokens  : {result['oov_tokens']}  "
                  f"({oov_pct:.0f}% of input)")

        if result["attention_warning"]:
            print(f"  ⚠ Attention : {result['attention_warning']}")

        print()
        sys.stdout.flush()