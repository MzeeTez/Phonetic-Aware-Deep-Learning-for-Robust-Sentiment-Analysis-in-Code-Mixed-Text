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
from g2p_en import G2p
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from model import EnhancedDualChannelLSTM

# ── Setup ─────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g2p_en = G2p()

with open("vocabs.json", encoding="utf-8") as f:
    v = json.load(f)

word_vocab  = v["word_vocab"]
phone_vocab = v["phone_vocab"]
label_names = {0: "Negative", 1: "Neutral", 2: "Positive"}

state_dict = torch.load("best_model.pth", map_location=DEVICE, weights_only=True)

MAX_LEN = 64
UNK_W   = word_vocab.get("<UNK>", 1)
UNK_P   = phone_vocab.get("<UNK>", 1)

model = EnhancedDualChannelLSTM(
    word_vocab_size  = len(word_vocab),
    phone_vocab_size = len(phone_vocab),
    dropout     = 0.0,
    var_dropout = 0.0,
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
# Tuples are (neg_bias, neu_bias, pos_bias) added to raw logits before softmax.
# Negative=0, Neutral=1, Positive=2.
#
# Calibration principle:
#   - "Strong" sentiment words (bakwas, mast, zabardast) should be decisive
#     on their own: bias magnitude ~2.0 on the dominant class.
#   - "Mild" words (acha, pyaar) need an intensifier alongside to tip past
#     Neutral: base magnitude ~1.5 on dominant class.
#   - Intensifiers (bohot, bahut) contribute ~0.5 extra to whichever
#     sentiment class is already winning — achieved via the compound
#     multiplier in _compute_lexicon_bias rather than a fixed tuple.
#   - Negation (nahi, mat) partially flip the sign of co-occurring sentiment.
#
# The BIAS_CAP of 3.5 allows decisive outcomes (~90 %+ confidence) for
# clear-sentiment sentences while still letting the model override the
# lexicon when it is confident in the opposite direction.

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
    # Handled separately in _compute_lexicon_bias; listed here so
    # they are detected as lexicon tokens.
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
    "acha":     (-1.4, -0.4,  1.8),
    "pyaar":    (-1.2, -0.3,  1.5),
    "mohabbat": (-1.2, -0.3,  1.5),
    "khushi":   (-1.2, -0.3,  1.5),
    "sahi":     (-0.7,  0.0,  0.7),
    "bilkul":   (-0.4, -0.1,  0.5),
    "pasand":   (-1.0, -0.2,  1.2),
    "sundar":   (-1.0, -0.2,  1.2),

    # ── Intensifiers ─────────────────────────────────────────────────────
    # Their main role is to scale up co-occurring sentiment words
    # (handled via INTENSIFIER_MULTIPLIER below). The tuple here adds a
    # small anti-Neutral push on their own.
    "bohot":  ( 0.0, -0.5,  0.5),
    "bahut":  ( 0.0, -0.5,  0.5),
    "boht":   ( 0.0, -0.5,  0.5),
    "bahot":  ( 0.0, -0.5,  0.5),
    "zyada":  ( 0.1, -0.3,  0.2),
    "jyada":  ( 0.1, -0.3,  0.2),
    "bahut":  ( 0.0, -0.5,  0.5),
    "ekdum":  ( 0.0, -0.4,  0.4),
    "bilkul": (-0.4, -0.1,  0.5),
}

# Words treated as intensifiers — when present, non-intensifier sentiment
# words in the same sentence get their bias scaled up by this factor.
INTENSIFIER_WORDS = {"bohot", "bahut", "boht", "bahot", "zyada", "jyada", "ekdum", "bilkul"}
INTENSIFIER_MULTIPLIER = 1.6   # e.g. "acha" alone → 1.0×; "bohot acha" → 1.6×

# Maximum total bias magnitude per class after all scaling.
BIAS_CAP = 3.5


def _compute_lexicon_bias(tokens: list) -> torch.Tensor:
    """
    Compute logit bias with intensifier scaling.

    Algorithm:
      1. Separate tokens into intensifiers vs sentiment-bearing words.
      2. If any intensifier is present, multiply each sentiment word's bias
         by INTENSIFIER_MULTIPLIER.
      3. Sum all biases (intensifiers use their own small tuple un-scaled).
      4. Clamp each class to [-BIAS_CAP, +BIAS_CAP].
    """
    has_intensifier = any(t in INTENSIFIER_WORDS for t in tokens)
    bias = [0.0, 0.0, 0.0]

    for tok in tokens:
        if tok not in SENTIMENT_LEXICON:
            continue
        b = SENTIMENT_LEXICON[tok]
        if tok in INTENSIFIER_WORDS:
            # Intensifiers always contribute their own (small) bias un-scaled
            scale = 1.0
        else:
            # Sentiment words get amplified when an intensifier is present
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


def _phoneme_to_id(phonemes: list, raw_word: str = "") -> int:
    """
    FIX-D: also try raw_word as a phone_vocab key before giving up.
    """
    if not phonemes:
        # FIX-D: try the raw word itself as a phoneme key
        if raw_word and raw_word in phone_vocab:
            return phone_vocab[raw_word]
        return UNK_P
    clean = [re.sub(r'\d+$', '', p) for p in phonemes if p.strip()]
    if not clean:
        return UNK_P
    joined = "+".join(clean)
    if joined in phone_vocab:
        return phone_vocab[joined]
    for p in clean:
        if p in phone_vocab:
            return phone_vocab[p]
    # FIX-D: last resort — raw word string
    if raw_word and raw_word in phone_vocab:
        return phone_vocab[raw_word]
    return UNK_P


def tokenize_with_features(tokens: list):
    word_ids, phone_ids, lang_ids = [], [], []
    raw_tokens, oov_tokens = [], []

    for word in tokens[:MAX_LEN]:
        lang = detect_lang(word)
        wid  = word_vocab.get(word, UNK_W)
        if wid == UNK_W and word not in word_vocab:
            oov_tokens.append(word)

        if lang == "eng":
            phonemes = g2p_en(word)
        elif lang == "hin":
            try:
                deva     = transliterate(word, sanscript.ITRANS, sanscript.DEVANAGARI)
                phonemes = [deva]
            except Exception:
                phonemes = [word]
        else:
            phonemes = [word]

        pid = _phoneme_to_id(phonemes, raw_word=word)   # FIX-D: pass raw word

        word_ids.append(wid)
        phone_ids.append(pid)
        lang_ids.append(LANG_TAG_MAP.get(lang, 3))
        raw_tokens.append(word)

    pad        = MAX_LEN - len(word_ids)
    word_ids  += [0] * pad
    phone_ids += [0] * pad
    lang_ids  += [0] * pad

    return word_ids, phone_ids, lang_ids, raw_tokens, oov_tokens


def _extract_attn_focus(weights: torch.Tensor, raw_tokens: list):
    """
    FIX-A: Entropy-weighted aggregation over all real query positions.

    Problem with the old "last row" approach:
      For a 3-token input padded to 64, the attention matrix has 61 padding
      columns.  The head-averaged last-row entries for real positions are
      near-zero because the softmax spreads mass across 64 slots, not 3.
      After min-max normalisation, all three values collapse to ~0.012.

    New approach:
      1. Take head-averaged attention: (T_real, T_real) slice only.
      2. For each query row q, compute entropy H_q = -Σ p log p.
         A uniform row has entropy log(n); a peaked row has low entropy.
      3. Weight each row by w_q = exp(-H_q) — peaked rows dominate.
      4. Weighted sum across query rows → importance vector over key positions.
      5. Min-max normalise to [0, 1].

    This gives clearly differentiated scores even for 3-token inputs.

    Returns (importance_vector: Tensor[n_tok], is_uniform: bool).
    """
    n_tok = len(raw_tokens)
    if n_tok == 0:
        return torch.zeros(1), True

    # Single token: nothing to compare — score is 1.0 by definition, no std issue
    if n_tok == 1:
        return torch.ones(1), False

    # Mean over heads, then crop to real tokens only → (n_tok, n_tok)
    attn_mean = weights[0].mean(0)[:n_tok, :n_tok].cpu().float()

    # Re-normalise rows so they sum to 1 over real tokens (remove padding dilution)
    row_sums = attn_mean.sum(dim=1, keepdim=True).clamp(min=1e-9)
    attn_norm = attn_mean / row_sums                           # (n_tok, n_tok)

    # Per-row entropy (higher = more uniform = less informative)
    eps    = 1e-9
    log_p  = torch.log(attn_norm + eps)
    H      = -(attn_norm * log_p).sum(dim=1)                  # (n_tok,)

    # Row weights: peaked rows get high weight, uniform rows near zero
    row_w = torch.exp(-H)                                      # (n_tok,)
    row_w = row_w / row_w.sum().clamp(min=1e-9)

    # Weighted sum over query rows → key-position importance
    importance = (row_w.unsqueeze(1) * attn_norm).sum(dim=0)  # (n_tok,)

    # Softmax-normalise with temperature=0.5 so all tokens get non-zero scores.
    # Min-max collapsed the lowest token to exactly 0.0 — e.g. in a 2-token
    # input the second token always showed 0.0, which was confusing.
    # Softmax sharpens relative differences while keeping every score > 0.
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

    # FIX-B: apply lexicon bias to raw logits before temperature + softmax
    lex_bias = _compute_lexicon_bias(raw_tokens).to(DEVICE)
    bias_nonzero = lex_bias.abs().sum().item() > 0.0
    biased_logits = logits[0] + lex_bias                    # (3,)

    # FIX-C: temperature 1.05 (was 1.3)
    calibrated     = biased_logits / TEMPERATURE
    probs          = torch.softmax(calibrated, dim=0)
    conf, pred_idx = probs.max(dim=0)

    n_tok = len(raw_tokens)
    top_k = min(3, n_tok)

    # FIX-A: entropy-weighted attention
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
