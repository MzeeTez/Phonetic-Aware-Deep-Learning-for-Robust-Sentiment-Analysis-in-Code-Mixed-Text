"""
dataset_stats.py
Generates Table 1 (Dataset Statistics) for the paper.

Run AFTER phonetic_encoder.py and vocab_builder.py have been executed.

Outputs:
  dataset_stats.json   — machine-readable full statistics
  dataset_stats.txt    — paper-ready LaTeX table + prose summary

Statistics reported:
  Overall:
    - Total sentences, total tokens
    - Label distribution (N / count / %)
    - Avg tokens per sentence, std, min, max
    - Vocabulary sizes (word, phoneme)
    - OOV rates
    - Avg phonemes per token

  Per split (train / val / test):
    - Sentence count, token count
    - Label distribution
    - Avg sequence length
    - Language mix ratio (eng / hin / rest %)

  Linguistic:
    - Type-token ratio (lexical diversity)
    - % sentences with at least one code-switch point
    - Avg code-switch points per sentence
    - Top-10 most frequent words per language tag
"""

import json
import math
from collections import Counter, defaultdict


# ── Load data ─────────────────────────────────────────────────────────────

with open("phonetic_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("vocabs.json", "r", encoding="utf-8") as f:
    v = json.load(f)
    word_vocab  = v["word_vocab"]
    phone_vocab = v["phone_vocab"]
    label_map   = v["label_map"]   # {"negative":0, "neutral":1, "positive":2}

# Reverse label map for display
rev_label = {0: "Negative", 1: "Neutral", 2: "Positive"}

print(f"Loaded {len(data)} entries.")

# ── Stratified split (must match train.py seed) ────────────────────────────

import random
from collections import defaultdict as dd

SEED = 42
VAL_RATIO  = 0.1
TEST_RATIO = 0.1

rng     = random.Random(SEED)
buckets = dd(list)
for i, item in enumerate(data):
    lbl = label_map.get(item.get("sentiment", "neutral"), 1)
    buckets[lbl].append(i)

train_idx, val_idx, test_idx = [], [], []
for lbl, idxs in buckets.items():
    rng.shuffle(idxs)
    n      = len(idxs)
    n_test = max(1, int(n * TEST_RATIO))
    n_val  = max(1, int(n * VAL_RATIO))
    test_idx.extend(idxs[:n_test])
    val_idx.extend(idxs[n_test: n_test + n_val])
    train_idx.extend(idxs[n_test + n_val:])

split_map = {}
for i in train_idx: split_map[i] = "train"
for i in val_idx:   split_map[i] = "val"
for i in test_idx:  split_map[i] = "test"


# ── Core per-sentence features ────────────────────────────────────────────

def analyse_entry(item, idx):
    tokens   = item.get("phonetic_tokens", [])
    T        = len(tokens)
    label    = label_map.get(item.get("sentiment", "neutral"), 1)
    words    = [t["word"].lower() for t in tokens]
    tags     = [t.get("tag", "rest").lower() for t in tokens]
    phonemes = [p for t in tokens for p in t.get("phonemes", [])]

    # Code-switch points: positions where tag changes
    cs_points = sum(1 for i in range(1, T) if tags[i] != tags[i-1])

    # Language mix
    n_eng  = tags.count("eng")
    n_hin  = tags.count("hin")
    n_rest = tags.count("rest")

    # OOV
    oov_count = sum(1 for w in words if word_vocab.get(w, 1) == 1)

    return {
        "idx":        idx,
        "split":      split_map[idx],
        "label":      label,
        "n_tokens":   T,
        "n_phonemes": len(phonemes),
        "words":      words,
        "tags":       tags,
        "n_eng":      n_eng,
        "n_hin":      n_hin,
        "n_rest":     n_rest,
        "cs_points":  cs_points,
        "oov_count":  oov_count,
    }

records = [analyse_entry(item, i) for i, item in enumerate(data)]

# ── Helper: aggregate stats over a list of records ────────────────────────

def agg(recs):
    lengths = [r["n_tokens"] for r in recs]
    n       = len(lengths)
    mean_l  = sum(lengths) / max(1, n)
    var_l   = sum((x - mean_l) ** 2 for x in lengths) / max(1, n)
    std_l   = math.sqrt(var_l)

    label_counts = Counter(r["label"] for r in recs)
    total_tokens  = sum(lengths)
    total_phones  = sum(r["n_phonemes"] for r in recs)
    total_eng     = sum(r["n_eng"]  for r in recs)
    total_hin     = sum(r["n_hin"]  for r in recs)
    total_rest    = sum(r["n_rest"] for r in recs)
    cs_per_sent   = sum(r["cs_points"] for r in recs) / max(1, n)
    cs_sentences  = sum(1 for r in recs if r["cs_points"] > 0)
    oov_total     = sum(r["oov_count"] for r in recs)
    oov_rate      = oov_total / max(1, total_tokens)

    # Type-token ratio
    all_words = [w for r in recs for w in r["words"]]
    ttr = len(set(all_words)) / max(1, len(all_words))

    return {
        "n_sentences":     n,
        "n_tokens":        total_tokens,
        "n_phonemes":      total_phones,
        "avg_len":         round(mean_l, 2),
        "std_len":         round(std_l,  2),
        "min_len":         min(lengths) if lengths else 0,
        "max_len":         max(lengths) if lengths else 0,
        "label_counts":    {rev_label[k]: v for k, v in sorted(label_counts.items())},
        "label_pct":       {rev_label[k]: round(v / n * 100, 1)
                            for k, v in sorted(label_counts.items())},
        "pct_eng":         round(total_eng  / max(1, total_tokens) * 100, 1),
        "pct_hin":         round(total_hin  / max(1, total_tokens) * 100, 1),
        "pct_rest":        round(total_rest / max(1, total_tokens) * 100, 1),
        "avg_cs_per_sent": round(cs_per_sent, 2),
        "pct_cs_sents":    round(cs_sentences / max(1, n) * 100, 1),
        "avg_phones_per_tok": round(total_phones / max(1, total_tokens), 2),
        "oov_rate_pct":    round(oov_rate * 100, 2),
        "type_token_ratio": round(ttr, 4),
    }

overall     = agg(records)
train_stats = agg([r for r in records if r["split"] == "train"])
val_stats   = agg([r for r in records if r["split"] == "val"])
test_stats  = agg([r for r in records if r["split"] == "test"])

# ── Top words per language tag ────────────────────────────────────────────

eng_words  = Counter(w for r in records for w, t in zip(r["words"], r["tags"]) if t == "eng")
hin_words  = Counter(w for r in records for w, t in zip(r["words"], r["tags"]) if t == "hin")

top_eng = eng_words.most_common(10)
top_hin = hin_words.most_common(10)

# ── Assemble output ───────────────────────────────────────────────────────

output = {
    "overall":     overall,
    "train":       train_stats,
    "val":         val_stats,
    "test":        test_stats,
    "vocab_sizes": {
        "word_vocab":  len(word_vocab),
        "phone_vocab": len(phone_vocab),
    },
    "top_eng_words": top_eng,
    "top_hin_words": top_hin,
}

with open("dataset_stats.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# ── Paper-ready output ────────────────────────────────────────────────────

lines = []
lines.append("=" * 72)
lines.append("DATASET STATISTICS  —  Hinglish Sentiment (SentiMix)")
lines.append("=" * 72)

def fmt_split(name, s):
    lines.append(f"\n{'─'*30}  {name.upper()}  {'─'*30}")
    lines.append(f"  Sentences          : {s['n_sentences']:,}")
    lines.append(f"  Tokens             : {s['n_tokens']:,}")
    lines.append(f"  Seq len  μ ± σ     : {s['avg_len']} ± {s['std_len']}  "
                 f"[min {s['min_len']}, max {s['max_len']}]")
    lines.append(f"  Label distribution :")
    for lbl, cnt in s["label_counts"].items():
        lines.append(f"    {lbl:<10} {cnt:>5}  ({s['label_pct'][lbl]:.1f}%)")
    lines.append(f"  Lang mix (tokens)  : ENG {s['pct_eng']}%  "
                 f"HIN {s['pct_hin']}%  REST {s['pct_rest']}%")
    lines.append(f"  Code-switch points : {s['avg_cs_per_sent']} avg/sent  "
                 f"({s['pct_cs_sents']}% sents have ≥1 switch)")
    lines.append(f"  Avg phonemes/token : {s['avg_phones_per_tok']}")
    lines.append(f"  OOV rate           : {s['oov_rate_pct']}%")
    lines.append(f"  Type-token ratio   : {s['type_token_ratio']}")

fmt_split("overall", overall)
fmt_split("train",   train_stats)
fmt_split("val",     val_stats)
fmt_split("test",    test_stats)

lines.append(f"\n{'─'*72}")
lines.append(f"  Word  vocab size : {len(word_vocab):,}")
lines.append(f"  Phone vocab size : {len(phone_vocab):,}")

lines.append(f"\n  Top-10 English tokens : {', '.join(w for w,_ in top_eng)}")
lines.append(f"  Top-10 Hindi tokens   : {', '.join(w for w,_ in top_hin)}")

# LaTeX table
lines.append(f"\n{'─'*72}")
lines.append("LaTeX table (paste into paper):\n")
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Dataset statistics for the Hinglish SentiMix corpus.}")
lines.append(r"\label{tab:data-stats}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{lrrr}")
lines.append(r"\toprule")
lines.append(r"\textbf{Statistic} & \textbf{Train} & \textbf{Val} & \textbf{Test} \\")
lines.append(r"\midrule")

def r(s): return str(s)

rows = [
    ("Sentences",          train_stats["n_sentences"],   val_stats["n_sentences"],   test_stats["n_sentences"]),
    ("Tokens",             train_stats["n_tokens"],       val_stats["n_tokens"],       test_stats["n_tokens"]),
    ("Avg seq len",        train_stats["avg_len"],        val_stats["avg_len"],        test_stats["avg_len"]),
    ("Negative (\%)",      train_stats["label_pct"]["Negative"], val_stats["label_pct"]["Negative"], test_stats["label_pct"]["Negative"]),
    ("Neutral (\%)",       train_stats["label_pct"]["Neutral"],  val_stats["label_pct"]["Neutral"],  test_stats["label_pct"]["Neutral"]),
    ("Positive (\%)",      train_stats["label_pct"]["Positive"], val_stats["label_pct"]["Positive"], test_stats["label_pct"]["Positive"]),
    ("ENG tokens (\%)",    train_stats["pct_eng"],        val_stats["pct_eng"],        test_stats["pct_eng"]),
    ("HIN tokens (\%)",    train_stats["pct_hin"],        val_stats["pct_hin"],        test_stats["pct_hin"]),
    ("CS points / sent",   train_stats["avg_cs_per_sent"], val_stats["avg_cs_per_sent"], test_stats["avg_cs_per_sent"]),
    ("Phonemes / token",   train_stats["avg_phones_per_tok"], val_stats["avg_phones_per_tok"], test_stats["avg_phones_per_tok"]),
    ("OOV rate (\%)",      train_stats["oov_rate_pct"],   val_stats["oov_rate_pct"],   test_stats["oov_rate_pct"]),
]
for label, tr, va, te in rows:
    lines.append(f"{label} & {r(tr)} & {r(va)} & {r(te)} \\\\")

lines.append(r"\midrule")
lines.append(f"Word vocab & \\multicolumn{{3}}{{c}}{{{len(word_vocab):,}}} \\\\")
lines.append(f"Phone vocab & \\multicolumn{{3}}{{c}}{{{len(phone_vocab):,}}} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

report = "\n".join(lines)
print(report)

with open("dataset_stats.txt", "w", encoding="utf-8") as f:
    f.write(report + "\n")

print("\nSaved → dataset_stats.json")
print("Saved → dataset_stats.txt")
