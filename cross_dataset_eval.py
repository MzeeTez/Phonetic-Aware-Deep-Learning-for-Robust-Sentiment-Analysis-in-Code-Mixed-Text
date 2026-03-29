"""
cross_dataset_eval.py
Zero-shot cross-dataset evaluation:
  Train set  : SentiMix  (model already trained — loads best_model.pth)
  Test  set  : LinCE Hinglish Sentiment (test.conll)

No retraining.  The model sees LinCE data for the first time at inference.

Outputs:
  cross_dataset_results.json   — full metrics
  cross_dataset_results.txt    — paper-ready comparison table

Steps performed:
  1. Load LinCE test.conll via lince_loader.py
  2. Preprocess (clean_token from preprocess.py)
  3. Run phonetic_encoder.py on the fly (uses cached G2P)
  4. Encode with the SentiMix vocabs.json (OOV tokens → <UNK>)
  5. Run model.forward() — same best_model.pth trained on SentiMix
  6. Report accuracy, macro-F1, per-class F1, confusion matrix
  7. Compare against SentiMix test-set numbers for the delta

Usage:
  python cross_dataset_eval.py --lince_test lince/hinglish_sentiment/test.conll

If you also have the SentiMix evaluation_results.json from evaluate.py,
the script will automatically include the in-domain vs cross-domain comparison.
"""

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score,
)
from collections import Counter

# ── Project imports ────────────────────────────────────────────────────────
from lince_loader    import load_lince_data
from preprocess      import clean_token
from phonetic_encoder import get_phonemes
from model           import EnhancedDualChannelLSTM

LABEL_NAMES  = ["Negative", "Neutral", "Positive"]
LABEL_MAP    = {"negative": 0, "neutral": 1, "positive": 2}
LANG_TAG_MAP = {"eng": 1, "hin": 2, "rest": 3}
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN  = 64
MAX_PHONES   = 8


# ── CLI ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--lince_test", default="lince/hinglish_sentiment/test.conll",
                    help="Path to LinCE test.conll")
parser.add_argument("--model_path",  default="best_model.pth")
parser.add_argument("--vocab_path",  default="vocabs.json")
parser.add_argument("--sentimix_results", default="evaluation_results.json",
                    help="Optional: in-domain results for delta table")
args = parser.parse_args()


# ── Load vocabs + model ────────────────────────────────────────────────────

print(f"Device: {DEVICE}")
with open(args.vocab_path, encoding="utf-8") as f:
    v = json.load(f)
word_vocab  = v["word_vocab"]
phone_vocab = v["phone_vocab"]

state_dict = torch.load(args.model_path, map_location=DEVICE, weights_only=True)
model = EnhancedDualChannelLSTM(
    word_vocab_size  = len(word_vocab),
    phone_vocab_size = len(phone_vocab),
    dropout=0.0, var_dropout=0.0,
).to(DEVICE)
model.load_state_dict(state_dict)
model.eval()
print(f"Model loaded from {args.model_path}")


# ── Load + preprocess LinCE ────────────────────────────────────────────────

raw_sentences = load_lince_data(args.lince_test)

def encode_sentence(sent):
    """
    Preprocess + phonetically encode one LinCE sentence, then map to IDs.
    Returns (word_ids, phone_ids, lang_ids, label) as lists.
    """
    tokens   = [clean_token(w) for w in sent["tokens"]]
    tags     = sent["tags"]
    label    = LABEL_MAP.get(sent["sentiment"], 1)

    # Zip and drop empty tokens
    pairs = [(w, t) for w, t in zip(tokens, tags) if w]

    # Truncate
    pairs = pairs[:MAX_SEQ_LEN]
    T     = len(pairs)

    word_ids_  = []
    phone_ids_ = []
    lang_ids_  = []

    unk_w = word_vocab.get("<UNK>", 1)
    unk_p = phone_vocab.get("<UNK>", 1)
    pad_p = phone_vocab.get("<PAD>", 0)

    for word, tag in pairs:
        # Word ID
        word_ids_.append(word_vocab.get(word, unk_w))
        # Phoneme IDs — full sequence, padded to MAX_PHONES
        phonemes = get_phonemes(word, tag)
        pids = [phone_vocab.get(p, unk_p) for p in phonemes[:MAX_PHONES]]
        pids += [pad_p] * (MAX_PHONES - len(pids))
        phone_ids_.append(pids)
        # Lang tag
        lang_ids_.append(LANG_TAG_MAP.get(tag, 3))

    # Pad sequence to MAX_SEQ_LEN
    pad_len    = MAX_SEQ_LEN - T
    word_ids_  += [0]                    * pad_len
    phone_ids_ += [[pad_p] * MAX_PHONES] * pad_len
    lang_ids_  += [0]                    * pad_len

    return word_ids_, phone_ids_, lang_ids_, label


print(f"\nEncoding {len(raw_sentences)} LinCE sentences...")

word_id_list, phone_id_list, lang_id_list, labels = [], [], [], []
skipped = 0
for sent in raw_sentences:
    try:
        w, p, l, lbl = encode_sentence(sent)
        word_id_list.append(w)
        phone_id_list.append(p)
        lang_id_list.append(l)
        labels.append(lbl)
    except Exception as e:
        skipped += 1

if skipped:
    print(f"  Skipped {skipped} sentences due to encoding errors.")

wids = torch.tensor(word_id_list, dtype=torch.long)   # (N, T)
pids = torch.tensor(phone_id_list, dtype=torch.long)  # (N, T, P)
lids = torch.tensor(lang_id_list, dtype=torch.long)   # (N, T)
labs = torch.tensor(labels, dtype=torch.long)         # (N,)

dataset = TensorDataset(wids, pids, lids, labs)
loader  = DataLoader(dataset, batch_size=64, shuffle=False)
print(f"  Encoding complete. {len(dataset)} usable samples.")


# ── Inference ──────────────────────────────────────────────────────────────

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for w_b, p_b, l_b, lab_b in loader:
        w_b  = w_b.to(DEVICE)
        p_b  = p_b.to(DEVICE)
        l_b  = l_b.to(DEVICE)
        logits, _, _ = model(w_b, p_b, l_b)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(lab_b.tolist())
        all_probs.extend(probs.tolist())

print("Inference complete.\n")


# ── Metrics ────────────────────────────────────────────────────────────────

acc         = accuracy_score(all_labels, all_preds)
macro_f1    = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
cm          = confusion_matrix(all_labels, all_preds)

label_counts = Counter(all_labels)
print("Label distribution in LinCE test:")
for k, name in enumerate(LABEL_NAMES):
    print(f"  {name:<10} : {label_counts[k]}")

print(f"\nAccuracy       : {acc:.4f}")
print(f"Macro F1       : {macro_f1:.4f}")
print(f"Weighted F1    : {weighted_f1:.4f}")
print("\nPer-class report:")
print(classification_report(all_labels, all_preds,
                             target_names=LABEL_NAMES, zero_division=0))


# ── OOV analysis on LinCE ─────────────────────────────────────────────────

lince_words   = [clean_token(w) for s in raw_sentences for w in s["tokens"] if clean_token(w)]
lince_oov     = sum(1 for w in lince_words if word_vocab.get(w, 1) == 1)
lince_oov_pct = lince_oov / max(1, len(lince_words)) * 100
print(f"\nLinCE OOV rate (against SentiMix vocab): {lince_oov_pct:.1f}%")
print(f"  ({lince_oov:,} / {len(lince_words):,} tokens are OOV)")
print("  This reflects true domain shift — the phoneme channel helps bridge it.")


# ── Load in-domain results for delta table ────────────────────────────────

indomain = None
if Path(args.sentimix_results).exists():
    with open(args.sentimix_results, encoding="utf-8") as f:
        indomain = json.load(f)


# ── Build paper-ready results table ──────────────────────────────────────

lines = []
lines.append("=" * 70)
lines.append("CROSS-DATASET EVALUATION  (Zero-Shot Transfer)")
lines.append("Train: SentiMix 2020   Test: LinCE Hinglish Sentiment")
lines.append("=" * 70)

if indomain:
    ind_acc = indomain.get("accuracy", "N/A")
    ind_f1  = indomain.get("macro_f1", "N/A")
    lines.append(f"\n{'Metric':<20} {'In-domain (SentiMix)':>22} {'Cross-domain (LinCE)':>22} {'Drop':>8}")
    lines.append("─" * 74)
    acc_drop = (float(ind_acc) - acc) if ind_acc != "N/A" else None
    f1_drop  = (float(ind_f1)  - macro_f1) if ind_f1 != "N/A" else None
    lines.append(f"{'Accuracy':<20} {float(ind_acc):>22.4f} {acc:>22.4f} "
                 f"{f'-{acc_drop:.4f}' if acc_drop else 'N/A':>8}")
    lines.append(f"{'Macro F1':<20} {float(ind_f1):>22.4f} {macro_f1:>22.4f} "
                 f"{f'-{f1_drop:.4f}' if f1_drop else 'N/A':>8}")
    lines.append(f"{'Weighted F1':<20} {'N/A':>22} {weighted_f1:>22.4f} {'N/A':>8}")
else:
    lines.append(f"\nAccuracy    : {acc:.4f}")
    lines.append(f"Macro F1    : {macro_f1:.4f}")
    lines.append(f"Weighted F1 : {weighted_f1:.4f}")

lines.append(f"\nLinCE OOV rate vs SentiMix vocab : {lince_oov_pct:.1f}%")
lines.append(f"LinCE test sentences             : {len(dataset):,}")

lines.append("\n\nLaTeX snippet (add to Table: Cross-Dataset Results):\n")
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Zero-shot cross-dataset transfer. Model trained on SentiMix 2020,")
lines.append(r"         evaluated on LinCE Hinglish Sentiment without retraining.}")
lines.append(r"\label{tab:cross-dataset}")
lines.append(r"\begin{tabular}{lcc}")
lines.append(r"\toprule")
lines.append(r"\textbf{Metric} & \textbf{SentiMix (in-domain)} & \textbf{LinCE (zero-shot)} \\")
lines.append(r"\midrule")

if indomain:
    lines.append(f"Accuracy  & {float(indomain['accuracy']):.4f} & {acc:.4f} \\\\")
    lines.append(f"Macro F1  & {float(indomain['macro_f1']):.4f}  & {macro_f1:.4f} \\\\")
else:
    lines.append(f"Accuracy  & --- & {acc:.4f} \\\\")
    lines.append(f"Macro F1  & --- & {macro_f1:.4f} \\\\")

lines.append(f"Weighted F1 & --- & {weighted_f1:.4f} \\\\")
lines.append(f"OOV rate  & --- & {lince_oov_pct:.1f}\\% \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

report = "\n".join(lines)
print("\n" + report)

with open("cross_dataset_results.txt", "w", encoding="utf-8") as f:
    f.write(report + "\n")

results = {
    "train_dataset":    "SentiMix 2020",
    "test_dataset":     "LinCE Hinglish Sentiment",
    "n_test":           len(dataset),
    "lince_oov_pct":    round(lince_oov_pct, 2),
    "accuracy":         round(float(acc),         4),
    "macro_f1":         round(float(macro_f1),    4),
    "weighted_f1":      round(float(weighted_f1), 4),
    "confusion_matrix": cm.tolist(),
    "indomain_for_delta": indomain,
}
with open("cross_dataset_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nSaved → cross_dataset_results.json")
print("Saved → cross_dataset_results.txt")
