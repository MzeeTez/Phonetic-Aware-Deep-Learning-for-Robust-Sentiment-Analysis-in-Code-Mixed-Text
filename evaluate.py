"""
Comprehensive evaluation script

Reports:
  - Accuracy, macro / weighted F1, precision, recall
  - Per-class ROC-AUC (one-vs-rest)
  - Confusion matrix (saved as PNG)
  - Attention heatmap for one correctly-predicted sample per class
  - Error analysis: top misclassified examples sorted by confidence
  - Saves all metrics to evaluation_results.json

Fixes applied:
  1. Checkpoint is a plain state_dict — load_state_dict() called directly,
     no checkpoint["model_state"] / checkpoint["cfg"] KeyErrors.
  2. Explicit MAX_SEQ_LEN / SEED constants — cfg no longer lives in checkpoint.
  3. UTF-8 encoding on all open() calls (vocabs.json, evaluation_results.json).
  4. roc_auc NaN handled correctly — round() not called before the nan check.
  5. Attention heatmap loop skips gracefully when a class has no correct pred.
  6. sys.stdout.flush() after every major print block — eliminates the Windows
     terminal scroll artifact that made output appear duplicated.
  7. zero_division=0 on all f1_score calls, consistent with train.py.
"""

import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score,
)
from sklearn.preprocessing import label_binarize
from model import EnhancedDualChannelLSTM
from dataset import CodeMixedDataset, stratified_split

LABEL_NAMES = ["Negative", "Neutral", "Positive"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# These must match the values used during training (train.py CFG)
MAX_SEQ_LEN = 64
SEED        = 42


# ── Load model + data ──────────────────────────────────────────────────────

with open("vocabs.json", encoding="utf-8") as f:
    v = json.load(f)

# train.py saves only model.state_dict() — load directly, weights_only=True
state_dict = torch.load("best_model.pth", map_location=DEVICE, weights_only=True)

model = EnhancedDualChannelLSTM(
    word_vocab_size  = max(v["word_vocab"].values()) + 1,
    phone_vocab_size = max(v["phone_vocab"].values()) + 1,
    dropout     = 0.0,   # always disable stochastic layers at eval time
    var_dropout = 0.0,
).to(DEVICE)
model.load_state_dict(state_dict)
model.eval()

dataset = CodeMixedDataset(
    "phonetic_data.json", "vocabs.json",
    max_seq_len=MAX_SEQ_LEN,
)
_, _, test_idx = stratified_split(dataset, seed=SEED)
loader = DataLoader(Subset(dataset, test_idx), batch_size=64, shuffle=False)

idx_to_word = {i: w for w, i in v["word_vocab"].items()}

print(f"Loaded {len(test_idx)} test samples on {DEVICE}")
sys.stdout.flush()


# ── Collect predictions ───────────────────────────────────────────────────

all_labels, all_preds, all_probs = [], [], []
sample_attention = {0: None, 1: None, 2: None}

with torch.no_grad():
    for batch in loader:
        wids = batch["word_ids"].to(DEVICE)
        pids = batch["phone_ids"].to(DEVICE)
        lids = batch["lang_ids"].to(DEVICE)
        labs = batch["label"]

        logits, w_weights, _ = model(wids, pids, lids)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().tolist()

        all_labels.extend(labs.tolist())
        all_preds.extend(preds)
        all_probs.extend(probs.tolist())

        for i, (lab, pred) in enumerate(zip(labs.tolist(), preds)):
            if sample_attention[lab] is None and lab == pred:
                attn_map   = w_weights[i].mean(0).cpu().numpy()
                token_list = [
                    idx_to_word.get(wids[i, j].item(), "?")
                    for j in range(wids.shape[1])
                    if wids[i, j].item() != 0
                ]
                sample_attention[lab] = {"attn": attn_map, "tokens": token_list}

print("Inference complete.")
sys.stdout.flush()


# ── Core metrics ──────────────────────────────────────────────────────────

acc         = accuracy_score(all_labels, all_preds)
macro_f1    = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

y_bin = label_binarize(all_labels, classes=[0, 1, 2])
try:
    roc_auc = roc_auc_score(y_bin, all_probs, multi_class="ovr", average="macro")
except Exception:
    roc_auc = float("nan")

print("\n" + "=" * 60)
print(f"Accuracy       : {acc:.4f}")
print(f"Macro F1       : {macro_f1:.4f}")
print(f"Weighted F1    : {weighted_f1:.4f}")
print(f"Macro ROC-AUC  : {'nan' if np.isnan(roc_auc) else f'{roc_auc:.4f}'}")
print("=" * 60)
print(classification_report(
    all_labels, all_preds,
    target_names=LABEL_NAMES,
    zero_division=0,
))
sys.stdout.flush()


# ── Confusion matrix ──────────────────────────────────────────────────────

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
    linewidths=0.5, ax=ax,
)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual",    fontsize=12)
ax.set_title("Confusion Matrix — Enhanced DualChannel LSTM", fontsize=13)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Confusion matrix saved → confusion_matrix.png")
sys.stdout.flush()


# ── Attention heatmaps ────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for cls_id, ax in zip([0, 1, 2], axes):
    info = sample_attention[cls_id]
    if info is None:
        ax.set_title(f"{LABEL_NAMES[cls_id]} (no correct prediction found)")
        ax.axis("off")
        continue

    toks = info["tokens"][:20]
    attn = info["attn"][:len(toks), :len(toks)]
    sns.heatmap(
        attn, xticklabels=toks, yticklabels=toks,
        cmap="YlOrRd", ax=ax, cbar=False, linewidths=0.1,
    )
    ax.set_title(f"Self-attention — {LABEL_NAMES[cls_id]}", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

plt.tight_layout()
plt.savefig("attention_heatmaps.png", dpi=150)
plt.close()
print("Attention heatmaps saved → attention_heatmaps.png")
sys.stdout.flush()


# ── Error analysis ────────────────────────────────────────────────────────

errors = [
    {
        "true":       LABEL_NAMES[t],
        "pred":       LABEL_NAMES[p],
        "confidence": float(max(prob)),
    }
    for t, p, prob in zip(all_labels, all_preds, all_probs)
    if t != p
]
errors.sort(key=lambda x: x["confidence"], reverse=True)

print(f"\nTotal errors : {len(errors)} / {len(all_labels)}")
print(f"Error rate   : {len(errors)/len(all_labels)*100:.1f}%")
print("Top 10 high-confidence errors:")
for e in errors[:10]:
    print(f"  true={e['true']:8s}  pred={e['pred']:8s}  conf={e['confidence']:.3f}")
sys.stdout.flush()


# ── Save full results ─────────────────────────────────────────────────────

roc_auc_safe = None if np.isnan(roc_auc) else round(float(roc_auc), 4)

results = {
    "accuracy":         round(float(acc),        4),
    "macro_f1":         round(float(macro_f1),    4),
    "weighted_f1":      round(float(weighted_f1), 4),
    "roc_auc":          roc_auc_safe,
    "confusion_matrix": cm.tolist(),
    "n_test":           len(all_labels),
    "n_errors":         len(errors),
    "error_rate":       round(len(errors) / len(all_labels), 4),
}

with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("\nFull results saved → evaluation_results.json")
sys.stdout.flush()