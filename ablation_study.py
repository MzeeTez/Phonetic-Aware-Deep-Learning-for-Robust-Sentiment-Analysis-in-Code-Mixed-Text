"""
ablation_study.py
Trains all ablation conditions and writes results to ablation_results.json.

Ablation table produced:
  ┌─────────────────────────┬──────────┬──────────┬──────────┬──────────┐
  │ Condition               │ Acc      │ Macro-F1 │ Neg-F1   │ Pos-F1   │
  ├─────────────────────────┼──────────┼──────────┼──────────┼──────────┤
  │ Full model              │          │          │          │          │
  │ − Phoneme channel       │          │          │          │          │
  │ − Cross-modal attention │          │          │          │          │
  │ − Language tags         │          │          │          │          │
  │ − Phoneme + Cross-attn  │          │          │          │          │
  └─────────────────────────┴──────────┴──────────┴──────────┴──────────┘

Usage:
    python ablation_study.py

Outputs:
    ablation_results.json   — machine-readable results for all conditions
    ablation_table.txt      — formatted table ready to paste into your paper
"""

import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score, classification_report
from ablation_model import AblationDualChannelLSTM
from dataset import CodeMixedDataset, stratified_split

# ── Config (must match your train.py CFG exactly) ─────────────────────────

CFG = dict(
    data_path     = "phonetic_data.json",
    vocab_path    = "vocabs.json",
    max_seq_len   = 64,
    batch_size    = 64,
    epochs        = 30,
    lr            = 3e-4,
    weight_decay  = 5e-2,
    warmup_epochs = 2,
    grad_clip     = 1.0,
    label_smooth  = 0.1,
    dropout       = 0.5,
    var_dropout   = 0.4,
    patience      = 4,
    seed          = 42,
    class_weights = [1.5, 1.5, 1.2],
)

torch.manual_seed(CFG["seed"])
torch.cuda.manual_seed_all(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}\n")

# ── Ablation conditions ───────────────────────────────────────────────────

ABLATIONS = [
    {
        "name":        "Full model",
        "short":       "full",
        "ablate_phoneme":    False,
        "ablate_cross_attn": False,
        "ablate_lang_tag":   False,
    },
    {
        "name":        "− Phoneme channel",
        "short":       "no_phoneme",
        "ablate_phoneme":    True,
        "ablate_cross_attn": False,
        "ablate_lang_tag":   False,
    },
    {
        "name":        "− Cross-modal attention",
        "short":       "no_cross_attn",
        "ablate_phoneme":    False,
        "ablate_cross_attn": True,
        "ablate_lang_tag":   False,
    },
    {
        "name":        "− Language tags",
        "short":       "no_lang_tag",
        "ablate_phoneme":    False,
        "ablate_cross_attn": False,
        "ablate_lang_tag":   True,
    },
    {
        "name":        "− Phoneme + Cross-attn (word-only)",
        "short":       "word_only",
        "ablate_phoneme":    True,
        "ablate_cross_attn": True,
        "ablate_lang_tag":   False,
    },
]

# ── Data (shared across all conditions) ───────────────────────────────────

with open(CFG["vocab_path"], encoding="utf-8") as f:
    v = json.load(f)

full_ds = CodeMixedDataset(
    CFG["data_path"], CFG["vocab_path"],
    max_seq_len=CFG["max_seq_len"], augment=False,
)
train_idx, val_idx, test_idx = stratified_split(full_ds, seed=CFG["seed"])

train_ds = CodeMixedDataset(
    CFG["data_path"], CFG["vocab_path"],
    max_seq_len=CFG["max_seq_len"], augment=True,
)

pin = DEVICE.type == "cuda"

train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=CFG["batch_size"],
                          shuffle=True,  num_workers=0, pin_memory=pin)
val_loader   = DataLoader(Subset(full_ds, val_idx),   batch_size=CFG["batch_size"],
                          shuffle=False, num_workers=0, pin_memory=pin)
test_loader  = DataLoader(Subset(full_ds, test_idx),  batch_size=CFG["batch_size"],
                          shuffle=False, num_workers=0, pin_memory=pin)

print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}\n")

# ── Loss ──────────────────────────────────────────────────────────────────

def smooth_ce_loss(logits, targets, smoothing=0.1, weight=None):
    n_classes = logits.size(-1)
    with torch.no_grad():
        smooth_targets = torch.full_like(logits, smoothing / (n_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(smooth_targets * log_probs).sum(dim=-1)
    if weight is not None:
        loss = loss * weight[targets]
    return loss.mean()

_class_weights = torch.tensor(CFG["class_weights"], dtype=torch.float)

# ── Train / eval helpers ──────────────────────────────────────────────────

def run_epoch(model, loader, optimizer=None, scaler=None, scheduler=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    weights = _class_weights.to(DEVICE)

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            wids = batch["word_ids"].to(DEVICE)
            pids = batch["phone_ids"].to(DEVICE)
            lids = batch["lang_ids"].to(DEVICE)
            labs = batch["label"].to(DEVICE)

            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                logits, _, _ = model(wids, pids, lids)
                loss = smooth_ce_loss(logits, labs,
                                      smoothing=CFG["label_smooth"], weight=weights)

            if training:
                scaler.scale(loss).backward()
                if DEVICE.type == "cuda":
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scale_before <= scaler.get_scale():
                    scheduler.step()

            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labs.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), macro_f1, all_labels, all_preds


def train_condition(ablation_cfg: dict) -> dict:
    """Train one ablation condition end-to-end. Returns test metrics dict."""
    name  = ablation_cfg["name"]
    short = ablation_cfg["short"]
    print(f"\n{'='*60}")
    print(f"  Condition: {name}")
    print(f"{'='*60}")

    # FIXED: Added max() + 1 to prevent CUDA Out of Bounds errors
    model = AblationDualChannelLSTM(
        word_vocab_size  = max(v["word_vocab"].values()) + 1,
        phone_vocab_size = max(v["phone_vocab"].values()) + 1,
        dropout          = CFG["dropout"],
        var_dropout      = CFG["var_dropout"],
        ablate_phoneme    = ablation_cfg["ablate_phoneme"],
        ablate_cross_attn = ablation_cfg["ablate_cross_attn"],
        ablate_lang_tag   = ablation_cfg["ablate_lang_tag"],
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total_params:,}")

    total_steps  = CFG["epochs"] * len(train_loader)
    warmup_steps = CFG["warmup_epochs"] * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CFG["lr"], weight_decay=CFG["weight_decay"],
                                  betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    best_f1, patience_ctr = 0.0, 0
    best_path = f"best_{short}.pth"

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _          = run_epoch(model, train_loader, optimizer, scaler, scheduler)
        vl_loss, vl_f1, vl_labs, vl_preds = run_epoch(model, val_loader)
        print(f"  Ep {epoch:02d}  tr_loss={tr_loss:.4f}  tr_f1={tr_f1:.4f}  "
              f"val_loss={vl_loss:.4f}  val_f1={vl_f1:.4f}  {time.time()-t0:.1f}s")

        if vl_f1 > best_f1:
            best_f1 = vl_f1
            patience_ctr = 0
            torch.save(model.state_dict(), best_path)
            print(f"    ✓ Saved (val_f1={best_f1:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= CFG["patience"]:
                print(f"  Early stop at epoch {epoch}")
                break

    # ── Test evaluation ───────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
    _, test_f1, test_labs, test_preds = run_epoch(model, test_loader)

    acc      = accuracy_score(test_labs, test_preds)
    per_cls  = classification_report(test_labs, test_preds,
                                     target_names=["Negative","Neutral","Positive"],
                                     output_dict=True, zero_division=0)
    results = {
        "condition":   name,
        "short":       short,
        "accuracy":    round(float(acc), 4),
        "macro_f1":    round(float(test_f1), 4),
        "neg_f1":      round(per_cls["Negative"]["f1-score"], 4),
        "neu_f1":      round(per_cls["Neutral"]["f1-score"],  4),
        "pos_f1":      round(per_cls["Positive"]["f1-score"], 4),
        "neg_prec":    round(per_cls["Negative"]["precision"], 4),
        "pos_prec":    round(per_cls["Positive"]["precision"], 4),
        "best_val_f1": round(best_f1, 4),
    }
    print(f"\n  TEST  acc={results['accuracy']:.4f}  macro_f1={results['macro_f1']:.4f}  "
          f"neg_f1={results['neg_f1']:.4f}  pos_f1={results['pos_f1']:.4f}")
    return results


# ── Run all conditions ────────────────────────────────────────────────────

all_results = []
for abl in ABLATIONS:
    res = train_condition(abl)
    all_results.append(res)

with open("ablation_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)
print("\n\nAblation results saved → ablation_results.json")

# ── Print formatted table ─────────────────────────────────────────────────

header = f"\n{'Condition':<32} {'Acc':>6} {'Mac-F1':>8} {'Neg-F1':>8} {'Neu-F1':>8} {'Pos-F1':>8}"
sep    = "-" * 74
rows   = [header, sep]
full   = next(r for r in all_results if r["short"] == "full")

for r in all_results:
    delta = f"  (Δ {r['macro_f1']-full['macro_f1']:+.4f})" if r["short"] != "full" else ""
    rows.append(
        f"{r['condition']:<32} {r['accuracy']:>6.4f} {r['macro_f1']:>8.4f}"
        f"{r['neg_f1']:>8.4f} {r['neu_f1']:>8.4f} {r['pos_f1']:>8.4f}{delta}"
    )

table = "\n".join(rows)
print(table)

with open("ablation_table.txt", "w", encoding="utf-8") as f:
    f.write(table + "\n")
print("\nFormatted table saved → ablation_table.txt")