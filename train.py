"""
Advanced training script for EnhancedDualChannelLSTM

Features:
  - Label-smoothing cross-entropy (reduces overconfidence)
  - AdamW with cosine annealing + linear warmup
  - Gradient clipping (norm=1.0)
  - Mixed-precision training (torch.amp) — GPU only, safely disabled on CPU
  - Stratified train / val / test split
  - Early stopping on val macro-F1
  - Best-model checkpoint saving (weights only, safe for sharing)
  - Training curve logging (JSON)

All fixes applied:
  1.  scaler.unscale_() guarded by device.type == "cuda" — no RuntimeError on CPU.
  2.  pin_memory conditional on CUDA — no wasted memory/latency on CPU machines.
  3.  torch.load uses weights_only=True — no arbitrary code-execution risk.
  4.  Checkpoint saves state_dict only — portable, no optimizer/CFG pickle.
  5.  Class-weight tensor moved to device inside run_epoch — survives device changes.
  6.  Scheduler steps ONLY when optimizer actually took a step (GradScaler skip-safe).
      Eliminates UserWarning about lr_scheduler.step() / optimizer.step() ordering.
  7.  Stale premature scheduler.step() prime removed — not needed with fix 6.
  8.  dropout 0.4->0.5, var_dropout 0.3->0.4, weight_decay 1e-2->5e-2 — cuts overfitting.
  9.  class_weights rebalanced [1.5, 1.5, 1.2] — Neutral was starved at 0.7,
      causing recall of only 0.44; equal weighting with Negative fixes this.
  10. patience tightened 5->4 — stops before the train/val gap widens further.
  11. torch.cuda.manual_seed_all added — full GPU reproducibility.
  12. num_workers=0 retained — prevents Windows multiprocessing (spawn) crash.
  13. UTF-8 encoding on all open() calls.
  14. Modern torch.amp.GradScaler / autocast syntax throughout.
"""

import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, classification_report
from model import EnhancedDualChannelLSTM
from dataset import CodeMixedDataset, stratified_split

# ── Config ────────────────────────────────────────────────────────────────

CFG = dict(
    data_path     = "phonetic_data.json",
    vocab_path    = "vocabs.json",
    max_seq_len   = 64,
    batch_size    = 64,
    epochs        = 30,
    lr            = 3e-4,
    weight_decay  = 5e-2,   # raised from 1e-2 — stronger L2 regularisation
    warmup_epochs = 2,
    grad_clip     = 1.0,
    label_smooth  = 0.1,
    dropout       = 0.5,    # raised from 0.4 — reduces overfitting
    var_dropout   = 0.4,    # raised from 0.3 — reduces overfitting
    patience      = 4,      # tightened from 5 — stops before gap widens further
    seed          = 42,
    # Rebalanced from [2.2, 0.7, 1.2].
    # The old Neutral weight of 0.7 actively penalised the model for getting
    # Neutral right, producing recall of only 0.44. Raising to 1.5 gives it
    # equal incentive alongside Negative.
    class_weights = [1.5, 1.5, 1.2],
)

# FIX 11: seed both CPU and all CUDA devices for full reproducibility
torch.manual_seed(CFG["seed"])
torch.cuda.manual_seed_all(CFG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Data ──────────────────────────────────────────────────────────────────

full_ds = CodeMixedDataset(
    CFG["data_path"], CFG["vocab_path"],
    max_seq_len=CFG["max_seq_len"], augment=False,
)
train_idx, val_idx, test_idx = stratified_split(full_ds, seed=CFG["seed"])

# Augmentation only on the training subset
train_ds = CodeMixedDataset(
    CFG["data_path"], CFG["vocab_path"],
    max_seq_len=CFG["max_seq_len"], augment=True,
)

# FIX 2: pin_memory only benefits GPU transfers; hurts CPU-only machines
pin = device.type == "cuda"

# num_workers=0: prevents Windows multiprocessing (spawn) crash
train_loader = DataLoader(
    Subset(train_ds, train_idx),
    batch_size=CFG["batch_size"], shuffle=True,  num_workers=0, pin_memory=pin,
)
val_loader = DataLoader(
    Subset(full_ds, val_idx),
    batch_size=CFG["batch_size"], shuffle=False, num_workers=0, pin_memory=pin,
)
test_loader = DataLoader(
    Subset(full_ds, test_idx),
    batch_size=CFG["batch_size"], shuffle=False, num_workers=0, pin_memory=pin,
)

print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

# ── Model ─────────────────────────────────────────────────────────────────

with open(CFG["vocab_path"], encoding="utf-8") as f:
    v = json.load(f)

model = EnhancedDualChannelLSTM(
    word_vocab_size  = len(v["word_vocab"]),
    phone_vocab_size = len(v["phone_vocab"]),
    dropout          = CFG["dropout"],
    var_dropout      = CFG["var_dropout"],
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

# ── Loss with label smoothing ─────────────────────────────────────────────

def smooth_ce_loss(logits, targets, smoothing=0.1, weight=None):
    """Cross-entropy with label smoothing."""
    n_classes = logits.size(-1)
    with torch.no_grad():
        smooth_targets = torch.full_like(logits, smoothing / (n_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(smooth_targets * log_probs).sum(dim=-1)
    if weight is not None:
        loss = loss * weight[targets]
    return loss.mean()

# FIX 5: keep as CPU tensor; .to(device) each epoch so it survives any
#         device change that might occur between checkpoint save and reload.
_class_weights = torch.tensor(CFG["class_weights"], dtype=torch.float)

# ── Optimizer + scheduler ─────────────────────────────────────────────────

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CFG["lr"],
    weight_decay=CFG["weight_decay"],
    betas=(0.9, 0.98),
)

total_steps  = CFG["epochs"] * len(train_loader)
warmup_steps = CFG["warmup_epochs"] * len(train_loader)


def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
# FIX 7: no premature scheduler.step() here. The GradScaler-skip-safe
# pattern in run_epoch (FIX 6) makes it unnecessary and avoids the warning.

# FIX 14: GradScaler enabled only on CUDA; disabled cleanly on CPU
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

# ── Training helpers ──────────────────────────────────────────────────────

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    # FIX 5: move weights to the active device each call
    weights = _class_weights.to(device)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            wids = batch["word_ids"].to(device)
            pids = batch["phone_ids"].to(device)
            lids = batch["lang_ids"].to(device)
            labs = batch["label"].to(device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, _, _ = model(wids, pids, lids)
                loss = smooth_ce_loss(
                    logits, labs,
                    smoothing=CFG["label_smooth"],
                    weight=weights,
                )

            if train:
                scaler.scale(loss).backward()

                # FIX 1: unscale_ only when AMP is active — raises RuntimeError on CPU
                if device.type == "cuda":
                    scaler.unscale_(optimizer)

                nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])

                # FIX 6: GradScaler-skip-safe scheduler stepping.
                #
                # When the scaler detects a gradient overflow it skips
                # optimizer.step() and reduces the internal scale value.
                # By comparing scale before vs after we know whether the
                # optimizer actually ran — and only advance the scheduler
                # if it did. This fully eliminates the UserWarning:
                # "lr_scheduler.step() called before optimizer.step()".
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # scale_before <= current scale  →  no overflow, optimizer ran
                # scale_before >  current scale  →  overflow, optimizer skipped
                if scale_before <= scaler.get_scale():
                    scheduler.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labs.cpu().tolist())

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, all_labels, all_preds


# ── Main training loop ────────────────────────────────────────────────────

best_f1, patience_ctr = 0.0, 0
history = []

for epoch in range(1, CFG["epochs"] + 1):
    t0 = time.time()
    tr_loss, tr_f1, _,       _        = run_epoch(train_loader, train=True)
    vl_loss, vl_f1, vl_labs, vl_preds = run_epoch(val_loader,   train=False)
    elapsed = time.time() - t0

    print(
        f"Epoch {epoch:02d}  "
        f"train_loss={tr_loss:.4f}  train_f1={tr_f1:.4f}  "
        f"val_loss={vl_loss:.4f}  val_f1={vl_f1:.4f}  "
        f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s"
    )
    history.append({
        "epoch":      epoch,
        "train_loss": tr_loss,
        "train_f1":   tr_f1,
        "val_loss":   vl_loss,
        "val_f1":     vl_f1,
    })

    if vl_f1 > best_f1:
        best_f1      = vl_f1
        patience_ctr = 0
        # FIX 3 & 4: state_dict only — no optimizer/CFG pickle,
        # so weights_only=True is safe on load (no code-execution risk).
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  ✓ Saved best model (val_f1={best_f1:.4f})")
    else:
        patience_ctr += 1
        if patience_ctr >= CFG["patience"]:
            print(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {CFG['patience']} epochs)"
            )
            break

with open("training_history.json", "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

# ── Final test evaluation ─────────────────────────────────────────────────

# FIX 3: weights_only=True — safe because checkpoint is a plain state_dict
model.load_state_dict(
    torch.load("best_model.pth", map_location=device, weights_only=True)
)

_, test_f1, test_labs, test_preds = run_epoch(test_loader, train=False)

print("\n" + "=" * 60)
print(f"TEST SET  macro-F1 = {test_f1:.4f}")
print("=" * 60)
print(classification_report(
    test_labs, test_preds,
    target_names=["Negative", "Neutral", "Positive"],
))