"""
train.py  (v2 — full phoneme sequences)

Changes from v1:
  - DataLoader now uses get_collate_fn() from dataset.py to handle the
    3-D phone_ids tensor (B, T, max_phones).
  - run_epoch passes phone_ids directly; model.forward() handles the new shape.
  - Everything else (label smoothing, AdamW, cosine schedule, AMP, early
    stopping, multi-seed support) is unchanged.

To run a single seed:
    python train.py

To run 5 seeds for paper results:
    python train_multiseed.py   (see separate file)
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
from dataset import CodeMixedDataset, stratified_split, get_collate_fn

# ── Config ────────────────────────────────────────────────────────────────

CFG = dict(
    data_path     = "phonetic_data.json",
    vocab_path    = "vocabs.json",
    max_seq_len   = 64,
    max_phones    = 8,          # max phonemes per token (new)
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Data ──────────────────────────────────────────────────────────────────

full_ds = CodeMixedDataset(
    CFG["data_path"], CFG["vocab_path"],
    max_seq_len=CFG["max_seq_len"],
    max_phones=CFG["max_phones"],
    augment=False,
)
train_idx, val_idx, test_idx = stratified_split(full_ds, seed=CFG["seed"])

train_ds = CodeMixedDataset(
    CFG["data_path"], CFG["vocab_path"],
    max_seq_len=CFG["max_seq_len"],
    max_phones=CFG["max_phones"],
    augment=True,
)

pin        = device.type == "cuda"
collate_fn = get_collate_fn()

train_loader = DataLoader(
    Subset(train_ds, train_idx), batch_size=CFG["batch_size"],
    shuffle=True, num_workers=0, pin_memory=pin, collate_fn=collate_fn,
)
val_loader = DataLoader(
    Subset(full_ds, val_idx), batch_size=CFG["batch_size"],
    shuffle=False, num_workers=0, pin_memory=pin, collate_fn=collate_fn,
)
test_loader = DataLoader(
    Subset(full_ds, test_idx), batch_size=CFG["batch_size"],
    shuffle=False, num_workers=0, pin_memory=pin, collate_fn=collate_fn,
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

# ── Optimiser + scheduler ─────────────────────────────────────────────────

optimizer    = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                 weight_decay=CFG["weight_decay"], betas=(0.9, 0.98))
total_steps  = CFG["epochs"] * len(train_loader)
warmup_steps = CFG["warmup_epochs"] * len(train_loader)

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler    = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

# ── Training loop ─────────────────────────────────────────────────────────

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    weights = _class_weights.to(device)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            wids = batch["word_ids"].to(device)
            pids = batch["phone_ids"].to(device)   # (B, T, max_phones)
            lids = batch["lang_ids"].to(device)
            labs = batch["label"].to(device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, _, _ = model(wids, pids, lids)
                loss = smooth_ce_loss(logits, labs,
                                      smoothing=CFG["label_smooth"], weight=weights)

            if train:
                scaler.scale(loss).backward()
                if device.type == "cuda":
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

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, all_labels, all_preds


best_f1, patience_ctr = 0.0, 0
history = []

for epoch in range(1, CFG["epochs"] + 1):
    t0 = time.time()
    tr_loss, tr_f1, _,       _        = run_epoch(train_loader, train=True)
    vl_loss, vl_f1, vl_labs, vl_preds = run_epoch(val_loader,   train=False)
    elapsed = time.time() - t0

    print(f"Epoch {epoch:02d}  train_loss={tr_loss:.4f}  train_f1={tr_f1:.4f}  "
          f"val_loss={vl_loss:.4f}  val_f1={vl_f1:.4f}  "
          f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s")
    history.append({"epoch": epoch, "train_loss": tr_loss, "train_f1": tr_f1,
                    "val_loss": vl_loss, "val_f1": vl_f1})

    if vl_f1 > best_f1:
        best_f1      = vl_f1
        patience_ctr = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  ✓ Saved best model (val_f1={best_f1:.4f})")
    else:
        patience_ctr += 1
        if patience_ctr >= CFG["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

with open("training_history.json", "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

# ── Final test ────────────────────────────────────────────────────────────

model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
_, test_f1, test_labs, test_preds = run_epoch(test_loader, train=False)

print("\n" + "=" * 60)
print(f"TEST SET  macro-F1 = {test_f1:.4f}")
print("=" * 60)
print(classification_report(test_labs, test_preds,
                             target_names=["Negative", "Neutral", "Positive"]))
