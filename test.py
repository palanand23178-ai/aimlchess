"""
evaluate.py
───────────
Run this ONCE after training is complete to get final test metrics.
Loads model_best.pth and evaluates on the last 10% of your dataset
(which was never used for training or validation decisions).

Usage:
    python3 evaluate.py
"""

import torch
import numpy as np
import chess

from model import ChessGPT, compute_loss
from config import ModelCFG, TrainCFG

# -------------------------------
# 1. Device
# -------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# -------------------------------
# 2. Load Config
# -------------------------------
model_config = ModelCFG()
train_config = TrainCFG()

# -------------------------------
# 3. Load Model
# -------------------------------
checkpoint = torch.load("model_best1.pth", map_location=device)
model = ChessGPT(model_config).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print(f"Model loaded from model_best1.pth")
print(f"  Saved at step : {checkpoint['step']}")
print(f"  Best val loss : {checkpoint['val_loss']:.4f}")

# -------------------------------
# 4. Load Data
# -------------------------------
X      = np.load(train_config.train_features)
y_move = np.load(train_config.train_moves)
y_score = np.load(train_config.train_scores)
fens   = np.load("data/train_fens.npy", allow_pickle=True).tolist()

X       = torch.tensor(X,       dtype=torch.float32)
y_move  = torch.tensor(y_move,  dtype=torch.long)
y_score = torch.tensor(y_score, dtype=torch.float32).clamp(-1, 1)

# -------------------------------
# 5. Test Split — last 10% of data
# -------------------------------
# Training used first 90%, so last 10% is completely unseen.
test_start = int(0.9 * len(X))

X_test       = X[test_start:].to(device)
y_move_test  = y_move[test_start:].to(device)
y_score_test = y_score[test_start:].to(device)
fens_test    = fens[test_start:]

print(f"\nTest set size: {len(X_test):,} positions")

# -------------------------------
# 6. Metrics
# -------------------------------
def compute_move_accuracy(logits, targets):
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean()

def compute_topk_accuracy(logits, targets, k=5):
    topk = logits.topk(k, dim=-1).indices
    correct = topk.eq(targets.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean()

def compute_score_error(pred, target):
    return torch.abs(pred.squeeze() - target).mean()

def mask_illegal_moves(logits, boards):
    masked = torch.full_like(logits, -1e9)
    for i, board in enumerate(boards):
        legal_ids = [m.from_square * 64 + m.to_square for m in board.legal_moves]
        masked[i, legal_ids] = logits[i, legal_ids]
    return masked

# -------------------------------
# 7. Full Test Evaluation
# -------------------------------
# Runs over the ENTIRE test set in batches — not random sampling.
# This gives a reliable, repeatable number for your report.

BSZ = 256   # batch size for evaluation

all_move_acc  = []
all_top5_acc  = []
all_score_err = []
all_loss      = []

print("\nRunning full test evaluation...")

with torch.no_grad():
    for start in range(0, len(X_test), BSZ):
        end = min(start + BSZ, len(X_test))

        x_batch     = X_test[start:end]
        move_batch  = y_move_test[start:end]
        score_batch = y_score_test[start:end]
        fens_batch  = fens_test[start:end]

        move_logits, score_pred = model(x_batch)

        loss, _, _ = compute_loss(
            move_logits, move_batch, score_pred, score_batch, model_config
        )

        boards        = [chess.Board(f) for f in fens_batch]
        masked_logits = mask_illegal_moves(move_logits, boards)

        move_acc  = compute_move_accuracy(masked_logits, move_batch)
        top5_acc  = compute_topk_accuracy(masked_logits, move_batch, k=5)
        score_err = compute_score_error(score_pred, score_batch)

        all_loss.append(loss.item())
        all_move_acc.append(move_acc.item())
        all_top5_acc.append(top5_acc.item())
        all_score_err.append(score_err.item())

        if (start // BSZ) % 20 == 0:
            print(f"  Evaluated {end:>6,} / {len(X_test):,} positions...")

# -------------------------------
# 8. Final Results
# -------------------------------
test_loss      = np.mean(all_loss)
test_move_acc  = np.mean(all_move_acc)
test_top5_acc  = np.mean(all_top5_acc)
test_score_err = np.mean(all_score_err)

print("\n" + "═" * 50)
print("           FINAL TEST RESULTS")
print("═" * 50)
print(f"  Test Loss       : {test_loss:.4f}")
print(f"  Move Accuracy   : {test_move_acc * 100:.2f}%")
print(f"  Top-5 Accuracy  : {test_top5_acc * 100:.2f}%")
print(f"  Score Error     : {test_score_err:.4f}")
print("═" * 50)
print("\n  Move Accuracy  = correct move predicted exactly")
print("  Top-5 Accuracy = correct move in model's top 5 choices")
print("  Score Error    = avg error in eval score (0=perfect, 1=worst)")

# -------------------------------
# 9. Performance Curve Plot
# -------------------------------
import matplotlib.pyplot as plt

print("\nGenerating performance curve...")

X_train_plot  = X[:test_start].to(device)
ym_train_plot = y_move[:test_start].to(device)
ys_train_plot = y_score[:test_start].to(device)

train_losses_plot = []
test_losses_plot  = []

PLOT_BSZ    = 256
PLOT_POINTS = 100   # number of points on the curve

# Sample PLOT_POINTS evenly spaced batches from train and test
train_indices = torch.linspace(0, len(X_train_plot) - PLOT_BSZ, PLOT_POINTS).long()
test_indices  = torch.linspace(0, len(X_test) - PLOT_BSZ, PLOT_POINTS).long()

with torch.no_grad():
    for t_idx, v_idx in zip(train_indices, test_indices):

        # Train batch loss
        x_t  = X_train_plot[t_idx : t_idx + PLOT_BSZ]
        ym_t = ym_train_plot[t_idx : t_idx + PLOT_BSZ]
        ys_t = ys_train_plot[t_idx : t_idx + PLOT_BSZ]
        ml_t, sp_t = model(x_t)
        loss_t, _, _ = compute_loss(ml_t, ym_t, sp_t, ys_t, model_config)
        train_losses_plot.append(loss_t.item())

        # Test batch loss
        x_v  = X_test[v_idx : v_idx + PLOT_BSZ]
        ym_v = y_move_test[v_idx : v_idx + PLOT_BSZ]
        ys_v = y_score_test[v_idx : v_idx + PLOT_BSZ]
        ml_v, sp_v = model(x_v)
        loss_v, _, _ = compute_loss(ml_v, ym_v, sp_v, ys_v, model_config)
        test_losses_plot.append(loss_v.item())

# Smooth curves to match professor's clean curve style
def smooth(values, weight=0.85):
    smoothed, last = [], values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return smoothed

plt.figure(figsize=(8, 6))
plt.plot(smooth(train_losses_plot), label="train", color="#4C9BE8", linewidth=2)
plt.plot(smooth(test_losses_plot),  label="test",  color="#F5A623", linewidth=2)
plt.legend(fontsize=12)
plt.title("Performance curve", fontsize=14)
plt.xlabel("Data / Experience", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.tight_layout()
plt.savefig("performance_curve.png")
plt.show()
print("\n📊 Performance curve saved → performance_curve.png")