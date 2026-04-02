import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import chess

from model import ChessGPT, compute_loss
from config import ModelCFG, TrainCFG
import os
os.makedirs("checkpoints", exist_ok=True)

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
# 3. Initialize Model
# -------------------------------
model = ChessGPT(model_config).to(device)
model.count_params()
optimizer = optim.AdamW(model.parameters(), lr=train_config.lr)

# -------------------------------
# 4. Load Data
# -------------------------------
X = np.load(train_config.train_features)
y_move = np.load(train_config.train_moves)
y_score = np.load(train_config.train_scores)
fens = np.load("data/train_fens.npy", allow_pickle=True).tolist()

# Convert to tensors — float32 enforced (MPS does not support float64)
X = torch.tensor(X, dtype=torch.float32)
y_move = torch.tensor(y_move, dtype=torch.long)
y_score = torch.tensor(y_score, dtype=torch.float32)

# Normalize score
y_score = y_score.clamp(-1, 1)

print("Data loaded:")
print("X:", X.shape)
print("Moves:", y_move.shape)
print("Scores:", y_score.shape)
print("FENs:", len(fens))

# -------------------------------
# 5. Train / Validation Split
# -------------------------------
split = int(0.9 * len(X))

X_train = X[:split].to(device)
y_move_train = y_move[:split].to(device)
y_score_train = y_score[:split].to(device)
fens_train = fens[:split]

X_val = X[split:].to(device)
y_move_val = y_move[split:].to(device)
y_score_val = y_score[split:].to(device)
fens_val = fens[split:]

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

# -------------------------------
# 7. Legal Move Masking
# -------------------------------
def mask_illegal_moves(logits, boards):
    masked_logits = torch.full_like(logits, -1e9)

    for i, board in enumerate(boards):
        legal_moves = list(board.legal_moves)
        legal_ids = [m.from_square * 64 + m.to_square for m in legal_moves]
        masked_logits[i, legal_ids] = logits[i, legal_ids]

    return masked_logits

# -------------------------------
# 8. LR Schedule — cosine decay with linear warmup
# -------------------------------
def get_lr(step: int) -> float:
    # Linear warmup
    if step < train_config.warmup_steps:
        return train_config.lr * (step + 1) / train_config.warmup_steps
    # Cosine decay from lr → lr_min
    progress = (step - train_config.warmup_steps) / max(
        1, train_config.max_steps - train_config.warmup_steps
    )
    cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())
    return train_config.lr_min + (train_config.lr - train_config.lr_min) * cosine

# -------------------------------
# 9. Training Loop
# -------------------------------
best_val_loss = float("inf")
no_improve_count = 0   # for early stopping

train_losses = []
move_accuracies = []
score_errors = []

model.train()

for step in range(train_config.max_steps):

    # ── LR update ────────────────────────────────────────────────────────────
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # ── Sample batch ─────────────────────────────────────────────────────────
    idx = torch.randint(0, len(X_train), (train_config.bsz,))

    x            = X_train[idx]
    move_target  = y_move_train[idx]
    score_target = y_score_train[idx]

    # ── Forward ──────────────────────────────────────────────────────────────
    move_logits, score_pred = model(x)

    # ── Loss (use original logits, not masked — masking is for metrics only) ─
    loss, move_loss, score_loss = compute_loss(
        move_logits,
        move_target,
        score_pred,
        score_target,
        model_config
    )

    # ── Backprop ─────────────────────────────────────────────────────────────
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
    optimizer.step()

    # ── Metrics (masked logits for realistic accuracy) ───────────────────────
    with torch.no_grad():
        boards       = [chess.Board(fens_train[i.item()]) for i in idx]
        masked_logits = mask_illegal_moves(move_logits.detach(), boards)

        move_acc  = compute_move_accuracy(masked_logits, move_target)
        top5_acc  = compute_topk_accuracy(masked_logits, move_target, k=5)
        score_err = compute_score_error(score_pred.detach(), score_target)

    train_losses.append(loss.item())
    move_accuracies.append(move_acc.item())
    score_errors.append(score_err.item())

    # ── Periodic checkpoint — driven by train_config.save_interval ───────────
    if step > 0 and step % train_config.save_interval == 0:
        ckpt_path = f"checkpoints/model_step_{step}.pth"
        torch.save({
            "step":        step,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_loss":    best_val_loss,
        }, ckpt_path)
        print(f"💾 Checkpoint saved → {ckpt_path}")

    # ── Logging + Validation ─────────────────────────────────────────────────
    if step % train_config.log_interval == 0:

        model.eval()
        with torch.no_grad():
            val_idx          = torch.randint(0, len(X_val), (train_config.bsz,))
            x_val            = X_val[val_idx]
            move_val_target  = y_move_val[val_idx]
            score_val_target = y_score_val[val_idx]

            move_logits_val, score_pred_val = model(x_val)

            val_loss, _, _ = compute_loss(
                move_logits_val,
                move_val_target,
                score_pred_val,
                score_val_target,
                model_config
            )

            boards_val        = [chess.Board(fens_val[i.item()]) for i in val_idx]
            masked_logits_val = mask_illegal_moves(move_logits_val, boards_val)

            val_acc       = compute_move_accuracy(masked_logits_val, move_val_target)
            val_top5      = compute_topk_accuracy(masked_logits_val, move_val_target, k=5)
            val_score_err = compute_score_error(score_pred_val, score_val_target)

            # ── Save best model ───────────────────────────────────────────────
            if val_loss.item() < best_val_loss:
                best_val_loss  = val_loss.item()
                no_improve_count = 0
                torch.save({
                    "step":        step,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_loss":    best_val_loss,
                }, "model_best1.pth")
                print("🏆 Best model saved!  val_loss =", round(best_val_loss, 4))
            else:
                no_improve_count += 1

        model.train()

        print(
            f"Step {step:>6} | lr {lr:.2e} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Acc: {move_acc.item():.4f} | Top5: {top5_acc.item():.4f} | "
            f"ScoreErr: {score_err.item():.4f} || "
            f"Val Loss: {val_loss.item():.4f} | "
            f"Acc: {val_acc.item():.4f} | Top5: {val_top5.item():.4f} | "
            f"ScoreErr: {val_score_err.item():.4f}"
        )

        # ── Early stopping ────────────────────────────────────────────────────
        if no_improve_count >= train_config.early_stop_patience:
            print(f"\n⏹  Early stopping triggered at step {step} "
                  f"(no improvement for {no_improve_count} log intervals).")
            break

# -------------------------------
# 10. Save Final Model
# -------------------------------
torch.save({
    "step":        step,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict(),
    "val_loss":    best_val_loss,
}, "model_final.pth")
print("\n✅ Final model saved!")

# -------------------------------
# 11. Plot Graphs
# -------------------------------
plt.figure(figsize=(10, 6))
plt.plot(train_losses,    label="Train Loss")
plt.plot(move_accuracies, label="Train Move Acc")
plt.plot(score_errors,    label="Train Score Err")

plt.legend()
plt.title("Training Performance")
plt.xlabel("Steps")
plt.ylabel("Value")

plt.savefig("training_plot1.png")
plt.show()