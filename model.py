import torch
import torch.nn as nn
import torch.nn.functional as F


# ── What this file does ───────────────────────────────────────────────────────
# This is the brain of the chess AI.
# It takes a chess board as input and outputs:
#   1. move_logits  — a score for every possible move (4096 moves)
#   2. score_pred   — who is winning (-1 = black, 0 = even, +1 = white)
# ─────────────────────────────────────────────────────────────────────────────


# -------------------------------
# 1. One Transformer Block
# -------------------------------
# Think of this like one "thinking layer".
# The model stacks 8 of these on top of each other.
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias=False):
        super().__init__()

        # Attention — lets each square "look at" all other squares
        # bias param now wired from config
        self.attn = nn.MultiheadAttention(
            n_embd, n_head, dropout=dropout, batch_first=True, bias=bias
        )

        # Feed-forward — processes each square individually after attention
        # GELU is smoother than ReLU, works better in transformers
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=bias),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=bias),
            nn.Dropout(dropout),
        )

        # LayerNorm — keeps numbers stable during training
        # Pre-LN style: normalize BEFORE attention (more stable than after)
        # elementwise_affine keeps scale/shift even when bias=False on linears
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # Attention step
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop(attn_out)   # residual connection — add input back in

        # Feed-forward step
        x = x + self.ff(self.ln2(x))  # residual connection again

        return x


# -------------------------------
# 2. The Full Chess Model
# -------------------------------
class ChessGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        bias    = getattr(config, 'bias',    False)
        dropout = getattr(config, 'dropout', 0.1)

        # Step 1 — project each square's 13 features into a bigger vector
        self.input_proj = nn.Linear(config.square_dim, config.n_embd, bias=bias)

        # Step 2 — positional embedding: tells the model WHERE each square is
        # Small random numbers so each square starts with a unique identity
        self.pos_emb = nn.Parameter(torch.randn(1, 65, config.n_embd) * 0.02)

        # Step 2b — extra_embedding: optional learned token added to every square
        # Enabled via CommonConfig.extra_embedding = True.
        # Acts as a global "board context" signal mixed in before the transformer.
        self.extra_emb: nn.Parameter | None = None
        if getattr(config, 'extra_embedding', False):
            self.extra_emb = nn.Parameter(torch.randn(1, 1, config.n_embd) * 0.02)

        # Dropout after input — randomly zeros some values to prevent overfitting
        self.input_drop = nn.Dropout(dropout)

        # Step 3 — stack of transformer blocks (the actual thinking)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.n_embd, config.n_head, dropout, bias=bias)
            for _ in range(config.n_layer)
        ])

        # Step 4 — final normalisation before output
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Step 5 — output heads
        # FIXED: was 4672, but masking uses from_sq*64+to_sq → max index 4095
        # so this MUST be 4096 to match the masking code in chess_app.py
        self.move_head  = nn.Linear(config.n_embd, 4096, bias=bias)
        self.score_head = nn.Linear(config.n_embd, 1,    bias=bias)

    def forward(self, x):
        # x shape: [batch, 65, 13]
        # 65 = 64 squares + 1 extra token, 13 = features per square

        x = self.input_proj(x)              # [B, 65, n_embd]
        x = x + self.pos_emb               # add positional info

        # If extra_embedding is enabled, broadcast and add the learned token
        if self.extra_emb is not None:
            x = x + self.extra_emb         # [B, 65, n_embd] + [1, 1, n_embd]

        x = self.input_drop(x)

        for block in self.blocks:           # pass through all 8 layers
            x = block(x)

        x = self.ln_f(x)
        x = x.mean(dim=1)                   # average all squares → [B, n_embd]

        move_logits = self.move_head(x)     # [B, 4096]
        score_pred  = self.score_head(x)    # [B, 1]

        return move_logits, score_pred

    def count_params(self):
        # Call this to see how big your model is: model.count_params()
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {n:,}  ({n/1e6:.1f}M)")


# -------------------------------
# 3. Loss Function
# -------------------------------
# Loss = how wrong the model was. Lower is better.
# Two parts:
#   move_loss  — was the predicted move correct?
#   score_loss — was the eval score close to the real one?
def compute_loss(move_logits, move_targets, score_pred, score_targets, config):

    # Cross entropy for move prediction (like a classification problem)
    move_loss = F.cross_entropy(move_logits, move_targets)

    # MSE for score prediction (like a regression problem)
    score_loss = F.mse_loss(score_pred.squeeze(-1), score_targets)

    # Combine both losses using weights from config
    total_loss = (
        config.weight_loss_move  * move_loss +
        config.weight_loss_score * score_loss
    )

    return total_loss, move_loss, score_loss