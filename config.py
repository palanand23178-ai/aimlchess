from dataclasses import dataclass
import torch


def get_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    print(" MPS not available,back to CPU.")
    return "cpu"

@dataclass
class CommonConfig:
    extra_embedding: bool = False
    device: str = get_device()


# Model configuration:
@dataclass
class ModelCFG:
    # Input features for each squre :
    # 12 piece channels (6 white + 6 black) + 1 side-to-move = 13:
    square_dim: int = 13

    # Transformer architecture 
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = False
    weight_loss_move: float  = 1.0
    weight_loss_score: float = 0.5


#train configuration
@dataclass
class TrainCFG:
    train_features: str = "data/train_features.npy"
    train_moves: str    = "data/train_moves.npy"
    train_scores: str   = "data/train_scores.npy"
    bsz: int = 256
    max_steps: int = 35_000
    log_interval: int = 100
    save_interval: int = 1_000

    # Learning rate 
    lr: float     = 3e-4
    lr_min: float = 1e-5
    grad_clip: float = 1.0
    warmup_steps: int = 500
    early_stop_patience: int = 20
    pin_memory: bool = False
    num_workers: int = 0
    dtype: str = "float32"
    compile_model: bool = False

    def __post_init__(self):
        assert self.lr_min < self.lr, \
            f"lr_min ({self.lr_min}) must be less than lr ({self.lr})"
        assert self.bsz > 0 and (self.bsz & (self.bsz - 1)) == 0, \
            f"bsz ({self.bsz}) should be a power of 2 for best MPS performance"
        assert self.warmup_steps < self.max_steps, \
            f"warmup_steps ({self.warmup_steps}) must be less than max_steps ({self.max_steps})"