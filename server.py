from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess
import torch
from model import ChessGPT
from config import ModelCFG
from pgn_to_data import board_to_features as board_to_tensor

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = ChessGPT(ModelCFG()).to(device)
checkpoint = torch.load("model_best1.pth", map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

class MoveRequest(BaseModel):
    fen: str

def decode_move(move_id):
    return chess.Move(move_id // 64, move_id % 64)

def mask_illegal_moves(logits, board):
    masked = torch.full_like(logits, -1e9)
    ids = [m.from_square * 64 + m.to_square for m in board.legal_moves]
    masked[0, ids] = logits[0, ids]
    return masked

@app.post("/ai_move")
def get_ai_move(req: MoveRequest):
    board = chess.Board(req.fen)
    x = torch.tensor(board_to_tensor(board), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        move_logits, score_pred = model(x)
    masked = mask_illegal_moves(move_logits, board)
    best_move = decode_move(masked.argmax().item())
    board.push(best_move)
    return {
        "move": best_move.uci(),
        "fen": board.fen(),
        "score": round(score_pred.item(), 4)
    }