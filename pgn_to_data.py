import chess
import chess.pgn
import numpy as np
import os
import io
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────

PGN_FILE      = "lichess.pgn"        # path to your downloaded .pgn file
OUT_DIR       = "data"               # where .npy files are saved
MAX_GAMES     = 10_000               # how many games to parse (start here)
MIN_ELO       = 1500                 # skip games below this Elo (noisy moves)
MAX_BLUNDER   = 3.0                  # skip position if eval swings > 3 pawns
SKIP_OPENING  = 5                    # skip first N half-moves (pure book = noise)
MAX_POSITIONS = 800_000              # hard cap on total positions

# Piece type → channel index (0-5 = white, 6-11 = black)
PIECE_TO_CH = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}


# ── Feature Extraction ────────────────────────────────────────────────────────

def board_to_features(board: chess.Board) -> np.ndarray:
    """
    Convert a chess.Board into a [65, 13] float32 array.

    Slots 0–63 : one row per square
        ch 0–11 : piece presence (1.0 if piece of that type/color is on square)
        ch 12   : side to move (1.0 = white to move, 0.0 = black to move)

    Slot 64 : global token (summary information)
        ch 0  : white can castle kingside
        ch 1  : white can castle queenside
        ch 2  : black can castle kingside
        ch 3  : black can castle queenside
        ch 4  : en-passant available (binary flag, not the square index)
        ch 5  : halfmove clock normalized to [0, 1]  (/ 100)
        ch 6–12 : zeros (unused, kept for shape consistency)
    """
    features = np.zeros((65, 13), dtype=np.float32)

    side_to_move = 1.0 if board.turn == chess.WHITE else 0.0

    # Per-square features
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is not None:
            ch = PIECE_TO_CH[(piece.piece_type, piece.color)]
            features[sq, ch] = 1.0
        features[sq, 12] = side_to_move   # same value broadcast to all squares

    # Global token (slot 64)
    features[64, 0] = float(board.has_kingside_castling_rights(chess.WHITE))
    features[64, 1] = float(board.has_queenside_castling_rights(chess.WHITE))
    features[64, 2] = float(board.has_kingside_castling_rights(chess.BLACK))
    features[64, 3] = float(board.has_queenside_castling_rights(chess.BLACK))
    features[64, 4] = float(board.ep_square is not None)
    features[64, 5] = min(board.halfmove_clock / 100.0, 1.0)

    return features


def parse_score(comment: str) -> Optional[float]:
    """
    Extract eval score from a Lichess PGN comment like [%eval 1.23] or [%eval #3].
    Returns a float in [-1, 1] or None if no eval found.
    Mate scores are mapped to ±1.0.
    """
    if "%eval" not in comment:
        return None
    try:
        part = comment.split("%eval")[1].strip().split("]")[0].strip()
        if part.startswith("#"):
            # Mate in N — map to ±1
            return 1.0 if not part.startswith("#-") else -1.0
        val = float(part)
        # Clamp pawns to [-10, 10] then normalize to [-1, 1]
        val = max(-10.0, min(10.0, val))
        return val / 10.0
    except (ValueError, IndexError):
        return None


# ── Main Parser ───────────────────────────────────────────────────────────────

def parse_pgn(pgn_path: str):
    features_list = []
    moves_list    = []
    scores_list   = []
    fens_list     = []
    seen_fens     = set()   # deduplication

    games_parsed   = 0
    positions_kept = 0
    positions_skip_blunder  = 0
    positions_skip_duplicate = 0
    positions_skip_opening  = 0

    print(f"Opening {pgn_path} ...")

    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:

        while games_parsed < MAX_GAMES and positions_kept < MAX_POSITIONS:

            game = chess.pgn.read_game(f)
            if game is None:
                print("Reached end of PGN file.")
                break

            # ── Elo filter ────────────────────────────────────────────────────
            try:
                white_elo = int(game.headers.get("WhiteElo", "0"))
                black_elo = int(game.headers.get("BlackElo", "0"))
                if white_elo < MIN_ELO or black_elo < MIN_ELO:
                    continue
            except ValueError:
                continue

            board = game.board()
            node  = game

            prev_score: Optional[float] = None
            half_move  = 0

            for move in game.mainline_moves():

                node = node.variation(move)
                half_move += 1

                # ── Skip opening moves ────────────────────────────────────────
                if half_move <= SKIP_OPENING:
                    board.push(move)
                    continue

                # ── Extract eval from comment ─────────────────────────────────
                comment = node.comment or ""
                score   = parse_score(comment)

                # ── Blunder filter ────────────────────────────────────────────
                if score is not None and prev_score is not None:
                    swing = abs(score - prev_score) * 10   # back to pawn units
                    if swing > MAX_BLUNDER:
                        board.push(move)
                        prev_score = score
                        positions_skip_blunder += 1
                        continue

                # ── Deduplication ─────────────────────────────────────────────
                fen = board.fen()
                if fen in seen_fens:
                    board.push(move)
                    positions_skip_duplicate += 1
                    continue
                seen_fens.add(fen)

                # ── Build features ────────────────────────────────────────────
                feat     = board_to_features(board)
                move_idx = move.from_square * 64 + move.to_square

                # Use score if available, else default to 0.0 (neutral)
                score_val = score if score is not None else 0.0
                if prev_score is None:
                    prev_score = score_val

                features_list.append(feat)
                moves_list.append(move_idx)
                scores_list.append(score_val)
                fens_list.append(fen)

                positions_kept += 1
                prev_score = score

                board.push(move)

                if positions_kept >= MAX_POSITIONS:
                    break

            games_parsed += 1

            if games_parsed % 500 == 0:
                print(f"  Games: {games_parsed:>6} | Positions kept: {positions_kept:>8,}")

    print(f"\n── Parse complete ───────────────────────────────")
    print(f"  Games parsed       : {games_parsed:>8,}")
    print(f"  Positions kept     : {positions_kept:>8,}")
    print(f"  Skipped (blunder)  : {positions_skip_blunder:>8,}")
    print(f"  Skipped (duplicate): {positions_skip_duplicate:>8,}")
    print(f"  Skipped (opening)  : {positions_skip_opening:>8,}")

    return (
        np.array(features_list, dtype=np.float32),
        np.array(moves_list,    dtype=np.int64),
        np.array(scores_list,   dtype=np.float32),
        np.array(fens_list,     dtype=object),
    )


# ── Save ──────────────────────────────────────────────────────────────────────

def save_data(features, moves, scores, fens, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "train_features.npy"), features)
    np.save(os.path.join(out_dir, "train_moves.npy"),    moves)
    np.save(os.path.join(out_dir, "train_scores.npy"),   scores)
    np.save(os.path.join(out_dir, "train_fens.npy"),     fens)

    print(f"\n✅ Saved to '{out_dir}/'")
    print(f"  features : {features.shape}  ({features.nbytes / 1e6:.1f} MB)")
    print(f"  moves    : {moves.shape}")
    print(f"  scores   : {scores.shape}")
    print(f"  fens     : {fens.shape}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(PGN_FILE):
        print(f" PGN file not found: '{PGN_FILE}'")
        print("   Place your Lichess .pgn file in the project folder and update PGN_FILE.")
        exit(1)

    features, moves, scores, fens = parse_pgn(PGN_FILE)

    if len(features) == 0:
        print("❌ No positions were extracted. Check MIN_ELO or PGN format.")
        exit(1)

    save_data(features, moves, scores, fens, OUT_DIR)