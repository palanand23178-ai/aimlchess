import streamlit as st
import chess
import torch
import json
import streamlit.components.v1 as components

# 1. FIXED: Removed the emoji "♟" which causes the PIL error on some systems
st.set_page_config(page_title="Chess AI", page_icon=":chess_pawn:", layout="wide")

st.title("♟ Chess AI")

# Check if model exists before trying to load it
import os
if not os.path.exists("model_best1.pth"):
    st.error("Model file 'model_best1.pth' not found in this folder!")
else:
    st.success("Model file found! The PIL error should be gone now.")

# Dummy board to test if UI renders
if "board" not in st.session_state:
    st.session_state.board = chess.Board()

st.write("Current FEN:", st.session_state.board.fen())
