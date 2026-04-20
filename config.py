# ============================================================
# config.py — Central configuration for the entire project
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "student-mat.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ── Target ───────────────────────────────────────────────────
TARGET_COL    = "G3"
PASS_THRESHOLD = 10          # G3 >= 10 → PASS (1), else FAIL (0)

# ── Features to DROP before training ─────────────────────────
# G1 & G2 are intermediate grades; keeping them would be data leakage
# for a real-world "predict early" scenario. Set INCLUDE_GRADES=True
# to include them (gives near-perfect accuracy but less realistic).
INCLUDE_GRADES = False
LEAKAGE_COLS   = ["G1", "G2", TARGET_COL]

# ── Categorical columns (will be one-hot encoded) ────────────
CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]

# ── Train / Test split ────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ── ANN hyper-parameters ─────────────────────────────────────
ANN_EPOCHS     = 80
ANN_BATCH_SIZE = 32

# ── XGBoost hyper-parameter search space ─────────────────────
XGB_PARAM_GRID = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [3, 5, 7],
    "learning_rate":     [0.05, 0.1, 0.2],
    "subsample":         [0.8, 1.0],
    "colsample_bytree":  [0.8, 1.0],
}
XGB_CV_FOLDS = 3

# ── Misc ──────────────────────────────────────────────────────
FIGURE_DPI = 150
