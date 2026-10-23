"""Constants and the bridge to the vendored official files.

Every competition-specific number (frame count, graded halves, pitch size) is read from the
vendored official `games.py`, so nothing here can silently disagree with the scorer.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL_DIR = os.path.join(ROOT, "third_party", "mtah_official")
if OFFICIAL_DIR not in sys.path:
    sys.path.insert(0, OFFICIAL_DIR)

import games as official_games  # noqa: E402
import possession_metrics as pm  # noqa: E402

GAME = official_games.GAME
FPS = official_games.GAMES[GAME]["fps"]
N_FRAMES = official_games.GAMES[GAME]["n_video_frames"]
PITCH = official_games.GAMES[GAME]["pitch"]  # (length, width) in metres

# Label codes, identical to possession_metrics.
H, A, D, X = pm.H, pm.A, pm.D, pm.X
LABELS = pm.LABELS  # ("H", "A", "D", "X")

PUBLIC_GT = os.path.join(OFFICIAL_DIR, "gt", "BAR-ATH_possession_gt_public.csv")
BASELINE_DIR = os.path.join(OFFICIAL_DIR, "baselines")


def graded_blocks():
    """[(start, stop)] half-open graded spans in video frames (the two halves)."""
    return [(first, last + 1) for _, first, last in official_games.half_video_blocks(GAME)]


def data_dir():
    """Where the gated match bundle lives (never inside this repository)."""
    return os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data")
