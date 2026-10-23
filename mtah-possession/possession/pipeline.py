"""End-to-end: competition ATD -> submission CSV, plus the open-data training of the final model.

Only aggregate statistics ever leave this module (schema report, label counts, possession
diagnostics). No frame-level tracking is written anywhere except the submission itself.
"""

import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

from .constants import (A, BASELINE_DIR, D, FPS, H, LABELS, N_FRAMES, OFFICIAL_DIR, PITCH,
                        PUBLIC_GT, ROOT, X, data_dir, graded_blocks, official_games, pm)
from .decode import decode_blocks, fit_duration_model
from .epts import load_epts
from .models import FrameModel, train

CONFIG_DEFAULT = os.path.join(ROOT, "configs", "final.json")


def load_config(path=None):
    with open(path or CONFIG_DEFAULT) as f:
        cfg = json.load(f)
    return cfg


# -- the competition match ---------------------------------------------------------------------

def load_competition(data_root=None):
    """The tactical ATD as a Match: row = video frame, blocks = the two graded halves."""
    root = data_root or data_dir()
    paths = official_games.asset_paths(data_dir=root)
    missing = [p for k, p in paths.items() if k in ("atd_meta", "atd_raw") and not os.path.isfile(p)]
    if missing:
        raise IOError("match data not found:\n  {}\nSet MTAH_DATA_DIR to the unpacked bundle "
                      "(see the official DATA.md).".format("\n  ".join(missing)))
    match, report = load_epts(paths["atd_meta"], paths["atd_raw"], pitch=PITCH,
                              name=official_games.GAME, n_frames=N_FRAMES)
    if report["first_frame_id"] != 1 or not report["contiguous"]:
        raise ValueError("ATD is not one record per video frame numbered from 1: {}".format(
            {k: report[k] for k in ("first_frame_id", "last_frame_id", "contiguous")}))
    match.blocks = graded_blocks()
    return match.validate(), report


# -- training the final model on the open matches ----------------------------------------------

def training_matches(sample_dir, seeds):
    from .external import degrade, load_open_match, noise_for_seed, visibility_runs_from_baseline
    runs = visibility_runs_from_baseline(os.path.join(BASELINE_DIR, "baseline_raw.csv"),
                                         graded_blocks())
    clean, degraded = [], []
    for g in (1, 2, 3):
        m = load_open_match(sample_dir, g)
        clean.append(m)
        for s in seeds:
            degraded.append(degrade(m, runs, seed=1000 * g + s, **noise_for_seed(s)))
    return clean, degraded


def train_final(cfg, sample_dir, out_path):
    clean, degraded = training_matches(sample_dir, cfg["train_seeds"])
    fm = train(degraded, drop=cfg.get("drop") or None, rounds=cfg["rounds"])
    fm.save(out_path)
    # duration model from all clean open labels
    cat, blocks, off = [], [], 0
    for m in clean:
        cat.append(m.labels)
        blocks += [(a + off, b + off) for a, b in m.blocks]
        off += m.n_frames
    dm = fit_duration_model(np.concatenate(cat), blocks, k=cfg["decoder"]["k"])
    with open(out_path + ".duration.json", "w") as f:
        json.dump({"mu": dm["mu"].tolist(), "trans": dm["trans"].tolist(),
                   "prior": dm["prior"].tolist(), "k": dm["k"]}, f, indent=1)
    return fm


def load_duration(path):
    with open(path) as f:
        d = json.load(f)
    return {"mu": np.array(d["mu"]), "trans": np.array(d["trans"]),
            "prior": np.array(d["prior"]), "k": int(d["k"])}


# -- prediction --------------------------------------------------------------------------------

def predict_labels(match, cfg, model_path):
    fm = FrameModel.load(model_path)
    dm = load_duration(model_path + ".duration.json")
    logp = fm.log_proba(match)
    codes = decode_blocks(logp, match.blocks, match.n_frames, dm, cfg["decoder"]["alpha"])
    return codes, logp


def write_submission(path, codes):
    """frame,label CSV, '\n' line endings, no index column. Deterministic bytes."""
    letters = np.array(LABELS)[codes]
    with open(path, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(letters))
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_official_validator(path):
    script = os.path.join(OFFICIAL_DIR, "validate_submission.py")
    out = subprocess.run([sys.executable, script, path], capture_output=True, text=True)
    return out.returncode, out.stdout + out.stderr


def diagnostics(codes, blocks):
    """Aggregate, non-reconstructive statistics of a prediction."""
    graded = np.zeros(len(codes), dtype=bool)
    for a, b in blocks:
        graded[a:b] = True
    spells = pm.possession_spells(codes, blocks)
    dur = np.array([b - a for a, b, _ in spells], dtype=float) / FPS
    counts = {LABELS[k]: int(np.sum(codes[graded] == k)) for k in range(4)}
    return {
        "n_rows": int(len(codes)),
        "graded_frames": int(graded.sum()),
        "label_counts_graded": counts,
        "label_counts_ungraded": {LABELS[k]: int(np.sum(codes[~graded] == k)) for k in range(4)},
        "n_possessions": int(len(spells)),
        "possessions_per_min": float(len(spells) / (graded.sum() / (60.0 * FPS))),
        "possession_duration_s": {q: float(np.percentile(dur, p)) for q, p in
                                  (("p10", 10), ("median", 50), ("p90", 90))} if len(dur) else {},
        "possessions_under_2s": int(np.sum(dur < 2.0)),
        "mean_possession_s": float(dur.mean()) if len(dur) else 0.0,
        "n_dead_segments": int(sum(1 for a, b, c in pm.segments(codes, blocks) if c == D)),
        "home_share_alive": float(np.sum(codes == H) / max(1, np.sum((codes == H) | (codes == A)))),
    }


def score_public(path):
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    pred = pm.label_codes(pm.read_label_csv(path))
    r = pm.evaluate(gt, pred, bootstrap_n=1000)
    return {k: r[k] for k in ("spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells",
                              "mean_spell_duration_s", "spell_bootstrap_low",
                              "spell_bootstrap_high", "alive_dead_f1", "team_acc_both_alive")}
