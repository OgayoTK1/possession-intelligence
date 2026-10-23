"""Command line entry point.

    python -m possession.cli reproduce --team TEAM_NAME   # the one command: predict, validate, report
    python -m possession.cli audit                        # schema / alignment audit (aggregates only)
    python -m possession.cli train                        # retrain the frame model from open data
    python -m possession.cli baseline                     # rebuild the official baseline here

Match data is read from $MTAH_DATA_DIR (default ./data). It is never written anywhere else.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from .constants import ROOT, graded_blocks, pm


def _report_path(name):
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    return os.path.join(ROOT, "reports", name)


def cmd_audit(args):
    from .pipeline import load_competition
    from .smart_tagging import load_intervals, summary as st_summary
    from .constants import official_games
    t0 = time.time()
    match, report = load_competition(args.data)
    vis = np.isfinite(match.ball[:, 0])
    graded = np.zeros(match.n_frames, dtype=bool)
    for a, b in match.blocks:
        graded[a:b] = True
    tracked = np.isfinite(match.px).sum(axis=1)
    report.update({
        "load_seconds": round(time.time() - t0, 1),
        "graded_blocks": match.blocks,
        "ball_present_graded": float(vis[graded].mean()),
        "ball_present_all": float(vis.mean()),
        "mean_tracked_graded": float(tracked[graded].mean()),
        "mean_tracked_ungraded": float(tracked[~graded].mean()),
        "extra_channels": sorted(match.extras.keys()),
        "documented_expectations": {"ball_present_all": 0.569, "ball_present_graded": 0.593,
                                    "mean_tracked_graded": 20.5, "n_slots": 46},
    })
    report["alignment_vs_official_baseline_raw"] = baseline_agreement(match)
    # Touch-detector calibration: the threshold was set from the open-data noise model. If the
    # real ATD's rate is far from the open reference, the ball noise level differs (REPORT §7).
    from .features import frame_features
    names, F = frame_features(match)
    ts = F[:, names.index("touch_sign")]
    ok = graded & np.isfinite(ts)
    report["touch_rate_per_visible_frame"] = float(np.mean(ts[ok] != 0)) if ok.any() else None
    report["touch_rate_open_degraded_reference"] = {"mean": 0.0918, "range": [0.0804, 0.1036]}
    paths = official_games.asset_paths(data_dir=args.data or None)
    if os.path.isfile(paths["smart_tagging"]):
        report["smart_tagging"] = st_summary(load_intervals(paths["smart_tagging"]))
    out = _report_path("audit_competition.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps(report, indent=1, default=str))
    print("\nwritten:", out, "(aggregate statistics only -- safe to share)")


def nearest_team_codes(match):
    """The official baseline's step 1 on our loader: 0/1 team of the nearest slot, -1 no ball."""
    bx, by = match.ball[:, 0], match.ball[:, 1]
    d = np.hypot(match.px - bx[:, None], match.py - by[:, None])
    filled = np.where(np.isfinite(d), d, np.inf)
    j = np.argmin(filled, axis=1)
    ok = np.isfinite(bx) & np.isfinite(filled[np.arange(len(j)), j])
    return np.where(ok, match.team_of_slot[j], -1).astype(np.int8)


def baseline_agreement(match):
    """Per-frame agreement with the committed, public baseline_raw.csv (all 199,319 frames).

    ~1.0 proves frame alignment (row = frame_id - 1), the slot -> team mapping and the unit
    handling on the real file. Lower values are reported, not corrected.
    """
    import pandas as pd
    from .constants import BASELINE_DIR
    ref = pd.read_csv(os.path.join(BASELINE_DIR, "baseline_raw.csv"))["label"].values
    mine = np.array(["H", "A", "D"])[np.where(nearest_team_codes(match) < 0, 2,
                                              nearest_team_codes(match))]
    if len(ref) != len(mine):
        return {"error": "length {} vs {}".format(len(mine), len(ref))}
    same = mine == ref
    shifted = {s: float(np.mean(np.roll(mine, s) == ref)) for s in (-1, 1)}
    return {"agreement": float(same.mean()),
            "agreement_ball_frames": float(same[ref != "D"].mean()) if (ref != "D").any() else None,
            "agreement_if_shifted_by_one": shifted}


def cmd_train(args):
    from .pipeline import load_config, train_final
    cfg = load_config(args.config)
    out = args.out or os.path.join(ROOT, cfg["model_path"])
    t0 = time.time()
    train_final(cfg, args.sample_dir, out)
    print("trained in {:.0f}s -> {}".format(time.time() - t0, out))


def cmd_reproduce(args):
    from .pipeline import (diagnostics, load_competition, load_config, predict_labels,
                           run_official_validator, score_public, sha256, write_submission)
    cfg = load_config(args.config)
    model_path = os.path.join(ROOT, cfg["model_path"])
    timing = {}
    t0 = time.time()
    match, schema = load_competition(args.data)
    timing["load_s"] = round(time.time() - t0, 1)
    t1 = time.time()
    codes, _ = predict_labels(match, cfg, model_path)
    timing["predict_s"] = round(time.time() - t1, 1)
    team = args.team or cfg.get("team_name", "TEAM_NAME")
    out_dir = args.out_dir or os.path.join(ROOT, "submission")
    os.makedirs(out_dir, exist_ok=True)
    path = write_submission(os.path.join(out_dir, "{}.csv".format(team)), codes)
    code, text = run_official_validator(path)
    audit = {
        "submission": os.path.relpath(path, ROOT),
        "sha256": sha256(path),
        "validator_exit_code": code,
        "validator_output": text.strip().splitlines(),
        "diagnostics": diagnostics(codes, match.blocks),
        "public_window": score_public(path),
        "timing": timing,
        "config": cfg,
        "schema": {k: schema[k] for k in ("n_rows", "first_frame_id", "last_frame_id",
                                          "contiguous", "n_slots", "player_channels",
                                          "ball_channels", "player_order_matches_layout")},
    }
    with open(_report_path("final_audit.json"), "w") as f:
        json.dump(audit, f, indent=1, default=str)
    print(text)
    print(json.dumps({k: audit[k] for k in ("submission", "sha256", "validator_exit_code",
                                            "diagnostics", "public_window", "timing")},
                     indent=1, default=str))
    if code != 0:
        sys.exit("official validator rejected the file")


def cmd_baseline(args):
    """The official nearest-player baseline on the shared loader, for a like-for-like check."""
    from .pipeline import load_competition, write_submission
    from .decode import majority_smooth
    match, _ = load_competition(args.data)
    from .features import frame_features
    names, F = frame_features(match)
    nih = F[:, names.index("nearest_is_home")]
    codes = np.where(np.isnan(nih), -1, np.where(nih > 0.5, 0, 1)).astype(np.int8)
    out_dir = args.out_dir or os.path.join(ROOT, "submission")
    os.makedirs(out_dir, exist_ok=True)
    for w in (0, 25, 50, 125):
        lab = np.full(match.n_frames, pm.X, dtype=np.int8)
        for a, b in match.blocks:
            lab[a:b] = majority_smooth(codes[a:b], w)
        p = write_submission(os.path.join(out_dir, "rebuilt_baseline_w{}.csv".format(w)), lab)
        print("wrote", p)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="possession")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("audit", "reproduce", "baseline", "train"):
        p = sub.add_parser(name)
        p.add_argument("--data", default=None, help="match bundle root (else $MTAH_DATA_DIR)")
        p.add_argument("--config", default=None)
        p.add_argument("--out-dir", default=None)
        p.add_argument("--team", default=None)
        p.add_argument("--out", default=None)
        p.add_argument("--sample-dir", default=os.path.join(ROOT, "external", "sample-data"))
    args = ap.parse_args(argv)
    {"audit": cmd_audit, "reproduce": cmd_reproduce, "baseline": cmd_baseline,
     "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
