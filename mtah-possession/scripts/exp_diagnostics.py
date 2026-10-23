"""Error taxonomy for the selected system on the held-out open matches (cached v3_full probs)."""
import json, os
import numpy as np
from _common import GAMES, SAMPLE_DIR, pm
from possession.external import load_open_match
from possession.decode import fit_duration_model, decode_blocks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
TAG, K, ALPHA = (sys.argv[1] if len(sys.argv) > 1 else "v4_touch"), 2, 1.0
clean = {g: load_open_match(SAMPLE_DIR, g) for g in GAMES}
buckets = [(0, 2), (2, 5), (5, 15), (15, 1e9)]
agg = {"recall_by_duration@IoU0.5": {str(b): [0, 0] for b in buckets}, "false_splits": 0,
       "merges": 0, "turnovers_gt": 0, "turnovers_missed_25f": 0, "dead_starts_gt": 0,
       "dead_start_abs_err_frames": [], "dead_end_abs_err_frames": [], "dead_bound_missed": 0,
       "pred_dur_s": [], "gt_dur_s": []}

for g in GAMES:
    labs = [clean[x] for x in GAMES if x != g]
    cat = np.concatenate([x.labels for x in labs]); off = 0; blocks = []
    for x in labs:
        blocks += [(a + off, b + off) for a, b in x.blocks]; off += x.n_frames
    dm = fit_duration_model(cat, blocks, k=K)
    for s in (0, 1):
        z = np.load(os.path.join(ROOT, "scripts", "cache", f"lp_{TAG}_g{g}_s{s}.npz"))
        lp, gt, bl = z["lp"], z["labels"], [tuple(b) for b in z["blocks"]]
        pred = decode_blocks(lp, bl, len(gt), dm, ALPHA)
        gs, ps = pm.possession_spells(gt, bl), pm.possession_spells(pred, bl)
        agg["pred_dur_s"] += [(b - a) / 25 for a, b, _ in ps]; agg["gt_dur_s"] += [(b - a) / 25 for a, b, _ in gs]
        ov = pm.spell_overlaps(gs, ps)
        flags = pm.matched_gt_flags(ov, len(gs), 0.5)
        for (a, b, c), f in zip(gs, flags):
            d = (b - a) / 25
            for lo, hi in buckets:
                if lo <= d < hi:
                    agg["recall_by_duration@IoU0.5"][str((lo, hi))][0] += int(f)
                    agg["recall_by_duration@IoU0.5"][str((lo, hi))][1] += 1
        # splits / merges (same-team overlaps covering >= 20% of the gt spell)
        for i, (a, b, c) in enumerate(gs):
            n = sum(1 for (pa, pb, pc) in ps if pc == c and min(b, pb) - max(a, pa) > 0.2 * (b - a))
            agg["false_splits"] += int(n >= 2)
        for (pa, pb, pc) in ps:
            n = sum(1 for (a, b, c) in gs if c == pc and min(b, pb) - max(a, pa) > 0.2 * (b - a))
            agg["merges"] += int(n >= 2)
        gtr = [t for t in pm.transitions(gt, bl) if {t[1], t[2]} == {0, 1}]
        ptr = [t for t in pm.transitions(pred, bl) if t[2] in (0, 1)]
        agg["turnovers_gt"] += len(gtr)
        for f, a, b in gtr:
            if not any(abs(pf - f) <= 25 and pb == b for pf, pa, pb in ptr):
                agg["turnovers_missed_25f"] += 1
        gd = [(a, b) for a, b, c in pm.segments(gt, bl) if c == 2]
        pd_ = [(a, b) for a, b, c in pm.segments(pred, bl) if c == 2]
        for a, b in gd:
            agg["dead_starts_gt"] += 1
            cand = [(pa, pb) for pa, pb in pd_ if min(b, pb) > max(a, pa)]
            if not cand:
                agg["dead_bound_missed"] += 1; continue
            pa, pb = max(cand, key=lambda x: min(b, x[1]) - max(a, x[0]))
            agg["dead_start_abs_err_frames"].append(abs(pa - a)); agg["dead_end_abs_err_frames"].append(abs(pb - b))

out = {
    "recall_by_duration@IoU0.5": {k: {"matched": v[0], "n": v[1], "recall": round(v[0] / max(1, v[1]), 3)}
                                 for k, v in agg["recall_by_duration@IoU0.5"].items()},
    "false_splits_per_100_gt": round(100 * agg["false_splits"] / len(agg["gt_dur_s"]), 1),
    "merges_per_100_pred": round(100 * agg["merges"] / len(agg["pred_dur_s"]), 1),
    "turnovers_gt": agg["turnovers_gt"],
    "turnover_miss_rate_25f": round(agg["turnovers_missed_25f"] / max(1, agg["turnovers_gt"]), 3),
    "dead_segments_gt": agg["dead_starts_gt"],
    "dead_segments_missed": agg["dead_bound_missed"],
    "dead_start_abs_err_median_frames": float(np.median(agg["dead_start_abs_err_frames"])),
    "dead_end_abs_err_median_frames": float(np.median(agg["dead_end_abs_err_frames"])),
    "duration_s_gt": {"median": float(np.median(agg["gt_dur_s"])), "mean": float(np.mean(agg["gt_dur_s"]))},
    "duration_s_pred": {"median": float(np.median(agg["pred_dur_s"])), "mean": float(np.mean(agg["pred_dur_s"]))},
}
json.dump(out, open(os.path.join(ROOT, "reports", f"diagnostics_{TAG}.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
