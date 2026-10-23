"""Predicted vs true dead-ball share and possession rate across tracking-noise levels (LOMO)."""
import json, os, sys
import numpy as np
from _common import GAMES, SAMPLE_DIR, pm, vis_runs
from possession.external import load_open_match, degrade, NOISE_LEVELS
from possession.decode import fit_duration_model, decode_blocks
from possession import models
tag = sys.argv[1] if len(sys.argv) > 1 else "v5_smooth"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runs = vis_runs(); clean = {g: load_open_match(SAMPLE_DIR, g) for g in GAMES}
from _common import open_matches
train_pool = open_matches(seeds=(4, 5, 6))
out = {}
for g in GAMES:
    fm = models.train([m for (gg, s), m in train_pool.items() if gg != g])
    labs = [clean[x] for x in GAMES if x != g]
    cat = np.concatenate([x.labels for x in labs]); off = 0; bl = []
    for x in labs:
        bl += [(a + off, b + off) for a, b in x.blocks]; off += x.n_frames
    dm = fit_duration_model(cat, bl, k=2)
    for li, (p, b, d) in enumerate(NOISE_LEVELS):
        m = degrade(clean[g], runs, seed=7000 + 10 * g + li, player_noise_m=p, ball_noise_m=b, ball_drift_m=d)
        pred = decode_blocks(fm.log_proba(m), m.blocks, m.n_frames, dm, 1.0)
        r = pm.evaluate(m.labels, pred); gr = m.labels != 3
        out.setdefault(f"noise_{p}", []).append({"spellF1": r["spell_f1_mean"], "D_true": float(np.mean(m.labels[gr] == 2)),
            "D_pred": float(np.mean(pred[gr] == 2)), "poss_pred": r["n_pred_spells"], "poss_true": r["n_gt_spells"]})
for k, v in out.items():
    agg = {kk: float(np.mean([x[kk] for x in v])) for kk in v[0]}
    out[k] = {"runs": v, "mean": agg}
    print(f"{k:10s} spellF1 {agg['spellF1']:.3f}  D share true {agg['D_true']:.3f} pred {agg['D_pred']:.3f}  possessions pred/true {agg['poss_pred']:.0f}/{agg['poss_true']:.0f}")
json.dump(out, open(os.path.join(ROOT, "reports", f"noise_robustness_{tag}.json"), "w"), indent=1)
