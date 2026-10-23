"""Paired per-run comparison of two cached prob sets under the same decoder."""
import sys, json, os
import numpy as np
from _common import GAMES, SAMPLE_DIR, pm
from possession.external import load_open_match
from possession.decode import fit_duration_model, decode_blocks
a_tag, b_tag = sys.argv[1], sys.argv[2]
clean = {g: load_open_match(SAMPLE_DIR, g) for g in GAMES}
rows = []
for g in GAMES:
    labs = [clean[x] for x in GAMES if x != g]
    cat = np.concatenate([x.labels for x in labs]); off = 0; bl = []
    for x in labs:
        bl += [(a + off, b + off) for a, b in x.blocks]; off += x.n_frames
    dm = fit_duration_model(cat, bl, k=2)
    for s in (0, 1):
        sc = []
        for tag in (a_tag, b_tag):
            z = np.load(f"cache/lp_{tag}_g{g}_s{s}.npz")
            pred = decode_blocks(z["lp"], [tuple(b) for b in z["blocks"]], len(z["labels"]), dm, 1.0)
            sc.append(pm.evaluate(z["labels"], pred)["spell_f1_mean"])
        rows.append((g, s, sc[0], sc[1], sc[1] - sc[0]))
        print(f"game {g} seed {s}: {a_tag} {sc[0]:.4f}  {b_tag} {sc[1]:.4f}  diff {sc[1]-sc[0]:+.4f}")
d = np.array([r[4] for r in rows])
print(f"mean diff {d.mean():+.4f}, runs improved {np.sum(d>0)}/{len(d)}, min {d.min():+.4f}")
json.dump({"a": a_tag, "b": b_tag, "runs": rows, "mean_diff": float(d.mean()), "n_improved": int(np.sum(d > 0))},
          open(f"../reports/paired_{a_tag}_vs_{b_tag}.json", "w"), indent=1)
