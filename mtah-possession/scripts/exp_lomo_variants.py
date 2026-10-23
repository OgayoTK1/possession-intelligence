"""Model-capacity and ensembling variants under the exact v5 protocol (same train pool, same
held-out runs), so every result pairs with v5_smooth.

    python exp_lomo_variants.py --tag v8_big --leaves 63 --rounds 600
    python exp_lomo_variants.py --tag v9_ens3 --seeds 7 11 13
"""
import argparse, os, time
import numpy as np
from _common import open_matches, GAMES
from possession import models

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--leaves", type=int, default=models.LGB_PARAMS["num_leaves"])
ap.add_argument("--rounds", type=int, default=models.N_ROUNDS)
ap.add_argument("--lr", type=float, default=models.LGB_PARAMS["learning_rate"])
ap.add_argument("--min-child", type=int, default=models.LGB_PARAMS["min_child_samples"])
ap.add_argument("--seeds", type=int, nargs="+", default=[models.LGB_PARAMS["seed"]])
args = ap.parse_args()

test = open_matches(seeds=(0, 1))
pool = open_matches(seeds=(4, 5, 6))
for g in GAMES:
    t0 = time.time()
    tr = [m for (gg, s), m in pool.items() if gg != g]
    fms = [models.train(tr, rounds=args.rounds,
                        params={"num_leaves": args.leaves, "learning_rate": args.lr,
                                "min_child_samples": args.min_child, "seed": sd})
           for sd in args.seeds]
    for s in (0, 1):
        m = test[(g, s)]
        p = np.mean([np.exp(fm.log_proba(m)) for fm in fms], axis=0)
        np.savez_compressed(f"cache/lp_{args.tag}_g{g}_s{s}.npz",
                            lp=np.log(np.clip(p, 1e-9, 1)).astype(np.float32),
                            labels=m.labels, blocks=np.array(m.blocks))
    print(f"game {g}: {len(fms)} model(s) in {time.time() - t0:.0f}s", flush=True)
