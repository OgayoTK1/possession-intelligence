"""Leave-one-match-out: train the frame model on two open matches, predict the third.
Caches log-probabilities for the decoder experiments."""
import argparse, json, os, time
import numpy as np
from _common import open_matches, GAMES
from possession import models

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="full")
ap.add_argument("--drop", nargs="*", default=[], help="feature-name prefixes to ablate")
ap.add_argument("--rounds", type=int, default=models.N_ROUNDS)
args = ap.parse_args()

test = open_matches(seeds=(0, 1))
train_pool = open_matches(seeds=(5, 6))
os.makedirs("cache", exist_ok=True)
log = {}
for g in GAMES:
    t0 = time.time()
    tr = [m for (gg, s), m in train_pool.items() if gg != g]
    fm = models.train(tr, drop=args.drop, rounds=args.rounds)
    t_train = time.time() - t0
    for s in (0, 1):
        m = test[(g, s)]
        t1 = time.time()
        lp = fm.log_proba(m)
        np.savez_compressed(f"cache/lp_{args.tag}_g{g}_s{s}.npz", lp=lp.astype(np.float32),
                            labels=m.labels, blocks=np.array(m.blocks))
        log[f"g{g}s{s}"] = {"train_s": round(t_train, 1), "predict_s": round(time.time() - t1, 1)}
    imp = sorted(zip(fm.booster.feature_importance("gain"), fm.names), reverse=True)[:15]
    print(f"game {g}: trained in {t_train:.0f}s on {len(tr)} matches; top features:",
          [n for _, n in imp[:10]], flush=True)
json.dump(log, open(f"cache/timing_{args.tag}.json", "w"), indent=1)
