"""Leave-one-Metrica-match-out with the 20 SkillCorner matches added to every training fold.
Test runs are identical to exp_lomo_train.py, so results pair with v5_smooth."""
import argparse, os, time
import numpy as np
from _common import open_matches, GAMES
from possession import models
from possession.skillcorner import list_matches, load_skillcorner_match

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="v6_sc"); ap.add_argument("--n-sc", type=int, default=20); ap.add_argument("--team-only", action="store_true")
args = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sc = [load_skillcorner_match(d) for d in list_matches(os.path.join(os.path.dirname(ROOT), "ext", "skc"))[:args.n_sc]]
print(len(sc), "SkillCorner matches loaded", flush=True)
test = open_matches(seeds=(0, 1)); pool = open_matches(seeds=(4, 5, 6))
for g in GAMES:
    t0 = time.time()
    tr = [m for (gg, s), m in pool.items() if gg != g] + sc
    fm = models.train(tr, team_only_prefixes=("sc",) if args.team_only else ())
    for s in (0, 1):
        m = test[(g, s)]
        np.savez_compressed(f"cache/lp_{args.tag}_g{g}_s{s}.npz", lp=fm.log_proba(m).astype(np.float32),
                            labels=m.labels, blocks=np.array(m.blocks))
    print(f"game {g}: trained on {len(tr)} matches in {time.time()-t0:.0f}s", flush=True)
