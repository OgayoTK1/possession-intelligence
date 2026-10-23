"""Decoder comparison on cached LOMO probabilities: argmax, smoothed argmax, HSMM grid."""
import argparse, json, time
import numpy as np
from _common import score, GAMES, SAMPLE_DIR
from possession.external import load_open_match
from possession.decode import fit_duration_model, decode_blocks, argmax_blocks, majority_smooth

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="full")
ap.add_argument("--alphas", nargs="*", type=float, default=[0.05, 0.1, 0.2, 0.35, 0.6, 1.0])
ap.add_argument("--ks", nargs="*", type=int, default=[1, 3, 6])
ap.add_argument("--out", default=None)
args = ap.parse_args()

clean = {g: load_open_match(SAMPLE_DIR, g) for g in GAMES}
dur = {}
for g in GAMES:
    # duration model from the *training* matches of this fold only
    labs = [clean[x] for x in GAMES if x != g]
    cat = np.concatenate([m.labels for m in labs]); off = 0; blocks = []
    for m in labs:
        blocks += [(a + off, b + off) for a, b in m.blocks]; off += m.n_frames
    dur[g] = (cat, blocks)

results = {}
def add(name, g, s, sc):
    results.setdefault(name, []).append(sc)

for g in GAMES:
    for s in (0, 1):
        z = np.load(f"cache/lp_{args.tag}_g{g}_s{s}.npz")
        lp, labels, blocks = z["lp"], z["labels"], [tuple(b) for b in z["blocks"]]
        T = len(labels)
        am = argmax_blocks(lp, blocks, T)
        add("argmax", g, s, score(labels, am))
        for w in (25, 50):
            # smoothed argmax: a team vote ignoring D, D kept where a majority says D
            pred = np.full(T, 3, dtype=np.int8)
            for a, b in blocks:
                p = np.exp(lp[a:b]); cs = np.cumsum(np.vstack([np.zeros(3), p]), 0)
                idx = np.arange(b - a); lo = np.maximum(0, idx - w); hi = np.minimum(b - a, idx + w + 1)
                pred[a:b] = np.argmax(cs[hi] - cs[lo], axis=1)
            add(f"meanprob_w{w}", g, s, score(labels, pred))
        for k in args.ks:
            model = fit_duration_model(*dur[g], k=k)
            for al in args.alphas:
                t0 = time.time()
                pred = decode_blocks(lp, blocks, T, model, al)
                sc = score(labels, pred); sc["runtime_s"] = time.time() - t0
                add(f"hsmm_k{k}_a{al}", g, s, sc)

print(f"{'method':22s} {'spellF1':>8s} {'sd':>6s} {'macroF1':>8s} {'poss/gt':>9s} {'dur_s':>6s} {'transF1':>8s} {'ADF1':>6s} {'team':>6s}")
summary = {}
for name, lst in results.items():
    sp = np.array([r["spellF1"] for r in lst])
    row = {key: float(np.mean([r[key] for r in lst])) for key in lst[0]}
    row["spell_sd_over_matches"] = float(np.std([np.mean(sp[i*2:i*2+2]) for i in range(3)]))
    summary[name] = row
    print(f"{name:22s} {row['spellF1']:8.4f} {row['spell_sd_over_matches']:6.3f} {row['macroF1']:8.4f} {row['n_pred']:5.0f}/{row['n_gt']:3.0f} {row['dur']:6.1f} {row['trans_f1_25']:8.3f} {row['alive_dead_f1']:6.3f} {row['team_acc']:6.3f}")
json.dump(summary, open(args.out or f"reports/decode_grid_{args.tag}.json", "w"), indent=1)
