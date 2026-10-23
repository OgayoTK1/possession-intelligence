"""Isolate the decoder: official nearest-player evidence (incl. 'no ball => D') fed to the HSMM.
Also records the official baseline (majority vote) with full metrics, for the experiment table."""
import json, time, os
import numpy as np
from _common import open_matches, score, GAMES, SAMPLE_DIR
from possession.external import load_open_match
from possession.features import frame_features
from possession.decode import fit_duration_model, decode_blocks, majority_smooth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ms = open_matches(seeds=(0, 1))
clean = {g: load_open_match(SAMPLE_DIR, g) for g in GAMES}
res = {}
for (g, s), m in ms.items():
    names, F = frame_features(m)
    nih = F[:, names.index("nearest_is_home")]
    codes = np.where(np.isnan(nih), -1, np.where(nih > 0.5, 0, 1)).astype(np.int8)
    for w in (25, 50, 125):
        pred = np.full(m.n_frames, 3, dtype=np.int8)
        for a, b in m.blocks:
            pred[a:b] = majority_smooth(codes[a:b], w)
        t0 = time.time(); sc = score(m.labels, pred); res.setdefault(f"official_baseline_w{w}", []).append(sc)
    labs = [clean[x] for x in GAMES if x != g]
    cat = np.concatenate([x.labels for x in labs]); off = 0; blocks = []
    for x in labs:
        blocks += [(a + off, b + off) for a, b in x.blocks]; off += x.n_frames
    for conf in (0.6, 0.8):
        p = np.full((m.n_frames, 3), (1 - conf) / 2)
        p[codes == 0, 0] = conf; p[codes == 1, 1] = conf; p[codes == -1, 2] = conf
        for al in (0.3, 1.0):
            t0 = time.time()
            pred = decode_blocks(np.log(p), m.blocks, m.n_frames, fit_duration_model(cat, blocks, k=2), al)
            sc = score(m.labels, pred); sc["runtime_s"] = time.time() - t0
            res.setdefault(f"baseline_evidence_hsmm_c{conf}_a{al}", []).append(sc)
summ = {}
for k, lst in res.items():
    row = {key: float(np.mean([r[key] for r in lst])) for key in lst[0]}
    sp = [r["spellF1"] for r in lst]; row["spell_sd_over_matches"] = float(np.std([np.mean(sp[i*2:i*2+2]) for i in range(3)]))
    summ[k] = row
    print(f"{k:38s} spellF1 {row['spellF1']:.4f}  macroF1 {row['macroF1']:.4f}  poss {row['n_pred']:.0f}/{row['n_gt']:.0f}  transF1 {row['trans_f1_25']:.3f}")
json.dump(summ, open(os.path.join(ROOT, "reports", "baseline_evidence.json"), "w"), indent=1)
