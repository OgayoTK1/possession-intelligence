"""Run the official baseline algorithm on degraded open matches; compare with its published
full-match scores to check the degradation is realistic."""
import numpy as np
from _common import open_matches, score
from possession.decode import majority_smooth
from possession.features import frame_features

PUBLISHED = {0: (0.0486, 0.6532), 25: (0.1996, 0.6752), 50: (0.2433, 0.6724), 125: (0.2242, 0.6356)}
ms = open_matches(seeds=(0, 1))
rows = []
for key, m in ms.items():
    names, F = frame_features(m)
    nih = F[:, names.index("nearest_is_home")]
    codes = np.where(np.isnan(nih), -1, np.where(nih > 0.5, 0, 1)).astype(np.int8)
    for w in PUBLISHED:
        pred = np.full(m.n_frames, 3, dtype=np.int8)
        for a, b in m.blocks:
            pred[a:b] = majority_smooth(codes[a:b], w)
        s = score(m.labels, pred); rows.append((key, w, s))
print("window  published(spell,macro)   open-degraded mean spellF1 [per match]   macroF1")
for w in PUBLISHED:
    sp = [s["spellF1"] for k, ww, s in rows if ww == w]; mf = [s["macroF1"] for k, ww, s in rows if ww == w]
    print(f"{w:5d}   {PUBLISHED[w][0]:.4f} {PUBLISHED[w][1]:.4f}        {np.mean(sp):.4f} {np.round(sp,3)}   {np.mean(mf):.4f}")
