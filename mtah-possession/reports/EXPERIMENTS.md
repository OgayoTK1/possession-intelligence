# Experiment table (all numbers measured)

Leave-one-match-out on Metrica's three open matches, each held-out match degraded with two
seeds (6 test runs; ~346 possessions per run, ~1,040 distinct possessions). spellF1 sd is
the standard deviation of the per-match mean across the 3 held-out matches. Scored with the
official possession_metrics.evaluate. These are NOT competition scores.

| system | spellF1 | sd | macroF1 | poss pred / gt | mean dur s | transF1@25 | notes |
|---|---|---|---|---|---|---|---|
| Official baseline, majority ±25 | 0.1808 | 0.013 | 0.4144 | 634 / 346 | 5.8 | 0.174 |  |
| Official baseline, majority ±50 | 0.2149 | 0.013 | 0.4286 | 483 / 346 | 8.1 | 0.152 | published floor |
| Official baseline, majority ±125 | 0.2051 | 0.006 | 0.4386 | 333 / 346 | 13.4 | 0.099 |  |
| B. Baseline evidence + semi-Markov decoder | 0.2203 | 0.021 | 0.4089 | 357 / 346 | 9.5 | 0.151 | 3.0 s decode; best of 4 settings shown |
| C1. LightGBM frame model, argmax | 0.2026 | 0.016 | 0.8120 | 1621 / 346 | 2.8 | 0.200 | no temporal decoding; v2 data |
| C2. LightGBM + probability window ±25 | 0.5581 | 0.020 | 0.8123 | 321 / 346 | 14.1 | 0.488 | fixed-window smoothing; v2 data |
| C3. LightGBM (v3 features) + semi-Markov (k=2, α=1.0) | 0.5973 | 0.023 | 0.8174 | 269 / 346 | 16.4 | 0.506 | 2.8 s decode; v3 data |
| C4. + touch features (v4) + semi-Markov (k=2, α=1.0) | 0.6199 | 0.022 | 0.8176 | 269 / 346 | 16.4 | 0.535 | 2.9 s decode; **selected / submitted** |
| C4 window-smoothing comparator (±25) | 0.5808 | 0.026 | 0.8088 | 306 / 346 | 14.9 | 0.504 |  |
| D. TCN / sequence net | not run | | | | | | see REPORT §6 |
| E. Ensemble | not run | | | | | | see REPORT §6 |

## Feature-group ablations (decoder k=2, α=1.0, same protocol; ablations relative to v3)

| variant | spellF1 | sd | macroF1 | poss pred / gt | mean dur s | transF1@25 | notes |
|---|---|---|---|---|---|---|---|
| v4: all features incl. touch (selected) | 0.6199 | 0.022 | 0.8176 | 269 / 346 | 16.4 | 0.535 | 2.9 s decode;  |
| v3: all except touch | 0.5973 | 0.023 | 0.8174 | 269 / 346 | 16.4 | 0.506 | 2.8 s decode;  |
| no ball-derived features (players only) | 0.4712 | 0.034 | 0.7574 | 238 / 346 | 18.4 | 0.282 | 2.9 s decode;  |
| no team-shape/speed features (ball-centric) | 0.5116 | 0.037 | 0.7439 | 256 / 346 | 16.9 | 0.447 | 2.9 s decode;  |
| no long context (±150/±400, 100-frame directional) | 0.5824 | 0.025 | 0.8084 | 308 / 346 | 14.3 | 0.493 | 2.8 s decode;  |
| no past/future directional windows | 0.5865 | 0.027 | 0.8141 | 269 / 346 | 16.4 | 0.484 | 2.7 s decode;  |
| sensitivity: stoppage-correlated ball missingness (train+test) | 0.6170 | 0.024 | 0.8210 | 278 / 346 | 15.9 | 0.547 | 2.6 s decode;  |

## Paired comparisons (same held-out runs, k=2, α=1.0)

| comparison | mean spellF1 diff (b − a) | runs improved | worst run |
|---|---|---|---|
| v3_full → v3_ballonly | -0.0857 | 0/6 | -0.1150 |
| v3_full → v3_noball | -0.1261 | 0/6 | -0.1451 |
| v3_full → v4_touch | +0.0226 | 6/6 | +0.0122 |
| v3_nodir → v3_full | +0.0108 | 5/6 | -0.0006 |
| v3_nolong → v3_full | +0.0148 | 4/6 | -0.0127 |

## Decoder grid (v2 data)

| decoder | spellF1 | sd | macroF1 | poss | dur | transF1 | |
|---|---|---|---|---|---|---|---|
| argmax | 0.2026 | 0.016 | 0.8120 | 1621 / 346 | 2.8 | 0.200 |  |
| hsmm_k1_a0.35 | 0.6003 | 0.022 | 0.8201 | 254 / 346 | 17.4 | 0.516 | 3.0 s decode;  |
| hsmm_k1_a0.6 | 0.6078 | 0.020 | 0.8201 | 275 / 346 | 16.0 | 0.524 | 2.8 s decode;  |
| hsmm_k1_a1.0 | 0.5948 | 0.016 | 0.8194 | 297 / 346 | 14.8 | 0.518 | 2.9 s decode;  |
| hsmm_k1_a1.5 | 0.5904 | 0.016 | 0.8186 | 313 / 346 | 14.0 | 0.515 | 3.0 s decode;  |
| hsmm_k1_a2.5 | 0.5760 | 0.014 | 0.8188 | 335 / 346 | 13.1 | 0.508 | 2.8 s decode;  |
| hsmm_k1_a4.0 | 0.5575 | 0.017 | 0.8186 | 363 / 346 | 12.1 | 0.499 | 2.7 s decode;  |
| hsmm_k2_a0.35 | 0.5863 | 0.026 | 0.8187 | 230 / 346 | 19.2 | 0.505 | 2.7 s decode;  |
| hsmm_k2_a0.6 | 0.6003 | 0.022 | 0.8201 | 254 / 346 | 17.3 | 0.515 | 2.9 s decode;  |
| hsmm_k2_a1.0 | 0.6076 | 0.021 | 0.8199 | 273 / 346 | 16.1 | 0.522 | 2.9 s decode;  |
| hsmm_k2_a1.5 | 0.5999 | 0.018 | 0.8197 | 290 / 346 | 15.1 | 0.520 | 2.9 s decode;  |
| hsmm_k2_a2.5 | 0.5902 | 0.018 | 0.8187 | 312 / 346 | 14.1 | 0.514 | 2.8 s decode;  |
| hsmm_k2_a4.0 | 0.5802 | 0.014 | 0.8188 | 330 / 346 | 13.3 | 0.509 | 2.9 s decode;  |
| hsmm_k3_a0.35 | 0.5766 | 0.018 | 0.8180 | 221 / 346 | 20.0 | 0.499 | 2.9 s decode;  |
| hsmm_k3_a0.6 | 0.5942 | 0.023 | 0.8204 | 240 / 346 | 18.4 | 0.511 | 2.7 s decode;  |
| hsmm_k3_a1.0 | 0.6042 | 0.022 | 0.8197 | 263 / 346 | 16.8 | 0.518 | 2.7 s decode;  |
| hsmm_k3_a1.5 | 0.6051 | 0.020 | 0.8197 | 278 / 346 | 15.8 | 0.523 | 2.7 s decode;  |
| hsmm_k3_a2.5 | 0.5942 | 0.016 | 0.8192 | 299 / 346 | 14.7 | 0.517 | 2.8 s decode;  |
| hsmm_k3_a4.0 | 0.5859 | 0.015 | 0.8187 | 319 / 346 | 13.8 | 0.511 | 2.7 s decode;  |
| meanprob_w25 | 0.5581 | 0.020 | 0.8123 | 321 / 346 | 14.1 | 0.488 |  |
| meanprob_w50 | 0.5501 | 0.024 | 0.8094 | 274 / 346 | 16.6 | 0.446 |  |
