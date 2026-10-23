# Experiment scripts → report files

All numbers in REPORT.md / reports/EXPERIMENTS.md come from these, run from this folder with the
open Metrica sample data at `../external/sample-data` (see the main README).

| script | produces |
|---|---|
| `exp_baseline_calibration.py` (env `DEGRADE_DEAD_FILL` = 1.0 / 0.5 / 0.0) | `reports/baseline_calibration.txt` |
| `exp_baseline_evidence_hsmm.py` | `reports/baseline_evidence.json` |
| `exp_lomo_train.py --tag T [--drop ...]` | `cache/lp_T_*.npz` (held-out probabilities), `reports/train_T.log` |
| `exp_decode_grid.py --tag T` | `reports/decode_T.{json,txt}`; `decode_grid_full` (pre-noise-fix, superseded) and `decode_grid_v2` (decoder grid) |
| `run_ablations.sh` | v3_full, v3_noball, v3_ballonly (its nolong/nodir runs dropped nothing — bug found and discarded) |
| `run_ablations3.sh` | v4_touch, v3_nolong, v3_nodir, sens_corr |
| `exp_paired.py A B` | `reports/paired_A_vs_B.json` |
| `exp_diagnostics.py TAG` | `reports/diagnostics_TAG.json` |
| `exp_noise_robustness.py TAG` | `reports/noise_robustness_TAG.json` (dead-ball share and spellF1 at 3 noise levels) |
| `exp_lomo_sc.py [--team-only]` | v6_sc / v7_scteam: SkillCorner matches added to training (both rejected) |
| `exp_lomo_variants.py` | v8_big (capacity) and v9_ens3 (3-seed ensemble), both rejected |
| `make_tables.py` | `reports/EXPERIMENTS.md` |

Feature-set versions: v2 = before pre-kick-off blocks; v3 = all features except `touch_*`;
v4 = all features; v5 = v4 + smoothed velocities and mixed-noise training (submitted).
