# Final audit

Two parts: what was verified in development (no competition data available, by design), and what
`python -m possession.cli reproduce` verifies on the real bundle, written to
`reports/final_audit.json` on your machine.

## A. Verified in development

| check | result | how |
|---|---|---|
| Vendored scorer/validator/games.py identical to upstream | pass | SHA-256 pinned in `tests/test_core.py` |
| Official baselines reproduced on the public window | pass | `run_evaluation.py`: smooth50 spellF1 0.3256, smooth125 0.4242, as published |
| Baseline smoothing port is exact | pass | rebuilds committed `baseline_smooth{25,50,125}.csv` byte-for-byte from `baseline_raw.csv` |
| EPTS reader follows the layout, not `<Players>` order | pass | unit test with swapped registers; Metrica open EPTS file loads (143,761 frames, contiguous) |
| Graded halves | pass | (11185, 90652), (110808, 190198) from official `games.py` |
| Possession rules for open-match labels | pass | unit tests: duel ≠ turnover, loose ball keeps team, dead until set piece, advantage ignored |
| Decoder: no 1-frame turnovers, keeps strong short possessions, team-symmetric, never crosses halftime, `X` only off-block | pass | unit tests |
| `X` forward-fill semantics of spellF1 | pass | unit test against the official `evaluate` |
| Submission contract | pass | 199,319 rows, header `frame,label`, frames 0..199318 in order, labels ⊂ {H,A,D,X}, no blanks, no index column; official validator exit 0 |
| Determinism (pipeline) | pass | two full runs on a full-size synthetic bundle: identical SHA-256 |
| Determinism (training) | pass | retraining reproduced `models/frame_model.txt` byte-for-byte |
| Runtime | measured | synthetic full-size bundle, 2 cores: load 3.5 s, predict 15.8 s, ~0.7 GB peak |
| No restricted data in the repository | pass | `.gitignore` blocks `data/`, all ATD/homography/tagging/video patterns, all CSVs except vendored public ones, and machine-local reports |

## B. What your run on the real bundle must show (`reports/final_audit.json`, `audit_competition.json`)

Run, in order:

```bash
python -m pytest -q tests                                  # 20 passed
python -m possession.cli audit                             # schema + alignment, aggregates only
python -m possession.cli reproduce --team machaji        # writes submission/machaji.csv
```

Expected values, and what to do if they differ:

| field | expect | if not |
|---|---|---|
| `n_rows`, `first_frame_id`, `contiguous` | 199319, 1, true | stop: alignment problem; report to organisers |
| `n_slots`, `n_home_slots` | 46, about half | check team ids in the metadata |
| `player_order_matches_layout` | either; reader handles both | if false, the official baseline mis-assigns teams — worth an issue upstream |
| `ball_present_all` / `ball_present_graded` | 0.569 / 0.593 (DATA.md) | stop: parsing problem |
| `mean_tracked_graded` | ≈ 20.5 (DATA.md) | stop: parsing problem |
| `alignment_vs_official_baseline_raw.agreement` | ≈ 1.0, and lower when shifted by one frame | < 0.99 means alignment, team mapping or units differ from the official baseline |
| `touch_rate_per_visible_frame` | near 0.08–0.10 (open-data reference) | much higher ⇒ real ball noisier than assumed; touch features less reliable (REPORT §7) |
| `validator_exit_code` | 0 | do not submit |
| `diagnostics.label_counts_graded.X` | 0 | — |
| `diagnostics.label_counts_ungraded` | all X (40,462) | — |
| `diagnostics.possessions_per_min` | roughly 2.5–4 (truth ≈ 3.4 in the public window) | far outside ⇒ inspect before submitting |
| `public_window.spell_f1_mean` | above the baselines' 0.33–0.42 would be reassuring | a low value is a warning, not a selection signal (17 possessions) |

Upload exactly the file whose SHA-256 is in `final_audit.json`.

## C. Unresolved risks

1. **Label-convention shift** between event-derived training labels and the competition's
   tracking-ownership truth. Unquantified until results day.
2. **Noise-model dependence** of the touch features (check `touch_rate_per_visible_frame`).
3. **Frame-level difficulty mismatch**: the degraded open data is harder at frame level than the
   real match (baseline macroF1 0.43 vs 0.67), so open-data scores are not competition estimates.
4. **Training diversity**: three anonymised matches from one provider.
