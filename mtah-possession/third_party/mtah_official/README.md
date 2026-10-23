# Vendored official files (MIT)

Copied unmodified from https://github.com/metrica-sports/more-than-a-hack-2027
(Copyright Metrica Sports, MIT licence — see LICENSE in this folder).

- `possession_metrics.py`, `run_evaluation.py`, `validate_submission.py`, `games.py`: the official
  scorer, validator and frame constants. `tests/test_official_vendor.py` pins their SHA-256 so any
  drift from upstream fails the test suite.
- `gt/`: the public 5-minute ground truth published in that repository.
- `baselines/`: the four committed baseline submissions. They are public (committed upstream) and
  are built from the ATD only; `baseline_raw.csv` is used here solely to measure the *pattern* of
  ball visibility (run lengths) for the degradation model and as a sanity input.

No restricted match data is contained here.
