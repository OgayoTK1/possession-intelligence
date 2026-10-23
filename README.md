# Possession Intelligence - More than a Hack 2027, Phase 1: Ball Possession

Frame-level possession (`H` / `A` / `D`) for FC Barcelona – Athletic Club from Metrica's automatic
tracking, optimised for the competition's ranking metric `spellF1`.

**Approach in one line:** position-only, team-symmetric features → LightGBM frame classifier
trained on *open* Metrica matches degraded to the competition's tracking quality → a duration-aware
(Erlang-expanded semi-Markov) Viterbi decoder → one CSV. Full method, evidence and limitations in
[REPORT.md](REPORT.md).

> **No match data is in this repository**, and none is needed to install, test or train it. The
> gated bundle is read from `$MTAH_DATA_DIR` at prediction time only.

## Reproduce the submission (one command)

```bash
git clone <https://github.com/OgayoTK1/possession-intelligence> && cd mtah-possession
python -m venv .venv && source .venv/bin/activate      # Python 3.13 tested
pip install -r requirements.txt

export MTAH_DATA_DIR=/path/to/unpacked/bundle          # contains tactical/BAR-ATH_tactical_atd.{xml,txt}
python -m possession.cli reproduce --team TEAM_NAME
```

This loads the tactical ATD, predicts every frame with the committed model
(`models/frame_model.txt`), writes `submission/TEAM_NAME.csv`, runs the **official**
`validate_submission.py` on that exact file, scores it on the public 5-minute window, and writes
`reports/final_audit.json` (validator output, SHA-256, label counts, possession-duration
diagnostics, timings). Measured on a full-size synthetic bundle (2 CPU cores): ~20 s wall time, ~0.7 GB peak RAM.

Determinism: the committed model is used as-is, the decoder is exact dynamic programming, and the
CSV writer is byte-stable — the same inputs give the same SHA-256.

## Other commands

```bash
python -m possession.cli audit                 # schema + frame-alignment audit -> reports/audit_competition.json
python -m possession.cli baseline              # official nearest-player baseline rebuilt on this loader
python -m possession.cli train                 # retrain models/frame_model.txt from the open matches
python -m pytest -q tests                      # 20 tests, no match data needed
```

`audit` includes a frame-by-frame comparison against the organisers' committed `baseline_raw.csv`;
agreement ≈ 1.0 proves frame alignment, slot→team mapping and units on the real file. Its output
contains aggregate statistics only.

### Retraining from scratch (optional)

```bash
git clone https://github.com/OgayoTK1/possession-intelligence
python -m possession.cli train
```

Training data are Metrica's three open sample matches only. Retraining on the same platform with
the pinned versions reproduces the committed model; across CPU architectures LightGBM may differ in
the last float bits, which is why reproduction uses the committed model.

## Repository map

```
possession/
  epts.py          schema-driven FIFA/EPTS reader (layout from DataFormatSpecification, not guessed)
  features.py      position-only, team-symmetric, NaN-preserving per-frame features
  external.py      open Metrica matches: event-derived labels + degradation to competition quality
  models.py        LightGBM frame classifier, team-swap augmentation, symmetric inference
  decode.py        official baseline smoothing (exact port) + Erlang semi-Markov Viterbi
  pipeline.py      competition load -> predict -> CSV -> official validator -> diagnostics
  smart_tagging.py audit-only reader for the Smart Tagging XML
  cli.py           commands above
scripts/           experiment drivers (leave-one-match-out training, decoder grid, ablations)
reports/           measured experiment outputs (JSON / text), produced by scripts/
third_party/mtah_official/   unmodified official scorer, validator, games.py, public GT (MIT, SHA-pinned)
tests/             unit + contract tests, synthetic bundle generator
```

## Data handling

* `.gitignore` excludes `data/`, every ATD/homography/tagging/video file pattern, all CSVs except
  the vendored public ones, and the two machine-local reports derived from the match.
* Per the competition T&Cs, the match data must not be uploaded to third-party hosted LLMs. This
  code was developed without access to it: every experiment in REPORT.md ran on open data.

## Licence

MIT for this code. Vendored official files: MIT, © Metrica Sports (see `third_party/`). Open
training data: Metrica Sports sample data, used with attribution as its README requests.
