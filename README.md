# possession-intelligence

**More than a Hack 2027 - Phase 1: Ball Possession.** Team **machaji**: Ogayo Andrew (team lead),
Maxwell Okoth, James Ngugi, Alvin Njoroge, Emmanuel Masambu.

All code lives in [`mtah-possession/`](mtah-possession/):

| | |
|---|---|
| How to reproduce the submission | [`mtah-possession/README.md`](mtah-possession/README.md) |
| Technical report | [`mtah-possession/REPORT.md`](mtah-possession/REPORT.md) |
| Every experiment, with measured numbers | [`mtah-possession/reports/EXPERIMENTS.md`](mtah-possession/reports/EXPERIMENTS.md) |
| Verification checklist | [`mtah-possession/reports/AUDIT.md`](mtah-possession/reports/AUDIT.md) |

```bash
cd mtah-possession
pip install -r requirements.txt
export MTAH_DATA_DIR=/path/to/unpacked/bundle     # folder containing tactical/
python -m possession.cli reproduce machaji        # -> submission/machaji.csv, validated
```

The match data is not in this repository; it is licensed for the competition only and must not be added.
