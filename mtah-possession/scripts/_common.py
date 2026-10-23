"""Shared setup for experiment scripts: the open validation matches, degraded reproducibly."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession.constants import BASELINE_DIR, graded_blocks, pm
from possession.external import load_open_match, degrade, noise_for_seed, visibility_runs_from_baseline

SAMPLE_DIR = os.environ.get("METRICA_SAMPLE_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "sample-data"))
GAMES = (1, 2, 3)

def vis_runs():
    return visibility_runs_from_baseline(os.path.join(BASELINE_DIR, "baseline_raw.csv"), graded_blocks())

def open_matches(seeds=(0,), games=GAMES):
    runs = vis_runs()
    out = {}
    for g in games:
        clean = load_open_match(SAMPLE_DIR, g)
        for s in seeds:
            kw = noise_for_seed(s) if os.environ.get("NOISE_MIX", "1") == "1" else {}
            out[(g, s)] = degrade(clean, runs, seed=1000 * g + s,
                                  dead_fill_prob=float(os.environ.get("DEGRADE_DEAD_FILL", "1.0")), **kw)
    return out

def score(labels, pred):
    r = pm.evaluate(labels, pred)
    return {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"], "n_pred": r["n_pred_spells"],
            "n_gt": r["n_gt_spells"], "dur": r["mean_spell_duration_s"],
            "trans_f1_25": r["transitions"][25]["f1"], "alive_dead_f1": r["alive_dead_f1"],
            "team_acc": r["team_acc_both_alive"]}
