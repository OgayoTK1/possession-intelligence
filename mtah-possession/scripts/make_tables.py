"""Collect measured results from reports/*.json into reports/EXPERIMENTS.md.

Every number in the table is read from a JSON written by an experiment script; nothing is typed in.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "reports")


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.isfile(p) else None


def row(label, d, note=""):
    if d is None:
        return "| {} | not run | | | | | | {} |".format(label, note)
    rt = d.get("runtime_s")
    return "| {} | {:.4f} | {:.3f} | {:.4f} | {:.0f} / {:.0f} | {:.1f} | {:.3f} | {}{} |".format(
        label, d["spellF1"], d.get("spell_sd_over_matches", float("nan")), d["macroF1"],
        d["n_pred"], d["n_gt"], d["dur"], d["trans_f1_25"],
        "{:.1f} s decode; ".format(rt) if rt else "", note)


def main():
    be = load("baseline_evidence.json") or {}
    v2 = load("decode_grid_v2.json") or {}
    lines = [
        "# Experiment table (all numbers measured)",
        "",
        "Leave-one-match-out on Metrica's three open matches, each held-out match degraded with two",
        "seeds (6 test runs; ~346 possessions per run, ~1,040 distinct possessions). spellF1 sd is",
        "the standard deviation of the per-match mean across the 3 held-out matches. Scored with the",
        "official possession_metrics.evaluate. These are NOT competition scores.",
        "",
        "| system | spellF1 | sd | macroF1 | poss pred / gt | mean dur s | transF1@25 | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for w in (25, 50, 125):
        lines.append(row("Official baseline, majority ±{}".format(w), be.get("official_baseline_w{}".format(w)),
                         "published floor" if w == 50 else ""))
    lines.append(row("B. Baseline evidence + semi-Markov decoder",
                     be.get("baseline_evidence_hsmm_c0.8_a0.3"), "best of 4 settings shown"))
    lines.append(row("C1. LightGBM frame model, argmax", v2.get("argmax"), "no temporal decoding; v2 data"))
    lines.append(row("C2. LightGBM + probability window ±25", v2.get("meanprob_w25"), "fixed-window smoothing; v2 data"))
    v4 = load("decode_v4_touch.json") or {}
    lines.append(row("C3. LightGBM (v3 features) + semi-Markov (k=2, α=1.0)", (load("decode_v3_full.json") or {}).get("hsmm_k2_a1.0"), "v3 data"))
    lines.append(row("C4. + touch features (v4) + semi-Markov (k=2, α=1.0)", v4.get("hsmm_k2_a1.0"), "**selected / submitted**"))
    lines.append(row("C4 window-smoothing comparator (±25)", v4.get("meanprob_w25"), ""))
    lines.append("| D. TCN / sequence net | not run | | | | | | see REPORT §6 |")
    lines.append("| E. Ensemble | not run | | | | | | see REPORT §6 |")
    lines += ["", "## Feature-group ablations (decoder k=2, α=1.0, same protocol; ablations relative to v3)", "",
              "| variant | spellF1 | sd | macroF1 | poss pred / gt | mean dur s | transF1@25 | notes |",
              "|---|---|---|---|---|---|---|---|"]
    for tag, label in (("v4_touch", "v4: all features incl. touch (selected)"), ("v3_full", "v3: all except touch"), ("v3_noball", "no ball-derived features (players only)"),
                       ("v3_ballonly", "no team-shape/speed features (ball-centric)"),
                       ("v3_nolong", "no long context (±150/±400, 100-frame directional)"),
                       ("v3_nodir", "no past/future directional windows"),
                       ("sens_corr", "sensitivity: stoppage-correlated ball missingness (train+test)")):
        d = load("decode_{}.json".format(tag))
        lines.append(row(label, d.get("hsmm_k2_a1.0") if d else None))
    lines += ["", "## Paired comparisons (same held-out runs, k=2, α=1.0)", "",
              "| comparison | mean spellF1 diff (b − a) | runs improved | worst run |", "|---|---|---|---|"]
    for f in sorted(os.listdir(R)):
        if f.startswith("paired_") and f.endswith(".json"):
            d = load(f)
            lines.append("| {} → {} | {:+.4f} | {}/{} | {:+.4f} |".format(
                d["a"], d["b"], d["mean_diff"], d["n_improved"], len(d["runs"]), min(r[4] for r in d["runs"])))
    lines += ["", "## Decoder grid (v2 data)", "", "| decoder | spellF1 | sd | macroF1 | poss | dur | transF1 | |",
              "|---|---|---|---|---|---|---|---|"]
    for k, d in sorted(v2.items()):
        lines.append(row(k, d))
    open(os.path.join(R, "EXPERIMENTS.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
