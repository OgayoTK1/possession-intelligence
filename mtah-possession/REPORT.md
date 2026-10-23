# Technical report — Ball possession from automatic tracking

More than a Hack 2027, Phase 1. Team: **machaji** — Ogayo Andrew (team lead), Maxwell Okoth, James Ngugi, Alvin Njoroge, Emmanuel Masambu. Repository: this one. One command regenerates
the submission: `python -m possession.cli reproduce --team machaji` (README).

## In brief

**Problem.** For every frame of FC Barcelona – Athletic Club, say which team has the ball, or that
the ball is dead. The tracking is automatic and noisy, and the ball is missing on 41% of graded
frames — often for tens of seconds at a time.

**Approach.** A model reads where the 22 players and the ball are, and how they move, and scores
each frame for "Barcelona in possession", "Athletic in possession" and "dead ball". A second step
turns those per-frame scores into whole possessions, the way the ranking metric counts them.
When the ball is invisible, the model falls back on team shape and movement.

**Why it should work.** It was trained and tested on three public Metrica matches, made as noisy
and as ball-less as this match. On matches it never saw, it recovers about three times as many
possessions as the official baseline under the same conditions.

**What it produces.** 3.3 possessions per minute averaging 12.7 s on the real match (the public
5 minutes: 3.4 per minute, 14.2 s), and on that public window a spellF1 of 0.54 against the
baselines' 0.33–0.42 — a correctness check, not a forecast.

**Pipeline.**

```
tactical tracking (EPTS)  ->  per-frame features   ->  LightGBM: P(H), P(A), P(D)  ->  semi-Markov decoder  ->  machaji.csv
46 tracks + ball, 25 fps      team shape, speeds,       trained on open Metrica         duration-aware,
                              ball-player geometry,     matches degraded to this        whole possessions
                              touches, time windows     match's quality
```

## 1. Summary

I treat possession as noisy, partially observed state inference over three states (`H`, `A`,
`D`). A LightGBM classifier estimates per-frame state probabilities from position-only,
team-symmetric features; a semi-Markov Viterbi decoder turns them into possessions with
data-fitted duration and transition priors. The classifier is trained only on **Metrica's three
open sample matches**, whose clean tracking I degrade to reproduce the competition ATD's measured
ball-visibility pattern and player coverage. Labels for those matches are derived algorithmically
from their open event data using the challenge's possession rules.

Selection evidence comes from leave-one-match-out validation on those ~1,040 open possessions,
not from the 17 public ones. Under identical degradation, the official baseline scores spellF1
0.215 there (its published full-match score is 0.243). The submitted system scores **0.642**
(sd 0.023 across held-out matches) and stays between 0.625 and 0.651 across three tracking-noise
levels. **This is not a prediction of the competition score**: the labelling conventions, teams
and tracking noise differ (§7).

On the real match, the first version over-predicted dead ball (46% of graded frames) because my
assumed tracking noise inflated speeds in training; §5.1 documents the diagnosis and the fix. The
submitted version predicts 3.29 possessions/min (public window truth: 3.4), and on the public
5-minute window — used only as a correctness check — scores spellF1 0.537 and macroF1 0.741 (the
official baselines: 0.33–0.42 and 0.58–0.62).

## 2. What the data audit changed

1. **Ball gaps are long, and the public window hides it.** From the committed `baseline_raw.csv`
   (public; `D` exactly where the ATD has no ball) the graded halves contain 150 missing-ball runs
   with a median of ~10 s and a maximum of 3.7 min; runs over 20 s hold 61–73% of all missing
   frames. In the public window only 6.5% of in-play frames lack the ball. A method tuned there
   would over-trust the ball. So the model has a **ball-free path** (team shape, speeds, line
   heights) and the validation data replays the real gap sequence.
2. **Missing is not dead.** Two of the four public dead-ball segments have the ball visible
   throughout (20.4 s and 14.5 s), so "no ball ⇒ D" (the baseline's rule) is wrong in both
   directions. The degradation therefore makes visibility independent of state (fills stoppage
   gaps with a noisy carried ball), so the model cannot learn that shortcut.
3. **The EPTS layout is the authority, not `<Players>` order.** In Metrica's open EPTS file the raw
   column order differs from the `<Players>` order; the official baseline assumes they match. My
   reader follows `DataFormatSpecification` (per frame range) and the audit reports whether the
   competition file agrees. `audit` also re-derives the baseline's nearest-team codes and compares
   them frame by frame with the organisers' `baseline_raw.csv` — a direct check of frame alignment,
   slot→team mapping and units on the real file.
4. **Graded but pre-kick-off.** The 2nd half is graded from 2,934 frames before its kick-off, all
   `D`. Training blocks include each open match's pre-kick-off stretch, labelled `D`.

## 3. Possession rules implemented (open-match labels)

Exactly as `possession/external.py` documents and `tests/test_core.py` checks:

* A team gains possession at the start of its own controlled action: pass, carry, recovery, set
  piece, shot. A challenge (duel) never changes possession on its own; nor does `BALL LOST`.
* Loose phases keep the previous team until the opponent's controlled action.
* The ball is dead from a ball-out (end frame), a foul received, or a goal, until the next set
  piece, which restarts play for the set-piece team. A stoppage followed by in-play actions
  without a restart (advantage) is discarded.

Aggregate agreement with the public ground truth's conventions: 3.3–4.0 possessions/min (public:
3.4), mean 12–15 s (14.2 s), dead-ball share 17–25% (20%).

## 4. Method

**Features** (`features.py`, 149 columns; positions only, speeds recomputed so open and
competition data are treated identically; missing stays `NaN`). Positions are smoothed with a
centred moving average (±6 frames for players, ±2 for the ball) before any velocity is computed,
so that speed features do not depend on the tracking-noise level (§5.1):
ball kinematics and pitch geometry; per-team nearest/second-nearest distance to the ball, counts
within 1.5 m / 5 m, relative velocity of each team's nearest player; team centroids, spreads,
mean speed and forward velocity, defensive/attacking line heights; whole-field speed and
fraction of near-stationary players (stoppages); ball-gap length features; **touch features** —
abrupt change of the smoothed ball velocity with a player within 2 m, and the team of the last and
next touch (carried through ball gaps); centred windows ±12/50/150/400 frames and past/future-only
windows of 25/100 frames. Each half is rotated so home attacks +x; slots are only ever aggregated by
team.

**Frame model** (`models.py`): LightGBM multiclass, 300 rounds, deterministic. Training rows
are every 4th graded frame of three open matches × three degradation seeds (one per noise level:
none, medium, high) × both team orientations
(features rebuilt on a team-swapped match). Inference averages the prediction with its
team-swapped counterpart, so the output is exactly team-equivariant: nothing is learned about
"home".

**Decoder** (`decode.py`): Erlang-expanded hidden semi-Markov model. Each state is a chain of
k = 2 sub-states, giving negative-binomial durations whose means and the H/A/D transition matrix
are fitted from training labels (H/A statistics pooled for symmetry). Emissions are
α·(log p(state|x) − log prior). Exact Viterbi per half; nothing crosses halftime. There is no
hand-set minimum duration and no frame-level correction.

**Abstention.** `X` is written only on ungraded frames. On graded frames the scorer forward-fills
`X` (tested in `tests/test_core.py`), so an abstention at a turnover carries the old team across
it; the decoder already outputs its best full path, so abstaining can only lose.

## 5. Validation protocol and results

Leave-one-match-out over the three open matches; each held-out match is degraded with two seeds
not used in training (6 runs, ~346 possessions each). Scores use the official
`possession_metrics.evaluate`. Full tables, all generated from JSON outputs:
[`reports/EXPERIMENTS.md`](reports/EXPERIMENTS.md).

**Does the validation set resemble the competition?** The official baseline algorithm on degraded
open matches: spellF1 0.023 / 0.181 / 0.215 / 0.205 for windows 0/25/50/125 vs published
full-match 0.049 / 0.200 / 0.243 / 0.224 — the same ordering and similar level on the ranking
metric. macroF1 is much lower on my data (0.39–0.44 vs 0.65–0.67) even when ball visibility is
made fully stoppage-correlated (at most 0.55), so my degraded data is harder at frame level than the real
match. I report this rather than tune the degradation to it.

| system (held-out open matches) | spellF1 | macroF1 | poss / true |
|---|---|---|---|
| Official baseline ±50 (published floor algorithm) | 0.215 | 0.429 | 483 / 346 |
| Baseline evidence + my semi-Markov decoder | 0.220 | 0.409 | 357 / 346 |
| LightGBM, argmax | 0.238 | 0.810 | 1403 / 346 |
| LightGBM + probability window ±25 | 0.581 | 0.809 | 306 / 346 |
| **LightGBM + semi-Markov, k=2, α=1.0 (selected)** | **0.620** | 0.818 | 269 / 346 |

What the comparisons show:

* **Evidence matters first.** The semi-Markov decoder on the baseline's own evidence is no better
  than majority voting (0.220 vs 0.215). The gain comes from the learned evidence.
* **Decoding matters second.** On the same LightGBM probabilities, duration-aware decoding beats
  fixed-window smoothing by +0.039 and argmax by +0.38.
* **The decoder setting is not fragile.** Over k ∈ {1,2,3} and α ∈ [0.35, 4], held-out spellF1
  stays within 0.56–0.61 (v2 grid); I took the centre of the plateau rather than its best cell.

Feature ablations (same protocol, decoder fixed):

| variant | spellF1 | paired vs reference |
|---|---|---|
| v4: all features incl. touch (**selected**) | **0.620** | +0.023 vs v3, 6/6 runs better |
| v3: all except touch | 0.597 | reference for the rows below |
| players only (no ball-derived features) | 0.471 | −0.126, 0/6 runs survive |
| ball-centric (no team-shape/speed features) | 0.512 | −0.086, 0/6 runs survive |
| no long context (±150/±400, 100-frame windows) | 0.582 | −0.015, worse in 4/6 runs |
| no past/future directional windows | 0.587 | −0.011, worse in 5/6 runs |
| sensitivity: stoppage-correlated ball missingness | 0.617 | conclusions unchanged |

Both evidence sources are necessary and complementary: a players-only model still reaches 0.47
(more than twice the baseline), which is what carries the 41% of graded frames without a ball.
Long and directional context are weakly supported; they are kept because they are only extra
columns in the same model, and the evidence is reported as weak.

**Paired test of the touch features**: 6/6 held-out runs improve, mean +0.023, minimum +0.012.

### 5.1 First run on the real match: a failure, its diagnosis and the fix

The first version (the v4 configuration above) produced a valid file, but its aggregate
statistics were implausible: 46% of graded frames dead ball (public truth 20%), 4.6
possessions/min, mean possession 7.0 s, and public-window spellF1 0.249 — below the baselines.

Diagnosis, from aggregate statistics only: on an open match, my degradation noise (0.25 m per
frame on players) raised the median whole-field player speed during play from 1.84 to 3.39 m/s
and during stoppages from 0.87 to 2.95 m/s. The model had learned that live play looks fast; the
real ATD is smoother, so live play looked like a stoppage.

Fix (no labels involved): velocities from smoothed positions, and training across three noise
levels. On held-out open matches the predicted dead-ball share is now 0.234 / 0.236 / 0.219
against a true 0.206 at no / medium / high noise (`reports/noise_robustness_v5_smooth.json`).

| real match, graded frames | first version | submitted version |
|---|---|---|
| dead-ball share | 46.0% | 30.6% |
| possessions per minute | 4.63 | 3.29 |
| mean possession | 7.0 s | 12.7 s |
| public window spellF1 / macroF1 | 0.249 / 0.549 | 0.537 / 0.741 |
| public window possessions (true 17) | 36 | 18 |

The public-window numbers are reported, not optimised: the change was decided from the
open-match diagnosis before the second real run, and the window's bootstrap interval
(0.35–0.71) is wide. A follow-up check showed the touch detector needs no recalibration: on the
real match 51% of visible-ball frames have a player within 2 m (open: 50–55%) and 8.4% exceed the
impulse threshold (open: 9–12%).

The ablations in §5 were measured before this change and are kept as measured.

## 6. Error analysis and what was not done

On the selected configuration (`reports/diagnostics_v4_touch.json`; v3 in brackets, which is
the diagnosis that motivated the touch features):

* Recall at IoU 0.5 by true possession length: 15.8% [11.5%] for < 2 s, 46% [43%] for 2–5 s,
  69% [68%] for 5–15 s, 76% [76%] for > 15 s. Short possessions remain the dominant loss.
* Merges outnumber splits: 12.2 merges per 100 predicted spells vs 1.8 splits per 100 true ones;
  predicted median duration 10.3 s vs 6.5 s true. 56% [59%] of turnovers are not localised
  within ±1 s.
* Dead-ball segments: 62 of 386 missed; matched ones start/end within a median 9 / 5 frames.

Raising α produces more, shorter possessions but loses more precision than it gains recall
(spellF1 falls beyond α ≈ 1). The limit is turnover evidence during long ball gaps, not decoding.

**Tried and dropped: more training matches from another provider.** SkillCorner's open data
(MIT, 20 A-League matches, broadcast tracking) was converted to the same labels with the same
rules (`possession/skillcorner.py`) and added to every training fold; the held-out Metrica runs
were unchanged, so the comparison is paired. All labels: −0.045 mean spellF1, 0/6 runs better
(team accuracy rose 0.902 → 0.911 but alive/dead F1 fell 0.941 → 0.928 — its stoppage convention
differs). Team labels only: −0.025, 1/6 runs better. Broadcast tracking (≈9 of 22 players
detected per frame, the rest extrapolated) does not transfer to a tactical camera, so the
submitted model uses Metrica's open matches only.

**Tried and dropped: more capacity and ensembling.** Same protocol, paired with the submitted
model. A larger LightGBM (63 leaves, 600 rounds): +0.000 mean spellF1, better in 3/6 runs.
Averaging three models with different seeds: +0.002, better in 4/6 runs (worst −0.013). Neither
clears the noise, and the ensemble would triple runtime, so neither is used. Together with the
SkillCorner results, this says the model is limited by the evidence in the tracking (long ball
gaps, short possessions), not by model capacity or training-set size from other providers.

**Not done, deliberately.** A temporal CNN/sequence model (D) was not trained: with three
training matches the frame model's errors are dominated by missing evidence, not by an inability
to model context the windows already provide, and I could not validate a higher-capacity model
honestly on this little data. No ensemble (E): there is no second strong, decorrelated
candidate. **Smart Tagging and homography are not used**: they exist only for the competition
match, so their effect could be measured on nothing but 17 public possessions. They are read by
the audit. The broadcast feed is not used (unsynchronised).

## 7. Limitations and risks

* **Label-convention shift.** The competition truth is "owning team" from corrected tracking;
  my training labels come from events. They agree in aggregate (§3) but may differ on touches and
  restart timing. This is the largest unquantified risk.
* **Degradation realism.** Visibility runs are the real ones; noise levels were my assumption
  and the first one was wrong (§5.1). Training now spans three levels and the speed features are
  smoothed; the audit prints the real match's speed and touch statistics for comparison.
* **Dead-ball share.** The submitted file predicts 30.6% dead ball on the real match. The truth
  for the full match is unknown (20% in the first 5 minutes; the 2nd half alone contributes
  2,934 pre-kick-off dead frames), so this may be slightly high.
* **Three training matches, one vendor, anonymised teams.** Barcelona's style (long possessions,
  high line) is not represented specifically.
* **Public window.** I did not select anything on it. Its score is reported by `reproduce` with a
  1000-draw bootstrap interval for correctness checking only.

## 8. Reproduction and runtime

`python -m possession.cli reproduce --team machaji` — loads the ATD, predicts, writes
`submission/machaji.csv`, runs the official validator on that exact file, writes
`reports/final_audit.json`. On the real match on a Windows laptop (Python 3.13): load 16 s,
features + model + decoding 47 s. The submitted file's SHA-256 is printed in `final_audit.json`;
the CSV writer is byte-stable, and the decoder is exact dynamic programming. LightGBM inference
can differ in the last floating-point bits across CPU architectures, which could in rare cases
change a frame; the SHA lets anyone compare. Retraining from open data: `python -m possession.cli train` (about 90 s on 2 cores; retraining reproduced the committed model file byte-for-byte).
Tests: `python -m pytest -q tests` (no match data required).

Assets used: tactical ATD (positions). Read but not used by the model: Smart Tagging (audit),
homography and both videos (not opened), broadcast ATD.
