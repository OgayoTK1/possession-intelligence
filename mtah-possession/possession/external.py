"""Openly published Metrica matches as an independent validation set.

Source: https://github.com/metrica-sports/sample-data (Sample Games 1-3, anonymised, synchronised
tracking + events, published by Metrica for community use with attribution). None of this touches
the competition fixture.

Labels are derived *algorithmically* from the open event data with the challenge's possession
rules, which are documented here once so the report can quote them:

  * A team gains possession at the start frame of one of its controlled actions:
    PASS, CARRY, RECOVERY, SET PIECE, SHOT. (Not CHALLENGE: a won duel without subsequent control
    is not a change of possession. Not BALL LOST: the loser's own event says nothing about who
    controls next.)
  * Loose phases keep the previous team: nothing changes between controlled actions.
  * The ball goes dead at a BALL OUT (its end frame, else start), at a FAULT RECEIVED, and at the
    end of a SHOT whose subtype contains GOAL. It stays dead until the next SET PIECE start frame,
    which restarts play for the set-piece team. A stoppage with an in-play controlled action
    before any set piece (advantage played, mis-tagged out) is ignored rather than invented.
  * Before the first kick-off and after the last event of each period: ungraded (X).

Domain shift, stated plainly: the competition labels come from corrected tracking ownership, not
from events, so short-touch and boundary conventions differ. These matches are used to choose
between methods and to fit transition/duration priors -- never to claim a competition score.
"""

import json
import os

import numpy as np
import pandas as pd

from .constants import A, D, H, X
from .match import Match

PITCH_OPEN = (105.0, 68.0)
CONTROL_TYPES = {"PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT"}


# -- event loading -----------------------------------------------------------------------------

def _events_csv(path):
    ev = pd.read_csv(path)
    out = pd.DataFrame({
        "team": ev["Team"].map({"Home": 0, "Away": 1}),
        "type": ev["Type"].str.upper().str.strip(),
        "subtype": ev["Subtype"].fillna("").str.upper(),
        "period": ev["Period"].astype(int),
        "start": ev["Start Frame"].astype(int),
        "end": ev["End Frame"].fillna(0).astype(int),
    })
    return out


def _events_json(path, home_team_id):
    data = json.load(open(path))["data"]
    rows = []
    for e in data:
        sub = e.get("subtypes")
        if isinstance(sub, list):
            subname = "-".join(s["name"] for s in sub if s)
        elif isinstance(sub, dict):
            subname = sub.get("name", "")
        else:
            subname = ""
        rows.append({
            "team": 0 if e["team"]["id"] == home_team_id else 1,
            "type": e["type"]["name"].upper().strip(),
            "subtype": (subname or "").upper(),
            "period": int(e["period"]),
            "start": int(e["start"]["frame"]),
            "end": int(e["end"]["frame"] or 0),
        })
    return pd.DataFrame(rows)


def derive_labels(events, n_frames, frame_base=1, period_starts=None, max_pre=3000):
    """Per-frame H/A/D/X codes from an event table (see module docstring for the rules).

    events.start / events.end are frame ids; row = frame_id - frame_base.
    period_starts: {period: first row of that period's tracking}. Frames from there up to the
    kick-off (at most max_pre) are labelled D and included in the block, mirroring the
    competition's 2nd half, which is graded from 2,934 frames before its kick-off.
    Returns (labels, blocks).
    """
    labels = np.full(n_frames, X, dtype=np.int8)
    blocks = []
    for period, ev in events.sort_values(["start", "end"]).groupby("period", sort=True):
        ev = ev.reset_index(drop=True)
        kick = ev[(ev["type"] == "SET PIECE") & ev["subtype"].str.contains("KICK OFF")]
        if not len(kick):
            continue
        first = int(kick["start"].iloc[0]) - frame_base
        last = int(max(ev["start"].max(), ev["end"].max())) - frame_base
        last = min(last, n_frames - 1)
        state = D
        dead_until_setpiece = False
        cursor = first
        # Change points: (frame, new_state)
        changes = []
        for _, e in ev.iterrows():
            f = int(e["start"]) - frame_base
            if f < first:
                continue
            etype, sub = e["type"], e["subtype"]
            if etype == "SET PIECE":
                changes.append((f, H if e["team"] == 0 else A))
                dead_until_setpiece = False
                continue
            if etype in CONTROL_TYPES and not dead_until_setpiece:
                changes.append((f, H if e["team"] == 0 else A))
                continue
            if etype in CONTROL_TYPES and dead_until_setpiece:
                # play went on without a restart: the stoppage was not real; undo it.
                if changes and changes[-1][1] == D:
                    changes.pop()
                dead_until_setpiece = False
                changes.append((f, H if e["team"] == 0 else A))
                continue
            if etype == "BALL OUT" or (etype == "FAULT RECEIVED"):
                g = (int(e["end"]) if etype == "BALL OUT" and e["end"] > 0 else int(e["start"])) \
                    - frame_base
                changes.append((max(g, f), D))
                dead_until_setpiece = True
                continue
            if etype == "SHOT" and "GOAL" in sub and "OFF" not in sub:
                g = (int(e["end"]) if e["end"] > 0 else int(e["start"])) - frame_base
                changes.append((g, D))
                dead_until_setpiece = True
        changes.sort(key=lambda c: c[0])
        seg_state = D
        prev = first
        for f, s in changes:
            f = max(f, first)
            if f > last:
                break
            labels[prev:f] = seg_state
            prev, seg_state = f, s
        labels[prev:last + 1] = seg_state
        block_start = first
        if period_starts and period in period_starts:
            block_start = max(int(period_starts[period]), first - max_pre,
                              blocks[-1][1] if blocks else 0)
            labels[block_start:first] = D
        blocks.append((block_start, last + 1))
    return labels, blocks


# -- tracking loading --------------------------------------------------------------------------

def _tracking_csv(path, team):
    raw = pd.read_csv(path, skiprows=2)
    cols = list(raw.columns)
    names = []
    for i, c in enumerate(cols):
        if c.startswith("Player"):
            names.append((c, cols[i + 1]))
    px = np.stack([raw[x].values for x, _ in names], axis=1).astype(np.float32)
    py = np.stack([raw[y].values for _, y in names], axis=1).astype(np.float32)
    bi = cols.index("Ball")
    ball = np.stack([raw[cols[bi]].values, raw[cols[bi + 1]].values], axis=1).astype(np.float32)
    frames = raw["Frame"].values.astype(np.int64)
    period = raw["Period"].values.astype(np.int64)
    return frames, px, py, ball, period


def load_open_match(sample_dir, game):
    """Sample Game 1, 2 or 3 as a labelled Match in metres."""
    length, width = PITCH_OPEN
    base = os.path.join(sample_dir, "data", "Sample_Game_{}".format(game))
    if game in (1, 2):
        stem = os.path.join(base, "Sample_Game_{}".format(game))
        fh, hx, hy, ball, period = _tracking_csv(stem + "_RawTrackingData_Home_Team.csv", 0)
        fa, ax, ay, _, _ = _tracking_csv(stem + "_RawTrackingData_Away_Team.csv", 1)
        if not np.array_equal(fh, fa):
            raise ValueError("home/away tracking frames differ")
        px = np.concatenate([hx, ax], axis=1) * length
        py = np.concatenate([hy, ay], axis=1) * width
        team = np.array([0] * hx.shape[1] + [1] * ax.shape[1], dtype=np.int8)
        ball = ball * np.array([length, width], dtype=np.float32)
        frame_base = int(fh[0])
        events = _events_csv(stem + "_RawEventsData.csv")
        period_starts = {int(p): int(np.flatnonzero(period == p)[0]) for p in np.unique(period)}
    else:
        from .epts import load_epts
        match, _ = load_epts(os.path.join(base, "Sample_Game_3_metadata.xml"),
                             os.path.join(base, "Sample_Game_3_tracking.txt"), pitch=PITCH_OPEN)
        px, py, team, ball = match.px, match.py, match.team_of_slot, match.ball
        frame_base = int(match.extras["first_frame_id"])
        home_id = "FIFATMA"
        events = _events_json(os.path.join(base, "Sample_Game_3_events.json"), home_id)
        # first_half_start / second_half_start from the EPTS metadata (frame ids 1 and 69662)
        period_starts = {1: 1 - frame_base, 2: 69662 - frame_base}
    labels, blocks = derive_labels(events, px.shape[0], frame_base, period_starts)
    m = Match(name="open{}".format(game), fps=25, pitch=PITCH_OPEN, ball=ball.astype(np.float32),
              px=px.astype(np.float32), py=py.astype(np.float32), team_of_slot=team,
              blocks=blocks, labels=labels)
    return m.validate()


# -- degradation: make clean open tracking look like the competition ATD ------------------------

def visibility_runs_from_baseline(baseline_raw_csv, blocks):
    """Alternating (visible?, length) ball-visibility runs of the competition match.

    Read from the *public, committed* baseline_raw.csv, where a frame is D exactly when the ATD
    had no ball. That is a property of the tracking, not of any label, so it is safe to reuse.
    """
    lab = pd.read_csv(baseline_raw_csv)["label"].values
    visible = lab != "D"
    out = []
    for start, stop in blocks:
        v = visible[start:stop].astype(np.int8)
        edges = np.flatnonzero(np.diff(np.concatenate(([-1], v, [-1]))) != 0)
        out.append([(bool(v[a]), int(b - a)) for a, b in zip(edges[:-1], edges[1:])])
    return out


NOISE_LEVELS = ((0.0, 0.0, 0.0), (0.10, 0.06, 0.12), (0.25, 0.15, 0.30))  # (player, ball, ball drift) m


def noise_for_seed(seed):
    """Degradation seeds cycle through noise levels, so the model never depends on one assumed
    tracking-noise level (the real ATD turned out smoother than the first assumption)."""
    p, b, d = NOISE_LEVELS[seed % len(NOISE_LEVELS)]
    return {"player_noise_m": p, "ball_noise_m": b, "ball_drift_m": d}


def degrade(match, vis_runs, seed, player_noise_m=0.25, player_drop_rate=0.07,
            dead_fill_prob=1.0, ball_noise_m=0.15, ball_drift_m=0.3):
    """A copy of `match` whose ball visibility follows the competition's real gap pattern.

    1. Ball: for each graded block, take one competition half's visibility sequence from a random
       start offset (wrapping), and hide the ball where it says missing. Where it says visible
       during a stoppage that the clean data left NaN, fill the ball by linear interpolation
       between the last and next seen positions (a ball carried to the restart spot) so that
       "ball missing" is no longer a free proxy for "dead" -- in the competition ATD it is not.
    2. Players: Gaussian position noise and burst drop-outs, giving ~20.5 tracked of 22 as in the
       competition ATD.
    """
    rng = np.random.RandomState(seed)
    ball = match.ball.copy()
    px, py = match.px.copy(), match.py.copy()
    t = match.n_frames

    # (1) fill clean-data gaps by interpolation, then impose the competition mask
    filled = ball.copy()
    for start, stop in match.blocks:
        seg = filled[start:stop]
        ok = np.isfinite(seg[:, 0])
        if ok.sum() >= 2:
            idx = np.arange(stop - start)
            for d in range(2):
                seg[~ok, d] = np.interp(idx[~ok], idx[ok], seg[ok, d])
        if dead_fill_prob < 1.0:
            drop = (~ok) & (rng.rand(stop - start) > dead_fill_prob)
            seg[drop] = np.nan
        filled[start:stop] = seg
    mask = np.ones(t, dtype=bool)
    for start, stop in match.blocks:
        runs = vis_runs[rng.randint(len(vis_runs))]
        seq = np.concatenate([np.full(n, v) for v, n in runs])
        offset = rng.randint(len(seq))
        seq = np.roll(seq, -offset)
        n = stop - start
        reps = int(np.ceil(n / float(len(seq))))
        mask[start:stop] = np.tile(seq, reps)[:n]
    # Ball measurement noise everywhere (white + slowly drifting AR(1) component). Without it the
    # interpolated stoppage fill moves in perfectly straight lines -- zero acceleration -- which a
    # model would learn as a dead-ball signature that the real ATD does not have.
    white = rng.normal(0, ball_noise_m, filled.shape)
    drift = np.zeros_like(filled)
    eps = rng.normal(0, ball_drift_m * np.sqrt(1 - 0.98 ** 2), filled.shape)
    for i in range(1, t):
        drift[i] = 0.98 * drift[i - 1] + eps[i]
    filled = filled + (white + drift).astype(np.float32)
    ball = np.where(mask[:, None], filled, np.nan).astype(np.float32)

    # (2) players
    present = np.isfinite(px)
    px = px + rng.normal(0, player_noise_m, px.shape).astype(np.float32)
    py = py + rng.normal(0, player_noise_m, py.shape).astype(np.float32)
    # burst drop-outs: per slot, a two-state Markov chain with mean burst 2 s
    p_end = 1.0 / 50.0
    p_start = p_end * player_drop_rate / (1 - player_drop_rate)
    for s in range(px.shape[1]):
        if not present[:, s].any():
            continue
        u = rng.rand(t)
        state = np.zeros(t, dtype=bool)
        cur = False
        for i in range(t):  # exact two-state chain; ~1.5e5 steps per slot
            if cur:
                if u[i] < p_end:
                    cur = False
            elif u[i] < p_start:
                cur = True
            state[i] = cur
        px[state, s] = np.nan
        py[state, s] = np.nan

    out = Match(name=match.name + "_deg{}".format(seed), fps=match.fps, pitch=match.pitch,
                ball=ball, px=px.astype(np.float32), py=py.astype(np.float32),
                team_of_slot=match.team_of_slot, blocks=list(match.blocks),
                labels=match.labels, extras={"vis_mask": mask})
    return out.validate()
