"""SkillCorner open data (MIT; A-League 2024/25, 20 matches) as extra labelled training matches.

Source: https://github.com/SkillCorner/opendata -- broadcast tracking at 10 fps with player and
ball positions (undetected players are extrapolated by SkillCorner), plus "dynamic events".

Labels are derived algorithmically from the dynamic events with the same possession rules as the
Metrica matches (external.py):

  * A team gains possession at the start frame of one of its `player_possession` events (a
    controlled possession: reception, interception, recovery). Duels without control never flip.
  * Loose phases keep the previous team.
  * Dead ball: after a player possession whose `game_interruption_after` is set (throw-in,
    corner, goal kick, free kick, goal) from AFTER_LAG after its end frame, until RESTART_LAG
    before the next possession whose `game_interruption_before` is set. That restart belongs to
    the possession's team when the interruption is "..._for", otherwise to the opponent.
  * Before the first possession of a period: dead (pre-kick-off), inside the graded block.

The two lags are approximations (the exact frame the ball crosses the line or is restarted is not
in the events): boundary noise of under a second on stoppages, stated in the report.

Tracking is resampled from 10 to 25 fps by linear interpolation so that every frame-based window
in features.py keeps its meaning. Ball is used only where SkillCorner marks it detected.
"""

import json
import os

import numpy as np
import pandas as pd

from .constants import A, D, H, X
from .match import Match

SRC_FPS = 10
FPS = 25
AFTER_LAG = 5      # 0.5 s at 10 fps: pass/clearance travelling out after the possession ends
RESTART_LAG = 10   # 1.0 s at 10 fps: throw/kick travelling to the receiver


def _load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def derive_labels_10fps(events, periods, n):
    """H/A/D/X labels at the source rate. periods: array of period per frame (0 = none)."""
    lab = np.full(n, X, dtype=np.int8)
    pp = events[events["event_type"] == "player_possession"].sort_values("frame_start")
    for period in (1, 2):
        idx = np.flatnonzero(periods == period)
        if not len(idx):
            continue
        p_lo, p_hi = int(idx[0]), int(idx[-1]) + 1
        ev = pp[pp["period"] == period]
        if not len(ev):
            continue
        cur = np.full(p_hi - p_lo, -1, dtype=np.int8)
        changes = [(p_lo, D)]
        first = True
        for _, e in ev.iterrows():
            team = H if e["team_is_home"] else A
            other = A if team == H else H
            s, t = int(e["frame_start"]), int(e["frame_end"])
            before = e["game_interruption_before"]
            if isinstance(before, str) and before or first:
                restart_team = team
                if isinstance(before, str) and before.endswith("_against"):
                    restart_team = other
                changes.append((max(p_lo, s - RESTART_LAG), restart_team))
                first = False
            changes.append((s, team))
            after = e["game_interruption_after"]
            if isinstance(after, str) and after:
                changes.append((t + AFTER_LAG, D))
        changes.sort(key=lambda c: c[0])
        state, prev = D, p_lo
        for f, st in changes:
            f = min(max(f, p_lo), p_hi)
            if f > prev:
                cur[prev - p_lo:f - p_lo] = state
            prev, state = max(prev, f), st
        cur[prev - p_lo:] = state
        lab[p_lo:p_hi] = cur
    return lab


def load_skillcorner_match(match_dir):
    mid = os.path.basename(os.path.normpath(match_dir))
    meta = json.load(open(os.path.join(match_dir, "{}_match.json".format(mid))))
    rows = _load_rows(os.path.join(match_dir, "{}_tracking_extrapolated.jsonl".format(mid)))
    events = pd.read_csv(os.path.join(match_dir, "{}_dynamic_events.csv".format(mid)),
                         low_memory=False)
    home_id = meta["home_team"]["id"]
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    team_of_player = {p["id"]: (0 if p["team_id"] == home_id else 1) for p in meta["players"]}
    pids = sorted(team_of_player)
    slot = {p: k for k, p in enumerate(pids)}
    events = events.copy()
    events["team_is_home"] = events["team_id"] == home_id

    n = len(rows)
    frames = np.array([r["frame"] for r in rows])
    if not np.array_equal(frames, np.arange(frames[0], frames[0] + n)):
        raise ValueError("non-contiguous frames in {}".format(mid))
    periods = np.array([r["period"] or 0 for r in rows], dtype=np.int8)
    px = np.full((n, len(pids)), np.nan, dtype=np.float32)
    py = np.full((n, len(pids)), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        b = r["ball_data"]
        if b and b.get("is_detected") and b.get("x") is not None:
            ball[i] = (b["x"] + length / 2, b["y"] + width / 2)
        for p in r["player_data"]:
            k = slot.get(p["player_id"])
            if k is not None and p["x"] is not None:
                px[i, k] = p["x"] + length / 2
                py[i, k] = p["y"] + width / 2
    # events are indexed by the tracking frame number
    offset = int(frames[0])
    events["frame_start"] = events["frame_start"] - offset
    events["frame_end"] = events["frame_end"] - offset
    lab10 = derive_labels_10fps(events, periods, n)

    # resample 10 -> 25 fps; positions linear (never across a NaN), labels nearest
    t_src = np.arange(n) / SRC_FPS
    m = int(np.floor(t_src[-1] * FPS)) + 1
    t_dst = np.arange(m) / FPS
    src_i = np.clip(np.round(t_dst * SRC_FPS).astype(int), 0, n - 1)
    out = {}
    for name, arr in (("px", px), ("py", py), ("ball", ball)):
        a2 = arr.reshape(n, -1)
        res = np.full((m, a2.shape[1]), np.nan, dtype=np.float32)
        lo = np.minimum(np.floor(t_dst * SRC_FPS).astype(int), n - 1)
        hi = np.minimum(lo + 1, n - 1)
        w = (t_dst * SRC_FPS - lo)[:, None]
        res[:] = (1 - w) * a2[lo] + w * a2[hi]   # NaN if either neighbour is NaN
        out[name] = res.reshape((m,) + arr.shape[1:])
    labels = lab10[src_i]
    per25 = periods[src_i]
    blocks = []
    for period in (1, 2):
        idx = np.flatnonzero((per25 == period) & (labels != X))
        if len(idx):
            blocks.append((int(idx[0]), int(idx[-1]) + 1))
    labels_g = np.full(m, X, dtype=np.int8)
    for a, b in blocks:
        labels_g[a:b] = labels[a:b]
    return Match(name="sc{}".format(mid), fps=FPS, pitch=(length, width), ball=out["ball"],
                 px=out["px"], py=out["py"], team_of_slot=np.array([team_of_player[p] for p in pids],
                                                                  dtype=np.int8),
                 blocks=blocks, labels=labels_g).validate()


def list_matches(root):
    base = os.path.join(root, "data", "matches")
    out = []
    for mid in sorted(os.listdir(base)):
        f = os.path.join(base, mid, "{}_tracking_extrapolated.jsonl".format(mid))
        if os.path.isfile(f) and os.path.getsize(f) > 10000:
            out.append(os.path.join(base, mid))
    return out
