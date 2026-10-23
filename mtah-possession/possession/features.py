"""Per-frame features for possession inference, built only from positions.

Design rules
------------
* Positions only. Speeds are recomputed from positions (clipped) rather than read from the ATD
  speed channel, so the open validation matches and the competition match get identical features.
* Missing stays missing. A missing ball gives NaN ball features (LightGBM routes NaN explicitly)
  plus explicit visibility / gap-length features; nothing is filled with zeros.
* Team-symmetric and orientation-free. Coordinates are rotated per half so that the home team
  attacks +x. Every team feature exists as a (home, away) pair, which is what makes the
  team-swap augmentation in models.py exact.
* Slots are never treated as people: every player feature is a team aggregate.
* Temporal context is bidirectional (offline inference over a finished match is allowed and
  reproducible), and never crosses a half boundary.
"""

import numpy as np

SPEED_CLIP = 12.0        # m/s, above any sprint: anything faster is an identity switch
BALL_SPEED_CLIP = 40.0   # m/s
VEL_HALF = 2             # centred difference over +/- 2 frames
SMOOTH_PLAYER = 6        # +/- frames of position smoothing before velocities (0.5 s window)
SMOOTH_BALL = 2          # +/- frames for the ball (kicks are abrupt; keep it short)
CONTROL_RADIUS = 1.5     # m
NEAR_RADIUS = 5.0        # m
GAP_CAP = 1500           # frames (60 s) cap for time-since / time-to ball features
WINDOWS = (12, 50, 150, 400)       # centred half-widths, frames
DIRECTIONAL = (25, 100)            # past-only / future-only widths, frames


# -- small numpy helpers -----------------------------------------------------------------------

def _smooth_nan(x, half=4):
    """Centred NaN-aware moving average over +/- half frames along axis 0 (1-D or 2-D).

    Positions are smoothed before differentiating, so measurement noise -- whose level differs
    between the open training matches and the competition ATD -- does not inflate speeds.
    A frame is NaN only if no value in its window is finite.
    """
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    pad = [(1, 0)] + [(0, 0)] * (x.ndim - 1)
    c = np.pad(np.cumsum(np.where(ok, x, 0.0), axis=0), pad)
    n = np.pad(np.cumsum(ok, axis=0), pad)
    t = x.shape[0]
    idx = np.arange(t)
    lo = np.maximum(0, idx - half)
    hi = np.minimum(t, idx + half + 1)
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (c[hi] - c[lo]) / cnt
    out = np.where(cnt > 0, out, np.nan)
    out = np.where(ok, out, np.nan)  # never invent a position where none was observed
    return out.astype(np.float32)


def _velocity(p, fps):
    """Centred finite difference along axis 0 over +/- VEL_HALF frames; NaN-propagating."""
    v = np.full(p.shape, np.nan, dtype=np.float32)
    k = VEL_HALF
    v[k:-k] = (p[2 * k:] - p[:-2 * k]) * (fps / (2.0 * k))
    return v


def _roll_mean(x, w):
    """Centred NaN-aware mean over [t-w, t+w]; also returns the count of finite values."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    n = np.concatenate(([0], np.cumsum(ok)))
    t = len(x)
    idx = np.arange(t)
    lo = np.maximum(0, idx - w)
    hi = np.minimum(t, idx + w + 1)
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (c[hi] - c[lo]) / cnt
    mean[cnt == 0] = np.nan
    return mean.astype(np.float32), cnt.astype(np.float32)


def _dir_mean(x, w, past):
    """NaN-aware mean over the w frames strictly before (past) or strictly after t."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    n = np.concatenate(([0], np.cumsum(ok)))
    t = len(x)
    idx = np.arange(t)
    if past:
        lo, hi = np.maximum(0, idx - w), idx
    else:
        lo, hi = np.minimum(t, idx + 1), np.minimum(t, idx + 1 + w)
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (c[hi] - c[lo]) / cnt
    mean[cnt == 0] = np.nan
    return mean.astype(np.float32)


TOUCH_RADIUS = 2.0       # m
TOUCH_IMPULSE = 4.5      # m/s over 6 frames; p99.9 of the degradation noise-only impulse is 4.4
TOUCH_AGE_CAP = 750      # frames


def _smooth5(x):
    """Centred 5-frame mean; NaN unless all five are present (no smoothing across gaps)."""
    out = np.full(len(x), np.nan, dtype=np.float32)
    if len(x) >= 5:
        out[2:-2] = (x[:-4] + x[1:-3] + x[2:-2] + x[3:-1] + x[4:]) / 5.0
    return out


def _last_event(sign, forward=True):
    """Sign of the most recent (forward) or next (backward) nonzero touch, and its age in frames.

    NaN frames (ball unseen) do not reset the memory: possession belongs to the last team that
    touched the ball even through a tracking gap. Age is capped; no touch yet -> sign 0.
    """
    s = np.where(np.isfinite(sign), sign, 0.0)
    order = np.arange(len(s)) if forward else np.arange(len(s))[::-1]
    pos = np.where(s[order] != 0, np.arange(len(s)), -1)
    pos = np.maximum.accumulate(pos)
    has = pos >= 0
    out_sign = np.zeros(len(s), dtype=np.float32)
    out_age = np.full(len(s), TOUCH_AGE_CAP, dtype=np.float32)
    idx = np.arange(len(s))
    out_sign_o = np.where(has, s[order][np.maximum(pos, 0)], 0.0)
    out_age_o = np.where(has, np.minimum(idx - pos, TOUCH_AGE_CAP), TOUCH_AGE_CAP)
    out_sign[order] = out_sign_o
    out_age[order] = out_age_o
    return out_sign, out_age


def _gap_lengths(vis):
    """(frames since ball last seen, frames until ball next seen), 0 when visible, capped."""
    t = len(vis)
    idx = np.arange(t)
    last = np.where(vis, idx, -10 ** 9)
    last = np.maximum.accumulate(last)
    since = np.minimum(idx - last, GAP_CAP)
    nxt = np.where(vis, idx, 10 ** 9)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    until = np.minimum(nxt - idx, GAP_CAP)
    return since.astype(np.float32), until.astype(np.float32)


def _kth_smallest(d, k):
    """k-th smallest along axis 1 ignoring NaN (k=0 nearest). NaN if fewer than k+1 finite."""
    filled = np.where(np.isfinite(d), d, np.inf)
    if filled.shape[1] <= k:
        return np.full(d.shape[0], np.nan, dtype=np.float32)
    part = np.partition(filled, k, axis=1)[:, k]
    part[np.isinf(part)] = np.nan
    return part.astype(np.float32)


def attack_direction(match, start, stop):
    """+1 if home attacks +x in this block, else -1.

    The deepest player towards x = 0 is nearly always a goalkeeper, so the team owning that
    player most of the time defends the x = 0 goal.
    """
    px = match.px[start:stop]
    filled = np.where(np.isfinite(px), px, np.inf)
    deepest = np.argmin(filled, axis=1)
    any_ok = np.isfinite(px).any(axis=1)
    home_deepest = match.team_of_slot[deepest[any_ok]] == 0
    return 1 if home_deepest.mean() >= 0.5 else -1


# -- the feature builder -----------------------------------------------------------------------

def frame_features(match):
    """(names, F) with F of shape (T, n_features), float32. Rows outside blocks are NaN."""
    t = match.n_frames
    length, width = match.pitch
    fps = match.fps
    cols = {}

    def put(name, values):
        if name not in cols:
            cols[name] = np.full(t, np.nan, dtype=np.float32)
        cols[name][sl] = values

    for start, stop in match.blocks:
        sl = slice(start, stop)
        sign = attack_direction(match, start, stop)
        bx, by = match.ball[sl, 0].astype(np.float32), match.ball[sl, 1].astype(np.float32)
        px, py = match.px[sl], match.py[sl]
        if sign < 0:  # rotate 180 degrees so home attacks +x
            bx, by = length - bx, width - by
            px, py = length - px, width - py
        n = stop - start
        team = match.team_of_slot
        vis = np.isfinite(bx) & np.isfinite(by)

        # ---- ball kinematics
        bvx, bvy = _velocity(_smooth_nan(bx, SMOOTH_BALL), fps), _velocity(_smooth_nan(by, SMOOTH_BALL), fps)
        bspeed = np.minimum(np.hypot(bvx, bvy), BALL_SPEED_CLIP)
        bacc = np.full(n, np.nan, dtype=np.float32)
        bacc[2:-2] = np.hypot(bvx[4:] - bvx[:-4], bvy[4:] - bvy[:-4]) * (fps / 4.0)
        put("ball_vis", vis.astype(np.float32))
        put("ball_x", bx)
        put("ball_y", by)
        put("ball_speed", bspeed)
        put("ball_acc", np.minimum(bacc, 500.0))
        put("ball_vx", np.clip(bvx, -BALL_SPEED_CLIP, BALL_SPEED_CLIP))
        out_x = np.maximum(np.maximum(-bx, bx - length), 0)
        out_y = np.maximum(np.maximum(-by, by - width), 0)
        put("ball_out_dist", np.where(vis, np.hypot(out_x, out_y), np.nan))
        put("ball_touchline_dist", np.minimum(by, width - by))
        put("ball_goalline_dist", np.minimum(bx, length - bx))
        since, until = _gap_lengths(vis)
        put("ball_gap_since", since)
        put("ball_gap_until", until)
        put("ball_gap_len", np.where(vis, 0, since + until))
        put("ball_still", np.where(vis, (bspeed < 0.5).astype(np.float32), np.nan))

        # ---- players: velocities, team aggregates
        pvx, pvy = _velocity(_smooth_nan(px, SMOOTH_PLAYER), fps), _velocity(_smooth_nan(py, SMOOTH_PLAYER), fps)
        pspeed = np.minimum(np.hypot(pvx, pvy), SPEED_CLIP)
        okp = np.isfinite(px) & np.isfinite(py)
        put("n_tracked", okp.sum(axis=1).astype(np.float32))
        put("all_speed_mean", _nanmean(pspeed, axis=1))
        put("all_slow_frac", _nanmean(np.where(np.isfinite(pspeed), pspeed < 1.0, np.nan), axis=1))

        d = np.hypot(px - bx[:, None], py - by[:, None])          # (n, S), NaN if ball missing
        rel = np.hypot(pvx - bvx[:, None], pvy - bvy[:, None])
        side = {0: "h", 1: "a"}
        d1 = {}
        for k in (0, 1):
            m = team == k
            s = side[k]
            dk = d[:, m]
            d1[k] = _kth_smallest(dk, 0)
            put("d1_" + s, d1[k])
            put("d2_" + s, _kth_smallest(dk, 1))
            put("n_ctrl_" + s, np.where(vis, np.sum(dk < CONTROL_RADIUS, axis=1), np.nan))
            put("n_near_" + s, np.where(vis, np.sum(dk < NEAR_RADIUS, axis=1), np.nan))
            # relative speed of this team's nearest player to the ball
            filled = np.where(np.isfinite(dk), dk, np.inf)
            j = np.argmin(filled, axis=1)
            relk = rel[:, m][np.arange(n), j]
            relk[~np.isfinite(d1[k])] = np.nan
            put("rel_speed_" + s, np.minimum(relk, BALL_SPEED_CLIP))

            xk, yk, sk = px[:, m], py[:, m], pspeed[:, m]
            cx = _nanmean(xk, axis=1)
            put("cx_" + s, cx)
            put("cy_" + s, _nanmean(yk, axis=1))
            put("sx_" + s, _nanstd(xk, axis=1))
            put("sy_" + s, _nanstd(yk, axis=1))
            put("speed_" + s, _nanmean(sk, axis=1))
            put("vx_" + s, _nanmean(pvx[:, m], axis=1))  # + = towards the home attack goal
            put("nplay_" + s, np.isfinite(xk).sum(axis=1).astype(np.float32))
            # line heights: 2nd-deepest and 2nd-highest outfielder towards each goal
            put("xlo_" + s, _kth_smallest(xk, 1))
            put("xhi_" + s, -_kth_smallest(-xk, 1))
            # ball-to-team-centroid distance, a ball-present proxy for "the team around it"
            put("dball_c_" + s, np.hypot(cx - bx, _nanmean(yk, axis=1) - by))

        ddiff = d1[0] - d1[1]
        put("d1_diff", ddiff)
        nearest_home = np.where(vis & np.isfinite(ddiff), (ddiff < 0).astype(np.float32), np.nan)
        put("nearest_is_home", nearest_home)
        ctrl = np.where(vis, np.where((np.fmin(d1[0], d1[1]) < CONTROL_RADIUS) & (bspeed < 8.0),
                                      np.where(ddiff < 0, 1.0, -1.0), 0.0), np.nan)
        put("control_sign", ctrl.astype(np.float32))

        # ---- touches: an abrupt change of ball velocity with a player close by
        # Kinematics on a 5-frame moving average, so measurement noise is not read as a kick.
        sbx, sby = _smooth5(bx), _smooth5(by)
        svx, svy = _velocity(sbx, fps), _velocity(sby, fps)
        impulse = np.full(n, np.nan, dtype=np.float32)
        impulse[3:-3] = np.hypot(svx[6:] - svx[:-6], svy[6:] - svy[:-6])
        close = np.fmin(d1[0], d1[1]) < TOUCH_RADIUS
        touch = np.where(np.isfinite(impulse), np.where(
            (impulse > TOUCH_IMPULSE) & close, np.where(ddiff < 0, 1.0, -1.0), 0.0), np.nan)
        put("touch_impulse", np.minimum(impulse, 40.0))
        put("touch_sign", touch.astype(np.float32))
        last_sign, last_age = _last_event(touch, forward=True)
        next_sign, next_age = _last_event(touch, forward=False)
        put("touch_last_sign", last_sign)
        put("touch_last_age", last_age)
        put("touch_next_sign", next_sign)
        put("touch_next_age", next_age)
        cols_touch_roll = ("touch_sign",)

        # ---- temporal context, inside the block only
        for w in WINDOWS:
            for name in ("nearest_is_home", "control_sign", "ball_vis", "all_speed_mean",
                         "ball_speed", "d1_diff", "vx_h", "vx_a", "cx_h", "cx_a", "sx_h", "sx_a",
                         "speed_h", "speed_a", "ball_out_dist", "ball_x", "all_slow_frac"):
                mean, cnt = _roll_mean(cols[name][sl], w)
                put("{}_m{}".format(name, w), mean)
            _, cnt = _roll_mean(cols["control_sign"][sl], w)
            ctrl_abs, _ = _roll_mean(np.abs(cols["control_sign"][sl]), w)
            put("ctrl_density_m{}".format(w), ctrl_abs * cnt / (2 * w + 1))
        for w in (12, 50):
            for name in cols_touch_roll:
                mean, _ = _roll_mean(cols[name][sl], w)
                put("{}_m{}".format(name, w), mean)
        for w in DIRECTIONAL:
            for name in ("control_sign", "nearest_is_home", "all_speed_mean", "ball_vis",
                         "touch_sign"):
                put("{}_past{}".format(name, w), _dir_mean(cols[name][sl], w, past=True))
                put("{}_fut{}".format(name, w), _dir_mean(cols[name][sl], w, past=False))

    names = sorted(cols)
    F = np.stack([cols[c] for c in names], axis=1)
    return names, F


def _nanmean(a, axis):
    ok = np.isfinite(a)
    cnt = ok.sum(axis=axis)
    s = np.where(ok, a, 0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / cnt
    out = np.where(cnt > 0, out, np.nan)
    return out.astype(np.float32)


def _nanstd(a, axis):
    m = _nanmean(a, axis)
    ok = np.isfinite(a)
    dev = np.where(ok, (a - np.expand_dims(m, axis)) ** 2, 0).sum(axis=axis)
    cnt = ok.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.sqrt(dev / cnt)
    return np.where(cnt > 1, out, np.nan).astype(np.float32)


# -- team swap, for augmentation ------------------------------------------------------------------

def swap_teams(match):
    """The same match with home and away exchanged (slots, labels). Features rebuilt from this are
    exactly the team-mirrored features, because attack_direction() re-orients per half."""
    from .match import Match
    labels = None
    if match.labels is not None:
        labels = match.labels.copy()
        labels[match.labels == 0] = 1
        labels[match.labels == 1] = 0
    return Match(name=match.name + "_swap", fps=match.fps, pitch=match.pitch, ball=match.ball,
                 px=match.px, py=match.py, team_of_slot=(1 - match.team_of_slot).astype(np.int8),
                 blocks=list(match.blocks), labels=labels, extras=dict(match.extras))
