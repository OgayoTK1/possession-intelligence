"""Turning per-frame evidence into a label sequence.

Two families:

* `majority_smooth`: the official baseline's centred majority vote, ported exactly (verified
  against the committed baseline CSVs in tests/), so the baseline can be rebuilt on any Match.

* `hsmm_viterbi`: a three-class hidden semi-Markov model approximated by Erlang state expansion.
  Each class c in {H, A, D} is a chain of k_c sub-states, so a stay in class c has a negative
  binomial duration with mean mu_c instead of the geometric (memoryless) duration of a plain HMM.
  That is the only "minimum duration" mechanism in the system: it makes a one-frame turnover
  improbable without forbidding a genuinely short possession, and its parameters are fitted from
  training labels rather than set by hand.

  Emissions are hybrid-HMM scaled likelihoods: alpha * (log p(c | x_t) - log prior(c)). alpha < 1
  compensates for consecutive frames not being independent evidence; it is the one decoder
  parameter chosen by cross-validated spellF1.
"""

import numpy as np

from .constants import A, D, H, X


# -- the official baseline's smoothing ---------------------------------------------------------

def majority_smooth(codes, window):
    """codes: 0 home / 1 away / -1 none. Identical to starter/baseline.py smooth()."""
    if window <= 0:
        return np.where(codes == 0, H, np.where(codes == 1, A, D)).astype(np.int8)
    home_c = np.concatenate(([0], np.cumsum(codes == 0)))
    away_c = np.concatenate(([0], np.cumsum(codes == 1)))
    idx = np.arange(len(codes))
    lo = np.maximum(0, idx - window)
    hi = np.minimum(len(codes), idx + window + 1)
    home = home_c[hi] - home_c[lo]
    away = away_c[hi] - away_c[lo]
    out = np.full(len(codes), D, dtype=np.int8)
    seen = (home + away) > 0
    out[seen] = np.where(home[seen] >= away[seen], H, A)
    return out


# -- duration-aware Viterbi --------------------------------------------------------------------

def fit_duration_model(labels, blocks, k=4):
    """Mean durations and class-to-class exit probabilities from labelled blocks.

    Returns dict(mu=(3,), trans=(3,3) row-stochastic with zero diagonal, prior=(3,), k=int).
    """
    from .constants import pm
    dur = {c: [] for c in (H, A, D)}
    trans = np.ones((3, 3)) * 0.5  # light Laplace smoothing
    np.fill_diagonal(trans, 0.0)
    for block in blocks:           # transitions are counted inside a block, never across halves
        prev = None
        for start, stop, c in pm.segments(labels, [block]):
            if c == X:
                prev = None
                continue
            dur[c].append(stop - start)
            if prev is not None:
                trans[prev, c] += 1
            prev = c
    trans = trans / trans.sum(axis=1, keepdims=True)
    mu = np.array([np.mean(dur[c]) for c in (H, A, D)])
    counts = np.array([np.sum(labels == c) for c in (H, A, D)], dtype=float)
    return {"mu": mu, "trans": trans, "prior": counts / counts.sum(), "k": int(k)}


def hsmm_viterbi(logp, model, alpha, team_symmetric=True):
    """Most likely H/A/D path given per-frame class log-probabilities.

    logp: (T, 3) natural-log class probabilities for H, A, D (rows sum to 1 in prob space).
    model: from fit_duration_model. alpha: emission scale.
    team_symmetric: average the H and A duration/transition statistics, so the decoder carries no
    home/away bias of its own (the training matches' home sides are not Barcelona).
    """
    k = int(model["k"])
    mu = np.array(model["mu"], dtype=float)
    trans = np.array(model["trans"], dtype=float)
    prior = np.array(model["prior"], dtype=float)
    if team_symmetric:
        mu[H] = mu[A] = 0.5 * (mu[H] + mu[A])
        prior[H] = prior[A] = 0.5 * (prior[H] + prior[A])
        t_ha = 0.5 * (trans[H, A] + trans[A, H])
        t_hd = 0.5 * (trans[H, D] + trans[A, D])
        t_dh = 0.5 * (trans[D, H] + trans[D, A])
        trans = np.array([[0, t_ha, t_hd], [t_ha, 0, t_hd], [t_dh, t_dh, 0]])
        trans = trans / trans.sum(axis=1, keepdims=True)

    n_cls = 3
    n_states = n_cls * k
    stay = np.clip(1.0 - k / np.maximum(mu, k + 1.0), 1e-6, 1 - 1e-9)  # per sub-state self-loop
    log_stay = np.log(stay)
    log_adv = np.log1p(-stay)
    log_trans = np.log(np.maximum(trans, 1e-300))

    emis = alpha * (logp - np.log(prior)[None, :])  # (T, 3)
    t_len = emis.shape[0]

    # state s = c * k + j
    delta = np.full(n_states, -np.inf)
    for c in range(n_cls):
        delta[c * k] = np.log(prior[c]) + emis[0, c]
    back = np.zeros((t_len, n_states), dtype=np.int16)

    cls_of = np.repeat(np.arange(n_cls), k)
    first = np.arange(n_cls) * k
    last = first + k - 1
    self_lp = log_stay[cls_of]
    adv_lp = log_adv[cls_of]
    idx = np.arange(n_states)

    for t in range(1, t_len):
        # candidate 1: stay in the same sub-state
        best = delta + self_lp
        arg = idx.copy()
        # candidate 2: advance j-1 -> j inside the class
        adv = np.full(n_states, -np.inf)
        adv[1:] = delta[:-1] + adv_lp[:-1]
        adv[first] = -np.inf
        better = adv > best
        best = np.where(better, adv, best)
        arg = np.where(better, idx - 1, arg)
        # candidate 3: exit the last sub-state of class c into the first of class c'
        for c2 in range(n_cls):
            s2 = first[c2]
            cand = delta[last] + log_adv[np.arange(n_cls)] + log_trans[:, c2]
            cand[c2] = -np.inf
            j = int(np.argmax(cand))
            if cand[j] > best[s2]:
                best[s2] = cand[j]
                arg[s2] = last[j]
        delta = best + emis[t, cls_of]
        back[t] = arg

    path = np.empty(t_len, dtype=np.int64)
    s = int(np.argmax(delta))
    for t in range(t_len - 1, -1, -1):
        path[t] = s
        s = back[t, s]
    return cls_of[path].astype(np.int8)


def decode_blocks(logp, blocks, n_frames, model, alpha, fill=X):
    """Decode each graded block independently; ungraded frames get `fill` (X)."""
    out = np.full(n_frames, fill, dtype=np.int8)
    for start, stop in blocks:
        out[start:stop] = hsmm_viterbi(logp[start:stop], model, alpha)
    return out


def argmax_blocks(logp, blocks, n_frames, fill=X):
    out = np.full(n_frames, fill, dtype=np.int8)
    for start, stop in blocks:
        out[start:stop] = np.argmax(logp[start:stop], axis=1)
    return out
