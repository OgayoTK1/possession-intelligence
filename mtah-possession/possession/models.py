"""Per-frame state classifier: LightGBM over the features in features.py.

Training data are the open Metrica matches, degraded to the competition's tracking quality,
each also team-swapped (features rebuilt from a mirrored Match), so the model has no notion of
"home" beyond what the frame shows. Deterministic: fixed seeds, deterministic=True, fixed threads.
"""

import json

import numpy as np

from .constants import X
from .features import frame_features, swap_teams

LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 400,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "max_bin": 127,
    "verbose": -1,
    "seed": 7,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 2,
}
N_ROUNDS = 300
STRIDE = 4


def rows_for_training(match, stride=STRIDE, drop=None):
    """(names, X, y) from the graded frames of a labelled match, every `stride`-th frame."""
    names, F = frame_features(match)
    keep = np.zeros(match.n_frames, dtype=bool)
    for start, stop in match.blocks:
        keep[start:stop:stride] = True
    keep &= match.labels != X
    cols = _keep_cols(names, drop)
    return [names[i] for i in cols], F[keep][:, cols], match.labels[keep].astype(np.int64)


def _dropped(name, pattern):
    """A pattern starting with '_' is a suffix group (e.g. '_m400', '_past'); else a prefix."""
    return pattern in name if pattern.startswith("_") else name.startswith(pattern)


def _keep_cols(names, drop):
    drop = [p for p in (drop or []) if p != "__none__"]
    if not drop:
        return list(range(len(names)))
    keep = [i for i, n in enumerate(names) if not any(_dropped(n, p) for p in drop)]
    unmatched = [p for p in drop if not any(_dropped(n, p) for n in names)]
    if unmatched:
        raise ValueError("ablation patterns matched no feature: {}".format(unmatched))
    return keep


def build_training_set(matches, augment_swap=True, drop=None):
    Xs, ys, names = [], [], None
    for m in matches:
        for mm in ([m, swap_teams(m)] if augment_swap else [m]):
            n, Xm, ym = rows_for_training(mm, drop=drop)
            names = n
            Xs.append(Xm)
            ys.append(ym)
    return names, np.concatenate(Xs), np.concatenate(ys)


def train(matches, params=None, rounds=N_ROUNDS, drop=None):
    import lightgbm as lgb
    names, Xtr, ytr = build_training_set(matches, drop=drop)
    p = dict(LGB_PARAMS)
    if params:
        p.update(params)
    booster = lgb.train(p, lgb.Dataset(Xtr, ytr, feature_name=names, free_raw_data=True),
                        num_boost_round=rounds)
    return FrameModel(booster, names, drop)


class FrameModel:
    def __init__(self, booster, names, drop=None):
        self.booster = booster
        self.names = names
        self.drop = drop or []

    def log_proba(self, match, symmetric=True):
        """(T, 3) log class probabilities; NaN rows outside blocks.

        symmetric: average with the team-swapped prediction (H/A exchanged back), so the output is
        exactly team-equivariant and carries no home bias.
        """
        p = self._proba(match)
        if symmetric:
            q = self._proba(swap_teams(match))
            p = 0.5 * (p + q[:, [1, 0, 2]])
        with np.errstate(divide="ignore"):
            return np.log(np.clip(p, 1e-9, 1.0))

    def _proba(self, match):
        names, F = frame_features(match)
        cols = [names.index(n) for n in self.names]
        out = np.full((match.n_frames, 3), np.nan)
        for start, stop in match.blocks:
            out[start:stop] = self.booster.predict(F[start:stop][:, cols])
        return out

    def save(self, path):
        self.booster.save_model(path)
        with open(path + ".json", "w") as f:
            json.dump({"names": self.names, "drop": self.drop}, f)

    @classmethod
    def load(cls, path):
        import lightgbm as lgb
        booster = lgb.Booster(model_file=path)
        meta = json.load(open(path + ".json"))
        return cls(booster, meta["names"], meta["drop"])
