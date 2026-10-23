"""The in-memory representation every other module works on.

A Match is source-agnostic: the competition ATD and the open Metrica matches both load into it, so
features, models and decoders never know which one they are looking at.

Conventions
-----------
- Positions are in metres, origin at a pitch corner, x along the length, y along the width.
- Missing observations are NaN (never 0).
- `team_of_slot[s]` is 0 for home and 1 for away. Slots are tracklets, not people.
- `blocks` are the half-open [start, stop) frame spans that are graded / labelled. Nothing temporal
  is ever computed across a block boundary.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Match:
    name: str
    fps: int
    pitch: Tuple[float, float]
    ball: np.ndarray                 # (T, 2) float32, metres, NaN when missing
    px: np.ndarray                   # (T, S) float32, metres
    py: np.ndarray                   # (T, S) float32, metres
    team_of_slot: np.ndarray         # (S,) int8, 0 home / 1 away
    blocks: List[Tuple[int, int]]    # graded half-open spans, one per half
    labels: Optional[np.ndarray] = None   # (T,) int8 H/A/D/X codes when known
    extras: dict = field(default_factory=dict)

    @property
    def n_frames(self):
        return self.ball.shape[0]

    def validate(self):
        """Cheap structural checks; raise with a readable message on the first failure."""
        t = self.n_frames
        if self.px.shape != self.py.shape or self.px.shape[0] != t:
            raise ValueError("player arrays {} / {} do not match {} frames".format(
                self.px.shape, self.py.shape, t))
        if self.team_of_slot.shape[0] != self.px.shape[1]:
            raise ValueError("team_of_slot has {} slots, positions have {}".format(
                self.team_of_slot.shape[0], self.px.shape[1]))
        if not set(np.unique(self.team_of_slot).tolist()) <= {0, 1}:
            raise ValueError("team_of_slot must be 0/1")
        for start, stop in self.blocks:
            if not 0 <= start < stop <= t:
                raise ValueError("block {} outside [0, {})".format((start, stop), t))
        if self.labels is not None and self.labels.shape[0] != t:
            raise ValueError("labels length {} != {}".format(self.labels.shape[0], t))
        return self
