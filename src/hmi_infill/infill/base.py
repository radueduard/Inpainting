"""Common interface for infill methods.

Every method is a callable taking an :class:`InfillInput` and returning the
filled image (same shape as ``damaged``). Methods may only change pixels
inside ``mask``; :func:`run` enforces this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..pipeline import Trial


@dataclass
class InfillInput:
    damaged: np.ndarray                  # decoded image with holes
    mask: np.ndarray                     # pixels to fill (True)
    valid: np.ndarray | None = None      # on-disk pixels (None = all)
    trial: Trial | None = None           # full decoder output (bounds, bit-planes, ...)
    past: list[np.ndarray] = field(default_factory=list)    # t-1, t-2, ... (nearest first)
    future: list[np.ndarray] = field(default_factory=list)  # t+1, ... (nearest first)
    past_masks: list[np.ndarray] = field(default_factory=list)
    future_masks: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def from_trials(cls, trials: list[Trial], index: int, n_past: int = 3, n_future: int = 0,
                    mask: str = "mask_lost") -> "InfillInput":
        """Build the input for frame `index` of a sequence, using damaged
        neighbouring frames as references (they suffered their own losses)."""
        t = trials[index]
        past = [trials[j] for j in range(index - 1, max(index - 1 - n_past, -1), -1)]
        fut = [trials[j] for j in range(index + 1, min(index + 1 + n_future, len(trials)))]
        return cls(
            damaged=t.damaged, mask=getattr(t, mask), valid=t.valid, trial=t,
            past=[p.damaged for p in past], future=[f.damaged for f in fut],
            past_masks=[getattr(p, mask) for p in past],
            future_masks=[getattr(f, mask) for f in fut],
        )


class InfillMethod(Protocol):
    name: str

    def __call__(self, x: InfillInput) -> np.ndarray: ...


def run(method: InfillMethod, x: InfillInput) -> np.ndarray:
    out = np.asarray(method(x), dtype=np.float64)
    return np.where(x.mask, out, x.damaged)
