"""Packet-erasure channel models."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class NoLoss:
    def loss_pattern(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.zeros(n, bool)

    def describe(self) -> dict:
        return {"model": "none"}


@dataclass
class Bernoulli:
    """Independent losses with probability p."""
    p: float

    def loss_pattern(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.random(n) < self.p

    def describe(self) -> dict:
        return {"model": "bernoulli", **asdict(self)}


@dataclass
class GilbertElliott:
    """Two-state Markov channel (good / bad) for bursty losses.

    p_gb: P(good -> bad), p_bg: P(bad -> good),
    loss_good / loss_bad: loss probability in each state.
    """
    p_gb: float
    p_bg: float
    loss_good: float = 0.0
    loss_bad: float = 1.0

    @classmethod
    def from_mean(cls, mean_loss: float, mean_burst: float) -> "GilbertElliott":
        """Gilbert model (no loss in good, all lost in bad) with the given
        average loss rate and average burst length in packets."""
        p_bg = 1.0 / mean_burst
        p_gb = mean_loss * p_bg / (1.0 - mean_loss)
        return cls(p_gb=p_gb, p_bg=p_bg)

    @property
    def mean_loss(self) -> float:
        pi_bad = self.p_gb / (self.p_gb + self.p_bg)
        return pi_bad * self.loss_bad + (1 - pi_bad) * self.loss_good

    def loss_pattern(self, n: int, rng: np.random.Generator) -> np.ndarray:
        pi_bad = self.p_gb / (self.p_gb + self.p_bg)
        bad = rng.random() < pi_bad
        u_state, u_loss = rng.random(n), rng.random(n)
        out = np.empty(n, bool)
        for i in range(n):
            out[i] = u_loss[i] < (self.loss_bad if bad else self.loss_good)
            bad = (u_state[i] >= self.p_bg) if bad else (u_state[i] < self.p_gb)
        return out

    def describe(self) -> dict:
        return {"model": "gilbert_elliott", **asdict(self), "mean_loss": self.mean_loss}


def transmit(packets: list[bytes], channel, rng: np.random.Generator):
    """Pass packets through the channel. Returns (received, lost_indices)."""
    lost = channel.loss_pattern(len(packets), rng)
    return [p for p, l in zip(packets, lost) if not l], np.flatnonzero(lost)
