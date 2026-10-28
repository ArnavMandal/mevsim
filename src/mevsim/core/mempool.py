"""Mempool of pending organic transactions."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from mevsim.core.config import SimConfig
from mevsim.core.types import Transaction, TxKind


class Mempool:
    """Holds pending organic txs and generates new arrivals."""

    def __init__(self, config: SimConfig, rng: np.random.Generator, ids: Iterator[int]):
        self.config = config
        self.rng = rng
        self._ids = ids
        self.pending: list[Transaction] = []

    def arrive(self, step: int, congestion: float) -> list[Transaction]:
        """Add Poisson(organic_rate) new txs; fees scale up with congestion."""
        c = self.config
        n = int(self.rng.poisson(c.organic_rate))
        gas = self.rng.integers(c.organic_gas_min, c.organic_gas_max + 1, size=n)
        median_fee = c.organic_fee_scale * (1.0 + c.congestion_sensitivity * congestion)
        fees = median_fee * self.rng.lognormal(0.0, c.organic_fee_sigma, size=n)
        new = [
            Transaction(next(self._ids), -1, int(g), float(f), step, TxKind.ORGANIC)
            for g, f in zip(gas, fees, strict=True)
        ]
        self.pending.extend(new)
        return new

    def remove(self, included: list[Transaction]) -> None:
        """Drop included txs from the pending set."""
        done = {tx.id for tx in included}
        self.pending = [tx for tx in self.pending if tx.id not in done]

    @property
    def pending_gas(self) -> int:
        return sum(tx.gas_used for tx in self.pending)
