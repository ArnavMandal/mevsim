"""Generation and bookkeeping of MEV opportunities."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from mevsim.core.config import SimConfig
from mevsim.core.types import Opportunity


class OpportunityPool:
    """Holds live MEV opportunities and generates new ones."""

    def __init__(self, config: SimConfig, rng: np.random.Generator, ids: Iterator[int]):
        self.config = config
        self.rng = rng
        self._ids = ids
        self.active: dict[int, Opportunity] = {}

    def arrive(self, step: int) -> list[Opportunity]:
        """Add Poisson(opportunity_rate) opportunities with Pareto-distributed values.

        Values are `scale * (1 + Lomax(alpha))`, i.e. Pareto with minimum `scale`.
        """
        c = self.config
        n = int(self.rng.poisson(c.opportunity_rate))
        values = c.opportunity_value_scale * (1.0 + self.rng.pareto(c.opportunity_value_alpha, n))
        new = [
            Opportunity(next(self._ids), float(v), step, step + c.opportunity_ttl) for v in values
        ]
        for opp in new:
            self.active[opp.id] = opp
        return new

    def remove_captured(self, opp_ids: list[int]) -> None:
        for oid in opp_ids:
            self.active.pop(oid, None)

    def expire(self, step: int) -> None:
        """Drop opportunities that cannot be captured after `step`."""
        self.active = {k: o for k, o in self.active.items() if o.expires_step > step}
