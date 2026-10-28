"""Auction mechanisms that decide who wins each MEV opportunity."""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

import numpy as np

from mevsim.core.types import AuctionResult, Bid


class Auction(Protocol):
    def resolve(self, bids: list[Bid], rng: np.random.Generator) -> list[AuctionResult]:
        """Return one result per opportunity that received at least one positive bid."""
        ...


def _group_valid_bids(bids: list[Bid]) -> dict[int, dict[int, float]]:
    """Group bids by opportunity, keeping each agent's highest positive bid."""
    grouped: dict[int, dict[int, float]] = defaultdict(dict)
    for bid in bids:
        if bid.amount <= 0:
            continue
        per_agent = grouped[bid.opportunity_id]
        per_agent[bid.agent] = max(per_agent.get(bid.agent, 0.0), bid.amount)
    return grouped


class SealedFirstPrice:
    """One sealed round per opportunity; highest bid wins and pays its own bid.

    Ties are broken uniformly at random using the simulator's generator.
    """

    def resolve(self, bids: list[Bid], rng: np.random.Generator) -> list[AuctionResult]:
        results = []
        grouped = _group_valid_bids(bids)
        for opp_id in sorted(grouped):
            per_agent = grouped[opp_id]
            top = max(per_agent.values())
            tied = sorted(a for a, amt in per_agent.items() if amt == top)
            winner = tied[int(rng.integers(len(tied)))] if len(tied) > 1 else tied[0]
            results.append(AuctionResult(opp_id, winner, top, len(per_agent)))
        return results


MECHANISMS: dict[str, type] = {"sealed_first_price": SealedFirstPrice}


def make_auction(name: str) -> Auction:
    """Build an auction mechanism by name."""
    if name not in MECHANISMS:
        raise ValueError(f"unknown mechanism {name!r}; choose from {sorted(MECHANISMS)}")
    return MECHANISMS[name]()
