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


def _pick_winner(per_agent: dict[int, float], rng: np.random.Generator) -> tuple[int, float]:
    """Return (winner, top bid); ties broken uniformly at random, independent of bid order."""
    top = max(per_agent.values())
    tied = sorted(a for a, amt in per_agent.items() if amt == top)
    winner = tied[int(rng.integers(len(tied)))] if len(tied) > 1 else tied[0]
    return winner, top


class SealedFirstPrice:
    """One sealed round per opportunity; highest bid wins and pays its own bid.

    Ties are broken uniformly at random using the simulator's generator.
    """

    def resolve(self, bids: list[Bid], rng: np.random.Generator) -> list[AuctionResult]:
        results = []
        grouped = _group_valid_bids(bids)
        for opp_id in sorted(grouped):
            per_agent = grouped[opp_id]
            winner, top = _pick_winner(per_agent, rng)
            results.append(AuctionResult(opp_id, winner, top, len(per_agent), top))
        return results


class SealedSecondPrice:
    """One sealed round per opportunity; highest bid wins and pays the second-highest bid.

    A sole bidder pays 0 (no reserve price). On a tie the winner (chosen at random) pays
    the tied amount, since the second-highest bid equals its own.
    """

    def resolve(self, bids: list[Bid], rng: np.random.Generator) -> list[AuctionResult]:
        results = []
        grouped = _group_valid_bids(bids)
        for opp_id in sorted(grouped):
            per_agent = grouped[opp_id]
            winner, top = _pick_winner(per_agent, rng)
            others = [amt for a, amt in per_agent.items() if a != winner]
            payment = max(others, default=0.0)
            results.append(AuctionResult(opp_id, winner, payment, len(per_agent), top))
        return results


MECHANISMS: dict[str, type] = {
    "sealed_first_price": SealedFirstPrice,
    "sealed_second_price": SealedSecondPrice,
}


def make_auction(name: str) -> Auction:
    """Build an auction mechanism by name."""
    if name not in MECHANISMS:
        raise ValueError(f"unknown mechanism {name!r}; choose from {sorted(MECHANISMS)}")
    return MECHANISMS[name]()
