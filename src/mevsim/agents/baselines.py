"""Baseline agents from SPEC section 5: random, greedy, and rule-based."""

from __future__ import annotations

from typing import Any

import numpy as np

from mevsim.core.types import Bid, Observation


class Agent:
    """Turns an observation into bids. Each agent owns a seeded generator."""

    name = "agent"

    def __init__(self, index: int, rng: np.random.Generator):
        self.index = index
        self.rng = rng

    def reset(self) -> None:
        """Clear any per-episode state."""

    def act(self, obs: Observation) -> list[Bid]:
        return [
            Bid(self.index, o.id, self.bid_fraction(obs) * o.value)
            for o in obs.visible_opportunities
        ]

    def bid_fraction(self, obs: Observation) -> float:
        raise NotImplementedError


class RandomAgent(Agent):
    """Bids a uniform random fraction of each visible opportunity's value."""

    name = "random"

    def act(self, obs: Observation) -> list[Bid]:
        opps = obs.visible_opportunities
        fractions = self.rng.uniform(0.0, 1.0, len(opps))
        return [Bid(self.index, o.id, f * o.value) for o, f in zip(opps, fractions, strict=True)]


class GreedyAgent(Agent):
    """Always bids a fixed fraction of visible value."""

    name = "greedy"

    def __init__(self, index: int, rng: np.random.Generator, fraction: float = 0.9):
        super().__init__(index, rng)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be in [0, 1]")
        self.fraction = fraction

    def bid_fraction(self, obs: Observation) -> float:
        return self.fraction


class RuleBasedAgent(Agent):
    """Bids just above the recent winning price, keeping a minimum margin.

    Tracks an EMA of winning payment / value from the previous block's public captures
    and bids `ratio + step_up` of value, capped at `1 - margin`. Under PGA it also acts
    as a classic PGA bot: when another agent holds the top bid on an opportunity, it
    raises to `top + step_up * value` (still capped at `1 - margin` of value).
    """

    name = "rule_based"

    def __init__(
        self,
        index: int,
        rng: np.random.Generator,
        margin: float = 0.1,
        step_up: float = 0.05,
        smoothing: float = 0.2,
        initial_ratio: float = 0.5,
    ):
        super().__init__(index, rng)
        if not 0.0 <= margin <= 1.0:
            raise ValueError("margin must be in [0, 1]")
        self.margin = margin
        self.step_up = step_up
        self.smoothing = smoothing
        self.initial_ratio = initial_ratio
        self.reset()

    def reset(self) -> None:
        self.ratio = self.initial_ratio

    track_bids = False  # learn from winning payments (False) or winning bids (True)

    def act(self, obs: Observation) -> list[Bid]:
        if obs.round == 0:  # once per block, not once per PGA round
            observed = obs.last_block_bids if self.track_bids else obs.last_block_captures
            for value, price in observed:
                if value > 0:
                    a = self.smoothing
                    self.ratio = a * (price / value) + (1.0 - a) * self.ratio
        base = self.bid_fraction(obs)
        bids = []
        for o in obs.visible_opportunities:
            fraction = base
            if o.id in obs.top_bids and o.id not in obs.leading and o.value > 0:
                outbid = obs.top_bids[o.id] / o.value + self.step_up
                fraction = max(base, min(outbid, 1.0 - self.margin))
            bids.append(Bid(self.index, o.id, fraction * o.value))
        return bids

    def bid_fraction(self, obs: Observation) -> float:
        return float(np.clip(self.ratio + self.step_up, 0.0, 1.0 - self.margin))


class RuleBasedBidAgent(RuleBasedAgent):
    """Rule-based agent that tracks winning *bids* instead of payments.

    Identical under first-price and PGA (bid == payment). Under second price the payment
    is the runner-up's bid, so tracking payments makes the plain rule-based agent ratchet
    its bids down; tracking the winning bid avoids that.
    """

    name = "rule_based_bid"
    track_bids = True


AGENTS: dict[str, type[Agent]] = {
    cls.name: cls for cls in (RandomAgent, GreedyAgent, RuleBasedAgent, RuleBasedBidAgent)
}


def make_agent(kind: str, index: int, rng: np.random.Generator, **params: Any) -> Agent:
    """Build an agent by name, e.g. `make_agent("greedy", 0, rng, fraction=0.8)`."""
    if kind not in AGENTS:
        raise ValueError(f"unknown agent {kind!r}; choose from {sorted(AGENTS)}")
    return AGENTS[kind](index, rng, **params)
