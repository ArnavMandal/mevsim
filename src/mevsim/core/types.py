"""Plain data types shared across the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TxKind(StrEnum):
    ORGANIC = "organic"
    AGENT = "agent"


@dataclass
class Transaction:
    """A transaction competing for block space.

    `owner` is -1 for organic users, otherwise the agent index.
    `opportunity_id` is set for agent txs that capture an MEV opportunity.
    """

    id: int
    owner: int
    gas_used: int
    max_fee_per_gas: float
    arrival_step: int
    kind: TxKind
    opportunity_id: int | None = None

    @property
    def total_fee(self) -> float:
        return self.gas_used * self.max_fee_per_gas


@dataclass
class Opportunity:
    """An MEV opportunity worth `value` to whoever captures it first."""

    id: int
    value: float
    arrival_step: int
    expires_step: int

    def is_active(self, step: int) -> bool:
        return self.arrival_step <= step <= self.expires_step


@dataclass(frozen=True)
class Bid:
    """A sealed bid: agent offers to pay `amount` (total, not per gas) for an opportunity."""

    agent: int
    opportunity_id: int
    amount: float


@dataclass(frozen=True)
class AuctionResult:
    """Outcome of the auction for one opportunity."""

    opportunity_id: int
    winner: int
    payment: float
    n_bidders: int


@dataclass
class Block:
    """A built block. `txs` are in inclusion order."""

    step: int
    gas_limit: int
    txs: list[Transaction] = field(default_factory=list)

    @property
    def gas_used(self) -> int:
        return sum(tx.gas_used for tx in self.txs)

    @property
    def builder_revenue(self) -> float:
        return sum(tx.total_fee for tx in self.txs)

    @property
    def clearing_price(self) -> float:
        """Lowest included fee per gas (0 for an empty block)."""
        return min((tx.max_fee_per_gas for tx in self.txs), default=0.0)


@dataclass
class Observation:
    """What one agent sees at the start of a step (partial and noisy)."""

    agent: int
    step: int
    balance: float
    visible_opportunities: list[Opportunity]
    top_fees: list[float]  # noisy top-k pending organic fees per gas, descending, zero-padded
    pending_gas: float  # noisy total pending organic gas
    gas_limit: int
    last_clearing_price: float
    # (opportunity value, payment) for each MEV capture in the previous block. Public:
    # on a real chain, included txs and the profit they extracted are visible after the fact.
    last_block_captures: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class StepResult:
    """Everything that happened in one block."""

    block: Block
    auction_results: list[AuctionResult]
    captured: list[Opportunity]
    rewards: list[float]  # per agent: captured value minus payment
