"""The core block-by-block simulator (no RL dependencies)."""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np

from mevsim.core.auction import make_auction, pga_leaders, standing_bids
from mevsim.core.builder import build_block
from mevsim.core.config import SimConfig
from mevsim.core.mempool import Mempool
from mevsim.core.opportunities import OpportunityPool
from mevsim.core.types import Bid, Block, Observation, StepResult, Transaction, TxKind


class Simulator:
    """Runs the per-step flow from SPEC section 2.

    Usage: `reset()`, then repeatedly `observe(i)` for each agent and `step(bids)`.
    After `reset()` and after each `step()`, arrivals for the current step have already
    happened, so observations always reflect the state agents bid against.

    Under PGA each block has `config.n_rounds` bidding rounds: call `bid_round(bids)` for
    rounds 0..n-2 (re-observing in between to see the top bids), then `step(bids)` for the
    last round. Each agent's observation is drawn once per block and reused across rounds,
    so what it can see does not change between rounds.

    All randomness comes from `rng`, split into independent child streams (organic
    arrivals, opportunities, auction ties, observations). Separate streams mean that
    changing how agents observe or bid never changes the organic tx sequence, which
    keeps a no-MEV counterfactual run comparable under the same seed.
    """

    def __init__(self, config: SimConfig, rng: np.random.Generator):
        self.config = config
        self.auction = make_auction(config.mechanism)
        self._root_rng = rng
        self._ready = False

    def reset(self) -> None:
        """Start a fresh episode at step 0 and generate step-0 arrivals."""
        c = self.config
        organic_rng, opp_rng, self._auction_rng, self._obs_rng = self._root_rng.spawn(4)
        tx_ids = itertools.count()
        self.mempool = Mempool(c, organic_rng, tx_ids)
        self.opportunities = OpportunityPool(c, opp_rng, itertools.count())
        self._tx_ids = tx_ids
        self.step_index = 0
        self.balances = [c.initial_balance] * c.n_agents
        self.congestion = 0.0
        self.last_clearing_price = 0.0
        self.last_block_captures: list[tuple[float, float]] = []
        self.last_block_bids: list[tuple[float, float]] = []
        self._seen_ids: set[int] = set()
        self._new_block()
        self._ready = True
        self._arrive()

    def _new_block(self) -> None:
        self.round = 0
        self._round_bids: list[Bid] = []
        self._obs_cache: dict[int, Observation] = {}
        self._newly_seen: list = []

    def _arrive(self) -> None:
        self.mempool.arrive(self.step_index, self.congestion)
        self.opportunities.arrive(self.step_index)

    def observe(self, agent: int) -> Observation:
        """Partial, noisy view of the current state for one agent."""
        if self.config.mechanism != "pga":
            return self._draw_observation(agent)
        if agent not in self._obs_cache:
            self._obs_cache[agent] = self._draw_observation(agent)
        base = self._obs_cache[agent]
        top_bids, leading = {}, set()
        standing = standing_bids(self._round_bids)
        for opp in base.visible_opportunities:
            if opp.id in standing:
                top, leaders = pga_leaders(standing[opp.id])
                top_bids[opp.id] = top
                if leaders == [agent]:  # sole leader; a tie is not a safe lead
                    leading.add(opp.id)
        return replace(
            base, round=self.round, n_rounds=self.config.n_rounds, top_bids=top_bids,
            leading=leading,
        )  # fmt: skip

    def _draw_observation(self, agent: int) -> Observation:
        c = self.config
        rng = self._obs_rng
        opps = sorted(self.opportunities.active.values(), key=lambda o: o.id)
        visible = [o for o in opps if rng.random() < c.p_see]
        for o in visible:
            if o.id not in self._seen_ids:
                self._seen_ids.add(o.id)
                self._newly_seen.append(o)

        fees = sorted((tx.max_fee_per_gas for tx in self.mempool.pending), reverse=True)
        top = np.zeros(c.top_k)
        k = min(c.top_k, len(fees))
        top[:k] = fees[:k]
        top[:k] = np.maximum(top[:k] * (1.0 + rng.normal(0.0, c.obs_noise_std, k)), 0.0)
        pending = self.mempool.pending_gas * (1.0 + rng.normal(0.0, c.obs_noise_std))

        return Observation(
            agent=agent,
            step=self.step_index,
            balance=self.balances[agent],
            visible_opportunities=visible,
            top_fees=sorted(top.tolist(), reverse=True),
            pending_gas=max(float(pending), 0.0),
            gas_limit=c.gas_limit,
            last_clearing_price=self.last_clearing_price,
            last_block_captures=list(self.last_block_captures),
            last_block_bids=list(self.last_block_bids),
            n_rounds=c.n_rounds,
        )

    def _check_bids(self, bids: list[Bid]) -> None:
        for bid in bids:
            if not 0 <= bid.agent < self.config.n_agents:
                raise ValueError(f"bid from unknown agent {bid.agent}")

    def bid_round(self, bids: list[Bid]) -> None:
        """Submit one non-final PGA round. Bids are binding and become public as top bids."""
        if not self._ready:
            raise RuntimeError("call reset() before bid_round()")
        if self.round >= self.config.n_rounds - 1:
            raise RuntimeError("no intermediate rounds left; call step() for the final round")
        self._check_bids(bids)
        active = self.opportunities.active
        self._round_bids += [
            replace(b, round=self.round) for b in bids if b.opportunity_id in active
        ]
        self.round += 1

    def step(self, bids: list[Bid]) -> StepResult:
        """Resolve the auction, build the block, pay rewards, then advance one step.

        Bids on unknown or expired opportunities are ignored. A winner pays only if its
        tx is actually included; if the block is full the opportunity stays live.
        Under PGA this is the final round and must follow `n_rounds - 1` `bid_round` calls.
        """
        if not self._ready:
            raise RuntimeError("call reset() before step()")
        c = self.config
        step = self.step_index
        if self.round != c.n_rounds - 1:
            raise RuntimeError(f"step() called in round {self.round} of {c.n_rounds}")
        self._check_bids(bids)

        active = self.opportunities.active
        final = [replace(b, round=self.round) for b in bids]
        valid = self._round_bids + [b for b in final if b.opportunity_id in active]
        results = self.auction.resolve(valid, self._auction_rng)

        agent_txs = [
            Transaction(
                id=next(self._tx_ids),
                owner=r.winner,
                gas_used=c.agent_tx_gas,
                max_fee_per_gas=r.payment / c.agent_tx_gas,
                priority_fee_per_gas=r.bid / c.agent_tx_gas,
                arrival_step=step,
                kind=TxKind.AGENT,
                opportunity_id=r.opportunity_id,
            )
            for r in results
        ]
        block: Block = build_block(self.mempool.pending + agent_txs, c.gas_limit, step)

        rewards = [0.0] * c.n_agents
        captured = []
        block_captures, block_bids = [], []
        for tx in block.txs:
            if tx.kind is TxKind.AGENT:
                opp = active[tx.opportunity_id]
                rewards[tx.owner] += opp.value - tx.total_fee
                captured.append(opp)
                block_captures.append((opp.value, tx.total_fee))
                block_bids.append((opp.value, tx.ordering_fee * tx.gas_used))
        for i, r in enumerate(rewards):
            self.balances[i] += r

        self.mempool.remove([tx for tx in block.txs if tx.kind is TxKind.ORGANIC])
        self.opportunities.remove_captured([o.id for o in captured])
        self.opportunities.expire(step)

        backlog = self.mempool.pending_gas / c.gas_limit
        a = c.congestion_smoothing
        self.congestion = a * backlog + (1.0 - a) * self.congestion
        self.last_clearing_price = block.clearing_price
        self.last_block_captures = block_captures
        self.last_block_bids = block_bids

        all_bids, newly_seen = self._round_bids + final, self._newly_seen
        self.step_index += 1
        self._new_block()
        self._arrive()
        return StepResult(block, results, captured, rewards, all_bids, newly_seen)
