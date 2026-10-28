"""Per-episode metrics from SPEC section 6 (profit, win rate, latency, congestion, shading).

Auction efficiency and welfare loss need a counterfactual run and arrive with M5.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mevsim.core.types import Bid, Opportunity, StepResult, TxKind

# Two-sided 95% Student-t critical values by degrees of freedom; 1.96 beyond the table.
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]  # fmt: skip


def mean_ci(values: list[float] | np.ndarray) -> tuple[float, float]:
    """Mean and 95% confidence half-width (t-distribution) across seeds."""
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else float("nan"), 0.0
    df = len(x) - 1
    t = _T95[df - 1] if df <= len(_T95) else 1.96
    return float(x.mean()), float(t * x.std(ddof=1) / np.sqrt(len(x)))


class EpisodeRecorder:
    """Accumulates one episode's steps and summarizes them per agent and per episode."""

    def __init__(self, n_agents: int, gas_limit: int):
        self.n_agents = n_agents
        self.gas_limit = gas_limit
        self.steps = 0
        self.profit = [0.0] * n_agents
        self.captures = [0] * n_agents
        self.bid_on: list[set[int]] = [set() for _ in range(n_agents)]
        self.bid_ratios: list[list[float]] = [[] for _ in range(n_agents)]
        self.latencies: list[int] = []
        self.backlog: list[float] = []
        self.gas_used: list[int] = []

    def record(
        self,
        bids: list[Bid],
        opportunities: dict[int, Opportunity],
        result: StepResult,
        pending_gas: float,
    ) -> None:
        """Log one step. `opportunities` are those active when bids were placed."""
        self.steps += 1
        for b in bids:
            opp = opportunities.get(b.opportunity_id)
            if opp is not None and opp.value > 0:
                self.bid_on[b.agent].add(opp.id)
                self.bid_ratios[b.agent].append(b.amount / opp.value)
        for i, r in enumerate(result.rewards):
            self.profit[i] += r
        for tx in result.block.txs:
            if tx.kind is TxKind.AGENT:
                self.captures[tx.owner] += 1
            else:
                self.latencies.append(result.block.step - tx.arrival_step)
        self.backlog.append(pending_gas / self.gas_limit)
        self.gas_used.append(result.block.gas_used)

    def agent_summaries(self) -> list[dict[str, Any]]:
        rows = []
        for i in range(self.n_agents):
            n_bid = len(self.bid_on[i])
            rows.append(
                {
                    "agent": i,
                    "profit": self.profit[i],
                    "profit_per_block": self.profit[i] / max(self.steps, 1),
                    "captures": self.captures[i],
                    "opportunities_bid": n_bid,
                    "win_rate": self.captures[i] / n_bid if n_bid else float("nan"),
                    "bid_shading": (
                        float(np.mean(self.bid_ratios[i])) if self.bid_ratios[i] else float("nan")
                    ),
                }
            )
        return rows

    def episode_summary(self) -> dict[str, Any]:
        lat = np.asarray(self.latencies, dtype=float)
        return {
            "blocks": self.steps,
            "organic_included": len(lat),
            "mean_latency": float(lat.mean()) if len(lat) else float("nan"),
            "p90_latency": float(np.percentile(lat, 90)) if len(lat) else float("nan"),
            "mean_congestion": float(np.mean(self.backlog)) if self.backlog else float("nan"),
            "mean_block_fullness": (
                float(np.mean(self.gas_used)) / self.gas_limit if self.gas_used else float("nan")
            ),
        }
