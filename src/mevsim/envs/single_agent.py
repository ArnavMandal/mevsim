"""Single-agent gymnasium env (SPEC section 4): one learner vs fixed baseline opponents."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from mevsim.core.config import SimConfig
from mevsim.core.types import Bid, Observation
from mevsim.episode import seeded_episode
from mevsim.metrics import EpisodeRecorder

OBS_CLIP = 10.0
_TRAIN_SEED_LOW = 2**20  # training episodes draw seeds at or above this; keep eval seeds below


def obs_size(config: SimConfig) -> int:
    return 6 + config.top_k + (3 if config.mechanism == "pga" else 0)


def featurize(obs: Observation, config: SimConfig, horizon: int) -> np.ndarray:
    """Normalized observation vector, every entry clipped to [0, OBS_CLIP].

    Layout: balance gain, best visible value, top-k fees, pending gas fraction, last
    clearing price, last block's mean winning payment/value, step fraction. Under PGA three
    more entries follow, about the best visible opportunity: current top bid / value,
    whether this agent leads it (0/1), and the round fraction.
    """
    fee_scale = config.organic_fee_scale
    best_opp = max(obs.visible_opportunities, key=lambda o: o.value, default=None)
    best = best_opp.value if best_opp is not None else 0.0
    ratios = [p / v for v, p in obs.last_block_captures if v > 0]
    x = np.array(
        [
            obs.balance / config.initial_balance - 1.0,
            best / config.opportunity_value_scale,
            *(f / fee_scale for f in obs.top_fees),
            obs.pending_gas / obs.gas_limit,
            obs.last_clearing_price / fee_scale,
            float(np.mean(ratios)) if ratios else 0.0,
            obs.step / horizon,
        ],
        dtype=np.float32,
    )
    if config.mechanism == "pga":
        top_ratio = leading = 0.0
        if best_opp is not None and best_opp.value > 0:
            top_ratio = obs.top_bids.get(best_opp.id, 0.0) / best_opp.value
            leading = float(best_opp.id in obs.leading)
        round_frac = obs.round / max(obs.n_rounds - 1, 1)
        x = np.append(x, np.array([top_ratio, leading, round_frac], dtype=np.float32))
    return np.clip(x, 0.0, OBS_CLIP)


class SingleAgentMEVEnv(gym.Env):
    """Agent 0 learns; opponents in the other slots are fixed baseline agents.

    Action: `bid_fraction` in [0, 1], applied to every visible opportunity (0 = skip).
    Reward: the learner's profit this block (captured value minus payment).
    Episodes last `horizon` blocks. Under PGA one env step is one bidding round, so an
    episode is `horizon * n_rounds` steps and reward arrives on each block's final round.
    `reset(seed=s)` builds the episode exactly as the experiment runner does for seed `s`;
    later seedless resets draw fresh episode seeds from the env's generator.
    The final step's `info` carries the episode metrics.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sim_config: SimConfig,
        opponents: list[str],
        agent_params: dict[str, dict[str, Any]] | None = None,
        horizon: int = 200,
    ):
        super().__init__()
        self.sim_config = sim_config
        self.opponents = list(opponents)
        self.agent_params = agent_params or {}
        self.horizon = horizon
        self.observation_space = gym.spaces.Box(
            0.0, OBS_CLIP, shape=(obs_size(sim_config),), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.sim = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(_TRAIN_SEED_LOW, 2**31))
        self.episode_seed = seed
        self.sim, agents = seeded_episode(
            self.sim_config, [None, *self.opponents], self.agent_params, seed
        )
        self.opponent_agents = agents[1:]
        self.recorder = EpisodeRecorder(self.sim.config.n_agents, self.sim.config.gas_limit)
        self._obs = self.sim.observe(0)
        return featurize(self._obs, self.sim.config, self.horizon), {"seed": seed}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.sim is None:
            raise RuntimeError("call reset() before step()")
        fraction = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], 0.0, 1.0))
        bids = [Bid(0, o.id, fraction * o.value) for o in self._obs.visible_opportunities]
        for agent in self.opponent_agents:
            bids += agent.act(self.sim.observe(agent.index))
        if self.sim.round < self.sim.config.n_rounds - 1:  # intermediate PGA round
            self.sim.bid_round(bids)
            self._obs = self.sim.observe(0)
            return featurize(self._obs, self.sim.config, self.horizon), 0.0, False, False, {}
        active = dict(self.sim.opportunities.active)
        result = self.sim.step(bids)
        self.recorder.record(result.bids, active, result, self.sim.mempool.pending_gas)

        terminated = self.sim.step_index >= self.horizon
        info: dict[str, Any] = {}
        if terminated:
            info["agent_metrics"] = self.recorder.agent_summaries()
            info["episode_metrics"] = self.recorder.episode_summary()
        self._obs = self.sim.observe(0)
        obs = featurize(self._obs, self.sim.config, self.horizon)
        return obs, float(result.rewards[0]), terminated, False, info
