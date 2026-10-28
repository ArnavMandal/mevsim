"""Multi-agent PettingZoo parallel env (SPEC section 4): N learners, optional fixed opponents."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv

from mevsim.core.config import SimConfig
from mevsim.core.types import Bid
from mevsim.envs.single_agent import _TRAIN_SEED_LOW, OBS_CLIP, featurize, obs_size
from mevsim.episode import seeded_episode
from mevsim.metrics import EpisodeRecorder


class MultiAgentMEVEnv(ParallelEnv):
    """Learners `agent_0..agent_{n-1}` fill sim slots 0..n-1; baseline opponents follow.

    Every learner has the single-agent env's observation, action (one bid fraction in
    [0, 1]) and reward (its own profit per block). Social welfare is reported, not
    rewarded: the final step's infos carry the episode metrics, including the split of
    captured MEV between agents and the builder. Seeding matches `SingleAgentMEVEnv`.
    """

    metadata = {"name": "mevsim_multi_v0", "render_modes": []}

    def __init__(
        self,
        sim_config: SimConfig,
        n_learners: int,
        opponents: list[str] | None = None,
        agent_params: dict[str, dict[str, Any]] | None = None,
        horizon: int = 200,
    ):
        if n_learners < 1:
            raise ValueError("need at least one learner")
        self.sim_config = sim_config
        self.n_learners = n_learners
        self.opponents = list(opponents or [])
        self.agent_params = agent_params or {}
        self.horizon = horizon
        self.possible_agents = [f"agent_{i}" for i in range(n_learners)]
        self.agents: list[str] = []
        self._obs_space = gym.spaces.Box(
            0.0, OBS_CLIP, shape=(obs_size(sim_config),), dtype=np.float32
        )
        self._act_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self._seed_rng = np.random.default_rng()
        self.sim = None

    def observation_space(self, agent: str) -> gym.spaces.Box:
        return self._obs_space

    def action_space(self, agent: str) -> gym.spaces.Box:
        return self._act_space

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        if seed is not None:
            self._seed_rng = np.random.default_rng(seed)
        else:
            seed = int(self._seed_rng.integers(_TRAIN_SEED_LOW, 2**31))
        self.episode_seed = seed
        lineup = [None] * self.n_learners + self.opponents
        self.sim, agents = seeded_episode(self.sim_config, lineup, self.agent_params, seed)
        self.opponent_agents = agents[self.n_learners :]
        self.recorder = EpisodeRecorder(self.sim.config.n_agents, self.sim.config.gas_limit)
        self.agents = list(self.possible_agents)
        return self._observe(), {a: {"seed": seed} for a in self.agents}

    def _observe(self) -> dict[str, np.ndarray]:
        self._obs = [self.sim.observe(i) for i in range(self.n_learners)]
        return {
            a: featurize(o, self.sim.config, self.horizon)
            for a, o in zip(self.possible_agents, self._obs, strict=True)
        }

    def step(self, actions: dict[str, np.ndarray]) -> tuple[dict, dict, dict, dict, dict]:
        if not self.agents:
            raise RuntimeError("call reset() before step()")
        bids = []
        for i, (agent, obs) in enumerate(zip(self.possible_agents, self._obs, strict=True)):
            a = np.asarray(actions[agent], dtype=float).reshape(-1)[0]
            fraction = float(np.clip(a, 0.0, 1.0))
            bids += [Bid(i, o.id, fraction * o.value) for o in obs.visible_opportunities]
        for agent in self.opponent_agents:
            bids += agent.act(self.sim.observe(agent.index))
        active = dict(self.sim.opportunities.active)
        result = self.sim.step(bids)
        self.recorder.record(bids, active, result, self.sim.mempool.pending_gas)

        done = self.sim.step_index >= self.horizon
        obs = self._observe()
        rewards = {a: float(result.rewards[i]) for i, a in enumerate(self.possible_agents)}
        terminations = {a: done for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos: dict[str, dict] = {a: {} for a in self.agents}
        if done:
            metrics = {
                "agent_metrics": self.recorder.agent_summaries(),
                "episode_metrics": self.recorder.episode_summary(),
            }
            infos = {a: dict(metrics) for a in self.agents}
            self.agents = []
        return obs, rewards, terminations, truncations, infos
