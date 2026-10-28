"""Seeded episode setup shared by the experiment runner and the RL envs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from mevsim.agents import Agent, make_agent
from mevsim.core.config import SimConfig
from mevsim.core.sim import Simulator


def seeded_episode(
    sim_config: SimConfig,
    lineup: list[str | None],
    agent_params: dict[str, dict[str, Any]],
    seed: int,
) -> tuple[Simulator, list[Agent | None]]:
    """Build a reset simulator and its agents from one seed.

    `n_agents` comes from the lineup. A `None` slot is left for an external policy (e.g. a
    learner) and gets no agent. The sim and agents use separate child streams, and slot i
    always gets agent stream i, so the same seed gives the same sim draws whatever fills
    each slot.
    """
    config = replace(sim_config, n_agents=len(lineup))
    sim_rng, agents_rng = np.random.default_rng(seed).spawn(2)
    agent_rngs = agents_rng.spawn(len(lineup))
    agents = [
        None if kind is None else make_agent(kind, i, rng, **agent_params.get(kind, {}))
        for i, (kind, rng) in enumerate(zip(lineup, agent_rngs, strict=True))
    ]
    sim = Simulator(config, sim_rng)
    sim.reset()
    return sim, agents
