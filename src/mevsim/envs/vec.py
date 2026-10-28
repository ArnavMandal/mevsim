"""SB3 VecEnv adapter for parameter sharing: each learner of each env copy is one slot."""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from mevsim.envs.multi_agent import MultiAgentMEVEnv


class SharedPolicyVecEnv(VecEnv):
    """Presents `len(envs) * n_learners` slots to one shared SB3 policy.

    Slot `k * n + i` is learner i of env copy k. All learners finish together, so a copy
    is auto-reset when its episode ends (`terminal_observation` set per SB3 convention),
    and the episode metrics are attached to the infos of all of that copy's slots.
    """

    def __init__(self, envs: list[MultiAgentMEVEnv]):
        self.envs = envs
        self.n = envs[0].n_learners
        agent = envs[0].possible_agents[0]
        super().__init__(
            len(envs) * self.n, envs[0].observation_space(agent), envs[0].action_space(agent)
        )
        self._actions: np.ndarray | None = None

    def _stack(self, obs: dict[str, np.ndarray], env: MultiAgentMEVEnv) -> list[np.ndarray]:
        return [obs[a] for a in env.possible_agents]

    def reset(self) -> np.ndarray:
        out = []
        for k, env in enumerate(self.envs):
            obs, _ = env.reset(seed=self._seeds[k * self.n] if self._seeds else None)
            out += self._stack(obs, env)
        self._reset_seeds()
        return np.stack(out)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        obs_out, rew_out, done_out, info_out = [], [], [], []
        for k, env in enumerate(self.envs):
            acts = self._actions[k * self.n : (k + 1) * self.n]
            actions = {a: acts[i] for i, a in enumerate(env.possible_agents)}
            obs, rewards, terms, truncs, infos = env.step(actions)
            done = all(terms.values()) or all(truncs.values())
            stacked = self._stack(obs, env)
            slot_infos = [dict(infos[a]) for a in env.possible_agents]
            if done:
                for info, o in zip(slot_infos, stacked, strict=True):
                    info["terminal_observation"] = o
                    info["TimeLimit.truncated"] = False
                new_obs, _ = env.reset()
                stacked = self._stack(new_obs, env)
            obs_out += stacked
            rew_out += [rewards[a] for a in env.possible_agents]
            done_out += [done] * self.n
            info_out += slot_infos
        return (
            np.stack(obs_out),
            np.asarray(rew_out, dtype=np.float32),
            np.asarray(done_out),
            info_out,
        )

    def close(self) -> None:
        pass

    def _slots(self, indices) -> list[int]:
        return list(range(self.num_envs)) if indices is None else list(np.atleast_1d(indices))

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        return [getattr(self.envs[i // self.n], attr_name, None) for i in self._slots(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        for i in self._slots(indices):
            setattr(self.envs[i // self.n], attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list[Any]:
        return [
            getattr(self.envs[i // self.n], method_name)(*args, **kwargs)
            for i in self._slots(indices)
        ]

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False] * len(self._slots(indices))
