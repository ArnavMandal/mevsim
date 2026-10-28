import csv

import numpy as np
import pytest

pytest.importorskip("pettingzoo")
pytest.importorskip("stable_baselines3")

from pettingzoo.test import parallel_api_test  # noqa: E402

from mevsim.core.config import SimConfig  # noqa: E402
from mevsim.envs import MultiAgentMEVEnv  # noqa: E402
from mevsim.envs.vec import SharedPolicyVecEnv  # noqa: E402
from mevsim.experiments.run import run_episode  # noqa: E402
from mevsim.training.train_ppo_multi import MultiPPOConfig, main  # noqa: E402

PARAMS = {"greedy": {"fraction": 0.75}}


def rollout(env, seed, fraction=0.75):
    obs, _ = env.reset(seed=seed)
    trace = []
    while env.agents:
        actions = {a: np.array([fraction], dtype=np.float32) for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        trace.append((obs, rewards))
    return trace, infos


@pytest.mark.parametrize("n", [1, 2, 5])
def test_parallel_api(n):
    parallel_api_test(MultiAgentMEVEnv(SimConfig(), n, ["random"], horizon=30), num_cycles=100)


def test_deterministic_under_seed():
    def run(seed):
        trace, _ = rollout(MultiAgentMEVEnv(SimConfig(), 3, horizon=40), seed)
        return [(tuple(o[a].tolist() for a in sorted(o)), r) for o, r in trace]

    assert run(2) == run(2)
    assert run(2) != run(3)


def test_matches_runner_with_fixed_fraction_learners():
    env = MultiAgentMEVEnv(SimConfig(), 2, ["random", "rule_based"], PARAMS, horizon=60)
    _, infos = rollout(env, seed=11)
    info = infos["agent_0"]
    rows, ep = run_episode(
        SimConfig(), ["greedy", "greedy", "random", "rule_based"], PARAMS, 60, seed=11
    )
    assert info["episode_metrics"] == ep
    assert [r["profit"] for r in info["agent_metrics"]] == [r["profit"] for r in rows]


def test_welfare_split_conserves_value():
    _, infos = rollout(MultiAgentMEVEnv(SimConfig(), 4, horizon=100), seed=0)
    ep = infos["agent_0"]["episode_metrics"]
    assert ep["mev_captured_per_block"] > 0
    assert ep["mev_captured_per_block"] == pytest.approx(
        ep["agent_profit_per_block"] + ep["mev_to_builder_per_block"]
    )


def test_vec_env_slots_and_auto_reset():
    venv = SharedPolicyVecEnv([MultiAgentMEVEnv(SimConfig(), 3, horizon=5) for _ in range(2)])
    assert venv.num_envs == 6
    venv.seed(0)
    obs = venv.reset()
    assert obs.shape == (6, venv.observation_space.shape[0])
    for _ in range(5):
        obs, rewards, dones, infos = venv.step(np.full((6, 1), 0.5, dtype=np.float32))
    assert dones.all()
    assert all("terminal_observation" in i and "agent_metrics" in i for i in infos)
    assert obs[:, -1].max() == 0.0  # step fraction back to 0 after auto-reset


CONFIG = """
name: tiny_multi
horizon: 10
n_learners: [2, 3, 4, 5]
baselines: [greedy]
agents:
  greedy: {fraction: 0.9}
train:
  total_timesteps: 200
  n_envs: 1
  ppo: {n_steps: 20, batch_size: 20, n_epochs: 1}
eval:
  seeds: 2
"""


def test_train_multi_smoke_n2_to_n5(tmp_path):
    p = tmp_path / "tiny.yaml"
    p.write_text(CONFIG)
    out = main(["--config", str(p), "--out", str(tmp_path / "res")])
    for n in [2, 3, 4, 5]:
        assert (out / f"n{n}" / "model.zip").exists()
        with open(out / f"n{n}" / "shading.csv") as f:
            rows = list(csv.DictReader(f))
        assert rows and {int(r["agent"]) for r in rows} == set(range(n))
    for name in ["summary.txt", "profit_vs_n.png", "shading_training.png", "eval_agents.csv"]:
        assert (out / name).exists()


def test_config_rejects_unknown_key(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(CONFIG.replace("n_envs: 1", "n_env: 1"))
    with pytest.raises(ValueError):
        MultiPPOConfig.from_yaml(p)
