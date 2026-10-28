import csv

import pytest

pytest.importorskip("stable_baselines3")

from mevsim.training.train_ppo import PPOConfig, main  # noqa: E402

CONFIG = """
name: tiny_ppo
horizon: 20
opponents: [random]
baselines: [random, greedy]
agents:
  random: {}
  greedy: {fraction: 0.9}
train:
  total_timesteps: 128
  n_envs: 2
  ppo: {n_steps: 32, batch_size: 32, n_epochs: 1}
eval:
  seeds: 3
"""


def test_train_ppo_smoke(tmp_path):
    p = tmp_path / "tiny.yaml"
    p.write_text(CONFIG)
    out = main(["--config", str(p), "--out", str(tmp_path / "res"), "--no-plots"])
    for name in ["config.yaml", "model.zip", "progress.csv", "eval_agents.csv", "summary.txt"]:
        assert (out / name).exists()
    with open(out / "eval_agents.csv") as f:
        rows = list(csv.DictReader(f))
    assert {r["scenario"] for r in rows} == {"ppo", "random", "greedy"}
    assert len(rows) == 3 * 3 * 2  # policies x seeds x agents


def test_config_rejects_unknown_train_key(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(CONFIG.replace("n_envs: 2", "n_env: 2"))
    with pytest.raises(ValueError):
        PPOConfig.from_yaml(p)
