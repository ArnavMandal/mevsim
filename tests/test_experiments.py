import csv

import pytest

from mevsim.experiments.run import ExperimentConfig, main, run_episode
from mevsim.metrics import mean_ci

CONFIG = """
name: tiny
sim_overrides: {gas_limit: 20}
horizon: 15
seeds: 2
agents:
  random: {}
  greedy: {fraction: 0.8}
  rule_based: {}
scenarios:
  mixed: [random, greedy, rule_based]
  duo: [greedy, greedy]
"""


@pytest.fixture
def config_path(tmp_path):
    p = tmp_path / "tiny.yaml"
    p.write_text(CONFIG)
    return p


def test_runner_writes_outputs(config_path, tmp_path):
    out = main(["--config", str(config_path), "--out", str(tmp_path / "res")])
    for name in ["config.yaml", "agents.csv", "episodes.csv", "profit.png", "latency.png"]:
        assert (out / name).exists()
    with open(out / "agents.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * 3 + 2 * 2  # seeds x agents, per scenario
    with open(out / "episodes.csv") as f:
        assert len(list(csv.DictReader(f))) == 4


def test_episode_deterministic_and_n_agents_from_lineup(config_path):
    cfg = ExperimentConfig.from_yaml(config_path)
    a = run_episode(cfg.sim, ["greedy", "random"], cfg.agents, 30, seed=3)
    b = run_episode(cfg.sim, ["greedy", "random"], cfg.agents, 30, seed=3)
    assert a == b
    rows, ep = a
    assert [r["agent_type"] for r in rows] == ["greedy", "random"]
    assert ep["blocks"] == 30


def test_config_rejects_undefined_agent(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(CONFIG.replace("duo: [greedy, greedy]", "duo: [greedy, oracle]"))
    with pytest.raises(ValueError):
        ExperimentConfig.from_yaml(p)


def test_mean_ci():
    mean, ci = mean_ci([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert ci == pytest.approx(4.303 * 1.0 / 3**0.5)
    assert mean_ci([5.0]) == (5.0, 0.0)
