import numpy as np
import pytest
import yaml
from test_agents import obs_with

from mevsim.agents import RuleBasedBidAgent
from mevsim.core.config import SimConfig
from mevsim.core.sim import Simulator
from mevsim.core.types import Bid
from mevsim.experiments.mechanisms import (
    MechanismsConfig,
    add_welfare,
    counterfactual,
    main,
)
from mevsim.experiments.run import run_episode


def test_newly_seen_reported_once_and_covers_captures():
    sim = Simulator(SimConfig(p_see=0.5, opportunity_rate=2.0), np.random.default_rng(0))
    sim.reset()
    seen_ids, captured_ids = [], []
    for _ in range(150):
        bids = []
        for i in range(sim.config.n_agents):
            obs = sim.observe(i)
            bids += [Bid(i, o.id, 0.5 * o.value) for o in obs.visible_opportunities]
        r = sim.step(bids)
        seen_ids += [o.id for o in r.newly_seen]
        captured_ids += [o.id for o in r.captured]
    assert len(seen_ids) == len(set(seen_ids))
    assert set(captured_ids) <= set(seen_ids)


def test_nothing_seen_without_observing():
    sim = Simulator(SimConfig(opportunity_rate=3.0), np.random.default_rng(0))
    sim.reset()
    assert all(sim.step([]).newly_seen == [] for _ in range(20))


def test_efficiency_in_unit_interval_and_drops_under_congestion():
    params = {"greedy": {"fraction": 0.9}}
    _, calm = run_episode(SimConfig(), ["greedy"] * 3, params, 200, seed=0)
    _, busy = run_episode(SimConfig(organic_rate=14.0), ["greedy"] * 3, params, 200, seed=0)
    assert calm["efficiency"] == pytest.approx(1.0)
    assert 0.0 < busy["efficiency"] < 0.9


def test_last_block_bids_publish_winning_bid_under_second_price():
    cfg = SimConfig(n_agents=2, p_see=1.0, organic_rate=0.0, mechanism="sealed_second_price")
    sim = Simulator(cfg, np.random.default_rng(0))
    sim.reset()
    opp = min(sim.opportunities.active.values(), key=lambda o: o.id)
    sim.step([Bid(0, opp.id, 3.0), Bid(1, opp.id, 1.0)])
    obs = sim.observe(0)
    assert obs.last_block_captures == [(opp.value, pytest.approx(1.0))]
    assert obs.last_block_bids == [(opp.value, pytest.approx(3.0))]


def test_rule_based_bid_agent_tracks_bids_not_payments():
    agent = RuleBasedBidAgent(0, np.random.default_rng(0), smoothing=0.5, initial_ratio=0.5)
    obs = obs_with([10.0], captures=[(10.0, 2.0)])
    obs.last_block_bids = [(10.0, 9.0)]
    agent.act(obs)
    assert agent.ratio == pytest.approx(0.7)


def test_counterfactual_has_no_mev_and_matches_zero_bid_run():
    cf = counterfactual(SimConfig(), 3, 50, seed=4)
    assert cf["mev_captured_per_block"] == 0.0 and cf["mev_seen_per_block"] == 0.0
    # A lineup that bids nothing leaves the organic side exactly as in the counterfactual.
    _, ep = run_episode(SimConfig(), ["greedy"] * 3, {"greedy": {"fraction": 0.0}}, 50, seed=4)
    assert ep["mean_congestion"] == cf["mean_congestion"]
    assert ep["mean_latency"] == cf["mean_latency"]


def test_add_welfare_arithmetic():
    ep = {
        "mev_captured_per_block": 4.0,
        "mev_to_builder_per_block": 3.0,
        "mev_seen_per_block": 5.0,
        "mean_congestion": 1.5,
    }
    row = add_welfare(ep, {"mean_congestion": 1.0}, gas_limit=30, delay_cost=0.1)
    assert row["builder_share"] == pytest.approx(0.75)
    assert row["extra_wait_per_block"] == pytest.approx(15.0)
    assert row["delay_loss_per_block"] == pytest.approx(1.5)
    assert row["welfare_loss_per_block"] == pytest.approx(4.5)
    assert row["uncaptured_mev_per_block"] == pytest.approx(1.0)


def write_config(tmp_path, **overrides):
    d = {
        "name": "mech_test",
        "horizon": 20,
        "seeds": 2,
        "mechanisms": ["sealed_first_price", "sealed_second_price", "pga"],
        "pga_rounds_sweep": [2, 5],
        "agents": {"random": {}, "greedy": {"fraction": 0.9}, "rule_based_bid": {}},
        "scenarios": {"mixed": ["random", "greedy", "rule_based_bid"]},
        **overrides,
    }
    path = tmp_path / "mech.yaml"
    path.write_text(yaml.safe_dump(d))
    return path


def test_runs_dedupe_sweep_against_main_pga(tmp_path):
    cfg = MechanismsConfig.from_yaml(write_config(tmp_path))
    labels = [(label, sim.mechanism, sim.n_rounds) for label, sim in cfg.runs()]
    assert labels == [
        ("sealed_first_price", "sealed_first_price", 1),
        ("sealed_second_price", "sealed_second_price", 1),
        ("pga", "pga", 5),
        ("pga_r2", "pga", 2),
    ]


def test_config_rejects_unknown_keys_and_mechanisms(tmp_path):
    with pytest.raises(ValueError):
        MechanismsConfig.from_yaml(write_config(tmp_path, typo=1))
    with pytest.raises(ValueError):
        MechanismsConfig.from_yaml(write_config(tmp_path, mechanisms=["dutch"]))


def test_main_smoke(tmp_path):
    out = main(["--config", str(write_config(tmp_path)), "--out", str(tmp_path / "res")])
    for f in ("config.yaml", "agents.csv", "episodes.csv", "summary.md", "mechanisms.png",
              "pga_rounds.png"):  # fmt: skip
        assert (out / f).exists(), f
    summary = (out / "summary.md").read_text()
    assert "| pga_r2 | mixed |" in summary and "| sealed_second_price | mixed |" in summary


def test_main_with_ppo_smoke(tmp_path):
    pytest.importorskip("stable_baselines3")
    ppo = {
        "enabled": True,
        "n_learners": 2,
        "train": {
            "total_timesteps": 256,
            "n_envs": 1,
            "ppo": {"n_steps": 64, "batch_size": 64, "n_epochs": 1},
        },
    }
    path = write_config(tmp_path, mechanisms=["sealed_first_price", "pga"], pga_rounds_sweep=[],
                        horizon=10, ppo=ppo)  # fmt: skip
    out = main(["--config", str(path), "--out", str(tmp_path / "res"), "--no-plots"])
    summary = (out / "summary.md").read_text()
    assert "| sealed_first_price | ppo |" in summary and "| pga | ppo |" in summary
    assert (out / "ppo_pga" / "model.zip").exists()
