import numpy as np
import pytest

pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from mevsim.core.config import SimConfig  # noqa: E402
from mevsim.envs import SingleAgentMEVEnv  # noqa: E402
from mevsim.experiments.run import run_episode  # noqa: E402

PARAMS = {"greedy": {"fraction": 0.75}}


def make_env(horizon=40):
    return SingleAgentMEVEnv(SimConfig(), ["random", "rule_based"], PARAMS, horizon=horizon)


def rollout(env, seed, fraction=0.75):
    obs, _ = env.reset(seed=seed)
    trace, done = [obs], False
    while not done:
        obs, reward, done, truncated, info = env.step(np.array([fraction], dtype=np.float32))
        assert not truncated
        trace.append((obs, reward))
    return trace, info


def test_check_env():
    check_env(make_env(), skip_render_check=True)


def test_reset_deterministic_under_seed():
    a, _ = rollout(make_env(), seed=4)
    b, _ = rollout(make_env(), seed=4)
    c, _ = rollout(make_env(), seed=5)
    assert np.array_equal(a[0], b[0])
    assert all(
        np.array_equal(x[0], y[0]) and x[1] == y[1] for x, y in zip(a[1:], b[1:], strict=True)
    )
    assert any(not np.array_equal(x[0], y[0]) for x, y in zip(a[1:], c[1:], strict=True))


def test_observations_in_space_and_episode_length():
    env = make_env(horizon=25)
    trace, info = rollout(env, seed=1)
    assert len(trace) == 26
    for obs in [trace[0]] + [o for o, _ in trace[1:]]:
        assert env.observation_space.contains(obs)
    assert info["episode_metrics"]["blocks"] == 25


def test_matches_runner_with_fixed_fraction_agent():
    """A constant-fraction learner must reproduce a greedy agent in the runner exactly."""
    _, info = rollout(make_env(horizon=60), seed=9, fraction=0.75)
    rows, ep = run_episode(SimConfig(), ["greedy", "random", "rule_based"], PARAMS, 60, seed=9)
    for env_row, run_row in zip(info["agent_metrics"], rows, strict=True):
        assert env_row["profit"] == pytest.approx(run_row["profit"])
        assert env_row["captures"] == run_row["captures"]
    assert info["episode_metrics"] == ep


PGA = SimConfig(mechanism="pga", pga_rounds=3)


def make_pga_env(horizon=20):
    return SingleAgentMEVEnv(PGA, ["random", "rule_based"], PARAMS, horizon=horizon)


def test_pga_check_env_and_episode_length():
    check_env(make_pga_env(), skip_render_check=True)
    env = make_pga_env(horizon=20)
    assert env.observation_space.shape == (6 + PGA.top_k + 3,)
    trace, info = rollout(env, seed=3)
    assert len(trace) - 1 == 20 * 3
    rewards = [r for _, r in trace[1:]]
    assert all(r == 0.0 for i, r in enumerate(rewards) if i % 3 != 2)
    assert info["episode_metrics"]["blocks"] == 20
    assert all(env.observation_space.contains(o) for o, _ in trace[1:])


def test_pga_env_matches_runner_with_fixed_fraction_agent():
    _, info = rollout(make_pga_env(horizon=40), seed=9, fraction=0.75)
    rows, ep = run_episode(PGA, ["greedy", "random", "rule_based"], PARAMS, 40, seed=9)
    assert [r["profit"] for r in info["agent_metrics"]] == [r["profit"] for r in rows]
    assert info["episode_metrics"] == ep


def test_sealed_obs_size_unchanged():
    assert make_env().observation_space.shape == (6 + SimConfig().top_k,)
