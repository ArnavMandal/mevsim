import numpy as np
import pytest

from mevsim.agents import GreedyAgent, RuleBasedAgent, make_agent
from mevsim.core.types import Observation, Opportunity


def obs_with(values, captures=()):
    opps = [Opportunity(i, v, 0, 2) for i, v in enumerate(values)]
    return Observation(
        agent=0,
        step=0,
        balance=100.0,
        visible_opportunities=opps,
        top_fees=[0.0] * 5,
        pending_gas=0.0,
        gas_limit=30,
        last_clearing_price=0.0,
        last_block_captures=list(captures),
    )


@pytest.mark.parametrize("kind", ["random", "greedy", "rule_based"])
def test_bids_within_value(kind):
    agent = make_agent(kind, 0, np.random.default_rng(0))
    for _ in range(50):
        obs = obs_with([1.0, 5.0, 20.0], captures=[(10.0, 9.9)])
        bids = agent.act(obs)
        assert [b.opportunity_id for b in bids] == [0, 1, 2]
        for b, o in zip(bids, obs.visible_opportunities, strict=True):
            assert b.agent == 0
            assert 0.0 <= b.amount <= o.value


@pytest.mark.parametrize("kind", ["random", "greedy", "rule_based"])
def test_agents_deterministic_under_seed(kind):
    def bids(seed):
        agent = make_agent(kind, 1, np.random.default_rng(seed))
        return [agent.act(obs_with([3.0, 7.0], captures=[(4.0, 2.0)])) for _ in range(20)]

    assert bids(5) == bids(5)


def test_no_bids_without_visible_opportunities():
    for kind in ["random", "greedy", "rule_based"]:
        assert make_agent(kind, 0, np.random.default_rng(0)).act(obs_with([])) == []


def test_greedy_bids_fixed_fraction():
    bids = GreedyAgent(0, np.random.default_rng(0), fraction=0.7).act(obs_with([10.0]))
    assert bids[0].amount == pytest.approx(7.0)


def test_rule_based_tracks_winning_ratio_and_caps_at_margin():
    agent = RuleBasedAgent(0, np.random.default_rng(0), margin=0.2, step_up=0.05, smoothing=0.5)
    assert agent.act(obs_with([10.0]))[0].amount == pytest.approx(5.5)  # initial 0.5 + 0.05
    agent.act(obs_with([], captures=[(10.0, 3.0)]))  # ratio -> 0.5*0.3 + 0.5*0.5 = 0.4
    assert agent.ratio == pytest.approx(0.4)
    assert agent.act(obs_with([10.0]))[0].amount == pytest.approx(4.5)
    for _ in range(20):
        agent.act(obs_with([], captures=[(10.0, 10.0)]))
    assert agent.act(obs_with([10.0]))[0].amount == pytest.approx(8.0)  # capped at 1 - margin
    agent.reset()
    assert agent.ratio == pytest.approx(0.5)


def test_make_agent_rejects_unknown():
    with pytest.raises(ValueError):
        make_agent("oracle", 0, np.random.default_rng(0))
