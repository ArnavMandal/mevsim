import numpy as np
import pytest

from mevsim.core.config import SimConfig
from mevsim.core.sim import Simulator
from mevsim.core.types import Bid, TxKind


def greedy_bids(sim, fraction=0.9):
    """Every agent bids `fraction` of each opportunity it sees."""
    bids = []
    for i in range(sim.config.n_agents):
        obs = sim.observe(i)
        bids += [Bid(i, o.id, fraction * o.value) for o in obs.visible_opportunities]
    return bids


def run(seed, steps=100, config=None, fraction=0.9):
    sim = Simulator(config or SimConfig(), np.random.default_rng(seed))
    sim.reset()
    results = []
    for _ in range(steps):
        for _ in range(sim.config.n_rounds - 1):
            sim.bid_round(greedy_bids(sim, fraction))
        results.append(sim.step(greedy_bids(sim, fraction)))
    return sim, results


def fingerprint(results):
    return [
        ([(t.id, t.owner, t.gas_used, t.max_fee_per_gas) for t in r.block.txs], r.rewards)
        for r in results
    ]


def test_deterministic_under_seed():
    _, a = run(42)
    _, b = run(42)
    _, c = run(43)
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(c)


def test_organic_stream_independent_of_agent_behavior():
    """Agents bidding differently must not change the organic random draws.

    Fees still respond to congestion (which agents affect), so turn that off here.
    """

    def all_organic(sim, results):
        included = [t for r in results for t in r.block.txs if t.kind is TxKind.ORGANIC]
        return sorted(
            (t.arrival_step, t.gas_used, t.max_fee_per_gas) for t in included + sim.mempool.pending
        )

    cfg = SimConfig(congestion_sensitivity=0.0, organic_rate=12.0)
    sim_a, res_a = run(7, steps=50, config=cfg, fraction=0.9)
    sim_b, res_b = run(7, steps=50, config=cfg, fraction=0.0)
    assert fingerprint(res_a) != fingerprint(res_b)
    assert all_organic(sim_a, res_a) == all_organic(sim_b, res_b)


@pytest.mark.parametrize("mechanism", ["sealed_first_price", "sealed_second_price", "pga"])
def test_capacity_and_value_conservation(mechanism):
    sim, results = run(1, steps=300, config=SimConfig(mechanism=mechanism))
    total_captured = total_profit = total_mev_payment = 0.0
    seen_opps: set[int] = set()
    for r in results:
        assert r.block.gas_used <= sim.config.gas_limit
        agent_txs = [t for t in r.block.txs if t.kind is TxKind.AGENT]
        assert len(agent_txs) == len(r.captured)
        for o in r.captured:
            assert o.id not in seen_opps, "opportunity captured twice"
            seen_opps.add(o.id)
        total_captured += sum(o.value for o in r.captured)
        total_profit += sum(r.rewards)
        total_mev_payment += sum(t.total_fee for t in agent_txs)
    assert total_captured > 0
    assert total_captured == pytest.approx(total_profit + total_mev_payment)
    balances_gain = sum(sim.balances) - sim.config.n_agents * sim.config.initial_balance
    assert balances_gain == pytest.approx(total_profit)


def test_first_price_reward_is_value_minus_bid():
    cfg = SimConfig(n_agents=2, p_see=1.0, organic_rate=0.0, opportunity_rate=3.0)
    sim = Simulator(cfg, np.random.default_rng(0))
    sim.reset()
    opp = min(sim.opportunities.active.values(), key=lambda o: o.id)
    r = sim.step([Bid(0, opp.id, 1.0), Bid(1, opp.id, 2.5)])
    assert r.rewards[0] == 0.0
    assert r.rewards[1] == pytest.approx(opp.value - 2.5)
    assert opp.id not in sim.opportunities.active


def test_second_price_reward_is_value_minus_second_bid():
    cfg = SimConfig(
        n_agents=3,
        p_see=1.0,
        organic_rate=0.0,
        opportunity_rate=3.0,
        mechanism="sealed_second_price",
    )
    sim = Simulator(cfg, np.random.default_rng(0))
    sim.reset()
    opp = min(sim.opportunities.active.values(), key=lambda o: o.id)
    r = sim.step([Bid(0, opp.id, 1.0), Bid(1, opp.id, 2.5), Bid(2, opp.id, 0.5)])
    assert r.rewards == [0.0, pytest.approx(opp.value - 1.0), 0.0]
    (tx,) = [t for t in r.block.txs if t.kind is TxKind.AGENT]
    assert tx.total_fee == pytest.approx(1.0)


def test_second_price_sole_winner_ordered_by_bid_under_congestion():
    # Paying 0 must not push the winner out of a full block: ordering uses the bid.
    cfg = SimConfig(
        n_agents=1,
        p_see=1.0,
        gas_limit=4,
        agent_tx_gas=2,
        organic_rate=50.0,
        organic_gas_min=2,
        organic_gas_max=2,
        organic_fee_scale=100.0,
        opportunity_rate=3.0,
        mechanism="sealed_second_price",
    )
    sim = Simulator(cfg, np.random.default_rng(0))
    sim.reset()
    assert sim.mempool.pending_gas > cfg.gas_limit
    opp = min(sim.opportunities.active.values(), key=lambda o: o.id)
    r = sim.step([Bid(0, opp.id, 1e6)])
    assert r.captured == [opp]
    assert r.rewards == [pytest.approx(opp.value)]


def test_winner_crowded_out_pays_nothing_and_opportunity_survives():
    cfg = SimConfig(
        n_agents=1,
        gas_limit=4,
        agent_tx_gas=2,
        organic_rate=50.0,
        organic_gas_min=2,
        organic_gas_max=2,
        organic_fee_scale=100.0,
        opportunity_rate=3.0,
        opportunity_ttl=5,
    )
    sim = Simulator(cfg, np.random.default_rng(0))
    sim.reset()
    opp = min(sim.opportunities.active.values(), key=lambda o: o.id)
    r = sim.step([Bid(0, opp.id, 0.01)])
    assert r.auction_results[0].winner == 0
    assert r.captured == [] and r.rewards == [0.0]
    assert opp.id in sim.opportunities.active


def test_expired_and_unknown_bids_ignored():
    cfg = SimConfig(n_agents=1, opportunity_rate=3.0, opportunity_ttl=0, organic_rate=0.0)
    sim = Simulator(cfg, np.random.default_rng(0))
    sim.reset()
    stale = min(sim.opportunities.active.values(), key=lambda o: o.id)
    sim.step([])
    assert stale.id not in sim.opportunities.active
    r = sim.step([Bid(0, stale.id, 1.0), Bid(0, 10_000, 1.0)])
    assert r.auction_results == [] and r.rewards == [0.0]


def test_bad_agent_and_step_before_reset():
    sim = Simulator(SimConfig(n_agents=2), np.random.default_rng(0))
    with pytest.raises(RuntimeError):
        sim.step([])
    sim.reset()
    with pytest.raises(ValueError):
        sim.step([Bid(5, 0, 1.0)])


def test_observation_is_partial():
    cfg = SimConfig(p_see=0.0)
    sim = Simulator(cfg, np.random.default_rng(0))
    sim.reset()
    obs = sim.observe(0)
    assert obs.visible_opportunities == []
    assert len(obs.top_fees) == cfg.top_k
    assert obs.top_fees == sorted(obs.top_fees, reverse=True)


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("gas_limt: 10\n")
    with pytest.raises(ValueError):
        SimConfig.from_yaml(p)


def test_default_yaml_matches_dataclass_defaults():
    from pathlib import Path

    path = Path(__file__).parents[1] / "configs" / "default.yaml"
    loaded = SimConfig.from_yaml(path)
    assert loaded == SimConfig()


def test_observation_reports_last_block_captures():
    sim = Simulator(SimConfig(), np.random.default_rng(3))
    sim.reset()
    assert sim.observe(0).last_block_captures == []
    seen = 0
    for _ in range(100):
        result = sim.step(greedy_bids(sim))
        expected = sorted(
            (tx.opportunity_id, tx.total_fee) for tx in result.block.txs if tx.kind is TxKind.AGENT
        )
        values = {o.id: o.value for o in result.captured}
        caps = sorted(sim.observe(0).last_block_captures)
        assert caps == sorted((values[oid], fee) for oid, fee in expected)
        seen += len(caps)
    assert seen > 0


def pga_config(**kw):
    base = dict(n_agents=3, p_see=1.0, organic_rate=0.0, opportunity_rate=3.0, mechanism="pga")
    return SimConfig(**{**base, **kw})


def test_pga_round_protocol_enforced():
    sim = Simulator(pga_config(pga_rounds=3), np.random.default_rng(0))
    sim.reset()
    with pytest.raises(RuntimeError):
        sim.step([])  # still round 0 of 3
    sim.bid_round([])
    sim.bid_round([])
    with pytest.raises(RuntimeError):
        sim.bid_round([])  # only the final round is left
    sim.step([])
    assert sim.round == 0 and sim.step_index == 1
    with pytest.raises(ValueError):
        sim.bid_round([Bid(7, 0, 1.0)])


def test_pga_rounds_reveal_top_bid_and_winner_pays_standing_bid():
    sim = Simulator(pga_config(pga_rounds=3), np.random.default_rng(0))
    sim.reset()
    opp = min(sim.opportunities.active.values(), key=lambda o: o.id)
    obs0 = sim.observe(1)
    assert obs0.round == 0 and obs0.n_rounds == 3 and obs0.top_bids == {}
    sim.bid_round([Bid(0, opp.id, 1.0), Bid(1, opp.id, 0.5)])
    obs1 = sim.observe(1)
    assert obs1.round == 1 and obs1.top_bids[opp.id] == 1.0 and opp.id not in obs1.leading
    assert opp.id in sim.observe(0).leading
    # Visibility and noisy fields are drawn once per block, not per round.
    assert obs1.visible_opportunities == obs0.visible_opportunities
    assert obs1.top_fees == obs0.top_fees and obs1.pending_gas == obs0.pending_gas
    sim.bid_round([Bid(1, opp.id, 2.0)])
    r = sim.step([Bid(0, opp.id, 0.1)])  # agent 0 cannot lower its standing bid of 1.0
    (res,) = [x for x in r.auction_results if x.opportunity_id == opp.id]
    assert (res.winner, res.payment) == (1, 2.0)
    assert r.rewards[1] == pytest.approx(opp.value - 2.0) and r.rewards[0] == 0.0
    assert [b.round for b in r.bids] == [0, 0, 1, 2]


def test_pga_earlier_bid_wins_tie_in_sim():
    sim = Simulator(pga_config(pga_rounds=2), np.random.default_rng(0))
    sim.reset()
    opp = min(sim.opportunities.active.values(), key=lambda o: o.id)
    sim.bid_round([Bid(2, opp.id, 1.5), Bid(0, opp.id, 1.5)])
    assert opp.id not in sim.observe(2).leading  # tied for the top: not a safe lead
    r = sim.step([Bid(1, opp.id, 1.5)])
    assert r.auction_results[0].winner in (0, 2)  # round-0 holders beat the round-1 bid
    assert r.rewards[1] == 0.0


def test_pga_one_round_equals_first_price():
    fp = run(5, steps=150, config=SimConfig(), fraction=0.8)[1]
    pga = run(5, steps=150, config=SimConfig(mechanism="pga", pga_rounds=1), fraction=0.8)[1]
    assert fingerprint(fp) == fingerprint(pga)


def test_pga_rejects_zero_rounds():
    with pytest.raises(ValueError):
        SimConfig(pga_rounds=0)
