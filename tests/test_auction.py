import numpy as np
import pytest

from mevsim.core.auction import (
    PriorityGasAuction,
    SealedFirstPrice,
    SealedSecondPrice,
    make_auction,
    pga_leaders,
    standing_bids,
)
from mevsim.core.types import Bid


def resolve(bids, seed=0):
    return SealedFirstPrice().resolve(bids, np.random.default_rng(seed))


def test_highest_bid_wins_and_pays_own_bid():
    (r,) = resolve([Bid(0, 7, 1.0), Bid(1, 7, 3.0), Bid(2, 7, 2.0)])
    assert (r.opportunity_id, r.winner, r.payment, r.n_bidders) == (7, 1, 3.0, 3)


def test_zero_and_negative_bids_are_skips():
    assert resolve([Bid(0, 1, 0.0), Bid(1, 1, -2.0)]) == []
    (r,) = resolve([Bid(0, 1, 0.0), Bid(1, 1, 0.5)])
    assert r.winner == 1 and r.n_bidders == 1


def test_duplicate_bids_keep_agents_max():
    (r,) = resolve([Bid(0, 1, 4.0), Bid(0, 1, 1.0), Bid(1, 1, 3.0)])
    assert r.winner == 0 and r.payment == 4.0 and r.n_bidders == 2


def test_each_opportunity_resolved_independently():
    results = resolve([Bid(0, 1, 2.0), Bid(1, 1, 1.0), Bid(1, 2, 5.0)])
    assert [(r.opportunity_id, r.winner) for r in results] == [(1, 0), (2, 1)]


def test_ties_are_seeded_and_fair():
    bids = [Bid(0, 1, 2.0), Bid(1, 1, 2.0), Bid(2, 1, 1.0)]
    assert resolve(bids, seed=3) == resolve(bids, seed=3)
    rng = np.random.default_rng(0)
    winners = [SealedFirstPrice().resolve(bids, rng)[0].winner for _ in range(2000)]
    assert set(winners) == {0, 1}
    assert 0.45 < winners.count(0) / len(winners) < 0.55


def test_tie_result_independent_of_bid_order():
    bids = [Bid(0, 1, 2.0), Bid(1, 1, 2.0)]
    assert resolve(bids, seed=5) == resolve(list(reversed(bids)), seed=5)


def test_unknown_mechanism():
    with pytest.raises(ValueError):
        make_auction("dutch")


def resolve2(bids, seed=0):
    return SealedSecondPrice().resolve(bids, np.random.default_rng(seed))


def test_second_price_winner_pays_second_highest():
    (r,) = resolve2([Bid(0, 7, 1.0), Bid(1, 7, 3.0), Bid(2, 7, 2.0)])
    assert (r.opportunity_id, r.winner, r.payment, r.n_bidders, r.bid) == (7, 1, 2.0, 3, 3.0)


def test_second_price_sole_bidder_pays_zero():
    (r,) = resolve2([Bid(0, 1, 0.0), Bid(1, 1, 2.5)])
    assert (r.winner, r.payment, r.n_bidders, r.bid) == (1, 0.0, 1, 2.5)


def test_second_price_ignores_winners_own_duplicate_bids():
    # Agent 0's lower duplicate must not set its own price.
    (r,) = resolve2([Bid(0, 1, 4.0), Bid(0, 1, 3.5), Bid(1, 1, 1.0)])
    assert r.winner == 0 and r.payment == 1.0 and r.n_bidders == 2


def test_second_price_tie_pays_tied_amount_and_is_fair():
    bids = [Bid(0, 1, 2.0), Bid(1, 1, 2.0), Bid(2, 1, 1.0)]
    assert resolve2(bids, seed=3) == resolve2(list(reversed(bids)), seed=3)
    rng = np.random.default_rng(0)
    results = [SealedSecondPrice().resolve(bids, rng)[0] for _ in range(2000)]
    assert all(r.payment == 2.0 for r in results)
    share = sum(r.winner == 0 for r in results) / len(results)
    assert 0.45 < share < 0.55


def test_payment_never_exceeds_bid_and_same_winner_as_first_price():
    rng = np.random.default_rng(1)
    for trial in range(200):
        bids = [
            Bid(int(a), int(o), float(x))
            for a, o, x in zip(
                rng.integers(0, 5, 12), rng.integers(0, 3, 12), rng.uniform(-1, 5, 12), strict=True
            )
        ]
        first, second = resolve(bids, seed=trial), resolve2(bids, seed=trial)
        assert [(r.opportunity_id, r.winner, r.bid) for r in first] == [
            (r.opportunity_id, r.winner, r.bid) for r in second
        ]
        for f, s in zip(first, second, strict=True):
            assert 0.0 <= s.payment <= s.bid == f.payment


def test_first_price_bid_equals_payment():
    (r,) = resolve([Bid(0, 1, 1.5), Bid(1, 1, 0.5)])
    assert r.bid == r.payment == 1.5


def test_make_auction_second_price():
    assert isinstance(make_auction("sealed_second_price"), SealedSecondPrice)


def resolve_pga(bids, seed=0):
    return PriorityGasAuction().resolve(bids, np.random.default_rng(seed))


def test_pga_highest_standing_bid_wins_and_pays_it():
    bids = [Bid(0, 1, 2.0, round=0), Bid(1, 1, 3.0, round=1), Bid(0, 1, 3.5, round=2)]
    (r,) = resolve_pga(bids)
    assert (r.winner, r.payment, r.bid, r.n_bidders) == (0, 3.5, 3.5, 2)


def test_pga_bids_are_binding():
    # Agent 0 "lowers" its bid in a later round; its round-0 bid still stands.
    bids = [Bid(0, 1, 4.0, round=0), Bid(0, 1, 1.0, round=1), Bid(1, 1, 3.0, round=1)]
    (r,) = resolve_pga(bids)
    assert r.winner == 0 and r.payment == 4.0


def test_pga_tie_goes_to_earliest_round():
    bids = [Bid(1, 1, 2.0, round=3), Bid(0, 1, 2.0, round=1), Bid(1, 1, 1.0, round=0)]
    for seed in range(20):
        assert resolve_pga(bids, seed)[0].winner == 0


def test_pga_repeated_amount_keeps_earliest_round():
    bids = [Bid(0, 1, 2.0, round=0), Bid(0, 1, 2.0, round=3), Bid(1, 1, 2.0, round=1)]
    assert standing_bids(bids)[1][0] == (2.0, 0)
    assert resolve_pga(bids)[0].winner == 0


def test_pga_same_round_tie_is_seeded_and_fair():
    bids = [Bid(0, 1, 2.0, round=2), Bid(1, 1, 2.0, round=2)]
    assert resolve_pga(bids, seed=4) == resolve_pga(list(reversed(bids)), seed=4)
    rng = np.random.default_rng(0)
    winners = [PriorityGasAuction().resolve(bids, rng)[0].winner for _ in range(2000)]
    assert 0.45 < winners.count(0) / len(winners) < 0.55


def test_pga_single_round_matches_first_price():
    rng = np.random.default_rng(2)
    for trial in range(200):
        bids = [
            Bid(int(a), int(o), float(x))
            for a, o, x in zip(
                rng.integers(0, 5, 12), rng.integers(0, 3, 12), rng.uniform(-1, 5, 12), strict=True
            )
        ]
        assert resolve_pga(bids, seed=trial) == resolve(bids, seed=trial)


def test_pga_leaders():
    assert pga_leaders({0: (2.0, 1), 1: (2.0, 1), 2: (2.0, 2), 3: (1.0, 0)}) == (2.0, [0, 1])
