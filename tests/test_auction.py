import numpy as np
import pytest

from mevsim.core.auction import SealedFirstPrice, make_auction
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
