from mevsim.core.builder import build_block
from mevsim.core.types import Transaction, TxKind


def tx(id, gas, fee, arrival=0):
    return Transaction(id, -1, gas, fee, arrival, TxKind.ORGANIC)


def test_sorted_by_fee_and_respects_capacity():
    block = build_block([tx(0, 4, 1.0), tx(1, 4, 3.0), tx(2, 4, 2.0)], gas_limit=8, step=0)
    assert [t.id for t in block.txs] == [1, 2]
    assert block.gas_used == 8
    assert block.clearing_price == 2.0
    assert block.builder_revenue == 4 * 3.0 + 4 * 2.0


def test_skips_too_large_tx_but_fills_with_smaller():
    block = build_block([tx(0, 5, 3.0), tx(1, 6, 2.0), tx(2, 2, 1.0)], gas_limit=8, step=0)
    assert [t.id for t in block.txs] == [0, 2]


def test_never_exceeds_capacity_randomized():
    import numpy as np

    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(0, 30))
        cands = [tx(i, int(rng.integers(1, 6)), float(rng.random())) for i in range(n)]
        block = build_block(cands, gas_limit=17, step=0)
        assert block.gas_used <= 17
        assert len({t.id for t in block.txs}) == len(block.txs)


def test_fee_ties_broken_by_arrival_then_id():
    block = build_block([tx(5, 2, 1.0, 1), tx(3, 2, 1.0, 1), tx(9, 2, 1.0, 0)], 4, step=2)
    assert [t.id for t in block.txs] == [9, 3]


def test_empty_block():
    block = build_block([], gas_limit=10, step=0)
    assert block.txs == [] and block.clearing_price == 0.0 and block.builder_revenue == 0.0
