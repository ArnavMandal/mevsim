"""Block builder: fills a block from candidate transactions."""

from __future__ import annotations

from mevsim.core.types import Block, Transaction


def build_block(candidates: list[Transaction], gas_limit: int, step: int) -> Block:
    """Sort by fee per gas (descending) and include each tx that still fits.

    Ties are broken by earlier arrival, then lower tx id, so ordering is deterministic.
    A tx that does not fit is skipped, and smaller txs further down may still be included.
    """
    ordered = sorted(candidates, key=lambda tx: (-tx.max_fee_per_gas, tx.arrival_step, tx.id))
    block = Block(step=step, gas_limit=gas_limit)
    remaining = gas_limit
    for tx in ordered:
        if tx.gas_used <= remaining:
            block.txs.append(tx)
            remaining -= tx.gas_used
    return block
