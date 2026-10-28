# Devlog

## 2026-10-26: M1 core simulator

**What changed**
- `core/`: `types.py` (dataclasses), `mempool.py`, `opportunities.py`, `auction.py`
  (`sealed_first_price` behind a pluggable interface), `builder.py` (fee-sorted, capacity-bound),
  `sim.py` (per-step flow from SPEC section 2), `config.py` (YAML, rejects unknown keys).
- 22 tests: capacity limits, fee and auction tie-breaking, first-price payment, value
  conservation (captured value = agent profit + builder MEV revenue), no double capture,
  expiry, crowd-out, determinism, partial observation, config loading.

**Design decisions**
- A bid is a total payment for one opportunity. The winner's tx enters the block with
  fee-per-gas = bid / `agent_tx_gas`, so it competes with organic txs for space. A winner
  pays only if its tx is included. Otherwise the opportunity stays live until it expires.
- Losing bids pay nothing (sealed, Flashbots-style). Reverted-tx costs are left to PGA / `fail_penalty`.
- Congestion is an EMA of backlog / capacity. Organic median fee = `scale * (1 + sens * congestion)`.
  Clearing price = lowest included fee per gas.
- The root generator spawns four child streams (organic, opportunities, auction ties, observations).

**Surprising**
- With one shared stream, simply calling `observe()` more often changed future organic
  arrivals. That would quietly break the no-MEV counterfactual comparison. Split streams fix
  the random draws, but organic *fees* still legitimately respond to congestion that agents cause.
- Sanity run (1000 blocks, 3 greedy agents bidding 90%): blocks ~56% full on average and the
  agents kept only ~10% of captured MEV. The builder got the rest, as first-price competition predicts.
