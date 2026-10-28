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

## 2026-10-28: M2 baseline agents + experiment runner

**What changed**
- `agents/`: `RandomAgent`, `GreedyAgent`, `RuleBasedAgent` behind `make_agent(kind, index, rng, **params)`.
  Each agent gets its own child generator, spawned separately from the simulator's, so agent
  randomness never shifts the sim's streams.
- `core/`: `Observation.last_block_captures` = (value, payment) for each MEV capture in the
  previous block. Public information on a real chain; the rule-based agent needs it, since
  `last_clearing_price` is set by organic fees and says nothing about MEV bids.
- `metrics/`: `EpisodeRecorder` (profit, win rate, bid shading, organic inclusion latency,
  congestion, block fullness) and `mean_ci` (t-based 95% CI). Efficiency and welfare wait for M5.
- `experiments/run.py` + `configs/baselines.yaml`: scenarios x seeds, CSVs, plots, resolved config.
  `matplotlib` added as a dependency.

**Results (default config, 10 seeds x 200 blocks, profit per block ± 95% CI)**
- Mixed lineup: greedy 0.34 ± 0.03, rule-based 0.28 ± 0.03, random 0.11 ± 0.02.
- Homogeneous: all-random 0.71 ± 0.07, all-rule-based 0.31 ± 0.02, all-greedy 0.22 ± 0.02.

**Surprising**
- Against each other greedy wins, but a population of random bidders earns ~3x a population
  of greedy ones: first-price competition among high bidders hands the surplus to the builder.
  Classic prisoner's dilemma. A good target for the M4 multi-agent question.
- The rule-based agent only ratchets up. Once others pay ~90%, its EMA climbs to the
  `1 - margin` cap, so in a homogeneous population it converges toward greedy
  (shading 0.86, still rising). It never probes downward after losses.
- Organic latency is ~0.01 blocks and identical across scenarios at default load (blocks ~56%
  full). Even near capacity (`organic_rate` 11.5, latency ~1 block) it barely differs by
  scenario: every capture adds one 2-gas tx no matter who wins, so the baselines' strategy
  can't move organic delay. Latency only becomes strategy-dependent once agents can send
  multiple txs (PGA rounds, failed txs) or bid amounts change crowd-out more.
