# SPEC: Multi-Agent RL for MEV Strategy Simulation

A stylized model of transaction ordering competition. This is NOT a faithful Ethereum client. It abstracts the mempool, block building, and fee auctions so that learning agents can be studied in a controlled setting.

## 1. Goals
1. Build a deterministic, tested block-auction simulator.
2. Compare baseline agents (random, greedy, rule-based) with RL agents.
3. Study emergent behavior with multiple learning agents under partial observability.
4. Quantify the welfare cost of MEV competition under different auction mechanisms.

## 2. Core simulation model

Time is discrete. One step = one block.

### Entities
- **Transaction**: `id, owner, gas_used, max_fee_per_gas (bid), arrival_step, kind` where kind is `organic` or `agent`.
- **MEV opportunity**: `id, value V, expires_step`. Example: an arbitrage worth V to whoever captures it. Only one agent can capture each opportunity.
- **Block**: gas limit `G` (e.g. 30 units, abstracted). Filled greedily by the builder using the active mechanism.
- **Builder/proposer**: receives bids as revenue. Rule-based, not learning (v1).

### Per-step flow
1. New organic transactions arrive (Poisson rate `lambda`, gas sizes from a configurable distribution, fees from a lognormal scaled by recent congestion).
2. New MEV opportunities arrive (Poisson rate `mu`, values from a heavy-tailed distribution).
3. Each agent receives its observation (partial) and chooses an action.
4. Auction resolves under the active mechanism (section 3).
5. Block is built, rewards are paid, included transactions leave the mempool, unincluded ones age.
6. Update congestion / base-fee signal.

### Partial observability
Each agent sees:
- Own balance and pending transactions
- Each opportunity with probability `p_see` (independent per agent)
- A noisy summary of the mempool: top-k fees, total pending gas (noise std configurable)
- Block capacity remaining, step index
- Previous block's clearing price

Agents do NOT see other agents' bids or private valuations.

## 3. Auction mechanisms (pluggable)

| Name | Description |
|------|-------------|
| `pga` | Priority Gas Auction: open first-price, R rounds of re-bidding within a block interval. Agents see current top bid each round. |
| `sealed_first_price` | One sealed round; highest bid wins, pays own bid. |
| `sealed_second_price` | One sealed round; highest bid wins, pays second-highest bid. |

Start with `sealed_first_price` (Week 2), add `pga` and second-price later as experiments.

Builder inclusion rule: sort candidate transactions by fee descending, include while capacity remains.

## 4. RL formulation

### Single-agent env (`gymnasium`)
- **Observation**: fixed-length float vector (normalized): own balance, best visible opportunity value (0 if none), top-k mempool fees, pending gas fraction, last clearing price, step fraction.
- **Action (v1, continuous)**: `bid_fraction in [0, 1]` of the visible opportunity value (0 = skip). Optionally a discrete variant: `{skip, bid 10%, 30%, 50%, 70%, 90%}` for DQN.
- **Reward**: `captured_value - payment` on win, 0 otherwise. Optional penalty term for failed/reverted inclusion (configurable `fail_penalty`).
- **Episode**: fixed horizon (e.g. 200 blocks).

### Multi-agent env (`pettingzoo`, parallel API)
- N agents with the same observation/action spaces.
- Each agent has an independent reward (profit). Social welfare is tracked as a metric, not a reward, in v1.
- Training: independent PPO with parameter sharing first; then separate policies to study non-stationarity.

## 5. Agents
| Agent | Behavior |
|-------|----------|
| Random | Uniform random bid fraction |
| Greedy | Always bids a fixed high fraction (e.g. 90%) of visible value |
| Rule-based | Bids just above estimated competitor clearing price (moving average of last winning bids), capped by value margin |
| PPO | SB3 PPO |
| DQN | SB3 DQN on the discrete variant (optional) |

## 6. Metrics
Computed per episode and aggregated across seeds (report mean and 95% CI):
- **Agent profit**: total and per-block
- **Win rate** on opportunities
- **Inclusion latency**: blocks between tx arrival and inclusion (organic txs)
- **Auction efficiency**: fraction of opportunities captured by the agent with the highest valuation
- **Congestion**: mean mempool backlog / capacity
- **Welfare loss due to MEV**: organic user surplus lost (delay cost x delay) + value burned in bidding wars (bids paid that went to the builder rather than being net surplus), vs. a no-MEV counterfactual run with the same seed
- **Bid shading**: mean bid / value ratio over time (shows strategy evolution)

## 7. Milestones

| # | Milestone | Done when |
|---|-----------|-----------|
| M1 | Core simulator + unit tests | Block building and auction tests pass; deterministic under seed |
| M2 | Baseline agents + experiment runner + plots | `run.py` produces profit/latency plots for 3 baselines |
| M3 | Gymnasium env + PPO | `check_env` passes; PPO beats Random and Greedy on average profit |
| M4 | PettingZoo multi-agent env | N=2..5 learning agents train without crashing; logs of bid shading |
| M5 | Mechanism comparison experiments | Table + plots comparing pga / first-price / second-price on welfare and efficiency |
| M6 | Calibration with real data (stretch) | Fee distribution fitted from public Ethereum / Flashbots data |
| M7 | Polish | README written like a short paper, diagrams, reproducible configs |

## 8. Experiments to run (M5)
1. RL vs baselines (single learner, others fixed).
2. Single vs multi-agent learning: does profit fall as the number of learners grows?
3. Mechanism comparison: efficiency and welfare under pga vs first-price vs second-price.
4. Observability sweep: vary `p_see` and noise; how much does information matter?
5. Congestion sweep: vary `lambda`; when do bidding wars blow up?

Each experiment: at least 10 seeds, fixed configs saved in `configs/`.

## 9. Non-goals (v1)
- No EVM execution, Solidity, or real signatures
- No real network or live mempool connection
- No multi-block strategies (reorgs, time-bandit attacks)
- No learning builder

## 10. Stretch ideas
- Sandwich-attack agent vs victim txs with slippage tolerance
- Learning builder/proposer that picks the mechanism
- Replay real historical gas price series as the organic fee process
- Validate a toy strategy on a local Anvil fork
- Small dashboard of bid shading over training
