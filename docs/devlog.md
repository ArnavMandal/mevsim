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

## 2026-10-28: M3 gymnasium env + PPO

**What changed**
- `envs/single_agent.py`: `SingleAgentMEVEnv` (learner in slot 0, baseline opponents in the
  rest). Observation: 6 + top_k normalized features clipped to [0, 10]: balance gain, best
  visible value, top-k fees, pending gas, clearing price, last block's mean winning
  payment/value, step fraction. Action: one bid fraction applied to every visible opportunity.
- `episode.py`: `seeded_episode()` shared by the runner and the env, so a seed builds the
  same episode in both. A test checks that a constant-fraction learner reproduces a greedy
  agent in the runner exactly.
- `training/train_ppo.py` + `configs/ppo_single.yaml`: SB3 PPO, then PPO and each baseline
  are evaluated in the learner slot on the same held-out seeds (0..19; training seeds are
  drawn from >= 2**20 so they never overlap).
- `gymnasium` and `stable-baselines3` as an optional `[rl]` extra, so `core/` installs and
  tests without them (env/training tests skip if missing).

**Surprising**
- 10k-step sanity run (not the real result): PPO already bids ~14% of value and matches
  Random's profit while winning only 6% of the opportunities it bids on. Against
  random + rule-based opponents, cheap bids on uncontested opportunities are almost as
  good as fighting for contested ones.
- With these opponents rule-based (0.36) beats greedy (0.33) in the learner slot, the
  reverse of the M2 mixed lineup, because the learner no longer faces a greedy opponent.

**Result (300k steps, 20 held-out seeds, learner-slot profit per block ± 95% CI)**
- PPO 0.346 ± 0.047, Random 0.294 ± 0.037, Greedy 0.333 ± 0.025, Rule-based 0.359 ± 0.023.
- Paired by seed (same arrivals for every policy): PPO beats Random by +0.052 ± 0.042
  (15/20 seeds), and is tied with Greedy (+0.012 ± 0.042, 12/20) and Rule-based
  (-0.014 ± 0.049, 9/20). The milestone's mean criterion passes, but only the win over
  Random is statistically clear.
- PPO settled on a near-constant bid of ~0.47 of value (shading 0.465 ± 0.002 across seeds):
  it wins few opportunities (12%) but keeps over half of each one. It has not yet learned
  to vary bids with the observation.
- Undertrained: training episode reward was still climbing steeply at the end
  (23 -> 41 -> 57 per 200-block episode at 4k / 188k / 303k steps). Next: a longer run.

**Result (1M steps, same 20 held-out seeds)**
- PPO 0.365 ± 0.053, now the highest mean (Rule-based 0.359, Greedy 0.333, Random 0.294).
- Paired by seed: vs Random +0.072 ± 0.046 (16/20 seeds), vs Greedy +0.032 ± 0.038 (13/20),
  vs Rule-based +0.006 ± 0.052 (9/20). Clearly better than Random; ahead of Greedy but the
  CI still just crosses zero; tied with Rule-based.
- Training plateaued around 500-600k steps (episode reward ~66-69 from there on), and the
  policy's action std fell from 0.97 to 0.08, so more steps alone won't help much.
  Still a near-constant bid (~0.50 of value). Better features (e.g. how many opportunities
  are visible, whether others likely saw one) or per-opportunity actions are the likely
  levers, plus more eval seeds to tighten the CIs.

## 2026-10-28: M4 multi-agent env + shared-policy PPO

**What changed**
- `envs/multi_agent.py`: `MultiAgentMEVEnv`, a PettingZoo parallel env. Learners fill slots
  0..N-1 with the M3 observation/action/reward, and optional baseline opponents fill the
  rest. Passes `parallel_api_test`; a fixed-fraction population reproduces the runner exactly.
- `envs/vec.py`: `SharedPolicyVecEnv`, a small SB3 VecEnv adapter (each learner of each env
  copy is one slot) used instead of supersuit, which has a history of version breakage
  with SB3 2.x / gymnasium 1.x.
- `metrics`: episodes now report `mev_captured_per_block` = `agent_profit_per_block` +
  `mev_to_builder_per_block` (welfare split; tested for conservation).
- `training/train_ppo_multi.py` + `configs/ppo_multi.yaml`: N = 2..5 sweep, same 600k
  agent-step budget per N, `ShadingLogger` callback -> `n<N>/shading.csv`, held-out
  evaluation vs homogeneous baseline populations.
- `pettingzoo` added to the `[rl]` extra. M3 evaluation raised to 100 seeds.

**Surprising**
- SB3's Gaussian policy starts with mean 0 and actions are clipped to [0, 1], so an
  untrained population bids ~0 about half the time. A shared-policy population therefore
  *starts* in a near-collusive state (everyone bids nothing and splits the MEV), and the
  question becomes whether competition erodes it. In a 20k-step smoke run PPO populations
  still bid 2-34% of value and earned 3-10x the greedy populations. That reflects the
  starting point, not learned collusion; `shading_training.png` shows which way it goes.
- Floating-point detail: a float32 action of 0.7 is 0.69999999, so the env and a 0.7 greedy
  agent disagree in the 8th digit. Equivalence tests use 0.75 (exact in float32).

**Result (2026-11-04, run `results/ppo_multi/20261104-192525`, 600k agent-steps per N, 100 held-out seeds)**

| N | PPO profit/block | PPO shading | builder share | best baseline profit | greedy profit |
|---|---|---|---|---|---|
| 2 | 0.995 ± 0.019 | 0.683 | 0.700 | random 1.279 | 0.332 |
| 3 | 0.193 ± 0.003 | 0.904 | 0.913 | random 0.681 | 0.221 |
| 4 | 0.068 ± 0.001 | 0.956 | 0.959 | random 0.417 | 0.166 |
| 5 | 0.071 ± 0.001 | 0.938 | 0.946 | random 0.273 | 0.133 |

- Competition erodes the near-collusive start. Mean training shading rises from ~0.35-0.44
  to 0.69 (N=2), 0.90 (N=3), 0.95 (N=4), 0.93 (N=5). For N >= 3 the learned population bids
  *above* greedy's 0.9 and earns less than an all-greedy population. Almost all captured MEV
  goes to the builder (91-96%), close to Bertrand-style full rent dissipation.
- Every learner's best response is to outbid, so the shared policy keeps less for each agent
  than even random bidding. All-random populations earn the most at every N because their
  average bid is only 0.5. This is the welfare-cost-of-competition effect that M5 is meant to measure.
- Profit falls steeply with N (SPEC experiment 2 answered qualitatively): 0.995 -> 0.193 -> 0.07.
- **Caveat: N=2 had not converged.** Shading was flat at ~0.36-0.38 for 300k steps, then
  climbed 0.44 -> 0.60 -> 0.70 over the last 300k. The N=2 numbers are a snapshot on the way
  to (probably) higher shading. N=3 is near convergence; N=4 and 5 plateaued.
  Same budget per N means fewer samples per learner slot at large N, but the opposite
  happened here: larger N converged faster, plausibly because the competitive pressure is stronger.
- Surprising: the climb is slow and late at N=2. With two bidders, undercutting is
  rewarded often enough that the gradient toward outbidding is weak until the policy std shrinks.

## 2026-11-04: M5 (in progress) second-price auction + PGA

**What changed**
- `sealed_second_price`: highest bid wins, pays the second-highest; a sole bidder pays 0 (no
  reserve). MEV txs are ordered in the block by the winning *bid* (new
  `Transaction.priority_fee_per_gas`) but charged the payment. Otherwise a sole second-price
  winner paying 0 would be crowded out of any congested block, which would confound the
  mechanism comparison.
- `pga` with `pga_rounds` (default 5): first-price over R binding rounds. `Simulator.bid_round`
  runs rounds 0..R-2 and `step` the last. Observations add `round`, `top_bids`, and `leading`
  (sole leader only), and are drawn once per block so visibility can't change between rounds.
  Ties go to the earliest round; same-round ties are random. With R=1 it reproduces sealed
  first-price exactly (tested). The rule-based agent outbids the top by `step_up` (capped at
  `1 - margin`) and updates its EMA once per block. Envs: one step per round, reward on the
  final round, and 3 extra obs features only under `pga` (sealed obs and M3/M4 runs unchanged).
- Bid shading now uses each agent's highest bid per opportunity per block (identical for
  sealed auctions, where each agent bids once).

**Surprising**
- First version of `leading` counted a tie for the top as leading. Identical rule-based bots
  then tied in round 0, all thought they led, and never raised: PGA reproduced first-price to
  the last digit. Counting only a *sole* leader restores the escalation (shading 0.857 -> 0.88
  at R=2, 0.90 at R=5).
- Quick baseline check (3 agents, 10 seeds): second-price roughly halves the builder share for
  random populations (0.68 -> 0.39). But the rule-based bot's EMA tracks *payments*, which are
  low under second price, so it spirals its bids down to ~0.48 of value. That is a property of
  the bot, not the mechanism, and worth keeping in mind when reading M5 baselines.
- Random agents under PGA effectively bid the max of R uniform draws (shading 0.83 at R=5).

## 2026-11-04: M5 mechanism comparison

**What changed**
- `experiments/mechanisms.py` + `configs/mechanisms.yaml` / `mechanisms_congested.yaml`:
  every population x mechanism on 20 seeds, a PGA rounds sweep, `summary.md` + plots, and
  optional PPO training per mechanism (M4 trainer).
- Efficiency = captured MEV / MEV some agent saw (`StepResult.newly_seen`, recorder
  `mev_seen_per_block`). Valuations are common, so "who valued it most" reduces to "did anyone
  who saw it get it".
- Welfare loss vs a no-MEV counterfactual (same seed, no bids; the organic stream is
  independent of agents): `delay_cost x` extra pending organic gas per block + MEV to builder.
- `rule_based_bid`: the rule-based agent tracking winning *bids* (new public
  `Observation.last_block_bids`) instead of payments.

**Results** (3 agents, 20 seeds; `results/mechanisms*/20261104-*`)
- Default config, builder share of MEV (all_random / mixed / all_rule_based_bid):
  first-price 0.68 / 0.89 / 0.86, second-price 0.38 / 0.53 / 0.77, PGA 0.91 / 0.91 / 0.90.
  Second price leaves agents the most; PGA the least. More PGA rounds push the builder share
  up (all_random 0.85 -> 0.92 -> 0.96 at R = 2/5/10, congested config).
- With the default config, efficiency is 1.000 and extra organic waiting is identical across
  mechanisms. Blocks are only ~56% full, so every winning MEV tx is included and the welfare
  differences are pure transfers to the builder.
- Congested config (`organic_rate: 12`, blocks ~99% full): efficiency ~0.81-0.87, and MEV txs
  push organic users back (~100 extra gas-blocks of waiting per block, vs ~0.1 by default).
  First- and second-price have identical efficiency and delay for the same population,
  because both order by bid. PGA is slightly *more* efficient for random bidders (0.87 vs 0.81)
  because escalation lifts low bids above organic fees.

**Surprising**
- Efficiency was exactly 1.000 everywhere on the first run. Not a bug: the default config is
  uncongested. Efficiency only bites when inclusion is contested, hence the congested config.
- Plain rule-based under second price "wins" on welfare loss (9.1 vs ~15.5 under congestion)
  only because its payment-tracking spiral bids so low that it loses block space and leaves
  MEV uncaptured (efficiency 0.64). The SPEC welfare formula doesn't count that, so
  `uncaptured_mev_per_block` is reported next to it. `rule_based_bid` fixes the spiral (0.77
  builder share, efficiency 0.86).
- The SPEC counts builder payments as welfare loss, but they are transfers, not deadweight
  loss. Only the delay term and uncaptured MEV are true efficiency losses here. Worth
  revisiting the definition in M7.
- PPO under each mechanism is wired up (`ppo.enabled`) but not yet run.
