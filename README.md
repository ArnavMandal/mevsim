# mevsim

A stylized multi-agent RL simulator of Ethereum-style transaction ordering and MEV.
Not a real Ethereum client: it abstracts the mempool, block building and fee auctions so
learning agents can be studied in a controlled setting. Design: [`SPEC.md`](SPEC.md).

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## M1: core simulator

```bash
pytest -q                          # auction, builder, and simulator tests
ruff check . && ruff format .
```

Minimal usage:

```python
import numpy as np
from mevsim.core.config import SimConfig
from mevsim.core.sim import Simulator
from mevsim.core.types import Bid

sim = Simulator(SimConfig.from_yaml("configs/default.yaml"), np.random.default_rng(0))
sim.reset()
for _ in range(100):
    bids = []
    for i in range(sim.config.n_agents):
        obs = sim.observe(i)
        bids += [Bid(i, o.id, 0.9 * o.value) for o in obs.visible_opportunities]
    result = sim.step(bids)  # result.block, result.rewards, result.captured
```

## M2: baseline agents and experiment runner

Random, greedy (fixed 90% bid) and rule-based (bids just above the recent winning
bid/value ratio, keeping a 10% margin) agents, compared over 10 seeds x 200 blocks:

```bash
python -m mevsim.experiments.run --config configs/baselines.yaml
```

Writes `results/baselines/<timestamp>/`: the resolved `config.yaml`, `agents.csv`
(profit, win rate, bid shading per agent and seed), `episodes.csv` (latency, congestion),
and `profit.png`, `latency.png`, `bid_shading.png`. Scenarios (a mixed lineup plus one
homogeneous lineup per agent type) are defined in the YAML and share seeds, so they see the
same organic and opportunity arrivals.

## M3: single-agent gymnasium env + PPO

```bash
pip install -e ".[dev,rl]"     # adds gymnasium + stable-baselines3 (pulls in PyTorch)
python -m mevsim.training.train_ppo --config configs/ppo_single.yaml   # ~10 min on a laptop CPU
```

`mevsim.envs.SingleAgentMEVEnv`: agent 0 learns a bid fraction in [0, 1] against fixed
baseline opponents (`opponents` in the config). After training, PPO and each baseline take
the learner slot on the same 20 held-out seeds. Writes `results/ppo_single/<timestamp>/`:
`model.zip`, SB3 `progress.csv`, `eval_agents.csv`, `eval_episodes.csv`, plots, and
`summary.txt` (learner-slot profit per policy, mean ± 95% CI).
