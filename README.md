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
    result = sim.step(bids)   # result.block, result.rewards, result.captured
```
