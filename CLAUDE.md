# CLAUDE.md

## Project
`mevsim`: a stylized multi-agent RL simulator of Ethereum-style transaction ordering and MEV.
Full design lives in `SPEC.md`. Read it before making design changes.

## Stack
- Python 3.11+
- numpy, pandas, matplotlib
- gymnasium (single-agent env), pettingzoo (multi-agent env)
- stable-baselines3 (PPO/DQN), supersuit if needed for PettingZoo -> SB3
- pytest, ruff, pyyaml

## Commands
- Install: `pip install -e ".[dev]"`
- Tests: `pytest -q`
- Lint/format: `ruff check . && ruff format .`
- Run a baseline experiment: `python -m mevsim.experiments.run --config configs/baselines.yaml`
- Train PPO: `python -m mevsim.training.train_ppo --config configs/ppo_single.yaml`

## Repo layout
```
src/mevsim/
  core/         # mempool, transactions, block builder, auctions (pure Python, no RL)
  envs/         # gymnasium + pettingzoo wrappers around core
  agents/       # random, greedy, rule_based, rl wrappers
  metrics/      # profit, efficiency, welfare, congestion, latency
  training/     # training scripts
  experiments/  # experiment runners + plotting
configs/        # YAML configs, one per experiment
tests/
results/        # gitignored; outputs of runs
```

## Rules
1. **Keep `core/` free of RL imports.** The simulator must be usable and testable without gym/SB3.
2. **Determinism.** Every random source takes a seed from a single `np.random.Generator`. No global `np.random.*` calls. Same seed + config = identical results.
3. **Test the auction logic.** Any change to `core/auction.py` or block building needs unit tests (capacity limits, tie-breaking, payment rules, conservation of value). Bugs here silently corrupt all RL results.
4. **Config-driven experiments.** No hard-coded hyperparameters in scripts; use YAML configs and log the resolved config next to results.
5. **Type hints and short docstrings** on public functions. Use dataclasses for Transaction, Block, Observation.
6. **Small diffs.** Implement one milestone at a time. Do not refactor unrelated code.
7. **Ask before adding dependencies.**

## Cost and workflow rules
- Do NOT launch long training runs yourself. Write the script, tell me the command, and I will run it and paste back a short summary.
- When reading logs, only read the last ~50 lines or a summary file, not full logs.
- Don't re-read files you already have in context; don't scan the whole repo unless asked.
- Before a big change, give a short plan (bullets) and wait for my OK.

## Definition of done for any milestone
- Tests pass, ruff clean
- A runnable command in the README for it
- A short note in `docs/devlog.md`: what changed, what was surprising
