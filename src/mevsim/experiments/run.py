"""Run baseline-agent experiments from a YAML config.

    python -m mevsim.experiments.run --config configs/baselines.yaml

Each scenario is a lineup of agents (one per sim agent slot). Every scenario runs on the
same seeds, so scenarios share organic and opportunity arrival streams (common random
numbers). Results go to `<out>/<name>/<timestamp>/`: the resolved config, per-agent and
per-episode CSVs, and plots.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mevsim.agents import make_agent
from mevsim.core.config import SimConfig
from mevsim.core.sim import Simulator
from mevsim.metrics import EpisodeRecorder


@dataclass
class ExperimentConfig:
    """Resolved experiment settings. `agents` maps agent type to constructor params."""

    name: str
    sim: SimConfig
    horizon: int
    seeds: list[int]
    agents: dict[str, dict[str, Any]]
    scenarios: dict[str, list[str]]
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        path = Path(path)
        with open(path) as f:
            d = yaml.safe_load(f)
        known = {"name", "sim_config", "sim_overrides", "horizon", "seeds", "agents", "scenarios"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown experiment keys: {sorted(unknown)}")
        sim = SimConfig()
        if "sim_config" in d:
            sim = SimConfig.from_yaml(_resolve(path, d["sim_config"]))
        sim = SimConfig.from_dict({**sim.to_dict(), **(d.get("sim_overrides") or {})})
        seeds = d.get("seeds", 10)
        seeds = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
        agents = {k: v or {} for k, v in d["agents"].items()}
        scenarios = d["scenarios"]
        for name, lineup in scenarios.items():
            missing = set(lineup) - set(agents)
            if missing:
                raise ValueError(f"scenario {name!r} uses undefined agents {sorted(missing)}")
        return cls(d["name"], sim, int(d["horizon"]), seeds, agents, scenarios, d)

    def resolved(self) -> dict[str, Any]:
        """Fully expanded config, saved next to results."""
        return {
            "name": self.name,
            "sim": self.sim.to_dict(),
            "horizon": self.horizon,
            "seeds": self.seeds,
            "agents": self.agents,
            "scenarios": self.scenarios,
        }


def _resolve(config_path: Path, ref: str) -> Path:
    """Resolve a referenced file relative to the CWD, falling back to the config's folder."""
    p = Path(ref)
    return p if p.exists() else config_path.parent / p.name


def run_episode(
    sim_config: SimConfig,
    lineup: list[str],
    agent_params: dict[str, dict[str, Any]],
    horizon: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one episode and return (per-agent rows, episode row)."""
    config = replace(sim_config, n_agents=len(lineup))
    sim_rng, agents_rng = np.random.default_rng(seed).spawn(2)
    agent_rngs = agents_rng.spawn(len(lineup))
    agents = [
        make_agent(kind, i, rng, **agent_params[kind])
        for i, (kind, rng) in enumerate(zip(lineup, agent_rngs, strict=True))
    ]
    sim = Simulator(config, sim_rng)
    sim.reset()
    recorder = EpisodeRecorder(config.n_agents, config.gas_limit)
    for _ in range(horizon):
        bids = [b for agent in agents for b in agent.act(sim.observe(agent.index))]
        active = dict(sim.opportunities.active)
        result = sim.step(bids)
        recorder.record(bids, active, result, sim.mempool.pending_gas)
    rows = recorder.agent_summaries()
    for row, kind in zip(rows, lineup, strict=True):
        row["agent_type"] = kind
    return rows, recorder.episode_summary()


def run_experiment(cfg: ExperimentConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run every scenario on every seed."""
    agent_rows, episode_rows = [], []
    for scenario, lineup in cfg.scenarios.items():
        for seed in cfg.seeds:
            rows, ep = run_episode(cfg.sim, lineup, cfg.agents, cfg.horizon, seed)
            agent_rows += [{"scenario": scenario, "seed": seed, **r} for r in rows]
            episode_rows.append({"scenario": scenario, "seed": seed, **ep})
    return agent_rows, episode_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="experiment YAML")
    parser.add_argument("--out", default="results", help="root output directory")
    parser.add_argument("--no-plots", action="store_true", help="skip plotting")
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config)
    out = Path(args.out) / cfg.name / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.resolved(), f, sort_keys=False)

    agent_rows, episode_rows = run_experiment(cfg)
    write_csv(out / "agents.csv", agent_rows)
    write_csv(out / "episodes.csv", episode_rows)
    if not args.no_plots:
        from mevsim.experiments.plots import plot_all

        plot_all(agent_rows, episode_rows, out)
    from mevsim.experiments.plots import summary_table

    print(summary_table(agent_rows, episode_rows))
    print(f"\nresults: {out}")
    return out


if __name__ == "__main__":
    main()
