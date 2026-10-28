"""Train PPO in the single-agent env, then evaluate it against the baselines.

    python -m mevsim.training.train_ppo --config configs/ppo_single.yaml

Evaluation puts each policy (PPO and every baseline) in the learner slot against the same
opponents on the same held-out seeds, so profits are directly comparable. Outputs go to
`<out>/<name>/<timestamp>/`: resolved config, `model.zip`, SB3 `progress.csv`,
`eval_agents.csv`, `eval_episodes.csv`, and plots.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from mevsim.core.config import SimConfig
from mevsim.envs import SingleAgentMEVEnv
from mevsim.experiments.run import load_sim_config, run_episode, write_csv


@dataclass
class PPOConfig:
    """Resolved settings for one PPO experiment."""

    name: str
    sim: SimConfig
    horizon: int
    opponents: list[str]
    agents: dict[str, dict[str, Any]]
    baselines: list[str]
    total_timesteps: int
    train_seed: int
    n_envs: int
    ppo: dict[str, Any] = field(default_factory=dict)
    eval_seeds: list[int] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PPOConfig:
        path = Path(path)
        with open(path) as f:
            d = yaml.safe_load(f)
        known = {"name", "sim_config", "sim_overrides", "horizon", "opponents", "agents",
                 "baselines", "train", "eval"}  # fmt: skip
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        train, ev = d.get("train") or {}, d.get("eval") or {}
        unknown = set(train) - {"total_timesteps", "seed", "n_envs", "ppo"}
        unknown |= set(ev) - {"seeds"}
        if unknown:
            raise ValueError(f"unknown train/eval keys: {sorted(unknown)}")
        agents = {k: v or {} for k, v in (d.get("agents") or {}).items()}
        missing = (set(d["opponents"]) | set(d.get("baselines", []))) - set(agents)
        if missing:
            raise ValueError(f"opponents/baselines use undefined agents {sorted(missing)}")
        seeds = ev.get("seeds", 20)
        return cls(
            name=d["name"],
            sim=load_sim_config(path, d),
            horizon=int(d.get("horizon", 200)),
            opponents=list(d["opponents"]),
            agents=agents,
            baselines=list(d.get("baselines", [])),
            total_timesteps=int(train["total_timesteps"]),
            train_seed=int(train.get("seed", 0)),
            n_envs=int(train.get("n_envs", 1)),
            ppo=dict(train.get("ppo") or {}),
            eval_seeds=list(range(seeds)) if isinstance(seeds, int) else list(seeds),
        )

    def resolved(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sim": self.sim.to_dict(),
            "horizon": self.horizon,
            "opponents": self.opponents,
            "agents": self.agents,
            "baselines": self.baselines,
            "train": {
                "total_timesteps": self.total_timesteps,
                "seed": self.train_seed,
                "n_envs": self.n_envs,
                "ppo": self.ppo,
            },
            "eval": {"seeds": self.eval_seeds},
        }

    def make_env(self) -> SingleAgentMEVEnv:
        return SingleAgentMEVEnv(self.sim, self.opponents, self.agents, self.horizon)


def train(cfg: PPOConfig, out: Path):
    """Train PPO and save `model.zip` and `progress.csv` in `out`."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.logger import configure

    venv = make_vec_env(cfg.make_env, n_envs=cfg.n_envs, seed=cfg.train_seed)
    model = PPO("MlpPolicy", venv, seed=cfg.train_seed, verbose=0, **cfg.ppo)
    model.set_logger(configure(str(out), ["stdout", "csv"]))
    model.learn(total_timesteps=cfg.total_timesteps)
    model.save(out / "model.zip")
    return model


def evaluate_policy(model, cfg: PPOConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the trained policy deterministically on each eval seed."""
    env = cfg.make_env()
    lineup = ["ppo", *cfg.opponents]
    agent_rows, episode_rows = [], []
    for seed in cfg.eval_seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
        for row, kind in zip(info["agent_metrics"], lineup, strict=True):
            agent_rows.append({"scenario": "ppo", "seed": seed, **row, "agent_type": kind})
        episode_rows.append({"scenario": "ppo", "seed": seed, **info["episode_metrics"]})
    return agent_rows, episode_rows


def evaluate_baselines(cfg: PPOConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Each baseline in the learner slot, same opponents and seeds as the PPO evaluation."""
    agent_rows, episode_rows = [], []
    for kind in cfg.baselines:
        lineup = [kind, *cfg.opponents]
        for seed in cfg.eval_seeds:
            rows, ep = run_episode(cfg.sim, lineup, cfg.agents, cfg.horizon, seed)
            agent_rows += [{"scenario": kind, "seed": seed, **r} for r in rows]
            episode_rows.append({"scenario": kind, "seed": seed, **ep})
    return agent_rows, episode_rows


def learner_table(agent_rows: list[dict[str, Any]]) -> str:
    """Learner-slot (agent 0) profit per block for each policy, mean ± 95% CI."""
    from mevsim.metrics import mean_ci

    lines = [f"{'policy':<14}{'profit_per_block':>24}{'win_rate':>20}{'bid_shading':>20}"]
    scenarios = list(dict.fromkeys(r["scenario"] for r in agent_rows))
    for s in scenarios:
        rows = [r for r in agent_rows if r["scenario"] == s and r["agent"] == 0]
        cells = []
        for m in ("profit_per_block", "win_rate", "bid_shading"):
            mean, ci = mean_ci([r[m] for r in rows if r[m] == r[m]])  # drop NaN
            cells.append(f"{mean:>12.3f} ± {ci:<5.3f}")
        lines.append(f"{s:<14}" + "".join(f"{c:>20}" for c in cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="PPO experiment YAML")
    parser.add_argument("--out", default="results", help="root output directory")
    parser.add_argument("--no-plots", action="store_true", help="skip plotting")
    args = parser.parse_args(argv)

    cfg = PPOConfig.from_yaml(args.config)
    out = Path(args.out) / cfg.name / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.resolved(), f, sort_keys=False)

    model = train(cfg, out)
    ppo_agents, ppo_eps = evaluate_policy(model, cfg)
    base_agents, base_eps = evaluate_baselines(cfg)
    agent_rows, episode_rows = ppo_agents + base_agents, ppo_eps + base_eps
    write_csv(out / "eval_agents.csv", agent_rows)
    write_csv(out / "eval_episodes.csv", episode_rows)
    if not args.no_plots:
        from mevsim.experiments.plots import plot_all

        plot_all(agent_rows, episode_rows, out)

    table = learner_table(agent_rows)
    (out / "summary.txt").write_text(table + "\n")
    print("\nLearner slot, held-out seeds:\n" + table)
    print(f"\nresults: {out}")
    return out


if __name__ == "__main__":
    main()
