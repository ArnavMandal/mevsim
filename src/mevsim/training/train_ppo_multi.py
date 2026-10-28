"""Train N learners with one shared PPO policy, for each N in a sweep.

    python -m mevsim.training.train_ppo_multi --config configs/ppo_multi.yaml

Per N: train with parameter sharing (independent rewards, shared network), logging each
learner's bid shading, win rate and profit per training episode. Then evaluate the learned
population on held-out seeds next to homogeneous baseline populations of the same size.
Outputs in `<out>/<name>/<timestamp>/`: resolved config, `n<N>/` (model, SB3 progress.csv,
shading.csv), `eval_agents.csv`, `eval_episodes.csv`, `summary.txt`, and plots.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from stable_baselines3.common.callbacks import BaseCallback

from mevsim.core.config import SimConfig
from mevsim.envs import MultiAgentMEVEnv
from mevsim.experiments.run import load_sim_config, run_episode, write_csv
from mevsim.metrics import mean_ci


@dataclass
class MultiPPOConfig:
    """Resolved settings for a multi-agent PPO sweep."""

    name: str
    sim: SimConfig
    horizon: int
    n_learners: list[int]
    opponents: list[str]
    agents: dict[str, dict[str, Any]]
    baselines: list[str]
    total_timesteps: int
    train_seed: int
    n_envs: int
    ppo: dict[str, Any] = field(default_factory=dict)
    eval_seeds: list[int] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> MultiPPOConfig:
        path = Path(path)
        with open(path) as f:
            d = yaml.safe_load(f)
        known = {"name", "sim_config", "sim_overrides", "horizon", "n_learners", "opponents",
                 "agents", "baselines", "train", "eval"}  # fmt: skip
        train, ev = d.get("train") or {}, d.get("eval") or {}
        unknown = (set(d) - known) | (set(train) - {"total_timesteps", "seed", "n_envs", "ppo"})
        unknown |= set(ev) - {"seeds"}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        agents = {k: v or {} for k, v in (d.get("agents") or {}).items()}
        opponents = list(d.get("opponents") or [])
        missing = (set(opponents) | set(d.get("baselines", []))) - set(agents)
        if missing:
            raise ValueError(f"opponents/baselines use undefined agents {sorted(missing)}")
        n_learners = d["n_learners"]
        n_learners = [n_learners] if isinstance(n_learners, int) else list(n_learners)
        seeds = ev.get("seeds", 20)
        return cls(
            name=d["name"],
            sim=load_sim_config(path, d),
            horizon=int(d.get("horizon", 200)),
            n_learners=[int(n) for n in n_learners],
            opponents=opponents,
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
            "n_learners": self.n_learners,
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

    def make_env(self, n: int) -> MultiAgentMEVEnv:
        return MultiAgentMEVEnv(self.sim, n, self.opponents, self.agents, self.horizon)


class ShadingLogger(BaseCallback):
    """Collects per-learner metrics from every finished training episode."""

    def __init__(self, n_learners: int):
        super().__init__()
        self.n = n_learners
        self.rows: list[dict[str, Any]] = []
        self.episodes = 0

    def _on_step(self) -> bool:
        for slot, info in enumerate(self.locals["infos"]):
            if slot % self.n or "agent_metrics" not in info:
                continue  # one record per env copy, taken from its first slot
            ep = info["episode_metrics"]
            for row in info["agent_metrics"][: self.n]:
                self.rows.append(
                    {
                        "timesteps": self.num_timesteps,
                        "episode": self.episodes,
                        "agent": row["agent"],
                        "bid_shading": row["bid_shading"],
                        "win_rate": row["win_rate"],
                        "profit_per_block": row["profit_per_block"],
                        "mev_captured_per_block": ep["mev_captured_per_block"],
                        "mev_to_builder_per_block": ep["mev_to_builder_per_block"],
                    }
                )
            self.episodes += 1
        return True


def train(cfg: MultiPPOConfig, n: int, out: Path):
    """Train one shared policy for N learners; returns (model, shading rows)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import VecMonitor

    from mevsim.envs.vec import SharedPolicyVecEnv

    venv = VecMonitor(SharedPolicyVecEnv([cfg.make_env(n) for _ in range(cfg.n_envs)]))
    model = PPO("MlpPolicy", venv, seed=cfg.train_seed, verbose=0, **cfg.ppo)
    model.set_logger(configure(str(out), ["stdout", "csv"]))
    logger = ShadingLogger(n)
    model.learn(total_timesteps=cfg.total_timesteps, callback=logger)
    model.save(out / "model.zip")
    return model, logger.rows


def evaluate_population(model, cfg: MultiPPOConfig, n: int):
    """All N learners run the shared policy deterministically on each eval seed."""
    env = cfg.make_env(n)
    lineup = ["ppo"] * n + cfg.opponents
    agent_rows, episode_rows = [], []
    for seed in cfg.eval_seeds:
        obs, _ = env.reset(seed=seed)
        while env.agents:
            actions, _ = model.predict(
                np.stack([obs[a] for a in env.possible_agents]), deterministic=True
            )
            obs, _, _, _, infos = env.step(dict(zip(env.possible_agents, actions, strict=True)))
        info = infos[env.possible_agents[0]]
        base = {"n_learners": n, "scenario": "ppo", "seed": seed}
        for row, kind in zip(info["agent_metrics"], lineup, strict=True):
            agent_rows.append({**base, **row, "agent_type": kind})
        episode_rows.append({**base, **info["episode_metrics"]})
    return agent_rows, episode_rows


def evaluate_baselines(cfg: MultiPPOConfig, n: int):
    """Homogeneous baseline populations of size N, same opponents and seeds."""
    agent_rows, episode_rows = [], []
    for kind in cfg.baselines:
        for seed in cfg.eval_seeds:
            rows, ep = run_episode(cfg.sim, [kind] * n + cfg.opponents, cfg.agents,
                                   cfg.horizon, seed)  # fmt: skip
            base = {"n_learners": n, "scenario": kind, "seed": seed}
            agent_rows += [{**base, **r} for r in rows]
            episode_rows.append({**base, **ep})
    return agent_rows, episode_rows


def _population_stats(agent_rows, episode_rows):
    """Per (N, scenario): mean ± CI of per-learner profit/block, bid shading, builder share."""
    stats: dict[tuple[int, str], dict[str, tuple[float, float]]] = {}
    keys = list(dict.fromkeys((r["n_learners"], r["scenario"]) for r in episode_rows))
    for n, s in keys:
        per_seed: dict[int, list[dict]] = {}
        for r in agent_rows:
            if r["n_learners"] == n and r["scenario"] == s and r["agent"] < n:
                per_seed.setdefault(r["seed"], []).append(r)
        profit = [np.mean([r["profit_per_block"] for r in rs]) for rs in per_seed.values()]
        shading = [np.nanmean([r["bid_shading"] for r in rs]) for rs in per_seed.values()]
        eps = [r for r in episode_rows if r["n_learners"] == n and r["scenario"] == s]
        share = [
            e["mev_to_builder_per_block"] / e["mev_captured_per_block"]
            for e in eps
            if e["mev_captured_per_block"] > 0
        ]
        stats[(n, s)] = {
            "profit_per_block": mean_ci(profit),
            "bid_shading": mean_ci(shading),
            "builder_share": mean_ci(share),
        }
    return stats


def summary_table(stats) -> str:
    cols = ("profit_per_block", "bid_shading", "builder_share")
    lines = [f"{'N':<4}{'policy':<14}" + "".join(f"{c:>22}" for c in cols)]
    for (n, s), st in stats.items():
        cells = "".join(f"{st[c][0]:>14.3f} ± {st[c][1]:<5.3f}" for c in cols)
        lines.append(f"{n:<4}{s:<14}{cells}")
    lines.append("\nprofit_per_block and bid_shading are per learner; builder_share is the")
    lines.append("fraction of captured MEV paid to the builder.")
    return "\n".join(lines)


def plot(stats, shading_rows: dict[int, list[dict]], out: Path) -> None:
    """profit_vs_n.png, builder_share_vs_n.png, and shading_training.png."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = sorted({n for n, _ in stats})
    scenarios = list(dict.fromkeys(s for _, s in stats))
    for metric, label, fname in [
        ("profit_per_block", "Profit per block, per learner", "profit_vs_n.png"),
        ("builder_share", "Share of captured MEV paid to builder", "builder_share_vs_n.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for s in scenarios:
            m = [stats[(n, s)][metric][0] for n in ns]
            c = [stats[(n, s)][metric][1] for n in ns]
            ax.errorbar(ns, m, yerr=c, marker="o", capsize=3, label=s)
        ax.set_xticks(ns)
        ax.set_xlabel("Number of agents N")
        ax.set_ylabel(label)
        ax.set_title(f"{label} (mean ± 95% CI across seeds)")
        ax.legend(title="population")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n, rows in sorted(shading_rows.items()):
        by_ep: dict[int, list[tuple[int, float]]] = {}
        for r in rows:
            if r["bid_shading"] == r["bid_shading"]:  # skip NaN
                by_ep.setdefault(r["episode"], []).append((r["timesteps"], r["bid_shading"]))
        if not by_ep:
            continue
        t = np.array([max(x for x, _ in v) for v in by_ep.values()])
        y = np.array([np.mean([s for _, s in v]) for v in by_ep.values()])
        w = max(1, len(y) // 50)
        y_smooth = np.convolve(y, np.ones(w) / w, mode="valid")
        ax.plot(t[w - 1 :], y_smooth, label=f"N={n}")
    ax.set_xlabel("Training timesteps (agent-steps)")
    ax.set_ylabel("Mean bid / value")
    ax.set_title("Bid shading during training (rolling mean over episodes)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "shading_training.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="multi-agent PPO YAML")
    parser.add_argument("--out", default="results", help="root output directory")
    parser.add_argument("--no-plots", action="store_true", help="skip plotting")
    args = parser.parse_args(argv)

    cfg = MultiPPOConfig.from_yaml(args.config)
    out = Path(args.out) / cfg.name / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.resolved(), f, sort_keys=False)

    agent_rows, episode_rows, shading_rows = [], [], {}
    for n in cfg.n_learners:
        print(f"\n=== N = {n} learners ===")
        run_dir = out / f"n{n}"
        run_dir.mkdir()
        model, rows = train(cfg, n, run_dir)
        shading_rows[n] = rows
        if rows:
            with open(run_dir / "shading.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        for a, e in (evaluate_population(model, cfg, n), evaluate_baselines(cfg, n)):
            agent_rows += a
            episode_rows += e

    write_csv(out / "eval_agents.csv", agent_rows)
    write_csv(out / "eval_episodes.csv", episode_rows)
    stats = _population_stats(agent_rows, episode_rows)
    if not args.no_plots:
        plot(stats, shading_rows, out)
    table = summary_table(stats)
    (out / "summary.txt").write_text(table + "\n")
    print("\nHeld-out seeds:\n" + table)
    print(f"\nresults: {out}")
    return out


if __name__ == "__main__":
    main()
