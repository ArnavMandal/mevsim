"""Compare auction mechanisms on profit, efficiency and welfare (M5).

    python -m mevsim.experiments.mechanisms --config configs/mechanisms.yaml

Every scenario (a lineup of baseline agents) runs under every mechanism on the same seeds,
plus a PGA rounds sweep. Welfare loss is measured against a no-MEV counterfactual: the same
seed with no agent bidding. Organic arrivals use their own random stream, so the
counterfactual sees the same users. Per block:

    extra_wait   = organic gas-blocks of waiting beyond the counterfactual
    delay_loss   = delay_cost * extra_wait
    welfare_loss = delay_loss + MEV paid to the builder   (SPEC section 6)

Optionally a shared-policy PPO population is trained under each mechanism (scenario
"ppo"). Outputs in `<out>/<name>/<timestamp>/`: resolved config, `agents.csv`,
`episodes.csv`, `summary.md`, `mechanisms.png`, `pga_rounds.png`, and `ppo_<mechanism>/`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mevsim.core.config import SimConfig
from mevsim.episode import seeded_episode
from mevsim.experiments.run import load_sim_config, run_episode, write_csv
from mevsim.metrics import EpisodeRecorder, mean_ci

# Categorical slots 1-5 of the reference palette, validated for CVD separation.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]


@dataclass
class MechanismsConfig:
    """Resolved settings for the mechanism comparison."""

    name: str
    sim: SimConfig
    horizon: int
    seeds: list[int]
    mechanisms: list[str]
    pga_rounds_sweep: list[int]
    delay_cost: float
    agents: dict[str, dict[str, Any]]
    scenarios: dict[str, list[str]]
    ppo: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> MechanismsConfig:
        path = Path(path)
        with open(path) as f:
            d = yaml.safe_load(f)
        known = {
            "name",
            "sim_config",
            "sim_overrides",
            "horizon",
            "seeds",
            "mechanisms",
            "pga_rounds_sweep",
            "delay_cost",
            "agents",
            "scenarios",
            "ppo",
        }
        ppo = d.get("ppo") or {}
        unknown = (set(d) - known) | (set(ppo) - {"enabled", "n_learners", "train"})
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        from mevsim.core.auction import MECHANISMS

        mechanisms = list(d["mechanisms"])
        bad = set(mechanisms) - set(MECHANISMS)
        if bad:
            raise ValueError(f"unknown mechanisms {sorted(bad)}")
        agents = {k: v or {} for k, v in d["agents"].items()}
        for name, lineup in d["scenarios"].items():
            missing = set(lineup) - set(agents)
            if missing:
                raise ValueError(f"scenario {name!r} uses undefined agents {sorted(missing)}")
        seeds = d.get("seeds", 20)
        return cls(
            name=d["name"],
            sim=load_sim_config(path, d),
            horizon=int(d.get("horizon", 200)),
            seeds=list(range(seeds)) if isinstance(seeds, int) else list(seeds),
            mechanisms=mechanisms,
            pga_rounds_sweep=[int(r) for r in d.get("pga_rounds_sweep") or []],
            delay_cost=float(d.get("delay_cost", 0.1)),
            agents=agents,
            scenarios=dict(d["scenarios"]),
            ppo=dict(ppo),
        )

    def resolved(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sim": self.sim.to_dict(),
            "horizon": self.horizon,
            "seeds": self.seeds,
            "mechanisms": self.mechanisms,
            "pga_rounds_sweep": self.pga_rounds_sweep,
            "delay_cost": self.delay_cost,
            "agents": self.agents,
            "scenarios": self.scenarios,
            "ppo": self.ppo,
        }

    def runs(self) -> list[tuple[str, SimConfig]]:
        """(label, sim config) per mechanism, then `pga_r<R>` for each extra sweep value."""
        runs = [(m, replace(self.sim, mechanism=m)) for m in self.mechanisms]
        for r in self.pga_rounds_sweep:
            if not ("pga" in self.mechanisms and r == self.sim.pga_rounds):
                runs.append((f"pga_r{r}", replace(self.sim, mechanism="pga", pga_rounds=r)))
        return runs


def counterfactual(sim_config: SimConfig, n_agents: int, horizon: int, seed: int) -> dict:
    """Episode metrics for the same seed with no agent bidding (no MEV competition)."""
    sim, _ = seeded_episode(sim_config, [None] * n_agents, {}, seed)
    recorder = EpisodeRecorder(n_agents, sim.config.gas_limit)
    for _ in range(horizon):
        for _ in range(sim.config.n_rounds - 1):
            sim.bid_round([])
        result = sim.step([])
        recorder.record([], {}, result, sim.mempool.pending_gas)
    return recorder.episode_summary()


def add_welfare(ep: dict[str, Any], cf: dict[str, Any], gas_limit: int, delay_cost: float):
    """Add builder share, extra waiting, delay loss and welfare loss to an episode row."""
    captured = ep["mev_captured_per_block"]
    extra_wait = gas_limit * (ep["mean_congestion"] - cf["mean_congestion"])
    delay_loss = delay_cost * extra_wait
    return {
        **ep,
        "builder_share": ep["mev_to_builder_per_block"] / captured if captured else float("nan"),
        "extra_wait_per_block": extra_wait,
        "delay_loss_per_block": delay_loss,
        "welfare_loss_per_block": delay_loss + ep["mev_to_builder_per_block"],
        # Not part of the SPEC welfare loss, but reported: MEV some agent saw yet nobody got.
        "uncaptured_mev_per_block": ep["mev_seen_per_block"] - captured,
    }


def run_comparison(cfg: MechanismsConfig, out: Path | None = None):
    """Run every (mechanism, scenario, seed); returns (agent rows, episode rows)."""
    agent_rows, episode_rows = [], []
    cf_cache: dict[tuple[int, int], dict] = {}

    def cf(n: int, seed: int) -> dict:
        if (n, seed) not in cf_cache:
            cf_cache[(n, seed)] = counterfactual(cfg.sim, n, cfg.horizon, seed)
        return cf_cache[(n, seed)]

    def add(label, sim, scenario, seed, n, rows, ep):
        base = {"mechanism": label, "pga_rounds": sim.n_rounds, "scenario": scenario, "seed": seed}
        agent_rows.extend({**base, **r} for r in rows)
        ep = add_welfare(ep, cf(n, seed), sim.gas_limit, cfg.delay_cost)
        episode_rows.append(
            {**base, "agent_profit_per_agent": ep["agent_profit_per_block"] / n, **ep}
        )

    for label, sim in cfg.runs():
        print(f"mechanism {label}")
        for scenario, lineup in cfg.scenarios.items():
            for seed in cfg.seeds:
                rows, ep = run_episode(sim, lineup, cfg.agents, cfg.horizon, seed)
                for row, kind in zip(rows, lineup, strict=True):
                    row["agent_type"] = kind
                add(label, sim, scenario, seed, len(lineup), rows, ep)

    if cfg.ppo.get("enabled"):
        for label, sim in cfg.runs()[: len(cfg.mechanisms)]:
            print(f"training PPO population under {label}")
            n, rows_by_seed = int(cfg.ppo.get("n_learners", 3)), train_ppo(cfg, sim, label, out)
            for seed, (rows, ep) in rows_by_seed.items():
                add(label, sim, "ppo", seed, n, rows, ep)
    return agent_rows, episode_rows


def train_ppo(cfg: MechanismsConfig, sim: SimConfig, label: str, out: Path | None):
    """Train a shared-policy population under one mechanism; eval rows keyed by seed."""
    from mevsim.training.train_ppo_multi import MultiPPOConfig, evaluate_population, train

    t = cfg.ppo.get("train") or {}
    n = int(cfg.ppo.get("n_learners", 3))
    mcfg = MultiPPOConfig(
        name=f"{cfg.name}_ppo_{label}",
        sim=sim,
        horizon=cfg.horizon,
        n_learners=[n],
        opponents=[],
        agents=cfg.agents,
        baselines=[],
        total_timesteps=int(t["total_timesteps"]),
        train_seed=int(t.get("seed", 0)),
        n_envs=int(t.get("n_envs", 1)),
        ppo=dict(t.get("ppo") or {}),
        eval_seeds=cfg.seeds,
    )
    run_dir = (out or Path(".")) / f"ppo_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model, shading = train(mcfg, n, run_dir)
    if shading:
        write_csv(run_dir / "shading.csv", shading)
    agent_rows, episode_rows = evaluate_population(model, mcfg, n)
    strip = {"n_learners", "scenario", "seed"}
    by_seed = {}
    for e in episode_rows:
        rows = [
            {k: v for k, v in r.items() if k not in strip}
            for r in agent_rows
            if r["seed"] == e["seed"]
        ]
        by_seed[e["seed"]] = (rows, {k: v for k, v in e.items() if k not in strip})
    return by_seed


METRICS = [
    ("agent_profit_per_agent", "Profit / agent / block"),
    ("builder_share", "Builder share of MEV"),
    ("efficiency", "Efficiency"),
    ("extra_wait_per_block", "Extra organic wait"),
    ("welfare_loss_per_block", "Welfare loss / block"),
    ("uncaptured_mev_per_block", "Uncaptured MEV / block"),
]


def aggregate(episode_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict]:
    """Mean ± 95% CI of each metric per (mechanism label, scenario)."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in episode_rows:
        groups.setdefault((r["mechanism"], r["scenario"]), []).append(r)
    stats = {}
    for key, rows in groups.items():
        stats[key] = {
            m: mean_ci([r[m] for r in rows if r[m] == r[m]])
            for m, _ in METRICS  # skip NaN
        }
        stats[key]["pga_rounds"] = rows[0]["pga_rounds"]
    return stats


def pga_sweep(stats) -> dict[int, str]:
    """PGA rounds -> mechanism label, covering the main `pga` run and the sweep runs."""
    return {
        st["pga_rounds"]: lbl
        for (lbl, _), st in stats.items()
        if lbl == "pga" or lbl.startswith("pga_r")
    }


def summary_markdown(cfg: MechanismsConfig, stats) -> str:
    header = "| mechanism | scenario | " + " | ".join(lbl for _, lbl in METRICS) + " |"
    sep = "|---" * (len(METRICS) + 2) + "|"

    def row(label, scenario):
        st = stats[(label, scenario)]
        cells = " | ".join(f"{st[m][0]:.3f} ± {st[m][1]:.3f}" for m, _ in METRICS)
        return f"| {label} | {scenario} | {cells} |"

    scenarios = list(dict.fromkeys(s for _, s in stats))
    lines = [
        f"# Mechanism comparison ({len(cfg.seeds)} seeds x {cfg.horizon} blocks)",
        "",
        header,
        sep,
    ]
    for m in cfg.mechanisms:
        lines += [row(m, s) for s in scenarios if (m, s) in stats]
    sweep = pga_sweep(stats)
    if len(sweep) > 1:
        lines += ["", "## PGA rounds sweep", "", header, sep]
        lines += [
            row(sweep[r], s) for r in sorted(sweep) for s in scenarios if (sweep[r], s) in stats
        ]
    lines += [
        "",
        "Mean ± 95% CI across seeds. Profit is per agent; builder share is the fraction of",
        "captured MEV paid to the builder; efficiency is captured MEV / MEV some agent saw;",
        "extra organic wait is gas-blocks of waiting per block beyond the no-MEV",
        f"counterfactual; welfare loss = {cfg.delay_cost} x extra wait + MEV paid to builder",
        "(SPEC definition; uncaptured MEV is reported separately, not included).",
    ]
    return "\n".join(lines)


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c9c8c2")
    ax.grid(axis="y", color="#e8e7e2", linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#55554f", labelsize=8)


def plot(cfg: MechanismsConfig, stats, out: Path) -> None:
    """mechanisms.png (grouped bars per metric) and pga_rounds.png (metric vs rounds)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = list(dict.fromkeys(s for _, s in stats))
    panels = [
        m
        for m in METRICS
        if m[0]
        in {"agent_profit_per_agent", "builder_share", "efficiency", "welfare_loss_per_block"}
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2.6 * len(panels)), sharex=True)
    width = 0.8 / len(cfg.mechanisms)
    x = np.arange(len(scenarios))
    for ax, (metric, label) in zip(axes, panels, strict=True):
        for j, mech in enumerate(cfg.mechanisms):
            vals = [stats.get((mech, s), {}).get(metric, (np.nan, 0)) for s in scenarios]
            ax.bar(
                x + (j - (len(cfg.mechanisms) - 1) / 2) * width,
                [v[0] for v in vals],
                width * 0.92,
                yerr=[v[1] for v in vals],
                color=COLORS[j % len(COLORS)],
                error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "#55554f"},
                label=mech,
            )
        ax.set_ylabel(label, fontsize=9, color="#2b2b28")
        _style(ax)
    axes[0].legend(
        title="mechanism",
        fontsize=8,
        title_fontsize=8,
        frameon=False,
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
    )
    axes[-1].set_xticks(x, [s.replace("_", " ") for s in scenarios], fontsize=8)
    fig.suptitle(f"Auction mechanisms (mean ± 95% CI, {len(cfg.seeds)} seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "mechanisms.png", dpi=150)
    plt.close(fig)

    sweep = pga_sweep(stats)
    if len(sweep) < 2:
        return
    rounds = sorted(sweep)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (metric, label) in zip(
        axes,
        [
            ("builder_share", "Builder share of MEV"),
            ("agent_profit_per_agent", "Profit / agent / block"),
        ],
        strict=True,
    ):
        for j, s in enumerate(scenarios):
            pts = [stats.get((sweep[r], s), {}).get(metric) for r in rounds]
            if any(p is None for p in pts):
                continue
            c = COLORS[j % len(COLORS)]
            ax.errorbar(
                rounds,
                [p[0] for p in pts],
                yerr=[p[1] for p in pts],
                color=c,
                marker=MARKERS[j % len(MARKERS)],
                markersize=6,
                linewidth=2,
                capsize=2,
                label=s.replace("_", " "),
            )
        ax.set_xticks(rounds)
        ax.set_xlabel("PGA bidding rounds per block", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.margins(x=0.1)
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("PGA: effect of more bidding rounds (mean ± 95% CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "pga_rounds.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="mechanisms YAML")
    parser.add_argument("--out", default="results", help="root output directory")
    parser.add_argument("--no-plots", action="store_true", help="skip plotting")
    args = parser.parse_args(argv)

    cfg = MechanismsConfig.from_yaml(args.config)
    out = Path(args.out) / cfg.name / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.resolved(), f, sort_keys=False)

    agent_rows, episode_rows = run_comparison(cfg, out)
    write_csv(out / "agents.csv", agent_rows)
    write_csv(out / "episodes.csv", episode_rows)
    stats = aggregate(episode_rows)
    table = summary_markdown(cfg, stats)
    (out / "summary.md").write_text(table + "\n")
    if not args.no_plots:
        plot(cfg, stats, out)
    print("\n" + table)
    print(f"\nresults: {out}")
    return out


if __name__ == "__main__":
    main()
