"""Plots and text summaries for baseline experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mevsim.metrics import mean_ci


def _group(rows: list[dict[str, Any]], keys: tuple[str, ...], metric: str) -> dict[tuple, list]:
    """Group a metric by keys. Agents of the same type in one episode are averaged first,
    so each seed contributes one sample."""
    per_seed: dict[tuple, dict[Any, list[float]]] = {}
    for r in rows:
        if np.isnan(r[metric]):
            continue
        per_seed.setdefault(tuple(r[k] for k in keys), {}).setdefault(r["seed"], []).append(
            r[metric]
        )
    return {k: [float(np.mean(v)) for v in seeds.values()] for k, seeds in per_seed.items()}


def summary_table(agent_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]]) -> str:
    """Mean +/- 95% CI per scenario and agent type."""
    keys = ("scenario", "agent_type")
    metrics = ("profit_per_block", "win_rate", "bid_shading")
    groups = {m: _group(agent_rows, keys, m) for m in metrics}
    lines = [f"{'scenario':<16}{'agent':<12}" + "".join(f"{m:>22}" for m in metrics)]
    for k in groups["profit_per_block"]:
        cells = []
        for m in metrics:
            mean, ci = mean_ci(groups[m].get(k, []))
            cells.append(f"{mean:>14.3f} ± {ci:<5.3f}")
        lines.append(f"{k[0]:<16}{k[1]:<12}" + "".join(cells))
    lines.append("")
    lines.append(f"{'scenario':<16}{'mean_latency':>20}{'congestion':>20}")
    lat = _group(episode_rows, ("scenario",), "mean_latency")
    cong = _group(episode_rows, ("scenario",), "mean_congestion")
    for k in lat:
        (lm, lc), (cm, cc) = mean_ci(lat[k]), mean_ci(cong[k])
        lines.append(f"{k[0]:<16}{lm:>12.3f} ± {lc:<5.3f}{cm:>12.3f} ± {cc:<5.3f}")
    return "\n".join(lines)


def plot_all(
    agent_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]], out: Path
) -> None:
    """Write profit.png, latency.png, and bid_shading.png to `out`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = list(dict.fromkeys(r["scenario"] for r in agent_rows))
    types = list(dict.fromkeys(r["agent_type"] for r in agent_rows))

    for metric, label, fname in [
        ("profit_per_block", "Profit per block", "profit.png"),
        ("bid_shading", "Mean bid / value", "bid_shading.png"),
    ]:
        groups = _group(agent_rows, ("scenario", "agent_type"), metric)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        width = 0.8 / len(types)
        x = np.arange(len(scenarios))
        for j, t in enumerate(types):
            stats = [mean_ci(groups.get((s, t), [])) for s in scenarios]
            present = [(s, t) in groups for s in scenarios]
            means = [m if p else 0.0 for (m, _), p in zip(stats, present, strict=True)]
            cis = [c if p else 0.0 for (_, c), p in zip(stats, present, strict=True)]
            ax.bar(x + (j - (len(types) - 1) / 2) * width, means, width, yerr=cis,
                   capsize=3, label=t)  # fmt: skip
        ax.set_xticks(x, scenarios)
        ax.set_ylabel(label)
        ax.set_title(f"{label} by scenario (mean ± 95% CI across seeds)")
        ax.legend(title="agent")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)

    lat = _group(episode_rows, ("scenario",), "mean_latency")
    stats = [mean_ci(lat.get((s,), [])) for s in scenarios]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(scenarios, [m for m, _ in stats], yerr=[c for _, c in stats], capsize=3,
           color="#4C72B0")  # fmt: skip
    ax.set_ylabel("Blocks from arrival to inclusion")
    ax.set_title("Organic tx inclusion latency (mean ± 95% CI across seeds)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "latency.png", dpi=150)
    plt.close(fig)
