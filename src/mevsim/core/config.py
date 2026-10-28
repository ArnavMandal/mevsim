"""Simulator configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SimConfig:
    """Parameters for the core simulator. Defaults mirror `configs/default.yaml`."""

    gas_limit: int = 30
    n_agents: int = 3
    mechanism: str = "sealed_first_price"
    organic_rate: float = 6.0
    organic_gas_min: int = 1
    organic_gas_max: int = 4
    organic_fee_scale: float = 1.0
    organic_fee_sigma: float = 0.5
    congestion_sensitivity: float = 0.5
    congestion_smoothing: float = 0.3
    opportunity_rate: float = 0.8
    opportunity_value_scale: float = 5.0
    opportunity_value_alpha: float = 2.5
    opportunity_ttl: int = 2
    agent_tx_gas: int = 2
    initial_balance: float = 100.0
    p_see: float = 0.8
    obs_noise_std: float = 0.1
    top_k: int = 5

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SimConfig:
        """Build from a dict, rejecting unknown keys so typos fail loudly."""
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SimConfig:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
