"""Declarative config: one TOML file describes a batch of reels.

A *reel* is a single (env, algo) job that produces one GIF. The file has an
optional ``[defaults]`` table and a list of ``[[reel]]`` entries; each reel
inherits the defaults and overrides what it needs. Env ids map to snake_case
GIF filenames by convention, so you rarely specify the output name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

try:  # stdlib on 3.11+, backport otherwise
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


@dataclass
class Reel:
    env: str
    algo: str
    steps: int = 100_000
    seed: int = 0
    group: str = "classic_control"
    sink: str = "gymnasium"
    hp: dict = field(default_factory=dict)
    name: str | None = None  # override filename stem; default derived from env

    @property
    def filename(self) -> str:
        stem = self.name or env_to_stem(self.env)
        return f"{stem}.gif"


def env_to_stem(env_id: str) -> str:
    """``CartPole-v1`` -> ``cart_pole``; ``MountainCarContinuous-v0`` -> ``mountain_car_continuous``."""
    base = re.sub(r"-v\d+$", "", env_id)
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", base).lower()
    return snake


def loads(text: str) -> list[Reel]:
    data = tomllib.loads(text)
    defaults = data.get("defaults", {})
    reels = []
    for entry in data.get("reel", []):
        merged = {**defaults, **entry}
        reels.append(Reel(**merged))
    return reels


def load(path: str | Path) -> list[Reel]:
    return loads(Path(path).read_text())
