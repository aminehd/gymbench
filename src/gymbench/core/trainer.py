"""Train a policy — env-agnostic, algo-agnostic.

Resolves the algo by name via the registry, builds the model, and runs
``.learn()``. Knows nothing about specific environments or GIFs.
"""

from __future__ import annotations

from .registry import get_algo


def train(env_id: str, algo_name: str, hp: dict | None, timesteps: int, seed: int):
    model = get_algo(algo_name)(env_id, hp or {}, seed)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    return model
