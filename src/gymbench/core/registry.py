"""The one extension point.

Algorithms register themselves under a short name with the ``@algo`` decorator;
everything else (the trainer, the CLI, configs) looks them up by that name and
never needs to know the concrete class. Add a new algorithm by writing a factory
and decorating it — the core stays closed for modification, open for extension.
"""

from __future__ import annotations

from typing import Callable, Dict

# name -> factory(env_id, hp, seed) -> a trained-or-trainable SB3 model
_ALGOS: Dict[str, Callable] = {}


def algo(name: str) -> Callable[[Callable], Callable]:
    """Register a model factory under ``name``.

    The factory takes ``(env_id, hp, seed)`` and returns an SB3 model ready to
    ``.learn()``. Example::

        @algo("ppo")
        def _ppo(env_id, hp, seed):
            from stable_baselines3 import PPO
            return PPO("MlpPolicy", env_id, seed=seed, verbose=0, **hp)
    """

    def register(fn: Callable) -> Callable:
        _ALGOS[name] = fn
        return fn

    return register


def get_algo(name: str) -> Callable:
    if name not in _ALGOS:
        raise KeyError(
            f"unknown algo {name!r}; registered: {', '.join(registered()) or '(none)'}"
        )
    return _ALGOS[name]


def registered() -> list[str]:
    return sorted(_ALGOS)
