"""Built-in algorithm factories.

Each is a thin wrapper over Stable-Baselines3 registered under a short name.
Imports are deferred inside the factories so that merely importing gymbench
doesn't pull in torch/SB3 until an algo is actually used.
"""

from __future__ import annotations

from .registry import algo


@algo("ppo")
def _ppo(env_id: str, hp: dict, seed: int):
    from stable_baselines3 import PPO

    return PPO("MlpPolicy", env_id, seed=seed, verbose=0, **hp)


@algo("dqn")
def _dqn(env_id: str, hp: dict, seed: int):
    from stable_baselines3 import DQN

    return DQN("MlpPolicy", env_id, seed=seed, verbose=0, **hp)


@algo("sac")
def _sac(env_id: str, hp: dict, seed: int):
    from stable_baselines3 import SAC

    return SAC("MlpPolicy", env_id, seed=seed, verbose=0, **hp)
