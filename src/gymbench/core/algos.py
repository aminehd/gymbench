"""Built-in algorithm factories.

Each is a thin wrapper over Stable-Baselines3 registered under a short name.
Imports are deferred inside the factories so that merely importing gymbench
doesn't pull in torch/SB3 until an algo is actually used.
"""

from __future__ import annotations

from .registry import algo


def _split_policy(hp: dict, default: str = "MlpPolicy") -> tuple[str, dict]:
    """Pop the ``policy`` name out of hp (default MlpPolicy) so image-based envs
    can request ``CnnPolicy`` from the config without a code change."""
    hp = dict(hp)
    return hp.pop("policy", default), hp


@algo("ppo")
def _ppo(env_id: str, hp: dict, seed: int):
    from stable_baselines3 import PPO

    policy, hp = _split_policy(hp)
    return PPO(policy, env_id, seed=seed, verbose=0, **hp)


@algo("dqn")
def _dqn(env_id: str, hp: dict, seed: int):
    from stable_baselines3 import DQN

    policy, hp = _split_policy(hp)
    return DQN(policy, env_id, seed=seed, verbose=0, **hp)


@algo("sac")
def _sac(env_id: str, hp: dict, seed: int):
    from stable_baselines3 import SAC

    policy, hp = _split_policy(hp)
    return SAC(policy, env_id, seed=seed, verbose=0, **hp)
