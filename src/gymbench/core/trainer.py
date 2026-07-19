"""Train a policy — env-agnostic, algo-agnostic.

Resolves the algo by name via the registry, builds the model, and runs
``.learn()``. Knows nothing about specific environments or GIFs.
"""

from __future__ import annotations

from .registry import get_algo


def build_env(env_id: str, n_envs: int, frame_stack: int, seed: int):
    """Return what SB3 should train on.

    For the simple case (one env, no stacking) we hand SB3 the bare id and let
    it build the env itself. Image envs like CarRacing need several parallel
    envs and a stack of recent frames (so the policy can perceive motion), so
    we build a ``VecFrameStack`` over a ``SubprocVecEnv`` and pass that instead.
    """
    if n_envs <= 1 and not frame_stack:
        return env_id

    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

    cls = SubprocVecEnv if n_envs > 1 else None
    venv = make_vec_env(env_id, n_envs=max(1, n_envs), seed=seed, vec_env_cls=cls)
    if frame_stack:
        venv = VecFrameStack(venv, frame_stack)
    return venv


def train(env_id: str, algo_name: str, hp: dict | None, timesteps: int, seed: int,
          n_envs: int = 1, frame_stack: int = 0):
    env = build_env(env_id, n_envs, frame_stack, seed)
    model = get_algo(algo_name)(env, hp or {}, seed)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    return model
