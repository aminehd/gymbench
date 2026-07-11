"""Roll out a policy and save a GIF in Gymnasium's exact docs format.

Mirrors ``docs/_scripts/gen_gifs.py`` from Gymnasium: 300 frames rendered with
render_mode="rgb_array", saved via PIL at duration=50, loop=0. The only
substantive difference from upstream is that the action comes from a trained
policy's ``predict`` instead of ``env.action_space.sample()``.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

# Match docs/_scripts/gen_gifs.py exactly.
LENGTH = 300


def _reset(env, start_state, seed=None):
    """Reset, optionally forcing a specific initial state (e.g. a tilted pole).

    For classic-control envs the observation is the internal state, so we can
    drop the agent into a harder starting pose to make the policy's recovery
    visible. Only meaningful for envs where obs == state (CartPole, etc.).
    """
    kw = {"seed": seed} if seed is not None else {}
    obs, _ = env.reset(**kw)
    if start_state is not None:
        env.state = np.array(start_state, dtype=np.float64)
        obs = np.array(start_state, dtype=np.float32)
    return obs


def render_gif(model, env_id: str, out_path: str | Path, seed: int = 42,
               hold_on_success: int = 0, start_state=None,
               deterministic: bool = True) -> Path:
    # .unwrapped drops the TimeLimit wrapper exactly as gen_gifs.py does, so a
    # policy that never terminates renders continuously (no mid-GIF reset).
    env = gym.make(env_id, render_mode="rgb_array").unwrapped
    obs = _reset(env, start_state, seed=seed)

    # Mirror gen_gifs.py's loop (`while len(frames) <= LENGTH`, 301 frames); the
    # changed line is the action source. For envs that end on success (Acrobot,
    # MountainCar) the terminal "success" pose is otherwise skipped by the reset,
    # so hold_on_success draws it and lingers, making the achievement visible.
    frames: list[Image.Image] = []
    while len(frames) <= LENGTH:
        frames.append(Image.fromarray(env.render()))

        # The one line that differs from upstream: trained policy, not random.
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            if terminated and hold_on_success:
                success_frame = Image.fromarray(env.render())  # the pose we just reached
                frames.extend([success_frame] * hold_on_success)
            obs = _reset(env, start_state)
    env.close()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
    )
    return out
