"""Roll out a policy and save a GIF in Gymnasium's exact docs format.

Mirrors ``docs/_scripts/gen_gifs.py`` from Gymnasium: 300 frames rendered with
render_mode="rgb_array", saved via PIL at duration=50, loop=0. The only
substantive difference from upstream is that the action comes from a trained
policy's ``predict`` instead of ``env.action_space.sample()``.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from PIL import Image

# Match docs/_scripts/gen_gifs.py exactly.
LENGTH = 300


def render_gif(model, env_id: str, out_path: str | Path, seed: int = 42) -> Path:
    # .unwrapped drops the TimeLimit wrapper exactly as gen_gifs.py does, so a
    # policy that never terminates renders continuously (no mid-GIF reset).
    env = gym.make(env_id, render_mode="rgb_array").unwrapped
    obs, _ = env.reset(seed=seed)

    # Mirror gen_gifs.py's loop verbatim (`while len(frames) <= LENGTH`, i.e.
    # 301 frames). The only changed line is the action source.
    frames: list[Image.Image] = []
    while len(frames) <= LENGTH:
        frames.append(Image.fromarray(env.render()))

        # The one line that differs from upstream: trained policy, not random.
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
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
