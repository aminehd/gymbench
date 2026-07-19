"""Greedy sanity metric — trust the number before you trust the GIF.

Runs a few deterministic episodes and returns the mean return. Headless
rendering can't tell you if a policy is good; this can.
"""

from __future__ import annotations

import gymnasium as gym

from .framestack import FrameStacker


def evaluate(model, env_id: str, episodes: int = 5, seed: int = 123,
             frame_stack: int = 0) -> float:
    env = gym.make(env_id)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        stacker = FrameStacker(frame_stack, obs) if frame_stack else None
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(stacker.observe() if stacker else obs,
                                      deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if stacker:
                stacker.push(obs)
            total += reward
            done = terminated or truncated
        returns.append(total)
    env.close()
    return sum(returns) / len(returns)
