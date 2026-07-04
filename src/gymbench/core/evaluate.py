"""Greedy sanity metric — trust the number before you trust the GIF.

Runs a few deterministic episodes and returns the mean return. Headless
rendering can't tell you if a policy is good; this can.
"""

from __future__ import annotations

import gymnasium as gym


def evaluate(model, env_id: str, episodes: int = 5, seed: int = 123) -> float:
    env = gym.make(env_id)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    env.close()
    return sum(returns) / len(returns)
