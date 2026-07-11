"""Tool #1 — trained-policy GIFs.

Wires the core plumbing together: train a policy, print a greedy sanity metric,
render 300 frames, and drop the GIF at its resolved sink. One reel in, one GIF
out. Used both by the batch runner and the one-off CLI path.
"""

from __future__ import annotations

from pathlib import Path

from ...core import Reel, evaluate, load, render_gif, resolve_sink, train


def load_bundled(name: str = "classic_control") -> list[Reel]:
    """Load a config from ``configs/<tool>/<name>.toml`` (mirrors this tool's path).

    ``configs/`` lives at the package root, outside src, so each tool's configs
    sit under a folder matching its ``tools/<tool>/`` package.
    """
    tool = __name__.rsplit(".", 1)[-1]  # "gif"
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return load(parent / "configs" / tool / f"{name}.toml")
    raise FileNotFoundError(f"could not locate configs/{tool}/{name}.toml")


def make(reel: Reel) -> Path:
    print(f"[{reel.env}] training {reel.algo} for {reel.steps} steps (seed={reel.seed})")
    model = train(reel.env, reel.algo, reel.hp, reel.steps, reel.seed)

    mean_return = evaluate(model, reel.env)
    print(f"[{reel.env}] mean greedy return (5 eps): {mean_return:.1f}")

    out = resolve_sink(reel.sink, reel.group, reel.filename)
    render_seed = reel.render_seed if reel.render_seed is not None else reel.seed + 42
    render_gif(model, reel.env, out, seed=render_seed, hold_on_success=reel.hold,
               start_state=reel.start_state)
    print(f"[{reel.env}] saved -> {out}")
    return out
