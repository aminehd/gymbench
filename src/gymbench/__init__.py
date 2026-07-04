"""gymbench — a gym experiment toolkit.

Public surface is the core plumbing; tools build on top of it. See
``gymbench.cli`` for the command-line entry point.
"""

from .core import (
    Reel,
    algo,
    env_to_stem,
    evaluate,
    get_algo,
    load,
    loads,
    registered,
    render_gif,
    resolve_sink,
    train,
)

__all__ = [
    "Reel",
    "algo",
    "env_to_stem",
    "evaluate",
    "get_algo",
    "load",
    "loads",
    "registered",
    "render_gif",
    "resolve_sink",
    "train",
]
