"""gymbench core — shared plumbing every tool reuses."""

# Importing algos here ensures the built-in factories self-register as soon as
# the core package is imported.
from . import algos  # noqa: F401
from .config import Reel, env_to_stem, load, loads
from .evaluate import evaluate
from .registry import algo, get_algo, registered
from .render import render_gif
from .sink import resolve as resolve_sink
from .trainer import train

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
