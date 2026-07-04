"""gymbench command line — git-style subcommands, one per tool.

    gymbench env-gif run                               # ALL classic control (bundled config)
    gymbench env-gif run --config classic_control      # same, by name
    gymbench env-gif run path/to/my.toml               # batch from your own config
    gymbench env-gif run --env Acrobot-v1 --algo dqn    # one-off from flags
    gymbench algos                                      # list registered algorithms

Thin by design: it parses args into a Reel (or loads them from TOML) and hands
off to the tool. All real work lives in the tools/ and core/ modules.

The user-facing command is hyphenated (``env-gif``) while the tool package is
``env_gif`` — Python identifiers can't contain hyphens.
"""

from __future__ import annotations

import argparse

from .core import Reel, load, registered
from .tools import env_gif


def _add_reel_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("config", nargs="?", help="TOML config path; omit to use a bundled set")
    p.add_argument("--config", dest="named", metavar="NAME",
                   help="bundled config by name (default: classic_control)")
    p.add_argument("--env", help="Gymnasium env id, e.g. CartPole-v1 (one-off mode)")
    p.add_argument("--algo", help=f"one of: {', '.join(registered())}")
    p.add_argument("--steps", type=int, help="training timesteps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--group", default="classic_control", help="env group -> docs subfolder")
    p.add_argument("--sink", default="gymnasium", help='"gymnasium" or an output directory')
    p.add_argument("--name", help="override GIF filename stem")


def _reels_from_args(args) -> list[Reel]:
    # A config file path wins if given.
    if args.config:
        return load(args.config)
    # One-off mode: both --env and --algo.
    if args.env and args.algo:
        return [_single_reel(args)]
    # Otherwise run a bundled batch (default: the full classic-control set).
    return env_gif.load_bundled(args.named or "classic_control")


def _single_reel(args) -> Reel:
    kw = dict(env=args.env, algo=args.algo, seed=args.seed, group=args.group, sink=args.sink)
    if args.steps:
        kw["steps"] = args.steps
    if args.name:
        kw["name"] = args.name
    return Reel(**kw)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gymbench", description="A gym experiment toolkit.")
    sub = parser.add_subparsers(dest="tool", required=True)

    gif_p = sub.add_parser("env-gif", help="render a trained policy as a GIF")
    gif_sub = gif_p.add_subparsers(dest="action", required=True)
    run_p = gif_sub.add_parser("run", help="train + render one or many reels")
    _add_reel_flags(run_p)

    sub.add_parser("algos", help="list registered algorithms")

    args = parser.parse_args(argv)

    if args.tool == "algos":
        print("\n".join(registered()))
        return

    if args.tool == "env-gif" and args.action == "run":
        for reel in _reels_from_args(args):
            env_gif.make(reel)


if __name__ == "__main__":
    main()
