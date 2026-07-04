# gymbench

A small, extensible toolkit for gym experiments. First tool: **`env-gif`** — train
a policy and render it as a GIF in Gymnasium's exact docs format (the thing that
replaces the random-agent GIFs in [issue #1610](https://github.com/Farama-Foundation/Gymnasium/issues/1610)).

## Install

```bash
pip install -e packages/gymbench      # from the gymlab monorepo root
```

## Use

```bash
# One command — train + render ALL classic-control envs into the gym fork
gymbench env-gif run

# Same, naming the bundled set explicitly
gymbench env-gif run --config classic_control

# Your own config
gymbench env-gif run path/to/my.toml

# One-off — from flags
gymbench env-gif run --env CartPole-v1 --algo ppo --steps 120000

# Render somewhere local instead of into the docs tree
gymbench env-gif run --env CartPole-v1 --algo ppo --sink ./out

# What algorithms are registered?
gymbench algos
```

Configs live at the package root under `configs/<tool>/`, mirroring
`src/gymbench/tools/<tool>/`. The env-gif tool's sets are in `configs/env_gif/`, so
`gymbench env-gif run` (which loads `configs/env_gif/classic_control.toml`) works
without pointing at a file.

## How it's built

Three ideas do all the work:

- **Registry (`@algo`)** — the one extension point. New algorithms self-register;
  the core never changes.
  ```python
  from gymbench import algo

  @algo("my_algo")
  def build(env_id, hp, seed):
      return MyModel("MlpPolicy", env_id, seed=seed, **hp)
  ```
- **Declarative config** — one TOML file is one reproducible batch. Env ids map to
  snake_case filenames by convention (`CartPole-v1` → `cart_pole.gif`).
- **Sinks** — `sink = "gymnasium"` writes into the bundled fork at
  `external/Gymnasium/docs/_static/videos/<group>/<name>.gif`; any other value is
  a plain output directory.

## Adding more environments

The framework is env-agnostic — **new environments need a new config, not new code.**

1. Add a TOML under `configs/env_gif/`, e.g. `box2d.toml`. The `group` field
   routes GIFs to the matching docs subfolder; filenames auto-derive from the env id.
   ```toml
   [defaults]
   sink = "gymnasium"
   group = "box2d"                 # -> docs/_static/videos/box2d/<name>.gif

   [[reel]]
   env = "LunarLander-v3"          # -> lunar_lander.gif
   algo = "dqn"
   steps = 500_000
   ```
2. Run it: `gymbench env-gif run --config box2d`.

Two things occasionally need more than a config — neither touches the core:

- **A new algorithm** (not `ppo`/`dqn`/`sac`) → add one factory in `core/algos.py`:
  ```python
  @algo("td3")
  def _td3(env_id, hp, seed):
      from stable_baselines3 import TD3
      return TD3("MlpPolicy", env_id, seed=seed, verbose=0, **hp)
  ```
- **System deps** for some env families — install the extra, then it just works:
  ```bash
  pip install "gymnasium[box2d]"   # Box2D (LunarLander, BipedalWalker, CarRacing)
  pip install "gymnasium[mujoco]"  # MuJoCo (Ant, HalfCheetah, Humanoid, …)
  ```

So: **new envs = new TOML. New algo = one decorator. The registry / config / sink
stay fixed.**

## Layout

```
configs/
  env_gif/
    classic_control.toml   config sets for the env-gif tool (mirrors tools/env_gif/)
src/gymbench/
  core/
    registry.py   @algo decorator + lookup — the extension point
    algos.py      built-in ppo / dqn / sac factories
    trainer.py    train(spec) -> model            (env-agnostic)
    render.py     model -> 300 frames -> GIF       (matches gen_gifs.py)
    evaluate.py   greedy mean return               (trust the number, then the GIF)
    sink.py       where the GIF lands
    config.py     TOML -> typed Reel specs
  tools/
    env_gif/      tool #1 — wires the core together (owns configs/env_gif/)
  cli.py          `gymbench env-gif run …`
```

A tool is a folder under `tools/<name>/` with a matching `configs/<name>/`; it
reuses `core/`. That's the whole framework.

> The CLI command is hyphenated (`env-gif`); the package folder is `env_gif`
> because Python import names can't contain hyphens.
