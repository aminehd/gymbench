# gymbench

Train a policy and render it as a GIF in Gymnasium's docs format.
CLI: `gymbench env-gif run`.

## Install

```bash
pip install -e .
```

## One config → its GIFs

A config is a TOML file listing reels — one `[[reel]]` per env. Running the config
trains a policy for each and renders its GIF:

```toml
# configs/env_gif/classic_control.toml
[defaults]
sink = "gymnasium"        # where GIFs are written
group = "classic_control" # docs subfolder

[[reel]]
env = "CartPole-v1"       # -> cart_pole.gif
algo = "ppo"              # ppo | dqn | sac
steps = 120_000

[[reel]]
env = "MountainCar-v0"
algo = "dqn"
steps = 120_000
hp = { learning_rate = 4e-3, gamma = 0.98 }   # optional per-reel hyperparams
```

```bash
gymbench env-gif run              # runs the bundled classic_control config -> all its GIFs
```

## Generate for other envs

Write another config and run it — no code changes.

```toml
# configs/env_gif/box2d.toml
[defaults]
group = "box2d"

[[reel]]
env = "LunarLander-v3"    # -> lunar_lander.gif
algo = "dqn"
steps = 500_000
```

```bash
gymbench env-gif run configs/env_gif/box2d.toml   # by path
gymbench env-gif run --config box2d                # or by name (configs/env_gif/box2d.toml)
```

## One-off from flags

```bash
gymbench env-gif run --env CartPole-v1 --algo ppo --steps 120000
gymbench env-gif run --env CartPole-v1 --algo ppo --sink ./out   # render locally
gymbench algos                                                    # list algorithms
```

## Flags

| flag | meaning |
|------|---------|
| `config` (positional) | path to a TOML config |
| `--config NAME` | bundled config by name (`configs/env_gif/NAME.toml`) |
| `--env` / `--algo` | one-off mode instead of a config |
| `--steps` | training timesteps |
| `--sink` | `"gymnasium"` (docs tree) or an output folder |
| `--group` | docs subfolder / env group |

## Layout

```
configs/env_gif/     config sets (mirror tools/)
src/gymbench/
  core/              registry · trainer · render · sink · config
  tools/env_gif/     the tool
  cli.py             gymbench env-gif run
```

New algorithm? Add a 4-line factory in `core/algos.py`. Env family with system
deps? `pip install "gymnasium[box2d]"` (or `[mujoco]`).

> CLI command is `env-gif`; the package folder is `env_gif` (imports can't have hyphens).
