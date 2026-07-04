# gymbench

Train a policy and render it as a GIF in Gymnasium's docs format. First tool: **`env-gif`**.
Built for [Gymnasium #1610](https://github.com/Farama-Foundation/Gymnasium/issues/1610).

## Config

Everything is driven by a TOML config — one `[[reel]]` per env:

```toml
[[reel]]
env = "CartPole-v1"   # -> cart_pole.gif
algo = "ppo"          # ppo | dqn | sac
steps = 120_000
```

Configs live in [`configs/env_gif/`](configs/env_gif/). Add an env = add a `[[reel]]`;
add a group = add a new TOML. No code changes.

## Run

```bash
pip install -e .
gymbench env-gif run                              # all envs in the config
gymbench env-gif run --env CartPole-v1 --algo ppo # one-off
gymbench algos                                     # list algorithms
```

## How it's built

- **`@algo` registry** — algorithms self-register; add one with a 4-line factory in `core/algos.py`.
- **`sink`** — `"gymnasium"` writes into the fork's docs tree; any other value is a plain folder.
- **[`core/render.py`](src/gymbench/core/render.py)** — the rollout loop, byte-for-byte
  `gen_gifs.py` except `action = model.predict(obs)` instead of `env.action_space.sample()`.

```
configs/env_gif/       config sets (mirror tools/)
src/gymbench/
  core/                registry · trainer · render · sink · config
  tools/env_gif/       the tool
  cli.py               gymbench env-gif run
```

> CLI command is `env-gif`; the package folder is `env_gif` (imports can't have hyphens).
