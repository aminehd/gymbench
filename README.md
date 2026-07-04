# gymbench

## The idea

Gymnasium's docs show a GIF for every environment — but they're recorded with a
**random** agent, so the little cart-pole just flops over and the mountain car
never climbs. [Issue #1610](https://github.com/Farama-Foundation/Gymnasium/issues/1610)
asks for GIFs of **trained** agents instead, so the docs show each env actually
being *solved*.

`gymbench` does exactly that: point it at an environment, it trains a small policy
(Stable-Baselines3), rolls it out, and saves a GIF in Gymnasium's exact docs
format. The rendering is identical to Gymnasium's own generator — the only change
is the agent picks actions with a trained policy instead of at random.

It's built as a tiny framework, not a one-off script: environments are **data**
(a config file), algorithms are **plugins** (one decorator), so the same tool
scales from CartPole to any other env without touching the core.

## Config

Everything is driven by a TOML config — one `[[reel]]` per env:

```toml
[[reel]]
env = "CartPole-v1"   # -> cart_pole.gif
algo = "ppo"          # ppo | dqn | sac
steps = 120_000
```

Configs live in [`configs/env_gif/`](configs/env_gif/). Add an env = add a `[[reel]]`;
add a whole new group (Box2D, MuJoCo…) = add a new TOML. No code changes.

## Run

```bash
pip install -e .
gymbench env-gif run                              # all envs in the config
gymbench env-gif run --env CartPole-v1 --algo ppo # one-off
gymbench algos                                     # list algorithms
```

## How it fits together

```
configs/env_gif/         config sets — WHAT to generate (mirrors tools/)
src/gymbench/
  core/
    registry.py          @algo decorator — how new algorithms plug in
    algos.py             built-in ppo / dqn / sac
    trainer.py           train a policy            (env-agnostic)
    render.py            policy -> GIF             (matches gen_gifs.py)
    evaluate.py          greedy score              (trust the number, then the GIF)
    sink.py              WHERE the GIF lands
    config.py            TOML -> typed specs
  tools/env_gif/         the tool — wires core together
  cli.py                 gymbench env-gif run
```

The three ideas that make it a framework:

- **Registry (`@algo`)** — algorithms self-register; add one with a 4-line factory
  in `core/algos.py`. The core never changes.
- **Declarative config** — one TOML = one reproducible batch. Env ids become
  snake_case filenames automatically (`CartPole-v1` → `cart_pole.gif`).
- **Sinks** — `"gymnasium"` writes into the fork's docs tree; any other value is a
  plain output folder.

The whole contribution in one line: same loop as Gymnasium's
[`gen_gifs.py`](src/gymbench/core/render.py) — `reset → render → step` — with
`action = model.predict(obs)` instead of `env.action_space.sample()`.

> CLI command is `env-gif`; the package folder is `env_gif` (imports can't have hyphens).
