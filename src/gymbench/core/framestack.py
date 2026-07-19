"""Single-env frame stacking that mirrors SB3's ``VecFrameStack``.

A policy trained with ``frame_stack=k`` expects k concatenated frames as its
observation. Training uses ``VecFrameStack`` over a vectorised env; when we
later drive a single env (to render a GIF or to evaluate) we must reproduce the
same stacking or ``model.predict`` sees the wrong shape. This keeps the last k
observations and concatenates them on the channel axis (newest last), and on
reset refills every slot with the fresh observation — exactly as SB3 does.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class FrameStacker:
    def __init__(self, k: int, first_obs):
        self.k = k
        self.buf = deque([first_obs] * k, maxlen=k)

    def observe(self):
        return np.concatenate(list(self.buf), axis=-1)

    def push(self, obs):
        self.buf.append(obs)

    def reset(self, obs):
        self.buf.clear()
        self.buf.extend([obs] * self.k)
