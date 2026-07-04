"""Where a rendered GIF lands.

Two sinks:

* ``"gymnasium"`` — drop it into the bundled Gymnasium fork's docs tree at
  ``external/Gymnasium/docs/_static/videos/<group>/<name>.gif``. This is the
  path Gymnasium's docs auto-embed, so a PR is literally this one file.
* any other string is treated as a plain output directory (e.g. ``"./out"``).

The monorepo root is the directory holding ``external/Gymnasium``. It's found
by walking up from this file, so sinks work regardless of the current working
directory (and regardless of the ``packages/gymbench`` nesting).
"""

from __future__ import annotations

from pathlib import Path

GYM_DOCS_VIDEOS = Path("external/Gymnasium/docs/_static/videos")


def find_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "external" / "Gymnasium").exists():
            return parent
    # Fallback: cwd, so a bare checkout still runs even before the submodule
    # is attached.
    return Path.cwd()


def resolve(sink: str, group: str, filename: str, root: Path | None = None) -> Path:
    root = root or find_root()
    if sink == "gymnasium":
        return root / GYM_DOCS_VIDEOS / group / filename
    return Path(sink) / filename
