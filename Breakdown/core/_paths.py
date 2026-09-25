"""
Cross-package import bootstrap.

There is no installable package in this repo, so `KeithleyGUI` is reached by
putting the repo root on `sys.path` -- the same approach used by
`SheetResistance/sheet_resistance_app.py:39-44`.

Import this module before importing anything from `KeithleyGUI`:

    from . import _paths  # noqa: F401
    from KeithleyGUI import keithley

Doing the path surgery here once keeps it out of every other module.
"""

import sys
from pathlib import Path

#: Repository root (the directory containing KeithleyGUI/, CV_MAP/, Breakdown/).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: KeithleyGUI package directory. `keithley.py` uses flat imports internally,
#: so this is also placed on the path for callers that want `import keithley`.
KEITHLEY_DIR = REPO_ROOT / "KeithleyGUI"


def _prepend(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


_prepend(REPO_ROOT)
_prepend(KEITHLEY_DIR)
