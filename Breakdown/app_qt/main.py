"""
Entry point.

    python Breakdown/app_qt/main.py

Run it from the repository root, or from anywhere -- the repo root is put on
sys.path below so ``Breakdown`` and ``KeithleyGUI`` both import cleanly.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PyQt5.QtCore import QSettings  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from Breakdown.app_qt.theme import apply_theme  # noqa: E402
from Breakdown.app_qt.wizard import BreakdownWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    app.setOrganizationName("NUSLab")
    app.setApplicationName("Breakdown")
    settings = QSettings()
    theme_name = settings.value("appearance/theme", "dark", type=str)
    apply_theme(app, theme_name)
    window = BreakdownWindow(settings=settings)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
