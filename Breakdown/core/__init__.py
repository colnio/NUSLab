"""
Headless core for the crosspoint breakdown suite.

Every module in this package is free of GUI imports. The PyQt5 shell in
``Breakdown/app_qt`` is one consumer; a web front-end could be another without
any instrument code changing.

Only :mod:`mfia` and :mod:`smu` touch instrument libraries (``zhinst`` and
``pyvisa`` respectively). Everything else is stdlib-only, so the test suite runs
on a machine with no instrument drivers installed.
"""
