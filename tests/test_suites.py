# -*- coding: utf-8 -*-
"""pytest wrapper around the three script-style suites.

Each suite is a standalone script (``python3 tests/test_x.py``) that prints
PASS/FAIL lines and exits non-zero on failure.  This module runs them as
subprocesses so ``pytest tests/`` covers exactly the same checks.
"""

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

SUITES = [
    "test_config.py",
    "test_transport.py",
    "test_provider_e2e.py",
]


@pytest.mark.parametrize("script", SUITES)
def test_suite_passes(script: str) -> None:
    """Run one suite and require a zero exit code."""
    result = subprocess.run(
        [sys.executable, str(HERE / script)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, (
        f"{script} failed\n--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
