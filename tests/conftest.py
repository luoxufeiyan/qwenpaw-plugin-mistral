# -*- coding: utf-8 -*-
"""Shared pytest setup: make the plugin modules importable from the repo root."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
