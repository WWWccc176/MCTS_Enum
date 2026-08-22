from __future__ import annotations

import importlib
import sys
from pathlib import Path

VERSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION_ROOT.parents[1]


def _prepend(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def load_mcts_backend():
    _prepend(VERSION_ROOT / "build")
    return importlib.import_module("mcts_enum_backend")


def load_heavy_backend():
    _prepend(PROJECT_ROOT / "HeavyBack" / "build")
    return importlib.import_module("heavy_backend")
