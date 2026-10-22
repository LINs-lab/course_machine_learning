from __future__ import annotations

import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TASK = REPO / "tasks" / "quad-optim"


@pytest.fixture
def demo_task(tmp_path: Path) -> Path:
    """A throwaway copy of the demo task bundle (tests never mutate tasks/)."""
    dst = tmp_path / "task" / "quad-optim"
    shutil.copytree(TASK, dst)
    return dst
