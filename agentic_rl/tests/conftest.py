"""Portable scratch directories (Windows sandbox cannot use pytest's 0700 ACL)."""

import shutil
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def scratch_dir():
    root = Path(__file__).resolve().parents[1] / "outputs" / "test_scratch"
    directory = root / uuid4().hex
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        # Only this fixture's newly created UUID directory is removed.
        assert directory.parent == root and len(directory.name) == 32
        shutil.rmtree(directory)
