"""Reproducibility metadata and stable hashing helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 hex digest."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash one file without changing it."""
    return sha256_bytes(Path(path).read_bytes())


def sha256_json(value: Any) -> str:
    """Hash canonical UTF-8 JSON."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256_bytes(payload)
