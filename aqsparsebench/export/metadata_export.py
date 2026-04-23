"""JSON sidecars for exported benchmark folders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path | str, payload: dict[str, Any], *, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=indent, default=str), encoding="utf-8")
