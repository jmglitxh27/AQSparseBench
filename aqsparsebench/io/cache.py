"""On-disk JSON cache for API responses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def cache_key_from_request(base_url: str, path: str, params: dict[str, Any]) -> str:
    """Stable key from URL path and sorted query parameters (excluding secrets)."""
    redacted = {
        k: v
        for k, v in sorted(params.items())
        if str(k).lower() not in ("email", "key", "apikey", "api_key")
    }
    payload = json.dumps({"base_url": base_url, "path": path, "params": redacted}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LocalCache:
    """Filesystem-backed cache for JSON-serializable API payloads."""

    def __init__(self, root: str | Path | None) -> None:
        self.root = Path(root).expanduser().resolve() if root else None

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def _path(self, key: str, *, service: str) -> Path:
        assert self.root is not None
        sub = key[:2]
        safe_service = service.replace("..", "").replace("/", "_").strip() or "default"
        dir_path = self.root / safe_service / sub
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{key}.json"

    def get_json(self, key: str, *, service: str = "aqs") -> Any | None:
        if not self.enabled:
            return None
        path = self._path(key, service=service)
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def set_json(self, key: str, value: Any, *, service: str = "aqs") -> None:
        if not self.enabled:
            return
        path = self._path(key, service=service)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
