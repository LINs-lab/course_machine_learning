# Ported (semantics preserved) from project_collab_auto_research/graph/config.py,
# trimmed to the keys Autoresearch-Base actually uses: file (config.yaml) values
# override a built-in allowlist registry; secrets are unreachable by construction.
"""Unified configuration: ``config.yaml`` (if present) > built-in defaults.

Only keys declared in ``_REGISTRY`` can ever be read — an unknown key is a
programming error, and credential-style keys are structurally unreachable
(this framework carries no credentials; real providers read env vars directly).
"""
from __future__ import annotations

from pathlib import Path

import yaml

# key -> default. The allowlist IS the contract: nothing outside it is readable.
_REGISTRY: dict = {
    "engine.cycles": 6,
    "engine.middle_every": 3,
    "engine.seed": 7,
    "engine.min_support": 2,
    "engine.saturation_threshold": 0.8,
    "engine.improver": "naive",          # naive | noop | <registered name>
    "engine.proposer": "heuristic",      # heuristic | <registered name>
    "llm.provider": "echo",              # echo | openai_compat
    "tasks.dir": "tasks",
}

# Credential-shaped keys are refused by construction, mirroring the source's
# _SENSITIVE frozenset rule: config.yaml is committed-adjacent, keys are not.
_SENSITIVE = frozenset({"llm.api_key", "llm.key", "api_key", "token", "secret"})

_FILE: Path | None = None
_CACHE: dict = {}
_LOADED = False


class ConfigError(RuntimeError):
    pass


def set_config_path(path: str | Path | None) -> None:
    """Point the loader at a config.yaml (None = defaults only). Resets cache."""
    global _FILE, _CACHE, _LOADED
    _FILE = Path(path) if path is not None else None
    _CACHE = {}
    _LOADED = False


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    merged: dict = {}
    if _FILE is not None and _FILE.exists():
        raw = yaml.safe_load(_FILE.read_text()) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{_FILE} must be a mapping")
        merged = _flatten(raw)
    _CACHE.clear()
    _CACHE.update(merged)
    _LOADED = True


def _flatten(nested: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in nested.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if key in _SENSITIVE:
            raise ConfigError(f"refusing to load credential-shaped key {key!r}")
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def get(key: str):
    """Read a registry key (file override > default). Unknown or
    credential-shaped keys raise — there is no silent fallback."""
    _ensure_loaded()
    if key in _SENSITIVE:
        raise ConfigError(f"key {key!r} is credential-shaped and unreadable")
    if key not in _REGISTRY:
        raise ConfigError(
            f"unknown config key {key!r}; known keys: {sorted(_REGISTRY)}")
    if key in _CACHE:
        return _CACHE[key]
    return _REGISTRY[key]


def get_int(key: str) -> int:
    raw = get(key)
    try:
        return int(raw)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"config key {key!r} must be an int, got {raw!r}") from e
