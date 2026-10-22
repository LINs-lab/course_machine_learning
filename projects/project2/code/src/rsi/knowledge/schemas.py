# Ported (semantics preserved) from
# project_collab_auto_research_shared_kb/protocol/__init__.py: YAML-authored
# JSON-Schema contracts are the single source of truth for the concept
# definitions; payloads are validated at the platform write boundary.
"""Contract loader for the knowledge-platform concept schemas.

The schemas under ``schemas/`` are the FROZEN concept definitions (I-Q
Question, I-H Hypothesis, I-E Evidence, I-K Knowledge, I-S Insight, I-X
RunProposal, plus the shared ChangeProposal envelope), authored in YAML and
loaded as JSON-Schema dicts. ``rsi.knowledge.concepts`` mirrors them as
dataclasses; contract tests assert the two never drift apart.
"""
from __future__ import annotations

import pathlib

import yaml

CONTRACT_VERSION = "0.2.0"

INTERFACES = ("iq", "ih", "ie", "ik", "is", "ix")

_SCHEMA_DIR = pathlib.Path(__file__).parent / "schemas"


def load_schema(key: str) -> dict:
    """Load a JSON-Schema dict. ``key`` is an interface key or a shared schema
    name (change_proposal, ik_theory)."""
    path = _SCHEMA_DIR / f"{key}.yaml"
    if not path.is_file():
        raise KeyError(f"unknown schema: {key!r}")
    return yaml.safe_load(path.read_text())
