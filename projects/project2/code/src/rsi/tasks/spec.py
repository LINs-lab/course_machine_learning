# Ported (semantics preserved) from project_collab_auto_research/graph/envs.py
# (EnvSpec/EnvMetric/load_env/read_result), trimmed to the single-variable
# contract: TaskSpec carries exactly the fields the base harness enforces.
# Intentionally dropped vs the source EnvSpec: multi_editable, share/federation,
# focus/capabilities/info_channels, regimes, collapse_guard, min_seeds,
# cross_family_judge, attestation — see docs/architecture.md (divergence table).
"""Task-bundle contract: a research task is a directory with a ``program.md``.

The manifest is a YAML frontmatter block; the markdown body below it is the
task protocol (human/agent-readable description of the research problem).

Hard invariants (fail-fast, mirroring the source loader):

- ``editable`` declares EXACTLY ONE file — every experiment is a controlled
  single-variable change on that file. Everything else in the bundle is
  read-only to the researcher.
- ``metric.direction`` ∈ {min, max}; the entrypoint must write
  ``metric.result_file`` containing the scalar at ``metric.result_key``.
- All declared paths are bundle-relative (no absolute paths, no ``..``).

The harness never imports task code: the entrypoint runs as a subprocess
(see rsi.tasks.runner) and the metric is read from the result file only.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


class TaskError(ValueError):
    """Raised when a task manifest is missing, malformed, or unsafe."""


@dataclass
class TaskMetric:
    name: str
    direction: str          # "min" | "max"
    result_file: str
    result_key: str


@dataclass
class TaskSpec:
    name: str
    entrypoint: list[str]
    time_budget: int                      # seconds; enforced by the runner
    metric: TaskMetric
    editable: list[str]                   # exactly ONE file (enforced)
    readonly: list[str]
    env_dir: Path
    protocol_body: str                    # markdown below the frontmatter
    schema_version: int = 1
    validate: list[str] = field(default_factory=list)   # optional argv; [] = none

    @property
    def editable_file(self) -> Path:
        return self.env_dir / self.editable[0]


def _check_rel(path: str, what: str) -> str:
    """Task-relative path only: non-empty, a string, not absolute, no ``..``
    escape. Returns the normalized posix form (``./solver.py`` →
    ``solver.py``) so downstream string comparisons match rglob output."""
    if not isinstance(path, str) or not path:
        raise TaskError(f"{what} must be a non-empty path string, got "
                        f"{path!r}")
    if Path(path).is_absolute() or ".." in Path(path).parts:
        raise TaskError(f"{what} must be a task-relative path, got {path!r}")
    return Path(path).as_posix()


def load_task(task_dir: str | Path) -> TaskSpec:
    """Parse ``<task_dir>/program.md`` into a TaskSpec. Fail-fast on shape."""
    env_dir = Path(task_dir)
    md = env_dir / "program.md"
    if not md.exists():
        raise TaskError(f"task at {env_dir} has no program.md")
    text = md.read_text()
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise TaskError(f"task {env_dir.name!r} program.md missing frontmatter block")
    raw = yaml.safe_load(m.group(1))
    if not isinstance(raw, dict):
        raise TaskError(f"task {env_dir.name!r} frontmatter must be a mapping")
    body = text[m.end():]
    name = env_dir.name

    for k in ("name", "entrypoint", "metric", "editable"):
        if k not in raw:
            raise TaskError(f"task {name!r} manifest missing field: {k}")
    if raw["name"] != name:
        raise TaskError(f"task name {raw['name']!r} != directory {name!r}")

    ep = raw["entrypoint"]
    if (not isinstance(ep, list) or not ep
            or not all(isinstance(x, str) for x in ep)):
        raise TaskError(
            f"task {name!r} entrypoint must be a non-empty argv list of strings")

    mt = raw["metric"]
    if not isinstance(mt, dict):
        raise TaskError(f"task {name!r} metric must be a mapping")
    for k in ("name", "direction", "result_file", "result_key"):
        if k not in mt:
            raise TaskError(f"task {name!r} metric missing field: {k}")
        if not isinstance(mt[k], str):
            raise TaskError(
                f"task {name!r} metric.{k} must be a string, got {mt[k]!r}")
    if mt["direction"] not in ("min", "max"):
        raise TaskError(
            f"task {name!r} metric.direction must be min|max, got {mt['direction']!r}")

    editable = raw["editable"]
    if (not isinstance(editable, list) or not editable
            or not all(isinstance(x, str) for x in editable)):
        raise TaskError(
            f"task {name!r} editable must be a non-empty list of strings")
    # Single authoritative single-file invariant (source: same rationale). The
    # researcher rewrites exactly one file and the loop attributes a metric
    # delta to one controlled change; reject here so the contract fails fast.
    if len(editable) != 1:
        raise TaskError(
            f"task {name!r} editable must declare exactly ONE file, got "
            f"{editable}; consolidate the research surface into a single seam")

    readonly = raw.get("readonly") or []
    if not isinstance(readonly, list) or not all(isinstance(x, str) for x in readonly):
        raise TaskError(f"task {name!r} readonly must be a list of strings")

    validate = raw.get("validate") or []
    if (not isinstance(validate, list)
            or not all(isinstance(x, str) for x in validate)):
        raise TaskError(
            f"task {name!r} validate must be an argv list of strings")

    mt_result_file = _check_rel(mt["result_file"], "metric.result_file")
    editable = [_check_rel(g, "editable path") for g in editable]
    readonly = [_check_rel(g, "readonly path") for g in readonly]
    # entrypoint argv tokens that LOOK like paths must stay inside the
    # bundle: the runner executes them with cwd=workdir, so a path-like
    # token is code execution and the same traversal rules apply (the
    # executable token itself, e.g. "python3", is exempt — PATH-resolved).
    for tok in ep[1:]:
        if "/" in tok or "\\" in tok:
            _check_rel(tok, "entrypoint argument")

    try:
        time_budget = int(raw.get("time_budget", 300))
    except (TypeError, ValueError):
        raise TaskError(f"task {name!r} time_budget must be an int") from None

    return TaskSpec(
        name=name,
        entrypoint=list(ep),
        time_budget=time_budget,
        metric=TaskMetric(
            name=str(mt["name"]), direction=str(mt["direction"]),
            result_file=mt_result_file, result_key=str(mt["result_key"])),
        editable=list(editable),
        readonly=list(readonly),
        env_dir=env_dir,
        protocol_body=body,
        schema_version=int(raw.get("schema_version", 1)),
        validate=list(validate),
    )


def _dig(data, dotpath: str):
    """Follow a dot-separated key path; None when any hop is missing."""
    cur = data
    for part in dotpath.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def read_result(spec: TaskSpec, workdir: str | Path) -> dict:
    """Read the optimized metric from the task's result file in ``workdir``.

    Returns ``{"metric": float|None, "observations": dict, "failure_mode":
    str|None}``. On any failure to obtain a finite float, metric is None with an
    explicit failure_mode ∈ {missing_result, invalid_metric, missing_metric}
    (source vocabulary; attestation modes are an intentional divergence — see
    docs/architecture.md). The result file is UNTRUSTED self-report:
    TODO(homework) task R-1 adds provenance/attestation (executor-attested
    results, signed metrics).
    """
    out = {"metric": None, "observations": {}, "failure_mode": None}
    rf = Path(workdir) / spec.metric.result_file
    if not rf.exists():
        out["failure_mode"] = "missing_result"
        return out
    try:
        data = json.loads(rf.read_bytes().decode("utf-8"))
    except (json.JSONDecodeError, OSError, ValueError, UnicodeDecodeError):
        out["failure_mode"] = "invalid_metric"
        return out
    val = _dig(data, spec.metric.result_key)
    if val is None:
        out["failure_mode"] = "missing_metric"
        return out
    try:
        f = float(val)
    except (TypeError, ValueError):
        out["failure_mode"] = "invalid_metric"
        return out
    if f != f or f in (float("inf"), float("-inf")):
        out["failure_mode"] = "invalid_metric"
        return out
    out["metric"] = f
    if isinstance(data, dict) and isinstance(data.get("observations"), dict):
        out["observations"] = data["observations"]
    return out
