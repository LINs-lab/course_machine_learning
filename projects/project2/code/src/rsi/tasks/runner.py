# Ported (semantics preserved) from project_collab_auto_research/graph/executors/
# {base.py,local.py} (miniaturized ComputeExecutor -> TaskRunner) plus the
# implement-stage scope guard (cli/implement.py guards: editable-only, valid
# python) expressed here as stage_workdir's pre-run diff check.
"""Task execution seam: stage an isolated workdir, dispatch the entrypoint.

- ``stage_workdir`` copies the PRISTINE task bundle into ``<out>/runs/<id>/``,
  applies the proposal's knob params to the editable file (on top of the
  incumbent's bytes, so keeps accumulate), and enforces the scope guard: after
  staging, ONLY the editable file may differ from the pristine bundle.
- ``LocalRunner.dispatch`` runs the entrypoint as a subprocess with the
  harness-authoritative benchmark identity injected (``RSI_EPOCH``,
  ``RSI_BENCHMARK_SEED`` — the epoch-freeze carrier; in-flight runs archive
  under their submission epoch), a wall-clock timeout, and a captured run.log.

The runner never imports task code; the only channel back is the result file
(see rsi.tasks.spec.read_result).
"""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rsi.tasks.spec import TaskSpec

# Knobs are located by AST (module-level Assign/AnnAssign with a plain Name
# target), NOT by regex: a column-0 regex also matches ``KNOB = ...`` lines
# inside module docstrings, and the harness would then attribute a metric
# delta to a change that never reached the executed code (single-variable
# contract broken silently). AST positions are docstring-immune.


def _knob_span(source: str, knob: str) -> tuple[int, int] | None:
    """1-based [start, end] line span of the module-level ``knob`` assignment,
    or None. Multi-target assigns (``A = B = 1``) are not knob sites."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
        else:
            continue
        if isinstance(target, ast.Name) and target.id == knob:
            return node.lineno, node.end_lineno or node.lineno
    return None


@dataclass
class DispatchResult:
    exit_code: int
    duration_s: float
    timed_out: bool
    log_path: Path


class TaskRunner(Protocol):
    def dispatch(self, spec: TaskSpec, workdir: Path,
                 env_extra: dict[str, str]) -> DispatchResult: ...


class TaskRunError(RuntimeError):
    pass


def apply_params(source: str, params: dict[str, object]) -> str:
    """Rewrite ``KNOB = <value>`` assignment lines in the editable source.

    Only exact-name top-level assignments are touched (single-variable control:
    the proposal names the knob it changed). Unknown knobs raise — a proposal
    may not fabricate variables that do not exist in the task's editable file.
    """
    out = source
    for knob, value in params.items():
        span = _knob_span(out, knob)
        if span is None:
            raise TaskRunError(
                f"knob {knob!r} has no assignment in the editable file; "
                f"editable knobs are the ones assigned at module level")
        lines = out.splitlines(keepends=True)
        lines[span[0] - 1:span[1]] = [f"{knob} = {value!r}\n"]
        out = "".join(lines)
    return out


def parse_params(source: str, knobs: list[str] | None = None) -> dict[str, object]:
    """Read module-level ``KNOB = value`` assignments (AST-located, literal
    eval of the RHS). ``knobs=None`` parses every top-level assignment."""
    found: dict[str, object] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        if knobs is not None and target.id not in knobs:
            continue
        segment = ast.get_source_segment(source, value)
        try:
            found[target.id] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            found[target.id] = (segment or "").strip()
    return found


def stage_workdir(spec: TaskSpec, workdir: Path,
                  incumbent_bytes: bytes | None,
                  params: dict[str, object]) -> Path:
    """Stage an isolated run directory: pristine copy + param application.

    ``incumbent_bytes`` carries the last-kept editable file (keeps accumulate);
    None means the pristine bundle starts as-is (baseline cycle).
    """
    workdir = Path(workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    shutil.copytree(spec.env_dir, workdir, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    editable = workdir / spec.editable[0]
    base = (incumbent_bytes if incumbent_bytes is not None
            else editable.read_bytes())
    new_source = apply_params(base.decode("utf-8"), params)
    editable.write_bytes(new_source.encode("utf-8"))

    _scope_guard(spec, workdir)
    return workdir


def _scope_guard(spec: TaskSpec, workdir: Path) -> None:
    """After staging, only the editable file may differ from the pristine task.

    Belt-and-suspenders check mirroring the implement-stage guards: the diff
    between the staged workdir and the pristine bundle must be confined to the
    single editable file. Anything else means the staging path is corrupt —
    fail loudly rather than run an uncontrolled experiment.
    """
    editable_name = spec.editable[0]
    for path in sorted(workdir.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(workdir).as_posix()
        pristine = spec.env_dir / rel
        if rel == editable_name:
            continue
        if not pristine.exists() or pristine.read_bytes() != path.read_bytes():
            raise TaskRunError(
                f"scope guard: non-editable file {rel!r} differs from the "
                f"pristine task bundle")
    compile((workdir / editable_name).read_text(), editable_name, "exec")


class LocalRunner:
    """Subprocess executor (the only shipped runner; the source system also had
    ssh/det GPU executors — out of scope here).

    Note: the wall-clock timeout kills only the direct child process; a task
    that spawns its own children may leave orphans writing into the workdir
    after dispatch returns (standard subprocess limitation — a sandboxed
    runner is homework territory).
    """

    def dispatch(self, spec: TaskSpec, workdir: Path,
                 env_extra: dict[str, str]) -> DispatchResult:
        argv = [a.replace("{python}", "python3") for a in spec.entrypoint]
        # The harness owns the RSI_* namespace: ambient shell variables like
        # RSI_STEPS/RSI_NOISE must NOT leak into the subprocess — a frozen
        # epoch means the benchmark is identical for every run, and only the
        # harness-injected identity (env_extra) may carry RSI_* values.
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("RSI_")}
        env.update({k: str(v) for k, v in env_extra.items()})
        log_path = Path(workdir) / "run.log"
        start = time.monotonic()
        try:
            proc = subprocess.run(
                argv, cwd=workdir, env=env, capture_output=True, text=True,
                timeout=spec.time_budget)
            timed_out = False
            code = proc.returncode
            stdout, stderr = proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            timed_out = True
            code = -1
            stdout = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr = (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        except OSError as e:
            # missing interpreter/binary, permission denied, ... — a dispatch
            # failure, never an engine crash (the run becomes a `fail`).
            timed_out = False
            code = 127
            stdout = ""
            stderr = f"dispatch error: {type(e).__name__}: {e}"
        duration = time.monotonic() - start
        log_path.write_text(
            f"# argv: {argv}\n# exit: {code} timed_out: {timed_out}\n"
            f"# stdout:\n{stdout}\n# stderr:\n{stderr}\n")
        return DispatchResult(exit_code=code, duration_s=duration,
                              timed_out=timed_out, log_path=log_path)
