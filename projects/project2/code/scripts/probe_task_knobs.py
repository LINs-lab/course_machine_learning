#!/usr/bin/env python3
"""Single-knob probe: run a task under several one-knob settings.

Useful before writing a proposer: check which knobs actually move the gate
metric, and by how much, under the frozen benchmark.

Usage:
    python scripts/probe_task_knobs.py <task> 'KNOB=v' 'KNOB=v' ...
    python scripts/probe_task_knobs.py house-price POLY_DEGREE=2 POLY_DEGREE=3
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rsi.tasks.runner import LocalRunner, stage_workdir  # noqa: E402
from rsi.tasks.spec import load_task, read_result  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def run(spec, tmp, params, tag):
    workdir = Path(tmp) / tag
    stage_workdir(spec, workdir, None, params)
    res = LocalRunner().dispatch(
        spec, workdir, {"RSI_BENCHMARK_SEED": "7", "RSI_EPOCH": "e001"})
    out = read_result(spec, workdir)
    return out["metric"], res.duration_s, out["failure_mode"], res.exit_code


def main(argv):
    name = argv[1]
    spec = load_task(ROOT / "tasks" / name)
    tmp = tempfile.mkdtemp(prefix=f"probe-{name}-")
    try:
        m, d, f, code = run(spec, tmp, {}, "base")
        print(f"{'baseline':<34}{m:>12.4f}  {d:6.1f}s {f or ''} exit={code}")
        for setting in argv[2:]:
            knob, val = setting.split("=", 1)
            try:
                v = float(val)
                v = int(v) if v == int(v) else v
            except ValueError:
                v = val
            m, d, f, code = run(spec, tmp, {knob: v}, setting.replace("/", "_"))
            print(f"{setting:<34}{m:>12.4f}  {d:6.1f}s {f or ''} exit={code}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
