"""Evaluation report CLI: replay a session's run log through the diagnostician.

Usage:
    python -m rsi.evaluation --runs <out>/state/runs.jsonl report
"""
from __future__ import annotations

import argparse
import json
import sys

from rsi.evaluation.diagnostics import NaiveDiagnostician


def _load_runs(path: str) -> list:
    from rsi.core.types import RunRecord
    runs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                runs.append(RunRecord.from_dict(json.loads(line)))
    return runs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rsi.evaluation")
    p.add_argument("--runs", required=True, help="path to state/runs.jsonl")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("report")
    args = p.parse_args(argv)

    runs = _load_runs(args.runs)
    diag = NaiveDiagnostician()
    epochs = sorted({r.epoch for r in runs})
    for epoch in epochs:
        report = diag.diagnose(runs, epoch)
        print(json.dumps({
            "epoch": report.epoch, "n_runs": report.n_runs,
            "saturation": report.saturation,
            "noise_floor": report.noise_floor,
            "discrimination": report.discrimination,
            "recommendation": report.recommendation,
            "details": report.details,
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
