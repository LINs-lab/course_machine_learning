"""Engine entry point: run the RSI loop end-to-end on a task bundle.

Usage:
    python -m rsi.core --task tasks/quad-optim --out out --cycles 6 \
        [--seed 7] [--middle-every 3] [--improver naive|noop] \
        [--question "..."]

Runs KEYLESS (the default proposer needs no LLM). Outputs land under --out:
state/ (strategy.py, runs.jsonl, ledger.jsonl, epochs.json), kb/ (the
knowledge-platform stores), runs/ (staged cycle workdirs), summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rsi.core.engine import EngineConfig, build_default_engine


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rsi.core")
    p.add_argument("--task", required=True, help="task bundle directory")
    p.add_argument("--out", required=True, help="output/session directory")
    p.add_argument("--cycles", type=int, default=6)
    p.add_argument("--middle-every", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--improver", choices=["naive", "noop"], default="naive")
    p.add_argument("--question", default=None,
                   help="root research question statement")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    if (out_dir / "kb").exists() or (out_dir / "state").exists():
        # a second run on a used session dir would crash on the first
        # question create ("already exists") — fail loud with guidance
        # instead (session resume is homework territory)
        p.error(f"--out {out_dir} already holds a session (kb/ or state/ "
                f"exists). Use a fresh --out, or delete the directory to "
                f"restart.")

    cfg = EngineConfig(
        task_dir=args.task, out_dir=args.out, cycles=args.cycles,
        middle_every=args.middle_every, seed=args.seed,
        improver=args.improver)
    engine = build_default_engine(cfg)
    direction = "Minimize" if engine.spec.metric.direction == "min" else "Maximize"
    question = args.question or (
        f"{direction} {engine.spec.metric.name} on {engine.spec.name} by "
        f"improving the editable {engine.spec.editable[0]} one knob at a time")
    summary = engine.run(question)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    with open(f"{args.out}/summary.json", "w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
