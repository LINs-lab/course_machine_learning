"""Read-only knowledge-base inspection CLI.

Usage:
    python -m rsi.knowledge --kb <dir> questions
    python -m rsi.knowledge --kb <dir> frontier [--max-saturation 1.0]
    python -m rsi.knowledge --kb <dir> question <id>
    python -m rsi.knowledge --kb <dir> hypothesis <id>
    python -m rsi.knowledge --kb <dir> evidence [--run <id> | --target <id>]
    python -m rsi.knowledge --kb <dir> insights
"""
from __future__ import annotations

import argparse
import json
import sys

from rsi.knowledge.platform import KnowledgePlatform


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rsi.knowledge")
    p.add_argument("--kb", required=True, help="knowledge-base directory")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("questions")
    f = sub.add_parser("frontier")
    f.add_argument("--max-saturation", type=float, default=1.0)
    q = sub.add_parser("question")
    q.add_argument("id")
    h = sub.add_parser("hypothesis")
    h.add_argument("id")
    e = sub.add_parser("evidence")
    e.add_argument("--run", default=None)
    e.add_argument("--target", default=None)
    sub.add_parser("insights")
    args = p.parse_args(argv)

    kb = KnowledgePlatform(args.kb)
    if args.cmd == "questions":
        items = kb.questions.all()
    elif args.cmd == "frontier":
        items = kb.frontier(max_saturation=args.max_saturation)
    elif args.cmd == "question":
        items = [kb.get_question(args.id)] if kb.get_question(args.id) else []
    elif args.cmd == "hypothesis":
        items = [kb.get_hypothesis(args.id)] if kb.get_hypothesis(args.id) else []
    elif args.cmd == "evidence":
        if args.run:
            items = kb.evidence_by_run(args.run)
        elif args.target:
            items = kb.evidence_for(args.target)
        else:
            items = kb.evidence.all()
    else:
        items = kb.insights.all()
    for item in items:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
