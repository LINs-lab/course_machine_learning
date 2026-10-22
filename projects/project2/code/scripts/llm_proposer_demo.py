#!/usr/bin/env python3
"""Worked example for homework task C-1: an LLM-guided proposer.

This is a reference implementation of an LLM-composed proposer (one knob
move per cycle), not part of the shipped baseline (the engine stays
keyless/naive). It wires the proposer seam exactly as the assignment asks:
build the default engine, swap the proposer, never touch the loop.

Requires an OpenAI-compatible endpoint: set the three env vars below
(course endpoint credentials are announced on Canvas). Without them the
script exits with a configuration error. Once a call fails to parse or
validate, that cycle falls back to the deterministic grid walk.

    export RSI_OPENAI_API_KEY= ...
    export RSI_OPENAI_BASE_URL= ...
    export RSI_OPENAI_MODEL= ...
    python scripts/llm_proposer_demo.py --task tasks/iris-ood \
        --out /tmp/iris-llm --cycles 24 --seed 7

The demo proposer asks the LLM for ONE knob move per cycle (the harness's
single-variable contract — whole-file rewrites are the deeper part of C-1),
with the deterministic grid walk as fallback on any parse/validation
failure. Every LLM decision is logged to <out>/llm_log.jsonl with the FULL
prompt and raw response; a content-addressed sqlite cache
(<out>/llm_cache.sqlite) replays identical calls without touching the
endpoint, so an interrupted session resumes deterministically (re-run the
same command — a state/checkpoint.json in --out triggers resume). Fresh
calls are made reproducible with temperature 0 + an explicit request seed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from rsi.core.engine import EngineConfig, build_default_engine
from rsi.core.proposer import HeuristicProposer, derive_knobs
from rsi.core.types import Proposal
from rsi.llm.cached import CachedProvider
from rsi.llm.openai_compat import OpenAICompatProvider

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_RE = re.compile(r"\{[^{}]*\}")


class LLMGuidedProposer:
    """Ask the LLM for the next single-knob move; grid-walk on any failure.

    The prompt carries the task protocol, the current knob values, the
    incumbent score, and this question's recent run history — the LLM does
    the evidence→proposal step the naive proposer lacks (homework C-1).
    """

    def __init__(
        self,
        provider,
        fallback: HeuristicProposer,
        log_path: Path,
        max_tokens: int = 2048,
    ):
        self.provider = provider
        self.fallback = fallback
        self.log_path = log_path
        self.max_tokens = max_tokens
        self._n = 900  # proposal ids distinct from the fallback's

    def propose(
        self, question, spec, incumbent_params, incumbent_value, history, strategy
    ):
        # Baseline cycle: no judgment needed, run the pristine file.
        if incumbent_value is None:
            return self.fallback.propose(
                question, spec, incumbent_params, incumbent_value, history, strategy
            )
        knobs = dict(incumbent_params or {})
        if not knobs:
            knobs = derive_knobs(spec.editable_file.read_text())
        if not knobs:
            return self.fallback.propose(
                question, spec, incumbent_params, incumbent_value, history, strategy
            )
        qid = question["id"]
        q_hist = [r for r in history if r.question_id == qid]
        tried = {
            (r.variable_changed, _param_key(r.params))
            for r in q_hist
            if r.status != "fail"
        }

        prompt = self._prompt(spec, knobs, incumbent_value, q_hist, tried, question)
        if hasattr(self.provider, "compose_cached"):
            raw, cache_hit = self.provider.compose_cached(prompt, self.max_tokens)
        else:
            raw, cache_hit = self.provider.compose(prompt, self.max_tokens), False
        move = self._parse(raw, knobs, tried, spec)
        self._log(question, prompt, raw, move, cache_hit)
        if move is None:
            return self.fallback.propose(
                question, spec, incumbent_params, incumbent_value, history, strategy
            )
        knob, value, rationale = move
        self._n += 1
        prefix = spec.editable[0].rsplit(".", 1)[0]
        return Proposal(
            proposal_id=f"llm-{self._n:03d}",
            question_id=qid,
            hypothesis_id=(q_hist[-1].hypothesis_id if q_hist else f"{qid}-h0"),
            variable_changed=f"{prefix}.{knob}",
            params={knob: value},
            rationale=f"llm: {rationale}",
            benchmark_ref=question["benchmark_ref"],
        )

    # -- prompt / parse -------------------------------------------------

    def _prompt(self, spec, knobs, incumbent_value, q_hist, tried, question) -> str:
        prefix = spec.editable[0].rsplit(".", 1)[0]
        tried_lines = (
            "\n".join(f"  {v} = {p}" for v, p in sorted(tried)) or "  (none yet)"
        )
        hist_lines = (
            "\n".join(
                f"  {r.run_id}: {r.variable_changed} {r.params} -> "
                f"{r.metric.value} ({r.status})"
                for r in q_hist[-14:]
            )
            or "  (no runs yet)"
        )
        direction = spec.metric.direction
        better = "LOWER" if direction == "min" else "HIGHER"
        return f"""You are the proposer in an automated research loop. Propose the NEXT single-variable experiment.

# Task protocol
{spec.protocol_body[:900]}

# State
- Metric: {spec.metric.name} ({better} is better). Current best (incumbent): {incumbent_value}
- Research question: {question["statement"]}
- Current knob values (the incumbent's): {json.dumps(knobs)}
- Already tried this question (do NOT repeat):
{tried_lines}
- Recent runs this question:
{hist_lines}

# Rules
- Change EXACTLY ONE knob from: {list(knobs)}. The harness rewrites `{spec.editable[0]}` for you.
- The new value must differ from the incumbent's value and from every tried pair above.
- Think about WHY recent runs improved or failed, and about the task's OOD structure.
- Answer with ONLY a JSON object, no prose, no markdown fence:
  {{"knob": "<name>", "value": <number>, "rationale": "<one sentence>"}}"""

    def _parse(self, raw, knobs, tried, spec):
        text = _THINK_RE.sub("", raw)
        m = None
        for match in _JSON_RE.finditer(text):
            try:
                cand = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(cand, dict) and "knob" in cand and "value" in cand:
                m = cand
        if m is None:
            return None
        knob, value = m.get("knob"), m.get("value")
        if (
            knob not in knobs
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            return None
        prefix = spec.editable[0].rsplit(".", 1)[0]
        key = (f"{prefix}.{knob}", _param_key({knob: value}))
        if key in tried or value == knobs[knob]:
            return None
        return knob, value, str(m.get("rationale", ""))[:200]

    def _log(self, question, prompt, raw, move, cache_hit) -> None:
        # the FULL agent context (prompt + raw response), not a tail: this
        # log plus the cache DB is the complete audit trail of the loop's
        # LLM decisions
        with self.log_path.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "question": question["id"],
                        "prompt": prompt,
                        "raw": raw,
                        "parsed": (
                            {"knob": move[0], "value": move[1], "rationale": move[2]}
                            if move
                            else None
                        ),
                        "fell_back": move is None,
                        "cache_hit": cache_hit,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # -- checkpoint hooks (engine resumes the id counter) -------------------

    def state_dict(self) -> dict:
        return {"n": self._n}

    def load_state_dict(self, state: dict) -> None:
        self._n = int(state.get("n", self._n))


def _param_key(params: dict) -> str:
    return ",".join(f"{k}={v!r}" for k, v in sorted(params.items()))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="llm_proposer_demo")
    p.add_argument("--task", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="cycle cap (default: 24, or unlimited when --hours is given)",
    )
    p.add_argument(
        "--hours",
        type=float,
        default=None,
        help="wall-clock cap in hours; the loop stops cleanly at "
        "the next cycle boundary (summary still written)",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--middle-every", type=int, default=3)
    p.add_argument(
        "--improver",
        choices=["naive", "noop"],
        default="noop",
        help="self-improvement loop: 'naive' enables guard-checked "
        "strategy edits (rotates knob priority when no_change "
        "waste dominates); 'noop' disables it (ablation "
        "control)",
    )
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    checkpoint = out_dir / "state" / "checkpoint.json"
    if checkpoint.exists():
        resume = True  # interrupted session: continue from the checkpoint
    elif (out_dir / "kb").exists() or (out_dir / "state").exists():
        p.error(
            f"--out {out_dir} already holds a session (no checkpoint to resume from)"
        )
    else:
        resume = False

    cycles = args.cycles if args.cycles is not None else (10**9 if args.hours else 24)
    cfg = EngineConfig(
        task_dir=args.task,
        out_dir=args.out,
        cycles=cycles,
        middle_every=args.middle_every,
        seed=args.seed,
        improver=args.improver,
        max_hours=args.hours,
        resume=resume,
    )
    engine = build_default_engine(cfg)
    # env-driven (RSI_OPENAI_* vars); vLLM/Qwen3 thinking is disabled so the
    # JSON answer lands inside the token budget. temperature 0 + an explicit
    # request seed make fresh calls reproducible; the sqlite cache replays
    # identical calls on resume without touching the endpoint.
    provider = CachedProvider(
        OpenAICompatProvider(
            {
                "llm.openai_compat.extra_body": {
                    "temperature": 0,
                    "seed": args.seed,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            }
        ),
        out_dir / "llm_cache.sqlite",
    )
    engine.proposer = LLMGuidedProposer(
        provider, HeuristicProposer(), out_dir / "llm_log.jsonl", max_tokens=1024
    )

    direction = "Minimize" if engine.spec.metric.direction == "min" else "Maximize"
    summary = engine.run(
        f"{direction} {engine.spec.metric.name} on {engine.spec.name} by "
        f"improving the editable {engine.spec.editable[0]} one knob at a time"
    )
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"llm cache: {provider.hits} hits, {provider.misses} misses "
        f"({'resumed' if resume else 'fresh'} session)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
