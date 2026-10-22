"""RSIEngine: the three-loop orchestrator and the composition root.

    ── FAST loop (every cycle) ─────────────────────────────────────────
    propose (reads strategy — the RSI re-entry point) → stage → dispatch
    → gate → keep ratchet → push evidence → confidence → maybe transition

    ── MIDDLE loop (every ``middle_every`` cycles) ──────────────────────
    diagnose (saturation / noise / discrimination) → write back to the
    question → maybe upgrade the benchmark (epoch freeze/upgrade) →
    efficiency report → arbitrate the live self-edit or propose a new one

    ── SLOW loop (event-driven) ─────────────────────────────────────────
    hypothesis terminal or question saturated or proposer exhausted →
    spawn child question (typed edge) → transition the question → pick
    the next frontier question → mount a fresh hypothesis → recurse

System-level self-improvement with RE-ENTRY: an applied edit goes to
``observing``; while observing, every cycle's proposal is already shaped by
the edited strategy (nothing else needs to happen — that IS the recursion);
the middle loop later arbitrates keep vs revert (Stage2-lite). The scorer
(gates) is never editable by the improver (trust boundary,
rsi.selfedit.guard.EDITABLE_SURFACES).

The engine depends ONLY on protocols (Proposer / SelfImprover /
Diagnostician / BenchmarkUpgrader / GateChain / TaskRunner); concrete naive
implementations are wired in ``build_default_engine`` — students rewire
that factory, not this loop.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from rsi.core.improver import (ImproverError, NaiveImprover, NoopImprover,
                               SelfImprover)
from rsi.core.ledger import EditLedger
from rsi.core.pipeline import failed_run, run_cycle
from rsi.core.proposer import (HeuristicProposer, Proposer,
                               ensure_strategy_file, load_strategy)
from rsi.core.types import RunRecord, build_efficiency_report
from rsi.evaluation.benchmark import (Benchmark, BenchmarkRef,
                                      BenchmarkUpgrader, EpochManager,
                                      SeedBumpUpgrader)
from rsi.evaluation.diagnostics import Diagnostician, NaiveDiagnostician
from rsi.evaluation.gates import (CollapseGate, GateChain, NoiseGate,
                                  NoChangeGate)
from rsi.knowledge.concepts import EvidenceLink, MachineEvidence
from rsi.knowledge.platform import KnowledgePlatform
from rsi.tasks.runner import LocalRunner
from rsi.tasks.spec import TaskSpec


@dataclass
class EngineConfig:
    task_dir: str
    out_dir: str
    cycles: int = 6
    middle_every: int = 3
    seed: int = 7
    min_support: int = 2
    saturation_threshold: float = 0.8
    improver: str = "naive"             # naive | noop
    max_hours: float | None = None      # wall-clock cap (None = cycle-bounded)
    resume: bool = False  # continue from state/checkpoint.json if present


class RSIEngine:
    def __init__(self, spec: TaskSpec, platform: KnowledgePlatform,
                 epochs: EpochManager, ledger: EditLedger,
                 strategy_path: Path, proposer: Proposer,
                 improver: SelfImprover, diagnostician: Diagnostician,
                 upgrader: BenchmarkUpgrader, runner: LocalRunner,
                 gates: GateChain, cfg: EngineConfig):
        self.spec = spec
        self.platform = platform
        self.epochs = epochs
        self.ledger = ledger
        self.strategy_path = strategy_path
        self.proposer = proposer
        self.improver = improver
        self.diagnostician = diagnostician
        self.upgrader = upgrader
        self.runner = runner
        self.gates = gates
        self.cfg = cfg
        # engine-local state
        self.runs: list[RunRecord] = []
        self.bench = Benchmark(
            name=spec.name, task=spec.name, metric_name=spec.metric.name,
            metric_direction=spec.metric.direction, seed=cfg.seed)
        self.incumbent_value: float | None = None
        self.incumbent_bytes: bytes | None = None
        self.incumbent_workdir: str | None = None
        self.noise_floor: float | None = None
        self._q_seq = 0
        self._stop_reason: str | None = None

    # -- run ----------------------------------------------------------------

    def run(self, question_statement: str) -> dict:
        out = Path(self.cfg.out_dir)
        (out / "state").mkdir(parents=True, exist_ok=True)
        ckpt = self._load_checkpoint()
        if ckpt is None:
            bench_ref = self.epochs.freeze(self.bench)
            question = self._create_question(question_statement, bench_ref)
            hypothesis_id = self._create_hypothesis(question)
            start_cycle, elapsed0 = 1, 0.0
        else:
            bench_ref, question, hypothesis_id = self._restore(ckpt)
            start_cycle, elapsed0 = ckpt["cycle"] + 1, ckpt["elapsed_s"]
        t0 = time.monotonic()

        for cycle in range(start_cycle, self.cfg.cycles + 1):
            if (self.cfg.max_hours is not None
                    and elapsed0 + time.monotonic() - t0
                    > self.cfg.max_hours * 3600):
                self._stop_reason = "wall_clock"
                break
            # ── FAST LOOP ────────────────────────────────────────────────
            strategy = load_strategy(self.strategy_path)  # RSI re-entry point
            proposal = self.proposer.propose(
                question, self.spec, self._current_params(),
                self.incumbent_value, self.runs, strategy)
            if proposal is None:
                # proposer exhausted this question — force the slow loop
                self._slow_loop(question, hypothesis_id, bench_ref)
                nxt = self.platform.frontier(limit=2)
                nxt = [q for q in nxt if q["id"] != question["id"]]
                if not nxt:
                    self._stop_reason = "no_frontier"
                    break
                question = nxt[0]
                hypothesis_id = self._create_hypothesis(question)
                self.incumbent_value = None
                self.noise_floor = None   # per-question, recomputed mid-loop
                bench_ref = self._switch_benchmark(question, bench_ref)
                self._save_checkpoint(cycle, question, hypothesis_id,
                                      bench_ref, elapsed0, t0)
                continue

            proposal.hypothesis_id = hypothesis_id
            try:
                run, _workdir = run_cycle(
                    self.spec, proposal, cycle, self.runner, self.gates,
                    self.incumbent_bytes, self.incumbent_value,
                    self.bench.seed, out, self.noise_floor)
            except Exception as e:  # noqa: BLE001 — a staging/scope failure
                # is ONE failed experiment (a student proposer WILL propose a
                # bad knob eventually); it must never kill the session.
                run = failed_run(self.spec, proposal, cycle, out,
                                 f"{type(e).__name__}: {e}")
            self.runs.append(run)
            self._append_run(run)

            if run.status == "keep" and run.metric.value is not None:
                self._ratchet(run)

            evidence_id = self._push_evidence(run, proposal)
            self.platform.update_confidence(hypothesis_id)
            hypothesis_id = self._maybe_promote(hypothesis_id,
                                                evidence_id, question)

            # ── MIDDLE LOOP ──────────────────────────────────────────────
            if (cycle % self.cfg.middle_every == 0
                    or cycle == self.cfg.cycles):
                # diagnosis is per-(question, epoch): a fresh question must
                # not inherit the parent's history (saturation/noise are
                # properties of THIS question's evidence)
                q_runs = [r for r in self.runs
                          if r.question_id == question["id"]]
                diag = self.diagnostician.diagnose(
                    q_runs, bench_ref.epoch, question["id"])
                if diag.mature:
                    # an immature report must not erase what the question
                    # has already learned (saturation 0.0 / noise 0.0 on
                    # "insufficient runs" would reset both)
                    self.noise_floor = diag.noise_floor
                    self.platform.update_question(
                        question["id"], {"saturation": diag.saturation})
                    question["saturation"] = diag.saturation
                if diag.mature and diag.recommendation == "upgrade_benchmark":
                    self.bench = self.upgrader.upgrade(self.bench, diag)
                    bench_ref = self.epochs.upgrade(self.bench)
                    self.platform.update_question(
                        question["id"],
                        {"benchmark_ref": {"name": bench_ref.name,
                                           "epoch": bench_ref.epoch},
                         # saturation was measured UNDER THE OLD EPOCH — the
                         # upgrade is precisely the decision "this question
                         # continues on a fresh benchmark", so reset it or the
                         # slow loop below would abandon the question in the
                         # very same cycle (threshold collision).
                         "saturation": 0.0})
                    # keep the local dict in sync: the proposer builds
                    # proposal.benchmark_ref from it, and run_cycle pairs
                    # that epoch tag with self.bench.seed — a stale dict
                    # mislabels every post-upgrade run.
                    question["benchmark_ref"] = {"name": bench_ref.name,
                                                 "epoch": bench_ref.epoch}
                    question["saturation"] = 0.0
                    # cross-epoch comparability reset: the incumbent SOLUTION
                    # persists; its value re-baselines under the new seed.
                    self.incumbent_value = None
                    self.noise_floor = None
                self._self_improve(bench_ref, question["id"])

            # ── SLOW LOOP ────────────────────────────────────────────────
            # A terminal hypothesis does NOT end the question: the fast loop
            # continues under a successor hypothesis (the research-process
            # shape — one question hosts many hypotheses over its life).
            # The question itself is only abandoned on saturation (here) or
            # proposer exhaustion (the None branch above).
            h = self.platform.get_hypothesis(hypothesis_id) or {}
            if h.get("status") in ("supported", "refuted"):
                hypothesis_id = self._create_hypothesis(question)
            if diag_saturation(question) >= self.cfg.saturation_threshold:
                self._slow_loop(question, hypothesis_id, bench_ref,
                                evidence_id)
                nxt = self.platform.frontier(limit=2)
                nxt = [q for q in nxt if q["id"] != question["id"]]
                if not nxt:
                    self._stop_reason = "no_frontier"
                    break
                question = nxt[0]
                hypothesis_id = self._create_hypothesis(question)
                self.incumbent_value = None
                self.noise_floor = None   # per-question, recomputed mid-loop
                bench_ref = self._switch_benchmark(question, bench_ref)

            self._save_checkpoint(cycle, question, hypothesis_id, bench_ref,
                                  elapsed0, t0)

        self._mark_done()
        return self._summary()

    # -- checkpoint / resume -------------------------------------------------
    #
    # A crash mid-run must not cost the session its trajectory: at the end
    # of every cycle the loop-carried state is checkpointed to
    # state/checkpoint.json (atomic write). With ``resume=True`` the engine
    # reloads the persisted runs/ledger/epochs/KB (all already on disk),
    # restores the loop variables, and continues at the next cycle. The
    # incumbent's editable bytes come back from the incumbent run's workdir
    # (workdirs persist under out/runs/).

    def _checkpoint_path(self) -> Path:
        return Path(self.cfg.out_dir) / "state" / "checkpoint.json"

    def _save_checkpoint(self, cycle: int, question: dict,
                         hypothesis_id: str, bench_ref: BenchmarkRef,
                         elapsed0: float, t0: float) -> None:
        proposer_state = {}
        getter = getattr(self.proposer, "state_dict", None)
        if callable(getter):
            proposer_state = getter()
        payload = {
            "version": 1,
            "done": False,
            "cycle": cycle,
            "question_id": question["id"],
            "hypothesis_id": hypothesis_id,
            "bench_ref": {"name": bench_ref.name, "epoch": bench_ref.epoch,
                          "task_version": bench_ref.task_version},
            "bench_seed": self.bench.seed,
            "incumbent_value": self.incumbent_value,
            "incumbent_workdir": self.incumbent_workdir,
            "noise_floor": self.noise_floor,
            "q_seq": self._q_seq,
            "elapsed_s": elapsed0 + (time.monotonic() - t0),
            "proposer_state": proposer_state,
        }
        path = self._checkpoint_path()
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        tmp.replace(path)  # atomic on POSIX: never a torn checkpoint

    def _load_checkpoint(self) -> dict | None:
        if not self.cfg.resume:
            return None
        path = self._checkpoint_path()
        if not path.exists():
            return None
        ckpt = json.loads(path.read_text())
        if ckpt.get("done"):
            raise SystemExit(
                f"error: {path} marks this session complete "
                f"({ckpt['cycle']} cycles) — nothing to resume")
        return ckpt

    def _restore(self, ckpt: dict):
        """Rebuild in-memory state from the checkpoint + persisted files."""
        runs_path = Path(self.cfg.out_dir) / "state" / "runs.jsonl"
        self.runs = [
            RunRecord.from_dict(json.loads(l))
            for l in runs_path.read_text().splitlines()
            if l.strip()
        ]
        self.incumbent_value = ckpt["incumbent_value"]
        self.incumbent_workdir = ckpt["incumbent_workdir"]
        self.incumbent_bytes = (
            (Path(self.incumbent_workdir) / self.spec.editable[0]).read_bytes()
            if self.incumbent_workdir
            else None
        )
        self.noise_floor = ckpt["noise_floor"]
        self._q_seq = ckpt["q_seq"]
        ref = ckpt["bench_ref"]
        self.bench = Benchmark(
            name=ref["name"], task=self.spec.name,
            metric_name=self.spec.metric.name,
            metric_direction=self.spec.metric.direction,
            seed=ckpt["bench_seed"])
        setter = getattr(self.proposer, "load_state_dict", None)
        if callable(setter):
            setter(ckpt.get("proposer_state") or {})
        question = self.platform.get_question(ckpt["question_id"])
        if question is None:
            raise SystemExit(
                f"error: checkpoint references question "
                f"{ckpt['question_id']} missing from the KB — the session "
                f"dir is inconsistent, refusing to resume")
        bench_ref = BenchmarkRef(**ref)
        return bench_ref, question, ckpt["hypothesis_id"]

    def _mark_done(self) -> None:
        path = self._checkpoint_path()
        if path.exists():
            ckpt = json.loads(path.read_text())
            ckpt["done"] = True
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2))
            tmp.replace(path)

    # -- fast-loop helpers ---------------------------------------------------

    def _switch_benchmark(self, question: dict,
                          bench_ref: BenchmarkRef) -> BenchmarkRef:
        """Bind the loop to the question's stored epoch — including its SEED.

        A frontier switch can land on a question bound to an older epoch;
        the run must be executed with the seed of the epoch it is tagged
        with, or the epoch tag lies (within-epoch comparability broken)."""
        ref = BenchmarkRef(
            name=question["benchmark_ref"]["name"],
            epoch=question["benchmark_ref"]["epoch"],
            task_version=bench_ref.task_version)
        seed = self.epochs.epoch_seed(ref.epoch)
        if seed is not None and seed != self.bench.seed:
            self.bench = Benchmark(
                name=self.bench.name, task=self.bench.task,
                metric_name=self.bench.metric_name,
                metric_direction=self.bench.metric_direction, seed=seed)
        return ref

    def _ratchet(self, run: RunRecord) -> None:
        """keep ∧ beats incumbent → incumbent bytes/value become this run's."""
        self.incumbent_value = run.metric.value
        self.incumbent_workdir = run.workdir
        editable = Path(run.workdir) / self.spec.editable[0]
        self.incumbent_bytes = editable.read_bytes()

    _FAILURE_CLASSES = {"timeout": "timeout", "crash": "harness",
                        "missing_result": "harness", "missing_metric": "harness",
                        "invalid_metric": "divergence"}

    def _push_evidence(self, run: RunRecord, proposal) -> str:
        editable_bytes = (Path(run.workdir) / self.spec.editable[0]).read_bytes()
        fm = run.metric.failure_mode
        evidence = MachineEvidence(
            id=f"ev-{run.run_id}", epoch=run.epoch,
            run_result={
                "run_id": run.run_id, "env": run.task, "status": run.status,
                "failure_mode": fm,
                "created": run.created, "duration_s": int(run.duration_s),
                "hypothesis": run.hypothesis_id,
                "variable_changed": run.variable_changed,
                "metric": {
                    "name": run.metric.name, "direction": run.metric.direction,
                    "value": run.metric.value, "baseline": run.metric.baseline,
                    "delta": run.metric.delta},
                "benchmark_ref": {**proposal.benchmark_ref,
                                  "task_version": "v1"},
                "executor": "local",
            },
            links=self._links(run),
            run={"revision": run.run_id,
                 "source_hash": hashlib.sha256(editable_bytes).hexdigest()},
            failure=({"class": self._FAILURE_CLASSES.get(fm, "unknown"),
                      "diagnosis": fm or "unknown"}
                     if run.status == "fail" else None),
        )
        evidence_id = self.platform.push_run(evidence.to_dict())
        # best-result ratchet on the question (direction-aware, EPOCH-aware:
        # cross-epoch scores are not comparable, so a keep from a newer
        # epoch replaces the record wholesale and the old score is never
        # used as a comparison baseline)
        q = self.platform.get_question(run.question_id) or {}
        cur = q.get("best_result")
        same_epoch = cur is not None and cur.get("epoch") == run.epoch
        better = (cur is None or not same_epoch
                  or (run.metric.value is not None
                      and (cur["score"] > run.metric.value
                           if run.metric.direction == "min"
                           else cur["score"] < run.metric.value)))
        if run.status == "keep" and better:
            self.platform.update_question(
                run.question_id,
                {"best_result": {"evidence_id": evidence_id,
                                 "score": run.metric.value,
                                 "epoch": run.epoch}})
        self.platform.attach_evidence(run.question_id, [evidence_id])
        return evidence_id

    def _links(self, run: RunRecord) -> list[EvidenceLink]:
        """The link policy: verdict + metric relation → supports/refutes/
        inconclusive on the hypothesis (and question)."""
        hid = run.hypothesis_id
        delta = run.metric.delta
        if run.status == "fail":
            return [EvidenceLink("inconclusive", hid, 0.3, 0.5)]
        if delta is None:  # baseline-establishing run: informs, does not test
            return [EvidenceLink("inconclusive", hid, 0.3, 0.5)]
        floor = self.noise_floor if self.noise_floor is not None else 0.0
        if run.status == "keep" and delta > floor:
            base = run.metric.baseline
            rel = (delta / abs(base)) if base not in (None, 0) else 1.0
            return [EvidenceLink("supports", hid, min(1.0, rel), 0.9)]
        if run.status == "discard" and delta <= 0:
            return [EvidenceLink("refutes", hid, 0.5, 0.5)]
        return [EvidenceLink("inconclusive", hid, 0.4, 0.5)]

    def _maybe_promote(self, hid: str, evidence_id: str,
                       question: dict) -> str:
        """Try the count-gated terminal transitions for the hypothesis.
        The QUESTION's own transition (partially_answered) is the slow
        loop's decision, made when the question is actually abandoned —
        not a side effect of its first hypothesis resolving.

        Policy (engine-level; the platform still enforces its own count
        thresholds): a hypothesis is only attempted as ``refuted`` on NET
        refutation (refute >= 2 and refute > support) — a single flat run
        must not terminalize a hypothesis that also has supporting keeps,
        or the slow loop churns questions without ever accumulating
        evidence."""
        h = self.platform.get_hypothesis(hid) or {}
        counts = h.get("evidence_counts") or {}
        support, refute = counts.get("support", 0), counts.get("refute", 0)
        to = "refuted" if (refute >= 2 and refute > support) else "supported"
        try:
            self.platform.transition_hypothesis(hid, to)
        except Exception:  # noqa: BLE001 — thresholds decide, not us
            return hid
        return hid

    # -- middle-loop helper: system-level self-improvement -------------------

    def _current_params(self) -> dict | None:
        """The incumbent's knob values (or the pristine file's on cycle 1)."""
        from rsi.tasks.runner import parse_params
        source = ((self.incumbent_bytes
                   if self.incumbent_bytes is not None
                   else self.spec.editable_file.read_bytes())
                  .decode("utf-8"))
        return {k: v for k, v in parse_params(source).items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}

    def _self_improve(self, bench_ref: BenchmarkRef,
                      question_id: str) -> None:
        eff = build_efficiency_report(self.runs)
        active = self.ledger.active_observation()
        if active is not None:
            applied_at = active.get("applied_at_cycle",
                                    len(self.runs)) or len(self.runs)
            # slice FIRST (the index refers to self.runs), then restrict to
            # one question and exclude all-abstain baseline re-runs (they
            # inflate keep-rate and mask degradation)
            def _arb(runs):
                return [r for r in runs
                        if r.question_id == question_id
                        and r.variable_changed != "baseline"]
            baseline = _arb(self.runs[:applied_at])
            if not baseline:
                # the question switched while the edit was observing: a
                # strategy edit is GLOBAL, so an empty same-question baseline
                # falls back to the whole pre-edit session — arbitrating
                # against keep_rate([]) == 0.0 would auto-keep anything.
                baseline = [r for r in self.runs[:applied_at]
                            if r.variable_changed != "baseline"]
            window = _arb(self.runs[applied_at:])
            self.improver.arbitrate(active["edit_id"], baseline, window)
            self.ledger.mark_terminated_at(active["edit_id"],
                                           len(self.runs))
            return
        if not eff.waste_patterns:
            return
        for cand in self.improver.propose(eff, self.strategy_path):
            try:
                edit_id = self.improver.apply(cand)
                self.ledger.mark_applied_at(edit_id, len(self.runs))
                break  # at most one live edit at a time
            except ImproverError:
                continue  # guard rejected this candidate; try the next

    # -- slow loop -----------------------------------------------------------

    def _slow_loop(self, question: dict, hypothesis_id: str,
                   bench_ref: BenchmarkRef, evidence_id: str | None = None,
                   ) -> None:
        """The question is being abandoned (saturated or proposer-exhausted):
        record its partial progress (conversion event) and ⑦ spawn a child
        question from the best evidence — edge extrapolation.

        TODO(homework): this naive strategy spawns ONE template child per
        abandonment. Real question-graph evolution — refine/decompose,
        challenges edges from contradictory evidence, merges of overlapping
        questions, typed-edge traversal in proposal context — is
        docs/homework.md task K-1."""
        # re-fetch: the caller's dict may predate best_result writes
        question = self.platform.get_question(question["id"]) or question
        best = question.get("best_result") or {}
        src = evidence_id or best.get("evidence_id")
        if not src:
            return
        if question["status"] == "open":
            try:
                self.platform.transition_question(
                    question["id"], "partially_answered",
                    conversion_event={"evidence_refs": [src],
                                      "hypothesis": hypothesis_id})
            except Exception:  # noqa: BLE001 — already transitioned is fine
                pass
        self._q_seq += 1
        child_id = f"{self.spec.name}-q{self._q_seq}"
        child = {
            "id": child_id,
            "statement": (f"Can {self.spec.metric.name} be pushed beyond "
                          f"{best.get('score')} with a different update-rule "
                          f"family on {self.spec.name}?"),
            "scope": question["scope"],
            "benchmark_ref": {"name": bench_ref.name,
                              "epoch": bench_ref.epoch},
            "status": "open", "version": 1, "saturation": 0.0,
            "uncertainty": [], "edges": [],
        }
        self.platform.spawn_question(
            question["id"], child,
            source={"evidence_id": src, "confidence": 0.6})

    # -- setup helpers ---------------------------------------------------------

    def _create_question(self, statement: str, bench_ref: BenchmarkRef) -> dict:
        return self.platform.create_question({
            "id": f"{self.spec.name}-q0",
            "statement": statement,
            "scope": self.spec.protocol_body[:400] or self.spec.name,
            "benchmark_ref": {"name": bench_ref.name, "epoch": bench_ref.epoch},
            "status": "open", "version": 1, "saturation": 0.0,
            "uncertainty": [], "edges": [],
        })

    def _create_hypothesis(self, question: dict) -> str:
        """Create a fresh hypothesis on the question. Ids are sequential per
        question (-h0, -h1, ...): revisiting an open question reuses the
        next free id instead of colliding with its earlier hypothesis."""
        n = 0
        while self.platform.get_hypothesis(f"{question['id']}-h{n}") is not None:
            n += 1
        hid = f"{question['id']}-h{n}"
        self.platform.create_hypothesis({
            "id": hid,
            "statement": (f"Tuning the editable knobs on "
                          f"{self.spec.name} improves {self.spec.metric.name} "
                          f"beyond the current incumbent"),
            "derived_from": [{"interface": "I-Q", "object_id": question["id"]}],
            "testability": {
                "procedure": (f"run {self.spec.entrypoint} via the harness "
                              f"with one knob changed per cycle"),
                "expected_evidence": (f"a keep run whose {self.spec.metric.name}"
                                      f" delta exceeds the epoch noise floor"),
                "falsifiable_by": ("a run of kept knob changes whose deltas "
                                   "stay within the noise floor"),
            },
            "status": "draft", "version": 1,
        })
        self.platform.transition_hypothesis(hid, "active")
        return hid

    def _append_run(self, run: RunRecord) -> None:
        path = Path(self.cfg.out_dir) / "state" / "runs.jsonl"
        with path.open("a") as fh:
            fh.write(json.dumps(run.to_dict()) + "\n")

    def _summary(self) -> dict:
        eff = build_efficiency_report(self.runs)
        folds = self.ledger.fold()
        out = {
            "task": self.spec.name,
            "cycles": len(self.runs),
            "keep": eff.keep, "discard": eff.discard, "fail": eff.fail,
            "keep_rate": eff.keep_rate,
            "n_new_bests": eff.n_new_bests,
            "improvements": {eid: row["status"] for eid, row in folds.items()},
            "questions": [q["id"] for q in self.platform.questions.all()],
            "epochs": self.epochs.epochs,
            "best": {q["id"]: q.get("best_result")
                     for q in self.platform.questions.all()},
        }
        if self._stop_reason is not None:
            out["stop_reason"] = self._stop_reason
        return out


def diag_saturation(question: dict) -> float:
    return float(question.get("saturation") or 0.0)


def build_default_engine(cfg: EngineConfig) -> RSIEngine:
    """Composition root wiring the NAIVE reference implementations.

    Students: this is the swap point — register your own Proposer /
    SelfImprover / Diagnostician / BenchmarkUpgrader / gates here (or add a
    config-driven registry) without touching RSIEngine.run.
    """
    from rsi.tasks.spec import load_task

    spec = load_task(cfg.task_dir)
    out = Path(cfg.out_dir)
    platform = KnowledgePlatform(out / "kb")
    epochs = EpochManager(out / "state" / "epochs.json")
    ledger = EditLedger(out / "state" / "ledger.jsonl")
    strategy_path = ensure_strategy_file(out / "state" / "strategy.py")
    improver = (NaiveImprover(ledger, strategy_path)
                if cfg.improver == "naive" else NoopImprover())
    return RSIEngine(
        spec=spec, platform=platform, epochs=epochs, ledger=ledger,
        strategy_path=strategy_path,
        proposer=HeuristicProposer(),
        improver=improver,
        diagnostician=NaiveDiagnostician(),
        upgrader=SeedBumpUpgrader(),
        runner=LocalRunner(),
        gates=GateChain([CollapseGate(), NoChangeGate(), NoiseGate()]),
        cfg=cfg)
