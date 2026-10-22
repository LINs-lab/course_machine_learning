# Ported (semantics preserved, Stage2-lite) from
# project_collab_auto_research/graph/self_improve/{meta_trigger,stage2}.
# py: the waste-signal → guarded edit → observe → arbitrate → kept|reverted
# cycle. The C/F split of the source design is preserved: the improver may
# rewrite ONLY the proposer's strategy artifact inside its sentinel region
# (EDITABLE_SURFACES in rsi.selfedit.guard); the scorer, gates, and every
# other module are frozen — the loop can never rewrite its own grader.
"""The SelfImprover seam: the system-level half of the RSI essence.

Flow (engine-driven, one live candidate at a time):

1. ``propose(efficiency_report, source_path)`` — turn waste signals into
   candidate edits of the strategy artifact's mutable region;
2. ``apply(proposal)`` — guard layers 0-3 (surface allowlist, sentinel
   integrity, byte-identity, statement-shape lock), then write + ledger
   ``applying → observing`` (prev bytes snapshotted for revert);
3. RE-ENTRY — nothing else happens by design: the next cycle's proposer
   re-reads the edited strategy, so a live edit immediately shapes the next
   proposal. That is the recursion;
4. ``arbitrate(edit_id, baseline_runs, window_runs)`` — Stage2-lite: compare
   keep-rate over the pre-edit window vs the post-edit window; degradation
   ⇒ ``reverted`` (bytes restored from the ledger), otherwise ``kept``
   (== promoted: the edit stays live and compounds).

The shipped ``NaiveImprover`` rotates the strategy's knob priority when
no-change waste dominates — a deliberately crude example. Real improvers
(prompt/parameter search, sentinel-region code edits, profile promotion)
are TODO(homework) (docs/homework.md, tasks C-2/C-4).
"""
from __future__ import annotations

import ast
import base64
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from rsi.core.ledger import EditLedger, EditRow
from rsi.core.types import EfficiencyReport, RunRecord
from rsi.evaluation.gates import Verdict
from rsi.selfedit import sentinels
from rsi.selfedit.guard import check_candidate

STAGE2_DEGRADE_FACTOR = 0.5   # keep_rate(new) < factor * keep_rate(base) ⇒ revert
MIN_WINDOW = 2                # runs needed before arbitration is meaningful


def keep_rate(runs: list[RunRecord]) -> float:
    if not runs:
        return 0.0
    return sum(1 for r in runs if r.status == "keep") / len(runs)


class ImproverError(RuntimeError):
    pass


@dataclass
class ImprovementProposal:
    edit_id: str
    surface: str                       # must be in EDITABLE_SURFACES
    target: Path
    rationale: str
    candidate_mutable: str             # NEW text for the sentinel region
    waste_refs: list[str] = field(default_factory=list)


class SelfImprover(Protocol):
    def propose(self, report: EfficiencyReport,
                source_path: Path) -> list[ImprovementProposal]: ...
    def apply(self, p: ImprovementProposal) -> str: ...
    def arbitrate(self, edit_id: str, baseline: list[RunRecord],
                  window: list[RunRecord]) -> Verdict: ...


class NoopImprover:
    """The honest baseline: never self-improves. Swap in a real one."""

    def propose(self, report: EfficiencyReport,
                source_path: Path) -> list[ImprovementProposal]:
        return []

    def apply(self, p: ImprovementProposal) -> str:
        raise ImproverError("NoopImprover never applies edits")

    def arbitrate(self, edit_id: str, baseline: list[RunRecord],
                  window: list[RunRecord]) -> Verdict:
        return Verdict("kept", ["noop"])


class NaiveImprover:
    """One crude trigger: when no_change waste dominates, rotate the knob
    priority in the strategy file (guard-checked, revertible)."""

    def __init__(self, ledger: EditLedger, strategy_path: str | Path):
        self.ledger = ledger
        self.strategy_path = Path(strategy_path)

    def propose(self, report: EfficiencyReport,
                source_path: Path) -> list[ImprovementProposal]:
        no_change = next((p["support"] for p in report.waste_patterns
                          if p["pattern"] == "no_change"), 0)
        if no_change == 0 or report.n_cycles == 0:
            return []
        if no_change / report.n_cycles < 0.3:
            return []
        _, mutable, _ = sentinels.split_regions(source_path.read_text())
        rotated = _rotate_first_list_value(mutable)
        if rotated is None:
            return []
        # content-addressed edit id: same candidate text = same edit (and
        # deterministic across runs — no uuid, no clock). A candidate whose
        # id already reached a terminal state is never re-proposed: a
        # reverted edit re-applying under the same id would conflate two
        # lifecycles in the append-only ledger.
        digest = hashlib.sha256(rotated.encode()).hexdigest()[:8]
        edit_id = f"ed-{digest}"
        if edit_id in self.ledger.fold():
            return []
        return [ImprovementProposal(
            edit_id=edit_id,
            surface="state/strategy.py", target=source_path,
            rationale="no_change waste dominates; rotate knob priority so the"
                      " grid walk explores a different axis first",
            candidate_mutable=rotated,
            waste_refs=["no_change"])]

    def apply(self, p: ImprovementProposal) -> str:
        # The guard checks the declared SURFACE label; the write target is
        # this improver's responsibility — they must be the same file, or
        # the trust boundary is one spoofed metadata field away.
        if p.target.resolve() != self.strategy_path.resolve():
            self.ledger.append(EditRow(
                edit_id=p.edit_id, surface=p.surface, target=str(p.target),
                rationale=p.rationale, status="rejected",
                waste_refs=p.waste_refs))
            raise ImproverError(
                f"guard rejected: target {p.target} is not the editable "
                f"surface {self.strategy_path}")
        canonical = p.target.read_text()
        candidate = sentinels.replace_mutable(canonical, p.candidate_mutable)
        verdict = check_candidate(canonical, candidate,
                                  surface=p.surface)
        if not verdict.ok:
            self.ledger.append(EditRow(
                edit_id=p.edit_id, surface=p.surface,
                target=str(p.target), rationale=p.rationale,
                status="rejected", waste_refs=p.waste_refs,
                prev_mutable_b64=_b64(sentinels.split_regions(canonical)[1]),
                candidate_mutable=p.candidate_mutable))
            raise ImproverError(f"guard rejected: {verdict.layer}: "
                                f"{verdict.reason}")
        # Ledger BEFORE the write: the applying row carries the revert
        # snapshot (prev_mutable_b64). Writing the file first would leave a
        # live, unrevertable edit if the process died in between.
        self.ledger.append(EditRow(
            edit_id=p.edit_id, surface=p.surface, target=str(p.target),
            rationale=p.rationale, status="applying",
            waste_refs=p.waste_refs,
            prev_mutable_b64=_b64(sentinels.split_regions(canonical)[1]),
            candidate_mutable=p.candidate_mutable))
        p.target.write_text(candidate)
        self.ledger.update_status(p.edit_id, "observing")
        return p.edit_id

    def arbitrate(self, edit_id: str, baseline: list[RunRecord],
                  window: list[RunRecord]) -> Verdict:
        if len(window) < MIN_WINDOW:
            # too little post-edit evidence to judge — keep by default, but
            # STILL record the terminal decision (an edit left 'observing'
            # forever would block every future improvement slot)
            self.ledger.update_status(edit_id, "kept")
            return Verdict("kept", ["window too short — keep by default"])
        base_rate = keep_rate(baseline)
        new_rate = keep_rate(window)
        if base_rate > 0 and new_rate < STAGE2_DEGRADE_FACTOR * base_rate:
            self._revert(edit_id)
            return Verdict("reverted",
                           [f"keep-rate degraded {base_rate:.2f}→{new_rate:.2f}"])
        self.ledger.update_status(edit_id, "kept")
        return Verdict("kept", [f"keep-rate {base_rate:.2f}→{new_rate:.2f}"])

    def _revert(self, edit_id: str) -> None:
        row = self.ledger.fold()[edit_id]
        canonical = Path(row["target"]).read_text()
        restored = sentinels.replace_mutable(
            canonical, _unb64(row["prev_mutable_b64"]))
        Path(row["target"]).write_text(restored)
        self.ledger.update_status(edit_id, "reverted")


def _rotate_first_list_value(mutable: str) -> str | None:
    """Deterministically rotate the first list literal's elements (shape
    preserved — an Assign of the same name to a new list literal)."""
    lines = mutable.splitlines(keepends=True)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith(("PRIORITY", "EXPLORE", "#")):
            continue
        if "=" not in stripped or "[" not in stripped:
            continue
        try:
            value = ast.literal_eval(stripped.split("=", 1)[1].strip())
        except (ValueError, SyntaxError):
            continue
        if not isinstance(value, list) or len(value) < 2:
            continue
        name = stripped.split("=", 1)[0].strip()
        rotated = value[1:] + value[:1]
        indent = line[:len(line) - len(line.lstrip())]
        lines[i] = f"{indent}{name} = {rotated!r}\n"
        return "".join(lines)
    return None


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def _unb64(text: str) -> str:
    return base64.b64decode(text.encode()).decode()
