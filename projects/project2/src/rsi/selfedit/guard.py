# Ported (semantics preserved) from
# project_collab_auto_research/graph/self_improve/guard.py, layers 1-3 plus the
# surface allowlist idea from check_gate_candidate's protected-files rule.
"""Source-edit guard: pre-filter for every self-improvement candidate.

Layers (first failure short-circuits):

0. surface allowlist — the candidate's target file must be in
   EDITABLE_SURFACES. This is the trust boundary made executable: the
   proposer/strategy layer may edit itself; the scorer and everything else
   may never be touched by the improver.
1. sentinel integrity — exactly one well-formed mutable region, no smuggled
   boundary tokens;
2. byte-identity — everything outside the mutable region is unchanged;
3. python validity + mutable-region statement shape locked to the canonical
   (a candidate may change a statement's INTERNALS but not the top-level
   statement structure — no smuggled assignments; the lock covers ALL
   assignment targets and the RHS value TYPE, so neither
   ``PRIORITY = os.system = [...]`` nor ``EXPLORE_WIDTH = "lots"`` passes).

Layer 4 (semantic override scan by an independent judge) is intentionally a
seam: TODO(homework) in docs/homework.md — the naive engine passes a trivial
judge that accepts only shape-identical data blocks.

NOTE (documented boundary, not a bug): layers 0-3 are STRUCTURAL. A knob's
right-hand side may be any expression of the same type — the guard cannot
tell ``EXPLORE_WIDTH = 6`` from ``EXPLORE_WIDTH = int(__import__("os").
popen("id").read()[:1])``. Semantic containment is layer 4's job (homework
C-3); until a real judge is wired, treat the mutable region as data-only by
convention and review kept edits in the ledger.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from rsi.selfedit import sentinels

# The ONLY surfaces a self-improvement candidate may target. Relative to the
# engine's state directory. Everything else — rsi/evaluation/*, the gates, the
# task scorer — is frozen by construction (guard rejects before any write).
EDITABLE_SURFACES = frozenset({"state/strategy.py"})


@dataclass
class GuardVerdict:
    ok: bool
    layer: str = ""
    reason: str = ""


def _rhs_sig(value: ast.expr):
    """The RHS signature locked by layer 3: expression kind, plus the python
    type for constants. Prevents same-shape type confusion (``EXPLORE_WIDTH
    = "lots"``) while leaving the VALUE free (data-only mutable region)."""
    if isinstance(value, ast.Constant):
        return ("Constant", type(value.value).__name__)
    return (type(value).__name__,)


def _mutable_shape(src: str):
    """The top-level statement SIGNATURE of a mutable region — (kind, names,
    rhs signature) per statement. The candidate must preserve the canonical's
    shape: it may change a statement's INTERNALS but not add/remove/retype a
    top-level statement (else a candidate could smuggle e.g.
    ``_build_prompt = <lambda>`` inside the sentinels — passes byte-identity
    + ast, hijacks behavior at import). ALL assignment targets are locked,
    not just plain Names: a non-Name target (``x = os.system = [...]``)
    contributes its dumped form, so multi-target smuggles change the shape."""
    out = []
    for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append((type(n).__name__, n.name, ()))
        elif isinstance(n, ast.Assign):
            names = tuple(t.id if isinstance(t, ast.Name) else ast.dump(t)
                          for t in n.targets)
            out.append(("Assign", names, _rhs_sig(n.value)))
        elif isinstance(n, ast.AnnAssign):
            target = (n.target.id if isinstance(n.target, ast.Name)
                      else ast.dump(n.target))
            out.append(("AnnAssign", (target,), _rhs_sig(n.value)
                        if n.value is not None else ()))
        else:
            out.append((type(n).__name__, None, ()))
    return tuple(out)


def check_candidate(canonical_source: str, candidate_source: str,
                    judge=None, surface: str | Path = "state/strategy.py",
                    ) -> GuardVerdict:
    """Run layers 0-3 (and 4 when a judge is given); first failure wins.

    ``judge(mutable_text) -> str`` must answer with ``override: yes|no``;
    any exception or unparseable answer is a REJECT (default-reject, never
    default-accept).
    """
    surface_key = str(surface).replace("\\", "/")
    if surface_key not in EDITABLE_SURFACES:
        return GuardVerdict(
            False, "surface-allowlist",
            f"target {surface_key!r} is not an editable surface; "
            f"editable: {sorted(EDITABLE_SURFACES)}")
    # 1. sentinel integrity of the candidate
    try:
        c_before, c_mut, c_after = sentinels.split_regions(candidate_source)
    except sentinels.SentinelError as e:
        return GuardVerdict(False, "sentinel-integrity", str(e))
    # 2. byte-identity of immutable source
    try:
        k_before, k_mut, k_after = sentinels.split_regions(canonical_source)
    except sentinels.SentinelError as e:
        return GuardVerdict(False, "sentinel-integrity", f"canonical: {e}")
    if c_before != k_before or c_after != k_after:
        return GuardVerdict(False, "byte-identity", "immutable source changed")
    # 3. python validity + statement-shape lock
    try:
        ast.parse(candidate_source)
        cand_shape = _mutable_shape(c_mut)
        canon_shape = _mutable_shape(k_mut)
    except SyntaxError as e:
        return GuardVerdict(False, "ast", f"syntax error: {e}")
    if cand_shape != canon_shape:
        return GuardVerdict(
            False, "mutable-shape",
            "mutable region changed its top-level statement structure "
            f"({canon_shape} -> {cand_shape})")
    # 4. semantic override scan (seam; default-reject on any uncertainty)
    if judge is not None:
        import re
        override_re = re.compile(r"override:\s*(yes|no)", re.IGNORECASE)
        try:
            raw = judge(c_mut)
            m = override_re.search(raw or "")
            if m is None or m.group(1).lower() == "yes":
                return GuardVerdict(
                    False, "override-scan",
                    "judge flagged override or returned no parseable verdict")
        except Exception as e:  # noqa: BLE001 — default-reject on any failure
            return GuardVerdict(
                False, "override-scan", f"judge unavailable: {type(e).__name__}")
    return GuardVerdict(True)
