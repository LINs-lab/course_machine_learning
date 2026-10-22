import pytest

from rsi.core.improver import (ImprovementProposal, ImproverError,
                               NaiveImprover, NoopImprover, keep_rate)
from rsi.core.ledger import EditLedger
from rsi.core.proposer import ensure_strategy_file, load_strategy
from rsi.core.types import EfficiencyReport, RunRecord
from rsi.evaluation.gates import MetricReading


@pytest.fixture
def strategy(tmp_path):
    return ensure_strategy_file(tmp_path / "state" / "strategy.py")


@pytest.fixture
def ledger(tmp_path):
    return EditLedger(tmp_path / "state" / "ledger.jsonl")


def _report(no_change=2, n=4):
    return EfficiencyReport(
        n_cycles=n, keep=n - no_change, discard=no_change, fail=0,
        keep_rate=(n - no_change) / n, wasted_fraction=no_change / n,
        n_new_bests=1,
        waste_patterns=[{"pattern": "no_change", "surface": "proposer",
                         "support": no_change, "est_cost": no_change}])


def _run(i, status, delta=None, value=1.0):
    return RunRecord(
        run_id=f"cycle-{i:03d}", proposal_id=f"p{i}", hypothesis_id="h",
        question_id="q", task="t", epoch="e001", status=status,
        metric=MetricReading(name="m", direction="min", value=value,
                             baseline=None if delta is None else value + delta),
        variable_changed=f"v{i}", created="", duration_s=0.1)


def test_noop_improver_never_proposes(strategy):
    assert NoopImprover().propose(_report(), strategy) == []
    with pytest.raises(ImproverError):
        NoopImprover().apply(ImprovementProposal(
            "ed-x", "state/strategy.py", strategy, "r", "X = 1\n"))


def test_naive_propose_triggers_on_waste(strategy):
    imp = NaiveImprover(EditLedger(strategy.parent / "ledger.jsonl"),
                        strategy)
    cands = imp.propose(_report(no_change=2, n=4), strategy)
    assert len(cands) == 1
    assert cands[0].surface == "state/strategy.py"
    # content-addressed: same input → same edit id
    again = imp.propose(_report(no_change=2, n=4), strategy)
    assert again[0].edit_id == cands[0].edit_id


def test_naive_no_trigger_without_waste(strategy):
    imp = NaiveImprover(EditLedger(strategy.parent / "l.jsonl"), strategy)
    assert imp.propose(_report(no_change=0, n=4), strategy) == []
    # below the 30% threshold
    assert imp.propose(_report(no_change=1, n=6), strategy) == []


def test_apply_writes_guarded_and_ledger_observing(strategy, ledger):
    imp = NaiveImprover(ledger, strategy)
    before = load_strategy(strategy)
    cand = imp.propose(_report(), strategy)[0]
    edit_id = imp.apply(cand)
    assert ledger.fold()[edit_id]["status"] == "observing"
    after = load_strategy(strategy)
    assert after["PRIORITY"] == before["PRIORITY"][1:] + before["PRIORITY"][:1]
    assert after["EXPLORE_WIDTH"] == before["EXPLORE_WIDTH"]


def test_arbitrate_kept_on_no_degradation(strategy, ledger):
    imp = NaiveImprover(ledger, strategy)
    edit_id = imp.apply(imp.propose(_report(), strategy)[0])
    baseline = [_run(1, "keep"), _run(2, "discard")]
    window = [_run(3, "keep"), _run(4, "keep")]
    v = imp.arbitrate(edit_id, baseline, window)
    assert v.status == "kept"
    assert ledger.fold()[edit_id]["status"] == "kept"
    assert load_strategy(strategy)["PRIORITY"][0] == "LR"  # edit stayed live


def test_arbitrate_reverted_restores_bytes(strategy, ledger):
    imp = NaiveImprover(ledger, strategy)
    original = strategy.read_text()
    edit_id = imp.apply(imp.propose(_report(), strategy)[0])
    assert strategy.read_text() != original
    baseline = [_run(1, "keep"), _run(2, "keep"), _run(3, "keep")]
    window = [_run(4, "discard"), _run(5, "discard")]
    v = imp.arbitrate(edit_id, baseline, window)
    assert v.status == "reverted"
    assert strategy.read_text() == original  # byte-exact restore
    assert ledger.fold()[edit_id]["status"] == "reverted"


def test_short_window_keeps_by_default(strategy, ledger):
    imp = NaiveImprover(ledger, strategy)
    edit_id = imp.apply(imp.propose(_report(), strategy)[0])
    v = imp.arbitrate(edit_id, [_run(1, "keep")], [_run(2, "discard")])
    assert v.status == "kept"


def test_apply_rejects_outside_surface(strategy, ledger, tmp_path):
    other = tmp_path / "scorer.py"
    other.write_text("# <<<SELF-IMPROVE-MUTABLE\nX = 1\n"
                     "# SELF-IMPROVE-MUTABLE>>>\n")
    imp = NaiveImprover(ledger, strategy)
    cand = ImprovementProposal(
        "ed-evil", "src/rsi/evaluation/gates.py", other, "malicious",
        "keep_everything = True\n")
    # rejected by the target check (the guard's surface-allowlist layer is
    # covered separately in test_guard) — either way: no write, ledger row
    with pytest.raises(ImproverError, match="guard rejected"):
        imp.apply(cand)
    assert ledger.fold()["ed-evil"]["status"] == "rejected"
    assert other.read_text().startswith("# <<<")  # untouched


def test_keep_rate_helper():
    runs = [_run(1, "keep"), _run(2, "discard"), _run(3, "keep")]
    assert keep_rate(runs) == pytest.approx(2 / 3)
    assert keep_rate([]) == 0.0
