import pytest

from rsi.evaluation.benchmark import (Benchmark, EpochFrozenError,
                                      EpochManager, SeedBumpUpgrader)


def _bench(seed=7):
    return Benchmark(name="quad-optim", task="quad-optim",
                     metric_name="final_loss", metric_direction="min",
                     seed=seed)


def test_freeze_opens_first_epoch(tmp_path):
    m = EpochManager(tmp_path / "epochs.json")
    ref = m.freeze(_bench())
    assert ref.epoch == "e001" and ref.task_version == "v1"


def test_refreeze_same_identity_ok(tmp_path):
    m = EpochManager(tmp_path / "epochs.json")
    m.freeze(_bench())
    assert m.freeze(_bench(seed=99)).epoch == "e001"  # seed re-read for runs


def test_refreeze_different_identity_rejected(tmp_path):
    m = EpochManager(tmp_path / "epochs.json")
    m.freeze(_bench())
    other = _bench()
    other.task_version = "v2"
    with pytest.raises(EpochFrozenError):
        m.freeze(other)


def test_upgrade_archives_and_increments(tmp_path):
    p = tmp_path / "epochs.json"
    m = EpochManager(p)
    m.freeze(_bench())
    ref2 = m.upgrade(_bench(seed=8))
    assert ref2.epoch == "e002"
    # persistence: a fresh manager sees the same history
    m2 = EpochManager(p)
    assert [e["epoch"] for e in m2.epochs] == ["e001", "e002"]
    assert m2.current().epoch == "e002"
    assert m2.epochs[0]["seed"] == 7 and m2.epochs[1]["seed"] == 8


def test_seed_bump_upgrader_deterministic():
    up = SeedBumpUpgrader()
    b2 = up.upgrade(_bench(seed=7), None)
    assert b2.seed == 8 and b2.name == "quad-optim"
