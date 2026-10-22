import pytest

from rsi.core.ledger import EditLedger, EditRow, LedgerError


def _row(edit_id="ed-1", status="applying"):
    return EditRow(edit_id=edit_id, surface="state/strategy.py",
                   target="state/strategy.py", rationale="r",
                   status=status, prev_mutable_b64="QUJD",
                   candidate_mutable="NEW")


def test_lifecycle_transitions(tmp_path):
    led = EditLedger(tmp_path / "ledger.jsonl")
    led.append(_row())
    led.update_status("ed-1", "observing")
    led.observe_cycle("ed-1", "cycle-004")
    row = led.update_status("ed-1", "kept")
    assert row["status"] == "kept"
    fold = led.fold()
    assert fold["ed-1"]["observed_cycles"] == ["cycle-004"]
    assert fold["ed-1"]["status"] == "kept"


def test_illegal_transition_rejected(tmp_path):
    led = EditLedger(tmp_path / "ledger.jsonl")
    led.append(_row())
    with pytest.raises(LedgerError, match="illegal transition"):
        led.update_status("ed-1", "kept")  # applying -> kept is illegal
    with pytest.raises(LedgerError):
        led.update_status("ghost", "observing")


def test_append_only_file_grows(tmp_path):
    p = tmp_path / "ledger.jsonl"
    led = EditLedger(p)
    led.append(_row())
    led.update_status("ed-1", "observing")
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2  # one row per event; never rewritten
    # reload sees the same fold
    assert EditLedger(p).fold()["ed-1"]["status"] == "observing"


def test_mark_applied_at(tmp_path):
    led = EditLedger(tmp_path / "ledger.jsonl")
    led.append(_row())
    led.update_status("ed-1", "observing")
    led.mark_applied_at("ed-1", 3)
    assert led.active_observation()["applied_at_cycle"] == 3


def test_rejected_is_terminal(tmp_path):
    led = EditLedger(tmp_path / "ledger.jsonl")
    led.append(_row(status="applying"))
    led.update_status("ed-1", "rejected")
    assert led.active_observation() is None
    with pytest.raises(LedgerError):
        led.update_status("ed-1", "observing")


def test_active_observation_ignores_stale_observing_rows(tmp_path):
    # regression: after a terminal row, older observing rows still exist in
    # the append-only log — active_observation must consult the FOLDED state
    led = EditLedger(tmp_path / "ledger.jsonl")
    led.append(_row())
    led.update_status("ed-1", "observing")
    led.mark_applied_at("ed-1", 3)
    assert led.active_observation()["applied_at_cycle"] == 3
    led.update_status("ed-1", "kept")
    assert led.active_observation() is None
    # a second, later edit becomes the active one; the first stays terminal
    led.append(_row(edit_id="ed-2"))
    led.update_status("ed-2", "observing")
    active = led.active_observation()
    assert active["edit_id"] == "ed-2"


def test_restore_bytes_roundtrip(tmp_path):
    import base64
    led = EditLedger(tmp_path / "ledger.jsonl")
    row = _row()
    row.prev_mutable_b64 = base64.b64encode(b"OLD = 1\n").decode()
    led.append(row)
    assert led.restore_bytes("ed-1") == b"OLD = 1\n"
