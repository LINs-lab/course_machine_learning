import pytest

from rsi.tasks.spec import TaskError, load_task, read_result


def test_load_demo_task(demo_task):
    spec = load_task(demo_task)
    assert spec.name == "quad-optim"
    assert spec.metric.direction == "min"
    assert spec.editable == ["solver.py"]
    assert spec.entrypoint[0] == "python3"
    assert "quadratic" in spec.protocol_body


def test_missing_manifest(tmp_path):
    with pytest.raises(TaskError):
        load_task(tmp_path / "nope")


def _manifest(tmp_path, body="hello"):
    d = tmp_path / "t1"
    d.mkdir()
    (d / "program.md").write_text(f"""---
schema_version: 1
name: t1
entrypoint: ["python3", "run.py"]
metric:
  name: score
  direction: min
  result_file: result.json
  result_key: score
editable: ["solver.py"]
---
{body}
""")
    return d


def test_two_editable_files_rejected(tmp_path):
    d = _manifest(tmp_path)
    text = (d / "program.md").read_text().replace(
        'editable: ["solver.py"]', 'editable: ["a.py", "b.py"]')
    (d / "program.md").write_text(text)
    with pytest.raises(TaskError, match="exactly ONE"):
        load_task(d)


def test_bad_direction_rejected(tmp_path):
    d = _manifest(tmp_path)
    text = (d / "program.md").read_text().replace("direction: min",
                                                  "direction: sideways")
    (d / "program.md").write_text(text)
    with pytest.raises(TaskError, match="min|max"):
        load_task(d)


def test_path_escape_rejected(tmp_path):
    d = _manifest(tmp_path)
    text = (d / "program.md").read_text().replace(
        'editable: ["solver.py"]', 'editable: ["../outside.py"]')
    (d / "program.md").write_text(text)
    with pytest.raises(TaskError, match="task-relative"):
        load_task(d)


def test_name_mismatch_rejected(tmp_path):
    d = _manifest(tmp_path)
    text = (d / "program.md").read_text().replace("name: t1", "name: other")
    (d / "program.md").write_text(text)
    with pytest.raises(TaskError, match="!="):
        load_task(d)


def test_read_result_failure_modes(demo_task, tmp_path):
    spec = load_task(demo_task)  # result_key = final_loss
    assert read_result(spec, tmp_path)["failure_mode"] == "missing_result"
    (tmp_path / "result.json").write_text("{not json")
    assert read_result(spec, tmp_path)["failure_mode"] == "invalid_metric"
    (tmp_path / "result.json").write_text('{"other": 1}')
    assert read_result(spec, tmp_path)["failure_mode"] == "missing_metric"
    (tmp_path / "result.json").write_text('{"final_loss": "NaN"}')
    assert read_result(spec, tmp_path)["failure_mode"] == "invalid_metric"
    (tmp_path / "result.json").write_text(
        '{"final_loss": 1.5, "observations": {"a": 1}}')
    out = read_result(spec, tmp_path)
    assert out["metric"] == 1.5 and out["observations"] == {"a": 1}
