from rsi.core.types import RunRecord
from rsi.evaluation.diagnostics import (CONTINUE, UPGRADE_BENCHMARK,
                                        NaiveDiagnostician)
from rsi.evaluation.gates import MetricReading


def _run(i, value, baseline=None, status="keep", epoch="e001", fail=False):
    return RunRecord(
        run_id=f"cycle-{i:03d}", proposal_id=f"p{i}", hypothesis_id="h",
        question_id="q", task="t", epoch=epoch, status=status,
        metric=MetricReading(name="m", direction="min", value=value,
                             baseline=baseline,
                             failure_mode="crash" if fail else None),
        variable_changed=f"v{i}", created="", duration_s=0.1,
        verdict_reasons=[])


def test_insufficient_runs_continues():
    diag = NaiveDiagnostician()
    r = diag.diagnose([_run(1, 1.0)], "e001", "q")
    assert r.recommendation == CONTINUE and r.n_runs == 1


def test_improving_series_not_saturated():
    diag = NaiveDiagnostician()
    runs = [_run(1, 10.0), _run(2, 5.0, baseline=10.0),
            _run(3, 2.0, baseline=5.0), _run(4, 1.0, baseline=2.0),
            _run(5, 0.5, baseline=1.0)]
    r = diag.diagnose(runs, "e001", "q")
    assert r.recommendation == CONTINUE
    assert r.saturation < 0.8


def test_plateau_series_recommends_upgrade():
    diag = NaiveDiagnostician()
    runs = ([_run(1, 10.0), _run(2, 5.0, baseline=10.0),
             _run(3, 4.9, baseline=5.0)] +
            [_run(i, 4.9, baseline=4.9, status="discard")
             for i in range(4, 10)])
    r = diag.diagnose(runs, "e001", "q")
    assert r.recommendation == UPGRADE_BENCHMARK
    assert r.saturation >= 0.8


def test_noise_floor_capped_by_scale():
    # a diverged discard (huge |delta|) must not poison the noise floor
    diag = NaiveDiagnostician()
    runs = [_run(1, 1.0), _run(2, 0.5, baseline=1.0),
            _run(3, 1e15, baseline=0.5, status="discard"),
            _run(4, 0.4, baseline=0.5), _run(5, 0.45, baseline=0.4,
                                             status="discard")]
    r = diag.diagnose(runs, "e001", "q")
    assert r.noise_floor <= 0.1 * 0.4 + 1e-9


def test_epoch_filtering():
    diag = NaiveDiagnostician()
    runs = [_run(1, 1.0, epoch="e001"), _run(2, 0.9, baseline=1.0,
                                             epoch="e002")]
    r = diag.diagnose(runs, "e001", "q")
    assert r.n_runs == 1
