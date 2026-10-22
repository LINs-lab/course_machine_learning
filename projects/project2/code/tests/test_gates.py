from rsi.evaluation.gates import (CollapseGate, GateChain, MetricReading,
                                  NoiseGate, NoChangeGate, RunContext)


def _reading(value=None, baseline=1.0, failure=None):
    return MetricReading(name="m", direction="min", value=value,
                         baseline=baseline, failure_mode=failure)


def _chain(floor=None):
    return GateChain([CollapseGate(), NoChangeGate(), NoiseGate()]), RunContext(
        baseline=1.0, noise_floor=floor, epoch="e001")


def test_collapse_fails():
    chain, ctx = _chain()
    v = chain.check(_reading(failure="timeout"), ctx)
    assert v.status == "fail" and v.reasons[0].startswith("collapse")

def test_nonfinite_fails():
    chain, ctx = _chain()
    v = chain.check(MetricReading(name="m", direction="min", value=float("nan"),
                                  baseline=1.0), ctx)
    assert v.status == "fail"


def test_worse_discards():
    chain, ctx = _chain()
    v = chain.check(_reading(value=1.5, baseline=1.0), ctx)
    assert v.status == "discard" and v.reasons == ["no_change"]


def test_small_gain_below_noise_discards():
    chain, ctx = _chain(floor=0.1)
    v = chain.check(_reading(value=0.95, baseline=1.0), ctx)
    assert v.status == "discard" and v.reasons == ["below_noise"]


def test_real_gain_keeps():
    chain, ctx = _chain(floor=0.1)
    v = chain.check(_reading(value=0.5, baseline=1.0), ctx)
    assert v.status == "keep"


def test_all_abstain_keeps():
    chain, ctx = _chain()
    # the baseline-establishing run: a value exists, no baseline to compare
    v = chain.check(_reading(value=1.0, baseline=None), ctx)
    assert v.status == "keep"


def test_max_direction_respected():
    chain, ctx = _chain()
    # direction max: 1.5 vs baseline 1.0 is an improvement → keep
    r = MetricReading(name="m", direction="max", value=1.5, baseline=1.0)
    assert chain.check(r, ctx).status == "keep"
    r2 = MetricReading(name="m", direction="max", value=0.5, baseline=1.0)
    assert chain.check(r2, ctx).status == "discard"
