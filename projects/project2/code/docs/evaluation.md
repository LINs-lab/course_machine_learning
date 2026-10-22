# Evaluation Interfaces (frozen)

These interfaces are deliberately decoupled from storage: `rsi.evaluation` library code imports nothing from `rsi.knowledge` / `rsi.core` and consumes plain run sequences. The course's separately arranged evaluation tasks drive them directly — treat the signatures in this module as a contract.

## Benchmark identity and epochs (`benchmark.py`)

```python
@dataclass(frozen=True)
class BenchmarkRef:      # carried by proposals, runs, and I-E evidence
    name: str            # benchmark name (default: the task name)
    epoch: str           # "e001", "e002", ... sequential, deterministic
    task_version: str    # "v1"

class EpochManager:      # persisted at <out>/state/epochs.json
    def freeze(bench) -> BenchmarkRef     # open/re-bind the current epoch
    def upgrade(bench) -> BenchmarkRef    # archive current, open the next
    def current() -> BenchmarkRef | None
```

Rules:

- Within one epoch the benchmark is **frozen** — same name, version, seed; every run in the epoch is directly comparable. Freezing is *carried to the task subprocess* via `RSI_EPOCH` / `RSI_BENCHMARK_SEED` env injection.
- Re-freezing with a different identity raises (`EpochFrozenError`) — that is an upgrade, not a freeze.
- In-flight runs archive under their **submission** epoch.
- Across epochs, scores are NOT directly comparable: the engine re-baselines the incumbent value (the incumbent *solution* persists) and cross-epoch reasoning must use normalized deltas. This is a deliberate teaching point of the middle loop.

`BenchmarkUpgrader` is the strategy seam. The shipped `SeedBumpUpgrader` changes only the observation seed; hardening the instance (harder problems, held-out splits, anti-shortcut variants) is homework task E-1.

## Verdict gates (`gates.py`)

```python
class Gate(Protocol):
    def check(reading: MetricReading, ctx: RunContext) -> GateResult
    # GateResult.verdict ∈ {keep, discard, fail, None(=abstain)}

class GateChain:   # ordered; first decisive verdict wins; all-abstain ⇒ keep
```

The trichotomy is frozen: **keep | discard | fail**. Gates *reject*, they do not bless — a real improvement needs no gate's permission, only no gate's veto. Shipped: `CollapseGate` (failure/non-finite ⇒ fail), `NoChangeGate` (delta ≤ 0 ⇒ discard), `NoiseGate` (0 < delta ≤ noise floor ⇒ discard). Homework: significance vs seed variance, shortcut/reward-hacking detection, held-out E_test (task E-2).

## Diagnostics (`diagnostics.py`)

```python
class Diagnostician(Protocol):
    def diagnose(history, epoch, question_id="") -> DiagnosisReport

@dataclass
class DiagnosisReport:
    saturation: float        # 0..1 — written back to the I-Q question node
    noise_floor: float       # feeds NoiseGate and the link policy
    discrimination: float    # separation of best keep vs median
    recommendation: str      # continue | upgrade_benchmark | saturate_question
```

The shipped `NaiveDiagnostician` uses sliding-window heuristics, caps the noise floor at 10% of the incumbent scale (a divergence is collapse, not noise), and excludes catastrophic outliers with a MEDIAN-anchored band (best-anchored exclusion goes blind when the best is ~0 — a solved min-task scores exactly 0.0). The recommendation routes the two saturated states apart: saturation high + discrimination low (the observer can't tell candidates apart) ⇒ `upgrade_benchmark` — the question CONTINUES on a fresh epoch, its saturation resets, the incumbent value re-baselines; saturation high + discrimination high (clean signal, no headroom) ⇒ `saturate_question` — the slow loop abandons the question. Real estimators — seed replication, paired tests, response surfaces — are homework task E-3.

## The report CLI

```bash
python -m rsi.evaluation --runs out/state/runs.jsonl report
# one JSON line per epoch: saturation / noise_floor / discrimination /
# recommendation
```
