"""Benchmark identity and the evaluation-epoch freeze/upgrade mechanics.

The middle loop's core contract (frozen interfaces; see docs/evaluation.md):

- Within one evaluation **epoch** the benchmark is FROZEN: same name, same
  task_version, same seed — every run in the epoch is directly comparable.
- Across epochs the benchmark may be UPGRADED (new seed, harder instance,
  tighter metric) — scores from different epochs are NOT directly comparable
  and cross-epoch reasoning must go through normalized deltas.
- In-flight runs archive under their SUBMISSION epoch (an upgrade never
  retags a run that already started).

The freeze is carried to the task subprocess via ``RSI_EPOCH`` /
``RSI_BENCHMARK_SEED`` env injection (see rsi.tasks.runner).

TODO(homework): BenchmarkUpgrader is the strategy seam — the naive
SeedBumpUpgrader just changes the observation seed; real upgraders harden the
instance (larger condition range, held-out splits, anti-shortcut variants).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class BenchmarkRef:
    """The frozen identity tuple carried by proposals, runs, and evidence
    (I-E ``benchmark_ref``: name + epoch + task_version)."""
    name: str
    epoch: str
    task_version: str

    def to_dict(self) -> dict:
        return {"name": self.name, "epoch": self.epoch,
                "task_version": self.task_version}

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkRef":
        return cls(name=d["name"], epoch=d["epoch"],
                   task_version=d.get("task_version", "v1"))


@dataclass
class Benchmark:
    name: str
    task: str
    metric_name: str
    metric_direction: str                      # min | max
    seed: int
    task_version: str = "v1"

    def ref(self, epoch: str) -> BenchmarkRef:
        return BenchmarkRef(name=self.name, epoch=epoch,
                            task_version=self.task_version)


class EpochFrozenError(RuntimeError):
    """Re-freezing the current epoch with a different benchmark identity."""


class EpochStateError(RuntimeError):
    """Corrupted/hand-edited epochs.json."""


class EpochManager:
    """Freeze/upgrade bookkeeping, persisted as JSON under the state dir.

    Epoch identity is (name, task_version); the seed is the observation seed
    opened with the epoch. Epoch ids are sequential and deterministic
    (``e001``, ``e002``…), derived from the highest existing id — no
    wall-clock in identities."""

    def __init__(self, state_path: str | Path):
        self.state_path = Path(state_path)
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                current, epochs = data["current"], data["epochs"]
                assert isinstance(epochs, list)
            except (json.JSONDecodeError, KeyError, TypeError,
                    AssertionError) as e:
                raise EpochStateError(
                    f"{self.state_path} is not a valid epochs state file "
                    f"({type(e).__name__}: {e}); delete it or restore it"
                ) from e
        else:
            current, epochs = None, []
        self._current: dict | None = current
        self._epochs: list[dict] = epochs

    def current(self) -> BenchmarkRef | None:
        if self._current is None:
            return None
        try:
            return BenchmarkRef(**self._current)
        except TypeError as e:
            raise EpochStateError(
                f"{self.state_path} current epoch entry is malformed "
                f"({e}); delete it or restore it") from e

    def epoch_seed(self, epoch: str) -> int | None:
        """The observation seed recorded for ``epoch`` (None if unknown).
        The engine needs this when the slow loop switches to a question
        bound to an older epoch: the run must be EXECUTED with the seed of
        the epoch it is TAGGED with, else within-epoch comparability breaks
        silently."""
        for e in self._epochs:
            if e.get("epoch") == epoch:
                return e.get("seed")
        return None

    def freeze(self, bench: Benchmark) -> BenchmarkRef:
        """Bind the current epoch to ``bench`` (or open the first epoch).
        Re-freezing with a DIFFERENT identity is refused — that is an
        upgrade, call upgrade()."""
        if self._current is not None:
            cur = BenchmarkRef(**self._current)
            if (cur.name != bench.name
                    or cur.task_version != bench.task_version):
                raise EpochFrozenError(
                    f"epoch {cur.epoch} frozen as {cur.name}/{cur.task_version};"
                    f" refusing re-freeze as {bench.name}/{bench.task_version}"
                    f" — use upgrade()")
            return cur
        ref = bench.ref("e001")
        self._current = ref.to_dict()
        self._epochs.append({**ref.to_dict(), "seed": bench.seed})
        self._save()
        return ref

    def upgrade(self, bench: Benchmark) -> BenchmarkRef:
        """Archive the current epoch and open the next (cross-epoch upgrade).
        In-flight runs keep their submission epoch: callers pass the OLD ref
        when archiving a run that started before the upgrade."""
        if self._current is None:
            return self.freeze(bench)
        # derive from the highest existing id (NOT the list length): a
        # hand-edited or partially-archived state must never produce a
        # non-monotonic id that collides with history
        highest = 0
        for e in self._epochs:
            try:
                highest = max(highest, int(str(e.get("epoch", "e0"))[1:]))
            except ValueError:
                continue
        ref = bench.ref(f"e{highest + 1:03d}")
        self._current = ref.to_dict()
        self._epochs.append({**ref.to_dict(), "seed": bench.seed})
        self._save()
        return ref

    @property
    def epochs(self) -> list[dict]:
        return list(self._epochs)

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(
            {"current": self._current, "epochs": self._epochs}, indent=2))


class BenchmarkUpgrader(Protocol):
    def upgrade(self, bench: Benchmark, diag) -> Benchmark: ...


class SeedBumpUpgrader:
    """Naive upgrader: same benchmark, next seed (deterministic derivation).
    Everything else — harder instances, held-out splits, anti-shortcut
    variants — is TODO(homework)."""

    def upgrade(self, bench: Benchmark, diag) -> Benchmark:
        return Benchmark(
            name=bench.name, task=bench.task,
            metric_name=bench.metric_name,
            metric_direction=bench.metric_direction,
            seed=bench.seed + 1, task_version=bench.task_version)
