#!/usr/bin/env python3
"""Multi-seed self-editing ablation comparison: self-editing ON vs OFF,
several seeds per arm.

    python scripts/compare_seeds.py \
        --on sessions/naive-seed7 --on sessions/naive-seed42 \
        --off sessions/noop-seed7 --off sessions/noop-seed42 \
        --task iris-ood --metric ood_accuracy --direction max --dest .

Reads the same session dirs as report.py. Writes figures/compare_seeds.pdf
(running-best trajectories: per-seed thin lines + per-arm median), and
figures/final_best.pdf (per-seed final kept best with mean ± sd), plus a
markdown table in compare_seeds.md. Style follows report.py's academic
figure convention (same rcParams, same palette: purple = ON / red = OFF).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib.pyplot as plt  # noqa: E402
from report import (  # noqa: E402  (import sets the academic rcParams)
    ANNOT_FS,
    FIG_H,
    FIG_W,
    INK,
    INK2,
    OURS,
    REFERENCE,
    S,
    _hours,
    _save,
    best_series,
    load_session,
    mean_running_best,
    metric_label,
)

ARM = {
    "on": {"label": "self-editing ON", "color": OURS, "dash": "-"},
    "off": {"label": "self-editing OFF", "color": REFERENCE,
            "dash": (0, (5, 3))},
}


def _seed_of(sess_dir: str) -> str:
    """Seed label for a session: the engine config isn't persisted, so the
    directory naming convention (<arm>-seed<N>) carries it."""
    name = Path(sess_dir).name
    return name.split("seed")[-1] if "seed" in name else name


def _median_curve(curves: list[tuple[list[float], list[float]]],
                  grid: list[float]) -> list[float]:
    """Median of running-best curves on a common time grid (forward-fill:
    a running best holds its last value until the next improvement)."""
    cols = []
    for t in grid:
        vals = []
        for xs, ys in curves:
            v = None
            for x, y in zip(xs, ys):
                if x > t:
                    break
                if y is not None:
                    v = y
            vals.append(v)
        vals = [v for v in vals if v is not None]
        cols.append(sorted(vals)[len(vals) // 2] if vals else None)
    return cols


def fig_running_best(arms: dict, direction: str, metric: str, dest: Path):
    """Running best (kept) vs wall-clock: per-seed thin lines, per-arm
    median thick (ON = reserved purple solid, OFF = red dashed)."""
    mlabel = metric_label(metric)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    t_max = 0.0
    for name, sessions in arms.items():
        curves = []
        for _, sess in sessions:
            runs = sess["runs"]
            xs = _hours(runs)
            best = best_series(runs, direction)
            curves.append((xs, best))
            t_max = max(t_max, xs[-1])
        st = ARM[name]
        for (sess_dir, sess), (xs, best) in zip(sessions, curves):
            ax.step(
                xs,
                [b if b is not None else float("nan") for b in best],
                where="post",
                color=st["color"],
                alpha=0.4,
                linewidth=S(1.1),
                zorder=3,
            )
        grid = [t_max * i / 400 for i in range(401)]
        med = _median_curve(curves, grid)
        gx = [t for t, m in zip(grid, med) if m is not None]
        gy = [m for m in med if m is not None]
        ax.step(gx, gy, where="post", color=st["color"], linestyle=st["dash"],
                linewidth=S(2.6), zorder=4,
                label=f"{st['label']} (median of {len(sessions)} seeds)")
    arrow = r"$\uparrow$" if direction == "max" else r"$\downarrow$"
    ax.set_xlabel("Wall-Clock Time (h)")
    ax.set_ylabel(f"Best {mlabel} {arrow}")
    ax.set_title(f"Running Best {mlabel} Across Seeds", pad=10)
    fig.legend(loc="lower center", ncols=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    _save(fig, dest)
    plt.close(fig)


def fig_final_best(arms: dict, direction: str, metric: str, dest: Path):
    """Per-seed final kept best per arm (jittered strip) with mean ± sd."""
    mlabel = metric_label(metric)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for xi, name in enumerate(("on", "off")):
        sessions = arms[name]
        finals = []
        for _, sess in sessions:
            vals = [r["metric"]["value"] for r in sess["runs"]
                    if r["status"] == "keep"
                    and r["metric"]["value"] is not None]
            if vals:
                finals.append(
                    max(vals) if direction == "max" else min(vals))
        if not finals:
            continue
        st = ARM[name]
        mean = sum(finals) / len(finals)
        sd = (sum((v - mean) ** 2 for v in finals) / max(1, len(finals) - 1)
              ) ** 0.5
        xs = [xi - 0.12 + 0.12 * i for i in range(len(finals))]
        ax.scatter(xs, finals, s=S(40) * 0.5, color=st["color"], zorder=4,
                   label=f"{st['label']} (n = {len(finals)} seeds)")
        ax.errorbar([xi], [mean], yerr=[sd], color=INK, capsize=4,
                    linewidth=S(1.4), zorder=5, marker="_", markersize=S(14))
        ax.annotate(
            f"{mean:.3f} ± {sd:.3f}",
            (xi, mean),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=ANNOT_FS,
            color=INK2,
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels([ARM["on"]["label"], ARM["off"]["label"]])
    ax.set_ylabel(f"Final Best {mlabel}")
    ax.set_title(f"Final Best {mlabel} per Seed", pad=10)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, dest)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="compare_seeds")
    p.add_argument("--on", action="append", required=True,
                   help="self-editing-ON session dir (repeatable)")
    p.add_argument("--off", action="append", required=True,
                   help="self-editing-OFF session dir (repeatable)")
    p.add_argument("--task", required=True)
    p.add_argument("--metric", default="metric")
    p.add_argument("--direction", choices=["min", "max"], default="max")
    p.add_argument("--dest", required=True)
    args = p.parse_args(argv)

    dest = Path(args.dest)
    (dest / "figures").mkdir(parents=True, exist_ok=True)
    arms = {
        "on": [(d, load_session(Path(d))) for d in args.on],
        "off": [(d, load_session(Path(d))) for d in args.off],
    }
    fig_running_best(arms, args.direction, args.metric,
                     dest / "figures" / "compare_seeds.pdf")
    fig_final_best(arms, args.direction, args.metric,
                   dest / "figures" / "final_best.pdf")

    # markdown table: per arm-seed + arm medians
    lines = [
        f"# {args.task} — multi-seed self-editing ablation", "",
        "| arm | seed | cycles | final best (kept) | keep rate | "
        "mean running best |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("on", "off"):
        for sess_dir, sess in arms[name]:
            runs = sess["runs"]
            keeps = [r["metric"]["value"] for r in runs
                     if r["status"] == "keep"
                     and r["metric"]["value"] is not None]
            best = ((max(keeps) if args.direction == "max" else min(keeps))
                    if keeps else None)
            nk = sum(1 for r in runs if r["status"] == "keep")
            auc = mean_running_best(runs, args.direction)
            best_s = f"{best:.4f}" if best is not None else "—"
            auc_s = f"{auc:.4f}" if auc is not None else "—"
            lines.append(
                f"| {ARM[name]['label']} | {_seed_of(sess_dir)} | "
                f"{len(runs)} | {best_s} | {nk / len(runs):.2%} | {auc_s} |"
            )
    lines += ["", "![running best](figures/compare_seeds.pdf)", "",
              "![final best per seed](figures/final_best.pdf)", ""]
    out = dest / "compare_seeds.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out} and figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
