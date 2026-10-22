#!/usr/bin/env python3
"""Session report generator: turn a (or two) RSI session dirs into figures
+ a markdown report (key trajectories, self-improvement ledger, per-edit
response, and a with/without-self-improvement comparison).

Usage (reproduce):
    python scripts/report.py --session main=OUT_DIR [--session control=OUT2] \
        --task iris-ood --metric ood_accuracy --direction max \
        --dest experiments/run

Reads: state/runs.jsonl, state/ledger.jsonl, state/epochs.json,
kb/{questions,hypotheses,evidence}.sqlite. Writes figures/*.pdf + report.md.
Figures are VECTOR PDF ONLY (fonttype 42, TrueType embedded), produced by
this script — never hand-edited.

Style follows the academic figure-production convention
(academic-latex-writing/references/figure-production.md): Times New Roman
serif, one reserved hue for OUR system (#7b1fae — forbidden to controls),
opacity hierarchy (ours 1.0 / others 0.85 / bands ~0.10), four-side gray
frame, grid at zorder 0 alpha 0.25, bold axis labels. Every standalone
figure carries a centered bold Times title (a reader must understand each
figure alone); grouped multi-panel figures use bold "(a) ..." panel labels
instead. Axis labels use standard display names ("OOD Accuracy",
"Experiment Cycle"), never raw internal identifiers.

Environment: conda env `autoresearch-base` (python 3.11, matplotlib,
graphviz). Last verified export: 2026-10-12.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sqlite3
import subprocess
import sys
import zlib
from datetime import datetime
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.patches import Patch  # noqa: E402
except ImportError:
    print(
        "error: matplotlib is required for report generation.\n"
        "Install it with: pip install -e .[report]  (or "
        "conda env update -f environment.yml --prune)",
        file=sys.stderr,
    )
    sys.exit(2)

# Palette per the figure-production convention. Exactly ONE reserved hue
# marks ours (the RSI system / main session); the strongest reference
# (control) takes a fixed contrasting hue; everything else is subordinate.
OURS = "#7b1fae"  # reserved — the RSI system (best line, keeps)
REFERENCE = "#c1272d"  # strongest baseline (control) in comparisons
ALERT = "#D14D4D"  # sign-aware alert red (fail / reverted edits)
SURFACE = "white"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
SPINE = "#D5D5D5"  # four-side frame, 1.2pt (dense-figure class)
# subordinate hues for knob/status families (the purple family is reserved)
SUB = ("#4a6fa5", "#2f8f83", "#b8860b", "#946b2d", "#5a5a8a", "#6b6b6b")

# display names for the canonical session roles: the CLI keys (main/control)
# are plumbing; figures and prose name the experimental condition
ARM_LABEL = {"main": "self-editing ON", "control": "self-editing OFF"}

# figure frame at FINAL PRINT SIZE (conference text width; height follows
# the golden ratio) — figures drop into a paper without rescaling, so the
# whole design (fonts, lines, markers) scales uniformly from the original
# 10.5in layout via S()/S2()
FIG_W, FIG_H = 5.5, 5.5 / 1.618
FS = FIG_W / 10.5


def S(x):
    """Uniform linear scale (line widths, marker diameters)."""
    return x * FS


FONT_BOOST = 1.5  # readability boost: fonts run 1.5x the uniform scale


def SF(x):
    """Font scale (uniform scale x the readability boost)."""
    return x * FS * FONT_BOOST


def S2(x):
    """Area scale (scatter `s` is marker area, so it shrinks quadratically)."""
    return x * FS * FS


# Uniform typography across every figure: legends one size, in-axes
# annotations one size (axis labels/ticks/titles come from rcParams).
LEGEND_FS = SF(11)
ANNOT_FS = SF(10)


def arm_label(name: str) -> str:
    return ARM_LABEL.get(name, name)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Liberation Serif",
            "Nimbus Roman",
            "DejaVu Serif",
        ],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,  # TrueType embedded
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        # font ladder, scaled with the frame + boost: base 15 / title 16 /
        # label 12 / tick 11 (at 5.5in with boost 1.5: ~12 / ~12.6 / ~9.4 / ~8.6pt)
        "font.size": SF(15),
        "text.color": INK,
        "axes.titlesize": SF(16),
        "axes.labelsize": SF(12),
        "xtick.labelsize": SF(11),
        "ytick.labelsize": SF(11),
        "axes.labelcolor": INK,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        # four-side equal-width gray frame; grid at zorder 0, alpha 0.25
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        # translucent rounded in-axes legend frames
        "legend.frameon": True,
        "legend.fancybox": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "lightgray",
        "legend.fontsize": LEGEND_FS,
    }
)


def load_session(session_dir: Path) -> dict:
    runs_path = session_dir / "state" / "runs.jsonl"
    epochs_path = session_dir / "state" / "epochs.json"
    for p in (runs_path, epochs_path):
        if not p.exists():
            raise SystemExit(
                f"error: {p} not found — is {session_dir} an RSI session "
                f"dir? (run `python -m rsi.core --task ... --out ...` first)"
            )
    runs = [json.loads(l) for l in runs_path.read_text().splitlines() if l.strip()]
    ledger_path = session_dir / "state" / "ledger.jsonl"
    ledger = (
        [json.loads(l) for l in ledger_path.read_text().splitlines() if l.strip()]
        if ledger_path.exists()
        else []
    )
    epochs = json.loads(epochs_path.read_text())

    def table(name):
        path = session_dir / "kb" / f"{name}.sqlite"
        if not path.exists():
            return []  # read-only + existence check: never CREATE db files
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        try:
            return [
                json.loads(r["json"])
                for r in db.execute(f"SELECT json FROM {name} ORDER BY rowid")
            ]
        except sqlite3.OperationalError:
            return []
        finally:
            db.close()

    return {
        "runs": runs,
        "ledger": ledger,
        "epochs": epochs,
        "questions": table("questions"),
        "hypotheses": table("hypotheses"),
        "evidence": table("evidence"),
        "strategy": (session_dir / "state" / "strategy.py").read_text()
        if (session_dir / "state" / "strategy.py").exists()
        else "",
    }


def _cycle(run_id: str) -> int:
    return int(run_id.split("-")[1])


def _hours(runs) -> list[float]:
    """Elapsed wall-clock hours since the session's first run — the fair
    budget axis when the arms' per-cycle costs differ (they do: the
    incumbent's STEPS inflation makes late cycles up to 9x slower)."""
    ts = [datetime.fromisoformat(r["created"]) for r in runs]
    t0 = ts[0]
    return [(t - t0).total_seconds() / 3600 for t in ts]


_ACRONYMS = {"ood", "id", "auc", "auroc", "mse", "rmse", "mae", "lr", "f1"}


def metric_label(metric: str) -> str:
    """Standard display name for an internal metric id:
    "ood_accuracy" -> "OOD Accuracy"."""
    return " ".join(
        w.upper() if w.lower() in _ACRONYMS else w.capitalize()
        for w in metric.replace("-", "_").split("_")
    )


def _save(fig, dest: Path) -> None:
    """Vector PDF only (fonttype 42), tight bounding box. No PNG output."""
    fig.savefig(dest, format="pdf", bbox_inches="tight")


def best_series(runs, direction):
    """Running best value by cycle (direction-aware)."""
    best, out = None, []
    for r in runs:
        v = r["metric"]["value"]
        if r["status"] == "keep" and v is not None:
            better = best is None or (v > best if direction == "max" else v < best)
            if better:
                best = v
        out.append(best)
    return out


def mean_running_best(runs, direction):
    """AUC of the running-best curve, normalized per cycle — sample
    efficiency: how much quality the loop holds on average over its budget,
    not just where it ends."""
    best = [b for b in best_series(runs, direction) if b is not None]
    return sum(best) / len(best) if best else None


def cycles_to_reach(runs, direction, target):
    """First cycle at which the running best matches/beats `target`
    (direction-aware); None if never. 'Cycles to match the control's final
    best' is the cleanest speed readout of an ablation."""
    if target is None:
        return None
    for r, b in zip(runs, best_series(runs, direction)):
        if b is not None and (b >= target if direction == "max" else b <= target):
            return _cycle(r["run_id"])
    return None


# -- self-edit response (did an edit cause measurable gain?) --------------------
#
# Three tiers, because "did self-improvement help" is three different
# questions:
#   L1  per-edit response — windowed pre/post contrast around each edit on
#       four response variables, attributed by difference-in-differences
#       (DiD) against the control arm;
#   L2  decision audit — does the ENGINE's kept/reverted call (its own
#       criterion: keep-rate over a >=2-run window, revert below 0.5x)
#       agree with L1's measured response;
#   L3  net-state contrast — after the last edit decision the arms differ
#       by one persistent strategy state; a tail comparison quantifies that
#       net intervention (this is the attribution compare.pdf deserves but
#       never states).
#
# Alignment: windows live INSIDE the question active at the edit cycle and
# the control is matched by within-question position — the arms' question
# sequences desynchronize early (an outcome, not a nuisance, but it would
# confound run-index-matched value/best contrasts with question switches).

EDIT_WINDOW = 100  # default half-window (runs) for the L1 contrast
TAIL_WINDOW = 500  # post-decision tail length for the L3 net contrast
BOOT_N = 2000  # bootstrap resamples for every reported CI
ROLL_W = 25  # rolling window for the edit-response figure curves


def _nanmean(xs):
    xs = [x for x in xs if x == x]  # NaN != NaN filters missing
    return sum(xs) / len(xs) if xs else float("nan")


def _var_series(runs, direction):
    """Per-run values of the four response variables: no-change waste (the
    edit's own declared target), keep, raw run value, running best."""
    best, v4 = None, []
    for r in runs:
        v = r["metric"]["value"]
        if (
            r["status"] == "keep"
            and v is not None
            and (best is None or (v > best if direction == "max" else v < best))
        ):
            best = v
        v4.append(best if best is not None else float("nan"))
    return {
        "no_change": [
            float("no_change" in (r.get("verdict_reasons") or [])) for r in runs
        ],
        "keep": [float(r["status"] == "keep") for r in runs],
        "value": [
            r["metric"]["value"] if r["metric"]["value"] is not None else float("nan")
            for r in runs
        ],
        "best": v4,
    }


# human labels + which direction is "gain" (value/best follow the metric)
VAR_LABEL = {
    "no_change": "no-change waste rate",
    "keep": "keep rate",
    "value": "mean run value",
    "best": "mean running best",
}
VAR_SIGN = {"no_change": -1, "keep": 1}


def _question_spans(runs):
    """Contiguous (question_id, start, end) spans over the run list."""
    spans, cur, start = [], None, 0
    for i, r in enumerate(runs):
        if r["question_id"] != cur:
            if cur is not None:
                spans.append((cur, start, i))
            cur, start = r["question_id"], i
    spans.append((cur, start, len(runs)))
    return spans


def _did_ci(a_pre, a_post, b_pre, b_post, seed, n_boot=BOOT_N):
    """Difference-in-differences point estimate + percentile bootstrap CI
    (resample runs within each window; deterministic seed per edit)."""
    a_pre = [x for x in a_pre if x == x]
    a_post = [x for x in a_post if x == x]
    b_pre = [x for x in b_pre if x == x]
    b_post = [x for x in b_post if x == x]
    if not (a_pre and a_post and b_pre and b_post):
        return float("nan"), float("nan"), float("nan")

    def delta(pre, post, rng):
        s1 = sum(pre[rng.randrange(len(pre))] for _ in pre) / len(pre)
        s2 = sum(post[rng.randrange(len(post))] for _ in post) / len(post)
        return s2 - s1

    rng = random.Random(seed)
    point = _nanmean(a_post) - _nanmean(a_pre) - _nanmean(b_post) + _nanmean(b_pre)
    diffs = sorted(
        delta(a_pre, a_post, rng) - delta(b_pre, b_post, rng) for _ in range(n_boot)
    )
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]
    return point, lo, hi


def _gain_label(point, lo, hi, sign):
    """gain ⇔ the DiD CI lies strictly on the improvement side of zero
    (sign: +1 up-is-better, -1 down-is-better)."""
    if not (point == point and lo == lo and hi == hi):
        return "n/a"
    if sign < 0:
        point, lo, hi = -point, -hi, -lo
    if lo > 0:
        return "gain"
    if hi < 0:
        return "harm"
    return "n.s."


def _local_best_series(runs, direction):
    """Running best WITHIN the current question (ratchet resets at every
    question switch) — the like-for-like progress variable when the arms
    are aligned by question rather than by run index."""
    best, out = None, []
    prev_q = None
    for r in runs:
        if r["question_id"] != prev_q:
            best, prev_q = None, r["question_id"]
        v = r["metric"]["value"]
        if (
            r["status"] == "keep"
            and v is not None
            and (best is None or (v > best if direction == "max" else v < best))
        ):
            best = v
        out.append(best if best is not None else float("nan"))
    return out


def edit_response(main, control, direction, window=EDIT_WINDOW):
    """Compute L1/L2/L3: per-edit within-question contrasts with a
    position-matched control DiD, the engine-decision audit context, and
    the post-decision net-state tail contrast. Returns a dict (see use in
    render()) — pure computation, no plotting."""
    runs = main["runs"]
    sm = _var_series(runs, direction)
    spans = _question_spans(runs)

    fold: dict[str, dict] = {}
    for row in main["ledger"]:
        fold[row["edit_id"]] = row  # last row wins (terminal last)
    edits = sorted(
        (r for r in fold.values() if r.get("applied_at_cycle") is not None),
        key=lambda r: r["applied_at_cycle"],
    )

    sc = _var_series(control["runs"], direction) if control else None
    spans_c = _question_spans(control["runs"]) if control else []
    span_c_map = {q: (a, b) for q, a, b in spans_c}

    out = []
    for i, e in enumerate(edits):
        c = e["applied_at_cycle"]
        qid, qs, qe = next((q, a, b) for q, a, b in spans if a <= c < b)
        prev_term = edits[i - 1]["terminated_at"] if i else qs
        next_apply = edits[i + 1]["applied_at_cycle"] if i + 1 < len(edits) else qe
        pre = range(max(prev_term or qs, c - window, qs), c)
        post = range(c, min(c + window, next_apply, qe))
        # position-matched control windows: same question, same offset
        ca, cb = span_c_map.get(qid, (None, None))
        if ca is not None:
            pre_c = [ca + (j - qs) for j in pre if ca + (j - qs) < cb]
            post_c = [ca + (j - qs) for j in post if ca + (j - qs) < cb]
        else:
            pre_c, post_c = [], []
        res = {
            "edit_id": e["edit_id"],
            "status": e["status"],
            "cycle": c,
            "question": qid,
            "terminated_at": e.get("terminated_at"),
            "vars": {},
        }
        for var in sm:
            p_m = [sm[var][j] for j in pre]
            q_m = [sm[var][j] for j in post]
            p_c = [sc[var][j] for j in pre_c] if sc else []
            q_c = [sc[var][j] for j in post_c] if sc else []
            sign = VAR_SIGN.get(var, 1 if direction == "max" else -1)
            delta_on = _nanmean(q_m) - _nanmean(p_m)
            if sc and p_c and q_c:
                did, lo, hi = _did_ci(
                    p_m, q_m, p_c, q_c, zlib.crc32(e["edit_id"].encode())
                )
            else:
                did, lo, hi = float("nan"), float("nan"), float("nan")
            res["vars"][var] = {
                "pre": _nanmean(p_m),
                "post": _nanmean(q_m),
                "n_pre": len(p_m),
                "n_post": len(q_m),
                "delta_on": delta_on,
                "did": did,
                "ci": (lo, hi),
                "verdict": _gain_label(did, lo, hi, sign),
            }
        out.append(res)

    # L3 net-state tail: both arms after the LAST edit decision — they
    # differ by whatever strategy state the improver left live. Caveat:
    # the arms' question mixes differ from here on (that divergence is
    # itself plausibly edit-caused, but it means this contrast bundles
    # the edit with its downstream question-sequencing effects).
    tail = {}
    t0 = t1 = None
    if edits and control:
        t0 = max((e.get("terminated_at") or e["applied_at_cycle"]) for e in edits)
        t1 = min(TAIL_WINDOW, len(runs), len(control["runs"]))
        if t0 < t1:
            for var in sm:
                x = [v for v in sm[var][t0:t1] if v == v]
                y = [v for v in sc[var][t0:t1] if v == v]
                sign = VAR_SIGN.get(var, 1 if direction == "max" else -1)
                rng = random.Random(zlib.crc32(b"tail" + var.encode()))
                diffs = sorted(
                    (lambda a, b: sum(a) / len(a) - sum(b) / len(b))(
                        [x[rng.randrange(len(x))] for _ in x],
                        [y[rng.randrange(len(y))] for _ in y],
                    )
                    for _ in range(BOOT_N)
                )
                lo = diffs[int(0.025 * len(diffs))]
                hi = diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]
                point = _nanmean(x) - _nanmean(y)
                tail[var] = {
                    "on": _nanmean(x),
                    "off": _nanmean(y),
                    "diff": point,
                    "ci": (lo, hi),
                    "verdict": _gain_label(point, lo, hi, sign),
                }
    recent_nocr = _nanmean([v for v in sm["no_change"][-500:]])
    # question-matched variant of the tail contrast: for every question
    # both arms reached, align each question's FIRST k runs (k<=100) across
    # arms — removes the composition confound (the arms spend the tail on
    # different question mixes, itself plausibly an edit effect, but not a
    # like-for-like quality comparison).
    tail_q = {}
    if edits and control:
        lq_m = _local_best_series(runs, direction)
        lq_c = _local_best_series(control["runs"], direction)
        for var in sm:
            xs_, ys_ = [], []
            for qid, qs_, qe_ in spans:
                if qe_ <= t0 or qid not in span_c_map:
                    continue
                ca_, cb_ = span_c_map[qid]
                k = min(100, qe_ - max(qs_, t0), cb_ - ca_)
                if k <= 0:
                    continue
                src_m = lq_m if var == "best" else sm[var]
                src_c = lq_c if var == "best" else sc[var]
                xs_ += [src_m[j] for j in range(max(qs_, t0), max(qs_, t0) + k)]
                ys_ += [src_c[j] for j in range(ca_, ca_ + k)]
            if xs_ and ys_:
                xs_ = [v for v in xs_ if v == v]
                ys_ = [v for v in ys_ if v == v]
                sign = VAR_SIGN.get(var, 1 if direction == "max" else -1)
                rng = random.Random(zlib.crc32(b"tailq" + var.encode()))
                diffs = sorted(
                    (lambda a, b: sum(a) / len(a) - sum(b) / len(b))(
                        [xs_[rng.randrange(len(xs_))] for _ in xs_],
                        [ys_[rng.randrange(len(ys_))] for _ in ys_],
                    )
                    for _ in range(BOOT_N)
                )
                lo = diffs[int(0.025 * len(diffs))]
                hi = diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]
                point = _nanmean(xs_) - _nanmean(ys_)
                tail_q[var] = {
                    "on": _nanmean(xs_),
                    "off": _nanmean(ys_),
                    "diff": point,
                    "ci": (lo, hi),
                    "verdict": _gain_label(point, lo, hi, sign),
                }
    # earliest visible reaction: when each arm left its first question
    q0_switch = {}
    for name, rs in (("on", runs), ("off", control["runs"] if control else [])):
        if rs:
            q0 = rs[0]["question_id"]
            q0_switch[name] = next(
                (i for i, r in enumerate(rs) if r["question_id"] != q0), len(rs)
            )
    return {
        "edits": out,
        "tail": tail,
        "tail_q": tail_q,
        "tail_start": t0,
        "tail_end": t1,
        "recent_no_change": recent_nocr,
        "q0_switch": q0_switch,
    }


def fig_trajectory(sess: dict, direction: str, metric: str, dest: Path, ylim=None):
    """Per-run metric + running best over the consumed-budget axis (one
    single-variable experiment per cycle). Standalone figure: centered bold
    title, standard axis names, baseline reference line."""
    runs = sess["runs"]
    mlabel = metric_label(metric)
    xs = _hours(runs)
    vals = [r["metric"]["value"] for r in runs]
    best = best_series(runs, direction)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    dense = len(runs) > 600  # long-run mode: thin the ink, keep the story

    # question bands: shading only, no per-band text labels (at print size
    # the q-labels crowd the top edge and add no quantitative information;
    # the question sequence is tabulated in the report)
    q_bounds, cur, start = [], None, 0
    for i, r in enumerate(runs):
        if r["question_id"] != cur:
            if cur is not None:
                q_bounds.append((cur, start, i - 1))
            cur, start = r["question_id"], i
    q_bounds.append((cur, start, len(runs) - 1))
    for i, (q, a, b) in enumerate(q_bounds):
        ax.axvspan(
            xs[a],
            xs[b],
            color=INK2,
            alpha=0.07 if i % 2 == 0 else 0.035,
            zorder=0,
            linewidth=S(0),
        )

    # per-run markers: status via shape + color + opacity hierarchy
    # (ours 1.0 / raw mass thinned in dense mode / alert for fail). Keeps
    # are drawn last and largest so the ratchet's evidence stays visible
    # above the discard mass; marker area grows with the font ladder.
    if dense:
        sizes = {"keep": S2(14), "discard": S2(9), "fail": S2(20)}
    else:
        sizes = {"keep": S2(40), "discard": S2(22), "fail": S2(50)}
    for status, color, marker, label in (
        ("discard", INK3, "X", "discarded"),
        ("fail", ALERT, "^", "failed"),
        ("keep", OURS, "o", "kept"),
    ):
        pts = [
            (x, v)
            for x, r, v in zip(xs, runs, vals)
            if r["status"] == status and v is not None
        ]
        if pts:
            alpha = 1.0 if status == "keep" else (0.85 if not dense else 0.35)
            ax.scatter(
                *zip(*pts),
                s=sizes[status],
                c=color,
                marker=marker,
                zorder=3 if status != "keep" else 4,
                alpha=alpha,
                label=f"{label} ($n$ = {len(pts)})",
                linewidths=0,
            )
    # best-so-far step line: OURS, top weight, star at each improvement
    bx = [x for x, b in zip(xs, best) if b is not None]
    by = [b for b in best if b is not None]
    ax.step(
        bx,
        by,
        where="post",
        color=OURS,
        linewidth=S(2.8),
        zorder=5,
        label="running best (kept)",
    )
    changed = [(bx[i], by[i]) for i in range(len(by)) if i == 0 or by[i] != by[i - 1]]
    if changed:
        ax.plot(
            *zip(*changed),
            linestyle="none",
            marker="*",
            color=OURS,
            markersize=S(12),
            zorder=6,
        )

    # epoch boundaries (labels staggered — nearby lines would otherwise
    # collide at the same height). Beyond ~24 epochs the lines become a
    # barcode: drop them entirely and say how many there were.
    epochs = sess["epochs"]["epochs"]
    if len(epochs) <= 24:
        for i, e in enumerate(epochs[1:]):
            ci = max(
                (j for j, r in enumerate(runs) if r["epoch"] < e["epoch"]), default=None
            )
            if ci is not None:
                ax.axvline(
                    xs[ci], color=INK3, linestyle=(0, (4, 3)), linewidth=S(1), zorder=2
                )
                ax.annotate(
                    e["epoch"],
                    (xs[ci], 0.92 - 0.05 * (i % 3)),
                    xycoords=("data", "axes fraction"),
                    fontsize=ANNOT_FS,
                    color=INK2,
                    va="top",
                    ha="left",
                )
    else:
        ax.annotate(
            f"{len(epochs)} epoch upgrades (dense — lines omitted)",
            (0.008, 0.03),
            xycoords="axes fraction",
            fontsize=ANNOT_FS,
            color=INK2,
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
        )

    # initial-baseline reference line (the first measured incumbent value):
    # everything above it is progress the loop actually made
    base = next(
        (
            v
            for r, v in zip(runs, vals)
            if r["variable_changed"] == "baseline" and v is not None
        ),
        None,
    )
    if base is not None:
        ax.axhline(base, color=INK3, linestyle=(0, (2, 3)), linewidth=S(1.1), zorder=2)
        ax.annotate(
            f"initial baseline {base:.3f}",
            (xs[0], base),
            xytext=(2, -12),
            textcoords="offset points",
            fontsize=ANNOT_FS,
            color=INK2,
            va="top",
            ha="left",
        )

    # direct label on the final best (selective, not every point) — placed
    # below-right of the end point so it stays clear of the curve end
    if by:
        ax.annotate(
            f"{by[-1]:.3f}",
            (bx[-1], by[-1]),
            xytext=(-2, -14),
            textcoords="offset points",
            fontsize=ANNOT_FS,
            color=INK,
            fontweight="bold",
            va="top",
            ha="right",
        )

    arrow = r"$\uparrow$" if direction == "max" else r"$\downarrow$"
    ax.set_xlabel("Wall-Clock Time (h)")
    ax.set_ylabel(f"{mlabel} {arrow}")
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        # no auto-margin below zero (a "-0" tick is not a quantity); leave
        # headroom at the top for the improvement stars
        finite = [v for v in vals if v is not None]
        if finite:
            lo, hi = min(finite), max(finite)
            span = hi - lo or 1.0
            ax.set_ylim(max(0.0, lo - 0.05 * span), hi + 0.08 * span)
    # dense canvases have no free corner — the legend goes below the axes
    # (same convention as the self-edit response panels). Scatter legend
    # markers ignore `s`, so build explicit handles at readable sizes.
    handles = [
        plt.Line2D([], [], color=OURS, linewidth=S(2.8), label="running best (kept)")
    ]
    for status, color, marker, label in (
        ("keep", OURS, "o", "kept"),
        ("discard", INK3, "X", "discarded"),
        ("fail", ALERT, "^", "failed"),
    ):
        n = sum(
            1
            for r, v in zip(runs, vals)
            if r["status"] == status and v is not None
        )
        if n:
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker=marker,
                    color=color,
                    markersize=S(8),
                    label=f"{label} ($n$ = {n})",
                )
            )
    fig.legend(
        handles=handles, loc="lower center", ncols=2, bbox_to_anchor=(0.5, 0.0)
    )
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    _save(fig, dest)
    plt.close(fig)


def fig_keeprate(sess: dict, dest: Path):
    """Standalone figure: rolling keep-rate (research efficiency) with the
    self-improvement ledger overlaid as decision events (application line +
    kept star / reverted cross at the top edge). The reserved hue marks the
    keep-rate curve and promoted edits. The trailing window scales with the
    run count: a fixed 9-run window is a barcode over thousands of cycles."""
    runs = sess["runs"]
    ledger = [r for r in sess["ledger"]]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    w = max(50, len(runs) // 60)
    xs = _hours(runs)
    kr = [
        sum(1 for r in runs[max(0, i - w + 1) : i + 1] if r["status"] == "keep")
        / len(runs[max(0, i - w + 1) : i + 1])
        for i in range(len(runs))
    ]
    ax.plot(
        xs[w - 1 :],
        kr[w - 1 :],  # skip the window warm-up (partial-window transient)
        color=OURS,
        linewidth=S(1.8),
        zorder=4,
        label=f"rolling keep rate (trailing {w} runs)",
    )
    ax.set_ylabel("Keep Rate")
    ax.set_ylim(0, 1)
    ax.set_xlim(xs[0], xs[-1])

    # ledger decision events (edits numbered in application order; ledger
    # hashes stay in the report text). Applied at the far left on an 8 h
    # axis, a lifecycle SPAN is an invisible sliver — a vertical marker
    # plus a top-edge verdict glyph stays readable at any horizon.
    fold: dict[str, dict] = {}
    for row in ledger:
        fold[row["edit_id"]] = row
    ordered = sorted(fold.values(), key=lambda r: r.get("applied_at_cycle") or 0)
    edit_no = {r["edit_id"]: i + 1 for i, r in enumerate(ordered)}
    n_labeled = 0
    for row in ordered:
        start = row.get("applied_at_cycle")
        status = row["status"]
        if start is None or start >= len(xs):
            continue
        kept = status == "kept"
        color = OURS if kept else (ALERT if status == "reverted" else INK3)
        ax.axvline(
            xs[start], color=color, linestyle=(0, (2, 2)), linewidth=S(1.2), zorder=2
        )
        if status in ("kept", "reverted"):
            y = 1.0 if n_labeled % 2 == 0 else 0.93
            ax.plot(
                [xs[start]],
                [y],
                linestyle="none",
                marker="*" if kept else "X",
                color=color,
                markersize=S(11),
                zorder=6,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )
            ax.annotate(
                f"edit {edit_no[row['edit_id']]} ({status})",
                (xs[start], y),
                xycoords=("data", "axes fraction"),
                xytext=(5, -3),
                textcoords="offset points",
                fontsize=ANNOT_FS,
                color=INK2,
                va="top",
                ha="left",
                annotation_clip=False,
            )
            n_labeled += 1

    # headline statistic: overall keep rate for the whole session
    n_keeps = sum(1 for r in runs if r["status"] == "keep")
    ax.annotate(
        f"overall keep rate: {n_keeps / len(runs):.1%} ({n_keeps}/{len(runs)})",
        (0.99, 0.04),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=LEGEND_FS,
        color=INK,
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
    )
    handles = [
        plt.Line2D([], [], color=OURS, linewidth=S(1.8), label="rolling keep rate")
    ]
    if any(r["status"] == "kept" for r in fold.values()):
        handles.append(
            plt.Line2D(
                [], [], linestyle="none", marker="*", color=OURS,
                markersize=S(11), label="edit kept",
            )
        )
    if any(r["status"] == "reverted" for r in fold.values()):
        handles.append(
            plt.Line2D(
                [], [], linestyle="none", marker="X", color=ALERT,
                markersize=S(9), label="edit reverted",
            )
        )
    fig.legend(handles=handles, loc="lower center", ncols=3,
               bbox_to_anchor=(0.5, 0.0))
    ax.set_xlabel("Wall-Clock Time (h)")
    ax.set_title("Rolling Keep Rate and Self-Edit Decisions", pad=10)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    _save(fig, dest)
    plt.close(fig)


def fig_exploration(sess: dict, dest: Path):
    """Standalone figure: rolling knob-usage share over the loop (trailing
    window 200, non-baseline runs), one line per knob. Legend labels carry
    each knob's trial count and keep rate — the per-knob payoff summary.
    Baseline re-measurements are bookkeeping at question/epoch switches and
    are excluded from the share. (A per-cycle scatter degenerates into
    solid stripes over thousands of cycles; the rolling share is the
    readable summary at this horizon.)"""
    runs = sess["runs"]
    knobs = sorted(
        {r["variable_changed"] for r in runs if r["variable_changed"] != "baseline"}
    )
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    W = 200
    xs = _hours(runs)
    nonbase = [r["variable_changed"] != "baseline" for r in runs]
    ymax = 0.0
    for i, k in enumerate(knobs):
        flags = [r["variable_changed"] == k for r in runs]
        share = []
        for j in range(len(runs)):
            lo = max(0, j - W + 1)
            den = sum(nonbase[lo : j + 1])
            share.append(sum(flags[lo : j + 1]) / den if den else float("nan"))
        n = sum(flags)
        nk = sum(
            1 for r in runs if r["variable_changed"] == k and r["status"] == "keep"
        )
        # skip the warm-up: before W runs the trailing share is a partial-
        # window transient (the t=0 spike to 1.0 is one run, not a mix)
        xs_k, share_k = xs[W - 1 :], share[W - 1 :]
        ymax = max(ymax, max((s for s in share_k if s == s), default=0.0))
        ax.plot(
            xs_k,
            share_k,
            color=SUB[i % 6],
            linewidth=S(1.6),
            zorder=3,
            label=f"{k.split('.')[-1]} ({n} trials; {nk / n:.1%} kept)",
        )
    ax.set_ylabel("Share of Exploration Runs")
    ax.set_xlabel("Wall-Clock Time (h)")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(0, max(0.5, 1.15 * ymax))
    ax.set_title("Exploration Mix over the Experiment Loop", pad=10)
    # window size as a subtitle line (inside the print-size frame; a
    # parenthetical in the title would overflow the tight bounding box)
    ax.text(
        0.5,
        1.015,
        f"trailing {W}-run window",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=ANNOT_FS,
        color=INK2,
    )
    # below the axes (same convention as the other dense figures); two
    # columns — three would be wider than the print-size frame
    fig.legend(loc="lower center", ncols=2, bbox_to_anchor=(0.5, 0.0))
    ax.grid(axis="x", visible=False)
    fig.tight_layout(rect=[0, 0.22, 1, 1])
    _save(fig, dest)
    plt.close(fig)


def fig_compare(sessions: dict, direction: str, dest: Path, metric: str):
    """Best-so-far curves, ours (reserved hue, solid, lw 2.8, star) vs the
    strongest reference (contrasting hue, dashed, lw 1.8). Standalone
    figure: centered bold title; each legend label carries its series'
    final value (end labels floating outside the frame collide whenever
    the arms tie, and they break the print-size bounding box)."""
    mlabel = metric_label(metric)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    style = {
        "main": dict(color=OURS, linestyle="-", linewidth=S(2.8)),
        "control": dict(color=REFERENCE, linestyle=(0, (5, 3)), linewidth=S(1.8)),
    }
    for name, sess in sessions.items():
        runs = sess["runs"]
        xs = _hours(runs)
        best = best_series(runs, direction)
        st = style.get(name, dict(color=SUB[0], linestyle="-", linewidth=S(1.8)))
        label = arm_label(name)
        if best and best[-1] is not None:
            label += f" (final: {best[-1]:.3f})"
        ax.step(xs, best, where="post", label=label, **st)
        if name == "main":  # ours: star marker, sparse
            every = max(1, len(xs) // 30)
            ax.plot(
                xs[::every],
                best[::every],
                linestyle="none",
                marker="*",
                color=OURS,
                markersize=S(9),
                zorder=5,
            )
    arrow = r"$\uparrow$" if direction == "max" else r"$\downarrow$"
    ax.set_xlabel("Wall-Clock Time (h)")
    ax.set_ylabel(f"Best {mlabel} {arrow}")
    # metric name lives on the y-axis; repeating it in the title would push
    # the title past the print-size frame (it sets the tight bbox width)
    ax.set_title("Running Best — Self-Editing Ablation", pad=10)
    fig.legend(
        loc="lower center",
        ncols=2,
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.0,
        handlelength=1.5,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    _save(fig, dest)
    plt.close(fig)


_EDIT_RESP_PANELS = [
    ("nochange", "no_change", "No-Change Waste Rate", "No-Change Rate"),
    ("keeprate", "keep", "Keep Rate", "Keep Rate"),
    ("value", "value", "Per-Run {m}", "{m}"),
    ("best", "best", "Running Best {m}", "Best {m}"),
]


def fig_edit_response(sessions: dict, direction: str, metric: str, dest_dir: Path):
    """The L1 figures — four STANDALONE files, one per response variable:
    rolling ON/OFF curves over the early region with the edit lifecycles
    marked (dotted apply line, shaded engine decision window, kept star /
    reverted cross). Each file stands alone: centered bold title, question
    bands, lower-right legend, one-line edit summary below the axis."""
    mlabel = metric_label(metric)
    main = sessions["main"]
    runs = main["runs"]
    sm = _var_series(runs, direction)
    control = sessions.get("control")
    sc = _var_series(control["runs"], direction) if control else None

    # the early region is a TIME window (consistent with the wall-clock
    # axis): the arms contribute however many runs fit into it
    T_END = 0.75
    hs = _hours(runs)
    n_on = next((i for i, h in enumerate(hs) if h > T_END), len(hs))
    xs = hs[:n_on]
    if control:
        hc = _hours(control["runs"])
        n_off = next((i for i, h in enumerate(hc) if h > T_END), len(hc))
        xc = hc[:n_off]
    else:
        xc = []

    def roll(vals):
        """Trailing rolling mean (ROLL_W), NaN-aware."""
        return [
            _nanmean(vals[max(0, i - ROLL_W + 1) : i + 1]) for i in range(len(vals))
        ]

    fold: dict[str, dict] = {}
    for row in main["ledger"]:
        fold[row["edit_id"]] = row
    edits = sorted(
        (r for r in fold.values() if r.get("applied_at_cycle") is not None),
        key=lambda r: r["applied_at_cycle"],
    )
    spans = _question_spans(runs[:n_on])

    for suffix, var, title_t, ylabel_t in _EDIT_RESP_PANELS:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        for i, (q, a, b) in enumerate(spans):
            ax.axvspan(
                xs[a],
                xs[min(b, n_on - 1)],
                color=INK2,
                alpha=0.05 if i % 2 == 0 else 0.0,
                zorder=0,
                linewidth=S(0),
            )
            # label a band only if its text fits (>=8% of the x window);
            # stagger two levels so adjacent labels never collide
            if xs[min(b, n_on - 1)] - xs[a] >= 0.08 * T_END:
                ax.annotate(
                    q.split("-")[-1],
                    (xs[(a + min(b, n_on - 1)) // 2], 0.02 + 0.06 * (i % 2)),
                    xycoords=("data", "axes fraction"),
                    ha="center",
                    va="bottom",
                    fontsize=ANNOT_FS,
                    color=INK2,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
                )
        ax.plot(xs, roll(sm[var][:n_on]), color=OURS, linewidth=S(2.2), zorder=4)
        if sc:
            ax.plot(
                xc,
                roll(sc[var][:n_off]),
                color=REFERENCE,
                linestyle=(0, (5, 3)),
                linewidth=S(1.8),
                zorder=3,
            )

        for e in edits:
            a = e["applied_at_cycle"]
            d = e.get("terminated_at")
            if a >= n_on:
                continue
            x0 = xs[a]
            x1 = xs[min(d - 1, n_on - 1)] if isinstance(d, int) and d > 0 else xs[-1]
            ax.axvspan(x0, x1, color=INK, alpha=0.14, zorder=1, linewidth=S(0))
            ax.axvline(
                x0, color=INK2, linestyle=(0, (2, 2)), linewidth=S(1.2), zorder=2
            )
            kept = e["status"] == "kept"
            ax.plot(
                [x1],
                [0.97],
                linestyle="none",
                marker="*" if kept else "X",
                color=OURS if kept else ALERT,
                markersize=S(12),
                zorder=6,
                transform=ax.get_xaxis_transform(),
            )

        # the encoding is shared by all four panels; placing the legend
        # below the axes keeps it clear of every response curve
        fig.legend(
            loc="lower center",
            ncols=3,
            bbox_to_anchor=(0.5, 0.0),
            handles=[
                plt.Line2D(
                    [], [], color=OURS, linewidth=S(2.2), label="self-editing ON"
                ),
                plt.Line2D(
                    [],
                    [],
                    color=REFERENCE,
                    linestyle=(0, (5, 3)),
                    linewidth=S(1.8),
                    label="self-editing OFF",
                ),
                plt.Line2D(
                    [],
                    [],
                    color=INK2,
                    linestyle=(0, (2, 2)),
                    linewidth=S(1.2),
                    label="self-edit applied",
                ),
                plt.Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="*",
                    color=OURS,
                    markersize=S(11),
                    label="edit kept",
                ),
                plt.Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="X",
                    color=ALERT,
                    markersize=S(9),
                    label="edit reverted",
                ),
                Patch(facecolor=INK, alpha=0.14, label="decision window"),
            ],
        )
        ax.set_xlabel("Wall-Clock Time (h)")
        ax.set_ylabel(ylabel_t.format(m=mlabel))
        ax.set_xlim(0, T_END)
        ax.set_title(f"Self-Edit Response: {title_t.format(m=mlabel)}", pad=10)
        fig.tight_layout(rect=[0, 0.15, 1, 1])
        _save(fig, dest_dir / f"edit_response_{suffix}.pdf")
        plt.close(fig)


# -- question graph ------------------------------------------------------------

# the frontier question is "ours" (reserved hue); resolved questions take
# subordinate gray-scale shades; light fills get dark node text
Q_STATUS_COLOR = {
    "open": OURS,
    "partially_answered": "#6b6b6b",
    "answered": "#3d3d3d",
    "refined": "#b8860b",
    "merged": SPINE,
    "retired": SPINE,
}
Q_LIGHT_FILL = {SPINE}
Q_EDGE_COLOR = {
    "spawns": INK2,
    "refines": "#b8860b",
    "challenges": ALERT,
    "depends_on": "#4a6fa5",
    "merged_into": INK3,
}


def fig_question_graph(sess: dict, dest: Path):
    """The research-question graph: graphviz hierarchical layout when `dot`
    is available, a dependency-free layered layout otherwise."""
    qs = sess["questions"]
    if not qs:
        return
    dot = shutil.which("dot")
    if dot is not None:
        _question_graph_dot(qs, dest, dot)
    else:
        _question_graph_mpl(qs, dest)


def _question_graph_dot(qs: list[dict], dest: Path, dot: str):
    """graphviz dot, rankdir=LR: spawn chains read as left→right timelines;
    refine/merge fan-outs get real DAG layering (crossing-minimized)."""
    known = {q["id"] for q in qs}
    lines = [
        "digraph questions {",
        f'  graph [rankdir=LR, bgcolor="{SURFACE}", pad="0.3", nodesep="0.55",'
        f' ranksep="0.95", labelloc="t", fontname="Times New Roman",'
        f' fontsize="14", fontcolor="{INK2}"];',
        f'  node [shape="circle", style="filled", fontname="Times New Roman",'
        f' width="0.72", fixedsize="true", penwidth="1.2",'
        f' color="{SURFACE}"];',
        f'  edge [fontname="Times New Roman", fontsize="10", arrowsize="0.7"];',
    ]
    for q in qs:
        fill = Q_STATUS_COLOR.get(q["status"], SPINE)
        ring = f', penwidth="2.6", color="{INK}"' if q["status"] == "open" else ""
        fontcolor = INK if fill in Q_LIGHT_FILL else "white"
        best = (q.get("best_result") or {}).get("score")
        label = (
            f'<font point-size="12" color="{fontcolor}">'
            f"<b>{q['id'].split('-')[-1]}</b></font>"
        )
        if best is not None:
            label += f'<br/><font point-size="8" color="{fontcolor}">{best:.3f}</font>'
        lines.append(
            f'  "{q["id"]}" [fillcolor="{fill}", fontcolor="{fontcolor}",'
            f" label=<{label}>{ring}];"
        )
    for q in qs:
        for e in q.get("edges", []):
            if e["to"] not in known:
                continue
            color = Q_EDGE_COLOR.get(e["type"], INK3)
            lbl = (
                f', label="{e["type"]}", fontcolor="{color}"'
                if e["type"] != "spawns"
                else ""
            )
            lines.append(f'  "{q["id"]}" -> "{e["to"]}" [color="{color}"{lbl}];')
    lines.append(
        '  label=<<b><font point-size="18">Research-Question Graph '
        '(Slow Loop)</font></b><br/><font point-size="11">purple = frontier '
        "(open, next up) &#183; dark gray = partially answered &#183; light "
        "gray = merged/retired &#183; number inside = best score &#183; "
        "unlabeled edges are spawns, other edge types are labeled</font>>;"
    )
    lines.append("}")
    src = dest.with_suffix(".gv")
    src.write_text("\n".join(lines) + "\n")
    subprocess.run([dot, "-Tpdf", "-o", str(dest), str(src)], check=True)


def _question_graph_mpl(qs: list[dict], dest: Path):
    """Dependency-free fallback: longest-path layering over the typed edges
    (a spawn chain becomes a straight left→right line; fan-outs stack
    vertically within their layer — no zigzag, no crossing labels)."""
    ids = [q["id"] for q in qs]
    known = set(ids)
    layer = {i: 0 for i in ids}
    for _ in range(len(ids)):  # relax edges until stable (DAG longest path)
        changed = False
        for q in qs:
            for e in q.get("edges", []):
                if e["to"] in known and layer[e["to"]] < layer[q["id"]] + 1:
                    layer[e["to"]] = layer[q["id"]] + 1
                    changed = True
        if not changed:
            break
    by_layer: dict[int, list[str]] = {}
    for i in ids:
        by_layer.setdefault(layer[i], []).append(i)
    n_layers = len(by_layer)
    max_depth = max(len(v) for v in by_layer.values())
    pos = {}
    for L in sorted(by_layer):
        members = by_layer[L]
        for j, i in enumerate(members):
            pos[i] = (
                L / max(1, n_layers - 1) if n_layers > 1 else 0.5,
                (j + 1) / (len(members) + 1),
            )
    fig, ax = plt.subplots(
        figsize=(max(8.5, 1.25 * n_layers), max(2.6, 1.15 * max_depth + 1.2))
    )
    for q in qs:  # edges under nodes
        for e in q.get("edges", []):
            if e["to"] not in pos:
                continue
            x1, y1 = pos[q["id"]]
            x2, y2 = pos[e["to"]]
            ax.annotate(
                "",
                (x2, y2),
                (x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=Q_EDGE_COLOR.get(e["type"], INK3),
                    lw=1.3,
                    shrinkA=17,
                    shrinkB=17,
                    mutation_scale=13,
                ),
            )
            if e["type"] != "spawns":
                ax.annotate(
                    e["type"],
                    ((x1 + x2) / 2, (y1 + y2) / 2),
                    xytext=(0, 5),
                    textcoords="offset points",
                    fontsize=SF(7.5),
                    ha="center",
                    color=Q_EDGE_COLOR.get(e["type"], INK3),
                )
    for q in qs:
        x, y = pos[q["id"]]
        fill = Q_STATUS_COLOR.get(q["status"], SPINE)
        open_ring = (
            {"edgecolors": INK, "linewidths": 2.2}
            if q["status"] == "open"
            else {"edgecolors": "none", "linewidths": 0}
        )
        ax.scatter([x], [y], s=900, zorder=3, color=fill, **open_ring)
        fontcolor = INK if fill in Q_LIGHT_FILL else "white"
        ax.annotate(
            q["id"].split("-")[-1],
            (x, y),
            ha="center",
            va="center",
            fontsize=SF(10),
            color=fontcolor,
            zorder=4,
            fontweight="bold",
        )
        # short caption (status is the node color, in the legend), staggered
        # on two levels so same-layer neighbors never collide
        best = (q.get("best_result") or {}).get("score")
        caption = ""
        if best is not None:
            caption = f"best {best:.3f}"
        if q.get("saturation"):
            caption += f"\nsat {q['saturation']:.2f}"
        if caption:
            level = list(pos).index(q["id"]) % 2
            ax.annotate(
                caption,
                (x, y),
                xytext=(0, -30 - 26 * level),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=SF(7.5),
                color=INK2,
                zorder=4,
            )
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=S(10), color=c, label=s)
        for s, c in Q_STATUS_COLOR.items()
        if any(q["status"] == s for q in qs)
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        ncols=min(4, len(handles)),
        handletextpad=0.1,
        columnspacing=0.9,
    )
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.3, 1.12)
    ax.axis("off")
    ax.set_title("Research-Question Graph (Slow Loop)", pad=10)
    fig.tight_layout()
    _save(fig, dest)
    plt.close(fig)


def key_events(sess: dict, direction: str) -> list[str]:
    events = []
    runs = sess["runs"]
    best = None
    for r in runs:
        v = r["metric"]["value"]
        c = _cycle(r["run_id"])
        if (
            r["status"] == "keep"
            and v is not None
            and (best is None or (v > best if direction == "max" else v < best))
        ):
            best = v
            events.append((c, f"new best {v:.4f} ({r['variable_changed']})"))
    prev_epoch = None
    for r in runs:
        if prev_epoch and r["epoch"] != prev_epoch:
            events.append(
                (
                    _cycle(r["run_id"]),
                    f"epoch {prev_epoch} → {r['epoch']} (benchmark "
                    f"upgraded, incumbent re-baselined)",
                )
            )
        prev_epoch = r["epoch"]
    prev_q = None
    for r in runs:
        if prev_q and r["question_id"] != prev_q:
            events.append(
                (
                    _cycle(r["run_id"]),
                    f"question {prev_q.split('-')[-1]} → "
                    f"{r['question_id'].split('-')[-1]}",
                )
            )
        prev_q = r["question_id"]
    fold: dict[str, dict] = {}
    for row in sess["ledger"]:
        if row["edit_id"] not in fold or row["status"] in ("kept", "reverted"):
            if row["status"] in ("kept", "reverted") or row["edit_id"] not in fold:
                fold[row["edit_id"]] = row
    ordered = sorted(fold.values(), key=lambda r: r.get("applied_at_cycle") or 0)
    edit_no = {r["edit_id"]: i + 1 for i, r in enumerate(ordered)}
    for row in ordered:
        if row["status"] in ("kept", "reverted"):
            events.append(
                (
                    row.get("applied_at_cycle") or 0,
                    f"self-edit {edit_no[row['edit_id']]} → "
                    f"{row['status']} (strategy priority rotation)",
                )
            )
    return [f"cycle {c}: {msg}" for c, msg in sorted(events)]


def render(sessions: dict[str, dict], args) -> Path:
    dest = Path(args.dest)
    (dest / "figures").mkdir(parents=True, exist_ok=True)
    main = sessions["main"]
    if not main["runs"]:
        raise SystemExit(
            "error: main session has no runs — nothing to "
            "report (did the engine run any cycles?)"
        )
    direction = args.direction
    metric = args.metric

    fig_trajectory(main, direction, metric, dest / "figures" / "trajectory.pdf")
    fig_keeprate(main, dest / "figures" / "keeprate.pdf")
    fig_exploration(main, dest / "figures" / "exploration.pdf")
    if "control" in sessions:
        fig_compare(sessions, direction, dest / "figures" / "compare.pdf", metric)
        fig_edit_response(sessions, direction, metric, dest / "figures")

    runs = main["runs"]
    keeps = [r for r in runs if r["status"] == "keep"]
    # the headline best is the RATCHET (kept runs only): a discarded run was
    # measured but rejected by the gates — reporting it as "best" would claim
    # a result the system itself refused to trust
    keep_vals = [
        r["metric"]["value"] for r in keeps if r["metric"]["value"] is not None
    ]
    best = (
        (max(keep_vals) if direction == "max" else min(keep_vals))
        if keep_vals
        else None
    )
    # ...but a rejected measurement ABOVE the ratchet is itself a finding
    # (gate/noise-floor miscalibration) — surface it, don't hide it
    all_vals = [r["metric"]["value"] for r in runs if r["metric"]["value"] is not None]
    rejected_best = None
    if all_vals and best is not None:
        cand = max(all_vals) if direction == "max" else min(all_vals)
        if (cand > best) if direction == "max" else (cand < best):
            rejected_best = cand
    base = next(
        (
            r["metric"]["value"]
            for r in runs
            if r["variable_changed"] == "baseline" and r["metric"]["value"] is not None
        ),
        None,
    )
    fold: dict[str, dict] = {}
    for row in main["ledger"]:
        fold[row["edit_id"]] = row
    edits_kept = sum(1 for r in fold.values() if r["status"] == "kept")
    edits_rev = sum(1 for r in fold.values() if r["status"] == "reverted")

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    improvement = (
        f"{best - base:+.4f}"
        if isinstance(best, (int, float)) and isinstance(base, (int, float))
        else "—"
    )

    title = (
        f"# {args.task} — self-editing ablation report"
        if "control" in sessions
        else f"# {args.task} — RSI session report"
    )
    lines = [
        title,
        "",
        f"- cycles: **{len(runs)}** · seed: **{args.seed}** · "
        f"epochs: **{len(main['epochs']['epochs'])}** · "
        f"questions: **{len(main['questions'])}** · "
        f"hypotheses: **{len(main['hypotheses'])}** · "
        f"evidence records: **{len(main['evidence'])}**",
    ]
    if "control" in sessions:
        lines.append(
            "- arms: **self-editing ON** (naive priority-rotation improver) "
            "vs **self-editing OFF** (noop improver)"
        )
    lines.append(
        f"- baseline {metric}: **{_fmt(base)}** → final best (kept): "
        f"**{_fmt(best)}** (improvement {improvement})"
    )
    if rejected_best is not None:
        lines.append(
            f"- note: a better value **{_fmt(rejected_best)}** was measured "
            f"but REJECTED by the gates (e.g. noise floor) — check whether "
            f"the gate or the measurement is wrong"
        )
    lines += [
        f"- keep-rate: **{len(keeps) / len(runs):.2%}** · "
        f"self-edits kept/reverted: **{edits_kept}/{edits_rev}**",
        "",
        "![trajectory](figures/trajectory.pdf)",
        "",
        f"*Trajectory of {metric} on the wall-clock axis (one "
        f"single-variable experiment per cycle; seed {args.seed}). Purple "
        "line + stars: running best (kept runs); purple dots: keeps; gray "
        "crosses: discards; red triangles: fails. Background bands are "
        "research questions; dashed verticals are epoch boundaries "
        "(omitted and counted instead beyond 24); the dashed horizontal is "
        "the initial baseline. Baseline re-measurements appear at every "
        "epoch upgrade and question switch — the incumbent solution "
        "persists, its value re-baselines.*",
        "",
        "![keeprate](figures/keeprate.pdf)",
        "",
        "*Rolling keep rate (trailing window scaled to the run count) "
        "over wall-clock time. Self-edit decisions are marked at their "
        "application time: dotted vertical, star = kept, cross = "
        "reverted. The overall keep rate is annotated at the lower "
        "right.*",
        "",
        "![exploration](figures/exploration.pdf)",
        "",
        "*Rolling share of each knob over wall-clock time (trailing window "
        "200 runs) — how the fast loop's exploration mix drifts. "
        "Legend labels carry each knob's trial count and keep rate (the "
        "per-knob payoff). Baseline re-measurements are bookkeeping at "
        "question/epoch switches and are excluded from the share.*",
        "",
    ]
    if "control" in sessions:
        c = sessions["control"]
        cruns = c["runs"]
        cvals = [
            r["metric"]["value"]
            for r in cruns
            if r["status"] == "keep" and r["metric"]["value"] is not None
        ]
        cbest = (max(cvals) if direction == "max" else min(cvals)) if cvals else None
        ckeeps = sum(1 for r in cruns if r["status"] == "keep")
        crate = f"{ckeeps / len(cruns):.2%}" if cruns else "—"
        # quantified self-improvement: sample efficiency (mean running best)
        # + speed (cycles for main to match the control's final best)
        auc_m = mean_running_best(runs, direction)
        auc_c = mean_running_best(cruns, direction)
        c2m = cycles_to_reach(runs, direction, cbest)
        lines += [
            "## Self-editing ablation",
            "",
            "![compare](figures/compare.pdf)",
            "",
            "*Running best on the wall-clock axis: self-editing ON "
            "(purple, solid, stars) vs self-editing OFF (red, dashed). "
            "Legend labels carry each series' final value.*",
            "",
            "| | self-editing ON | self-editing OFF |",
            "|---|---|---|",
            f"| final best {metric} | **{_fmt(best)}** | {_fmt(cbest)} |",
            f"| keep-rate | {len(keeps) / len(runs):.2%} | {crate} |",
            f"| mean running best (AUC/cycle) | **{_fmt(auc_m)}** | {_fmt(auc_c)} |",
            f"| cycles to match OFF final best | "
            f"**{c2m if c2m is not None else 'never'}** | — |",
            f"| self-edits kept/reverted | **{edits_kept}/{edits_rev}** | 0/0 |",
            "",
        ]
        # -- self-edit response: the defined notion of "did an edit cause
        # gain", per the three-tier definition (L1/L2/L3 above)
        er = edit_response(main, c, direction)
        if er["edits"]:
            # edits are numbered in application order; ledger ids stay in a
            # footnote for traceability
            edit_no = {ed["edit_id"]: i + 1 for i, ed in enumerate(er["edits"])}

            def _cell(v, fmt="{:+.3f}"):
                return fmt.format(v) if v == v else "—"

            lines += [
                "## Self-edit response — did an edit cause measurable gain?",
                "",
                "![no-change response](figures/edit_response_nochange.pdf)",
                "",
                "![keep-rate response](figures/edit_response_keeprate.pdf)",
                "",
                "![run-value response](figures/edit_response_value.pdf)",
                "",
                "![running-best response](figures/edit_response_best.pdf)",
                "",
                "**Definition (L1).** For each edit applied at run index "
                "$c_e$: pre-window $P=[\\max(\\text{prev termination},"
                " c_e-100)\\dots c_e)$, post-window $Q=[c_e \\dots "
                "\\min(c_e+100, \\text{next application}))$, both clipped "
                "to the question active at $c_e$; the control contributes "
                "position-matched windows from the *same* question (its "
                "question sequence desynchronizes from the ON arm's, so "
                "raw run-index matching would confound the value/best "
                "contrasts with question switches). Variables: no-change "
                "waste rate (**primary** — the edit's declared target), "
                "keep rate, mean run value, mean running best. **Gain on a "
                "variable ⇔ the difference-in-differences "
                "$\\mathrm{DiD}=\\Delta_{ON}-\\Delta_{OFF}$ has a 95% "
                "bootstrap CI (2,000 within-window run resamples) lying "
                "strictly on the improvement side of zero.**",
                "",
                "*Rolling (window 25 runs) response variables over the first "
                "0.75 h: self-editing ON (purple) vs self-editing OFF "
                "(red, dashed). Dotted verticals mark edit application; "
                "shaded bands are the arbitrator's decision windows (≥2 "
                "runs); star/cross mark the kept/reverted call. Background "
                "bands show the ON arm's question spans.*",
                "",
                "| edit | arbitrator decision | variable | pre (n) | "
                "post (n) | Δ ON | DiD [95% CI] | verdict |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for ed in er["edits"]:
                wlen = (
                    (ed["terminated_at"] - ed["cycle"])
                    if isinstance(ed["terminated_at"], int)
                    else None
                )
                decision = (
                    f"{ed['status']} ({wlen}-run window)" if wlen else ed["status"]
                )
                for var, label in VAR_LABEL.items():
                    v = ed["vars"][var]
                    pre = f"{v['pre']:.2f} ({v['n_pre']})"
                    post = f"{v['post']:.2f} ({v['n_post']})"
                    did = (
                        (f"{_cell(v['did'])} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
                        if v["did"] == v["did"]
                        else "—"
                    )
                    star = "**" if var == "no_change" else ""
                    lines.append(
                        f"| {star}edit {edit_no[ed['edit_id']]}{star} | "
                        f"{decision} | {star}{label}{star} | {pre} | "
                        f"{post} | {_cell(v['delta_on'])} | {did} | "
                        f"{star}{v['verdict']}{star} |"
                    )
                    decision = ""  # edit/decision only on the first row
            lines.append(
                "*edits numbered in application order; ledger ids: "
                + ", ".join(f"edit {n} = {eid}" for eid, n in edit_no.items())
                + ".*"
            )
            audit_parts = []
            for ed in er["edits"]:
                v = ed["vars"]["no_change"]
                wlen = (
                    (ed["terminated_at"] - ed["cycle"])
                    if isinstance(ed["terminated_at"], int)
                    else None
                )
                agree = {
                    "gain": "corroborated"
                    if ed["status"] == "kept"
                    else "contradicted",
                    "harm": "contradicted"
                    if ed["status"] == "kept"
                    else "corroborated",
                }.get(v["verdict"], "not resolvable at this window")
                audit_parts.append(
                    f"edit {edit_no[ed['edit_id']]}: arbitrator "
                    f"**{ed['status']}** after a {wlen}-run observation "
                    f"window; measured primary response {v['verdict']} "
                    f"(DiD {_cell(v['did'])}) — {agree}."
                )
            lines += ["", "**Decision audit (L2).** " + " ".join(audit_parts)]
            lines += [
                "",
                "**Net-state contrast (L3, cycles "
                f"{er['tail_start']}–{er['tail_end']}).** After the last "
                "edit decision the arms differ by one persistent strategy "
                "state (the kept rotation):",
                "",
                "| variable | ON | OFF | diff [95% CI] | verdict |",
                "|---|---|---|---|---|",
            ]
            for var, label in VAR_LABEL.items():
                t = er["tail"].get(var)
                if not t:
                    continue
                star = "**" if var == "no_change" else ""
                lines.append(
                    f"| {star}{label}{star} | {t['on']:.2f} | {t['off']:.2f} "
                    f"| {_cell(t['diff'])} [{t['ci'][0]:+.3f}, "
                    f"{t['ci'][1]:+.3f}] | {star}{t['verdict']}{star} |"
                )
            if er["tail_q"]:
                lines += [
                    "",
                    "Same contrast, **question-matched** — each question's "
                    "first ≤100 runs in both arms, which removes the "
                    "question-mix composition difference (running best is "
                    "the within-question ratchet here):",
                    "",
                    "| variable | ON | OFF | diff [95% CI] | verdict |",
                    "|---|---|---|---|---|",
                ]
                for var, label in VAR_LABEL.items():
                    t = er["tail_q"].get(var)
                    if not t:
                        continue
                    star = "**" if var == "no_change" else ""
                    lines.append(
                        f"| {star}{label}{star} | {t['on']:.2f} | "
                        f"{t['off']:.2f} | {_cell(t['diff'])} "
                        f"[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] | "
                        f"{star}{t['verdict']}{star} |"
                    )
            q0 = er["q0_switch"]
            lines += [
                "",
                f"*Caveat: the arms' question mixes differ over this tail "
                f"(the ON arm left its first question after "
                f"{q0.get('on', '—')} runs vs {q0.get('off', '—')} in the "
                "control — an early and possibly edit-caused reaction, but "
                "proposer temperature makes single-event attribution "
                "noisy), so the run-indexed table above bundles the edit "
                "with its downstream question-sequencing effects; the "
                "question-matched table is the like-for-like readout.*",
                "",
                f"*Structural note: the self-editing policy has been "
                f"silent since run {er['tail_start']} — its only edit "
                "family (knob-priority rotation) has two distinct "
                "candidates for three knobs, both terminal in the "
                "content-addressed ledger, while its trigger (no-change "
                f"share ≥ 30%) stayed satisfied ({er['recent_no_change']:.0%}"
                " over the last 500 runs). Self-editing in this run = two "
                "early edits + one persistent rotation, not ongoing "
                "self-modification.*",
                "",
            ]
    lines += (
        ["## Key events", "", "```"]
        + key_events(main, direction)
        + [
            "```",
            "",
            "## Final knowledge state",
            "",
            "| question | status | saturation | best |",
            "|---|---|---|---|",
        ]
    )
    for q in main["questions"]:
        b = (q.get("best_result") or {}).get("score")
        lines.append(
            f"| {q['id']} | {q['status']} | {q.get('saturation', 0):.2f} | {b:.4f} |"
            if b is not None
            else f"| {q['id']} | {q['status']} | {q.get('saturation', 0):.2f} | — |"
        )
    lines += [
        "",
        "| hypothesis | status | confidence | support/refute |",
        "|---|---|---|---|",
    ]
    for h in main["hypotheses"][:12]:
        cnt = h.get("evidence_counts") or {}
        lines.append(
            f"| {h['id']} | {h['status']} | "
            f"{h.get('confidence', 0):.2f} | "
            f"{cnt.get('support', 0)}/{cnt.get('refute', 0)} |"
        )
    if len(main["hypotheses"]) > 12:
        lines.append(f"| … ({len(main['hypotheses']) - 12} more hypotheses) | | | |")
    lines += [
        "",
        "## Strategy artifact (final state)",
        "",
        "```python",
        main["strategy"].strip(),
        "```",
        "",
    ]
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            or "unknown"
        )
    except OSError:
        commit = "unknown"
    lines += [
        "---",
        "",
        f"*Generated by `scripts/report.py` (git `{commit}`) from "
        f"sessions: {'; '.join(args.session)}.*",
        "",
    ]
    report = dest / "report.md"
    report.write_text("\n".join(lines))
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--session",
        action="append",
        required=True,
        help="name=session_dir (first must be name=main)",
    )
    p.add_argument("--task", required=True)
    p.add_argument("--metric", default="metric")
    p.add_argument("--direction", choices=["min", "max"], default="max")
    p.add_argument("--seed", default="7")
    p.add_argument("--dest", required=True)
    args = p.parse_args(argv)

    sessions = {}
    for spec in args.session:
        name, _, path = spec.partition("=")
        sessions[name] = load_session(Path(path))
    if "main" not in sessions:
        p.error("first --session must be main=<dir>")
    report = render(sessions, args)
    print(f"report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
