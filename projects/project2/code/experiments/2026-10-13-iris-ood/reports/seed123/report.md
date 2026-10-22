# iris-ood — self-editing ablation report

- cycles: **3525** · seed: **123** · epochs: **132** · questions: **50** · hypotheses: **1657** · evidence records: **3525**
- arms: **self-editing ON** (naive priority-rotation improver) vs **self-editing OFF** (noop improver)
- baseline ood_accuracy: **0.3556** → final best (kept): **0.9778** (improvement +0.6222)
- keep-rate: **6.10%** · self-edits kept/reverted: **1/1**

![trajectory](figures/trajectory.pdf)

*Trajectory of ood_accuracy on the wall-clock axis (one single-variable experiment per cycle; seed 123). Purple line + stars: running best (kept runs); purple dots: keeps; gray crosses: discards; red triangles: fails. Background bands are research questions; dashed verticals are epoch boundaries (omitted and counted instead beyond 24); the dashed horizontal is the initial baseline. Baseline re-measurements appear at every epoch upgrade and question switch — the incumbent solution persists, its value re-baselines.*

![keeprate](figures/keeprate.pdf)

*Rolling keep rate (trailing window scaled to the run count) over wall-clock time. Self-edit decisions are marked at their application time: dotted vertical, star = kept, cross = reverted. The overall keep rate is annotated at the lower right.*

![exploration](figures/exploration.pdf)

*Rolling share of each knob over wall-clock time (trailing window 200 runs) — how the fast loop's exploration mix drifts. Legend labels carry each knob's trial count and keep rate (the per-knob payoff). Baseline re-measurements are bookkeeping at question/epoch switches and are excluded from the share.*

## Self-editing ablation

![compare](figures/compare.pdf)

*Running best on the wall-clock axis: self-editing ON (purple, solid, stars) vs self-editing OFF (red, dashed). Legend labels carry each series' final value.*

| | self-editing ON | self-editing OFF |
|---|---|---|
| final best ood_accuracy | **0.9778** | 1.0000 |
| keep-rate | 6.10% | 17.97% |
| mean running best (AUC/cycle) | **0.8783** | 0.8095 |
| cycles to match OFF final best | **never** | — |
| self-edits kept/reverted | **1/1** | 0/0 |

## Self-edit response — did an edit cause measurable gain?

![no-change response](figures/edit_response_nochange.pdf)

![keep-rate response](figures/edit_response_keeprate.pdf)

![run-value response](figures/edit_response_value.pdf)

![running-best response](figures/edit_response_best.pdf)

**Definition (L1).** For each edit applied at run index $c_e$: pre-window $P=[\max(\text{prev termination}, c_e-100)\dots c_e)$, post-window $Q=[c_e \dots \min(c_e+100, \text{next application}))$, both clipped to the question active at $c_e$; the control contributes position-matched windows from the *same* question (its question sequence desynchronizes from the ON arm's, so raw run-index matching would confound the value/best contrasts with question switches). Variables: no-change waste rate (**primary** — the edit's declared target), keep rate, mean run value, mean running best. **Gain on a variable ⇔ the difference-in-differences $\mathrm{DiD}=\Delta_{ON}-\Delta_{OFF}$ has a 95% bootstrap CI (2,000 within-window run resamples) lying strictly on the improvement side of zero.**

*Rolling (window 25 runs) response variables over the first 0.75 h: self-editing ON (purple) vs self-editing OFF (red, dashed). Dotted verticals mark edit application; shaded bands are the arbitrator's decision windows (≥2 runs); star/cross mark the kept/reverted call. Background bands show the ON arm's question spans.*

| edit | arbitrator decision | variable | pre (n) | post (n) | Δ ON | DiD [95% CI] | verdict |
|---|---|---|---|---|---|---|---|
| **edit 1** | kept (3-run window) | **no-change waste rate** | 0.33 (3) | 0.83 (6) | +0.500 | -0.000 [-0.833, +0.833] | **n.s.** |
| edit 1 |  | keep rate | 0.67 (3) | 0.17 (6) | -0.500 | +0.000 [-0.833, +0.833] | n.s. |
| edit 1 |  | mean run value | 0.38 (3) | 0.55 (6) | +0.174 | +0.000 [-0.159, +0.152] | n.s. |
| edit 1 |  | mean running best | 0.39 (3) | 0.59 (6) | +0.207 | +0.000 [-0.141, +0.141] | n.s. |
| **edit 2** | reverted (3-run window) | **no-change waste rate** | 1.00 (3) | 0.75 (89) | -0.247 | -0.028 [-0.156, +0.101] | **n.s.** |
| edit 2 |  | keep rate | 0.00 (3) | 0.02 (89) | +0.022 | -0.002 [-0.050, +0.044] | n.s. |
| edit 2 |  | mean run value | 0.60 (3) | 0.61 (89) | +0.006 | +0.004 [-0.147, +0.147] | n.s. |
| edit 2 |  | mean running best | 0.67 (3) | 0.67 (89) | -0.000 | -0.000 [-0.000, -0.000] | harm |
*edits numbered in application order; ledger ids: edit 1 = ed-c50ec327, edit 2 = ed-19fe9c13.*

**Decision audit (L2).** edit 1: arbitrator **kept** after a 3-run observation window; measured primary response n.s. (DiD -0.000) — not resolvable at this window. edit 2: arbitrator **reverted** after a 3-run observation window; measured primary response n.s. (DiD -0.028) — not resolvable at this window.

**Net-state contrast (L3, cycles 12–500).** After the last edit decision the arms differ by one persistent strategy state (the kept rotation):

| variable | ON | OFF | diff [95% CI] | verdict |
|---|---|---|---|---|
| **no-change waste rate** | 0.78 | 0.73 | +0.049 [-0.004, +0.102] | **n.s.** |
| keep rate | 0.12 | 0.18 | -0.053 [-0.098, -0.008] | harm |
| mean run value | 0.58 | 0.63 | -0.048 [-0.061, -0.035] | harm |
| mean running best | 0.67 | 0.67 | -0.002 [-0.003, -0.002] | harm |

Same contrast, **question-matched** — each question's first ≤100 runs in both arms, which removes the question-mix composition difference (running best is the within-question ratchet here):

| variable | ON | OFF | diff [95% CI] | verdict |
|---|---|---|---|---|
| **no-change waste rate** | 0.87 | 0.79 | +0.071 [+0.034, +0.107] | **harm** |
| keep rate | 0.08 | 0.14 | -0.057 [-0.088, -0.027] | harm |
| mean run value | 0.67 | 0.69 | -0.016 [-0.028, -0.004] | harm |
| mean running best | 0.74 | 0.76 | -0.019 [-0.030, -0.008] | harm |

*Caveat: the arms' question mixes differ over this tail (the ON arm left its first question after 98 runs vs 91 in the control — an early and possibly edit-caused reaction, but proposer temperature makes single-event attribution noisy), so the run-indexed table above bundles the edit with its downstream question-sequencing effects; the question-matched table is the like-for-like readout.*

*Structural note: the self-editing policy has been silent since run 12 — its only edit family (knob-priority rotation) has two distinct candidates for three knobs, both terminal in the content-addressed ledger, while its trigger (no-change share ≥ 30%) stayed satisfied (89% over the last 500 runs). Self-editing in this run = two early edits + one persistent rotation, not ongoing self-modification.*

## Key events

```
cycle 1: new best 0.3556 (baseline)
cycle 3: new best 0.4444 (model.REG)
cycle 3: self-edit 1 → kept (strategy priority rotation)
cycle 6: new best 0.6667 (model.CLIP)
cycle 9: self-edit 2 → reverted (strategy priority rotation)
cycle 19: epoch e001 → e002 (benchmark upgraded, incumbent re-baselined)
cycle 34: epoch e002 → e003 (benchmark upgraded, incumbent re-baselined)
cycle 100: question q0 → q1
cycle 139: epoch e003 → e004 (benchmark upgraded, incumbent re-baselined)
cycle 142: epoch e004 → e005 (benchmark upgraded, incumbent re-baselined)
cycle 145: epoch e005 → e006 (benchmark upgraded, incumbent re-baselined)
cycle 160: epoch e006 → e007 (benchmark upgraded, incumbent re-baselined)
cycle 169: epoch e007 → e008 (benchmark upgraded, incumbent re-baselined)
cycle 178: epoch e008 → e009 (benchmark upgraded, incumbent re-baselined)
cycle 181: epoch e009 → e010 (benchmark upgraded, incumbent re-baselined)
cycle 184: epoch e010 → e011 (benchmark upgraded, incumbent re-baselined)
cycle 187: epoch e011 → e012 (benchmark upgraded, incumbent re-baselined)
cycle 190: epoch e012 → e013 (benchmark upgraded, incumbent re-baselined)
cycle 193: epoch e013 → e014 (benchmark upgraded, incumbent re-baselined)
cycle 196: epoch e014 → e015 (benchmark upgraded, incumbent re-baselined)
cycle 214: epoch e015 → e016 (benchmark upgraded, incumbent re-baselined)
cycle 217: epoch e016 → e017 (benchmark upgraded, incumbent re-baselined)
cycle 220: epoch e017 → e018 (benchmark upgraded, incumbent re-baselined)
cycle 223: epoch e018 → e019 (benchmark upgraded, incumbent re-baselined)
cycle 226: epoch e019 → e020 (benchmark upgraded, incumbent re-baselined)
cycle 259: epoch e020 → e021 (benchmark upgraded, incumbent re-baselined)
cycle 262: epoch e021 → e022 (benchmark upgraded, incumbent re-baselined)
cycle 265: epoch e022 → e023 (benchmark upgraded, incumbent re-baselined)
cycle 268: epoch e023 → e024 (benchmark upgraded, incumbent re-baselined)
cycle 271: epoch e024 → e025 (benchmark upgraded, incumbent re-baselined)
cycle 274: epoch e025 → e026 (benchmark upgraded, incumbent re-baselined)
cycle 277: epoch e026 → e027 (benchmark upgraded, incumbent re-baselined)
cycle 280: epoch e027 → e028 (benchmark upgraded, incumbent re-baselined)
cycle 289: epoch e028 → e029 (benchmark upgraded, incumbent re-baselined)
cycle 292: epoch e029 → e030 (benchmark upgraded, incumbent re-baselined)
cycle 295: epoch e030 → e031 (benchmark upgraded, incumbent re-baselined)
cycle 301: epoch e031 → e032 (benchmark upgraded, incumbent re-baselined)
cycle 304: epoch e032 → e033 (benchmark upgraded, incumbent re-baselined)
cycle 315: question q1 → q2
cycle 337: epoch e033 → e034 (benchmark upgraded, incumbent re-baselined)
cycle 340: epoch e034 → e035 (benchmark upgraded, incumbent re-baselined)
cycle 355: epoch e035 → e036 (benchmark upgraded, incumbent re-baselined)
cycle 421: epoch e036 → e037 (benchmark upgraded, incumbent re-baselined)
cycle 424: epoch e037 → e038 (benchmark upgraded, incumbent re-baselined)
cycle 427: epoch e038 → e039 (benchmark upgraded, incumbent re-baselined)
cycle 439: epoch e039 → e040 (benchmark upgraded, incumbent re-baselined)
cycle 442: epoch e040 → e041 (benchmark upgraded, incumbent re-baselined)
cycle 445: epoch e041 → e042 (benchmark upgraded, incumbent re-baselined)
cycle 448: epoch e042 → e043 (benchmark upgraded, incumbent re-baselined)
cycle 466: question q2 → q3
cycle 469: epoch e043 → e044 (benchmark upgraded, incumbent re-baselined)
cycle 478: epoch e044 → e045 (benchmark upgraded, incumbent re-baselined)
cycle 481: epoch e045 → e046 (benchmark upgraded, incumbent re-baselined)
cycle 484: epoch e046 → e047 (benchmark upgraded, incumbent re-baselined)
cycle 487: epoch e047 → e048 (benchmark upgraded, incumbent re-baselined)
cycle 490: epoch e048 → e049 (benchmark upgraded, incumbent re-baselined)
cycle 493: epoch e049 → e050 (benchmark upgraded, incumbent re-baselined)
cycle 526: epoch e050 → e051 (benchmark upgraded, incumbent re-baselined)
cycle 538: question q3 → q4
cycle 553: epoch e051 → e052 (benchmark upgraded, incumbent re-baselined)
cycle 556: epoch e052 → e053 (benchmark upgraded, incumbent re-baselined)
cycle 559: epoch e053 → e054 (benchmark upgraded, incumbent re-baselined)
cycle 562: epoch e054 → e055 (benchmark upgraded, incumbent re-baselined)
cycle 565: epoch e055 → e056 (benchmark upgraded, incumbent re-baselined)
cycle 574: epoch e056 → e057 (benchmark upgraded, incumbent re-baselined)
cycle 589: epoch e057 → e058 (benchmark upgraded, incumbent re-baselined)
cycle 592: epoch e058 → e059 (benchmark upgraded, incumbent re-baselined)
cycle 595: epoch e059 → e060 (benchmark upgraded, incumbent re-baselined)
cycle 607: epoch e060 → e061 (benchmark upgraded, incumbent re-baselined)
cycle 610: epoch e061 → e062 (benchmark upgraded, incumbent re-baselined)
cycle 613: epoch e062 → e063 (benchmark upgraded, incumbent re-baselined)
cycle 616: epoch e063 → e064 (benchmark upgraded, incumbent re-baselined)
cycle 619: epoch e064 → e065 (benchmark upgraded, incumbent re-baselined)
cycle 622: epoch e065 → e066 (benchmark upgraded, incumbent re-baselined)
cycle 628: epoch e066 → e067 (benchmark upgraded, incumbent re-baselined)
cycle 631: epoch e067 → e068 (benchmark upgraded, incumbent re-baselined)
cycle 640: new best 0.7111 (model.LR)
cycle 667: question q4 → q5
cycle 757: epoch e068 → e069 (benchmark upgraded, incumbent re-baselined)
cycle 760: epoch e069 → e070 (benchmark upgraded, incumbent re-baselined)
cycle 763: epoch e070 → e071 (benchmark upgraded, incumbent re-baselined)
cycle 766: epoch e071 → e072 (benchmark upgraded, incumbent re-baselined)
cycle 769: epoch e072 → e073 (benchmark upgraded, incumbent re-baselined)
cycle 772: epoch e073 → e074 (benchmark upgraded, incumbent re-baselined)
cycle 775: epoch e074 → e075 (benchmark upgraded, incumbent re-baselined)
cycle 778: epoch e075 → e076 (benchmark upgraded, incumbent re-baselined)
cycle 781: epoch e076 → e077 (benchmark upgraded, incumbent re-baselined)
cycle 784: epoch e077 → e078 (benchmark upgraded, incumbent re-baselined)
cycle 786: question q5 → q6
cycle 787: new best 0.7333 (model.STEPS)
cycle 789: new best 0.7778 (model.LR)
cycle 823: epoch e078 → e079 (benchmark upgraded, incumbent re-baselined)
cycle 829: epoch e079 → e080 (benchmark upgraded, incumbent re-baselined)
cycle 829: new best 0.8889 (baseline)
cycle 831: new best 0.9333 (model.LR)
cycle 862: question q6 → q7
cycle 907: epoch e080 → e081 (benchmark upgraded, incumbent re-baselined)
cycle 910: epoch e081 → e082 (benchmark upgraded, incumbent re-baselined)
cycle 952: question q7 → q8
cycle 1031: question q8 → q9
cycle 1084: epoch e082 → e083 (benchmark upgraded, incumbent re-baselined)
cycle 1087: epoch e083 → e084 (benchmark upgraded, incumbent re-baselined)
cycle 1117: epoch e084 → e085 (benchmark upgraded, incumbent re-baselined)
cycle 1141: question q9 → q10
cycle 1216: question q10 → q11
cycle 1219: epoch e085 → e086 (benchmark upgraded, incumbent re-baselined)
cycle 1222: epoch e086 → e087 (benchmark upgraded, incumbent re-baselined)
cycle 1295: question q11 → q12
cycle 1369: question q12 → q13
cycle 1387: epoch e087 → e088 (benchmark upgraded, incumbent re-baselined)
cycle 1390: epoch e088 → e089 (benchmark upgraded, incumbent re-baselined)
cycle 1393: epoch e089 → e090 (benchmark upgraded, incumbent re-baselined)
cycle 1420: epoch e090 → e091 (benchmark upgraded, incumbent re-baselined)
cycle 1423: epoch e091 → e092 (benchmark upgraded, incumbent re-baselined)
cycle 1426: epoch e092 → e093 (benchmark upgraded, incumbent re-baselined)
cycle 1429: epoch e093 → e094 (benchmark upgraded, incumbent re-baselined)
cycle 1432: epoch e094 → e095 (benchmark upgraded, incumbent re-baselined)
cycle 1442: question q13 → q14
cycle 1453: epoch e095 → e096 (benchmark upgraded, incumbent re-baselined)
cycle 1492: question q14 → q15
cycle 1544: question q15 → q16
cycle 1597: question q16 → q17
cycle 1666: question q17 → q18
cycle 1738: question q18 → q19
cycle 1787: question q19 → q20
cycle 1837: question q20 → q21
cycle 1887: question q21 → q22
cycle 1938: question q22 → q23
cycle 2012: question q23 → q24
cycle 2062: question q24 → q25
cycle 2114: question q25 → q26
cycle 2166: question q26 → q27
cycle 2220: question q27 → q28
cycle 2271: question q28 → q29
cycle 2323: question q29 → q30
cycle 2375: question q30 → q31
cycle 2449: question q31 → q32
cycle 2515: question q32 → q33
cycle 2564: question q33 → q34
cycle 2614: question q34 → q35
cycle 2667: question q35 → q36
cycle 2721: question q36 → q37
cycle 2775: question q37 → q38
cycle 2845: question q38 → q39
cycle 2919: question q39 → q40
cycle 2970: question q40 → q41
cycle 2995: epoch e096 → e097 (benchmark upgraded, incumbent re-baselined)
cycle 2998: epoch e097 → e098 (benchmark upgraded, incumbent re-baselined)
cycle 3004: epoch e098 → e099 (benchmark upgraded, incumbent re-baselined)
cycle 3007: epoch e099 → e100 (benchmark upgraded, incumbent re-baselined)
cycle 3013: new best 0.9556 (model.REG)
cycle 3022: epoch e100 → e101 (benchmark upgraded, incumbent re-baselined)
cycle 3037: epoch e101 → e102 (benchmark upgraded, incumbent re-baselined)
cycle 3040: epoch e102 → e103 (benchmark upgraded, incumbent re-baselined)
cycle 3043: epoch e103 → e104 (benchmark upgraded, incumbent re-baselined)
cycle 3049: epoch e104 → e105 (benchmark upgraded, incumbent re-baselined)
cycle 3062: question q41 → q42
cycle 3079: epoch e105 → e106 (benchmark upgraded, incumbent re-baselined)
cycle 3088: epoch e106 → e107 (benchmark upgraded, incumbent re-baselined)
cycle 3091: epoch e107 → e108 (benchmark upgraded, incumbent re-baselined)
cycle 3153: question q42 → q43
cycle 3199: question q43 → q44
cycle 3241: question q44 → q45
cycle 3304: question q45 → q46
cycle 3370: epoch e108 → e109 (benchmark upgraded, incumbent re-baselined)
cycle 3373: epoch e109 → e110 (benchmark upgraded, incumbent re-baselined)
cycle 3376: epoch e110 → e111 (benchmark upgraded, incumbent re-baselined)
cycle 3379: epoch e111 → e112 (benchmark upgraded, incumbent re-baselined)
cycle 3382: epoch e112 → e113 (benchmark upgraded, incumbent re-baselined)
cycle 3382: new best 0.9778 (baseline)
cycle 3385: epoch e113 → e114 (benchmark upgraded, incumbent re-baselined)
cycle 3387: question q46 → q47
cycle 3433: epoch e114 → e115 (benchmark upgraded, incumbent re-baselined)
cycle 3454: epoch e115 → e116 (benchmark upgraded, incumbent re-baselined)
cycle 3463: epoch e116 → e117 (benchmark upgraded, incumbent re-baselined)
cycle 3466: epoch e117 → e118 (benchmark upgraded, incumbent re-baselined)
cycle 3475: question q47 → q48
cycle 3478: epoch e118 → e119 (benchmark upgraded, incumbent re-baselined)
cycle 3490: epoch e119 → e120 (benchmark upgraded, incumbent re-baselined)
cycle 3493: epoch e120 → e121 (benchmark upgraded, incumbent re-baselined)
cycle 3496: epoch e121 → e122 (benchmark upgraded, incumbent re-baselined)
cycle 3499: epoch e122 → e123 (benchmark upgraded, incumbent re-baselined)
cycle 3502: epoch e123 → e124 (benchmark upgraded, incumbent re-baselined)
cycle 3505: epoch e124 → e125 (benchmark upgraded, incumbent re-baselined)
cycle 3532: epoch e125 → e126 (benchmark upgraded, incumbent re-baselined)
cycle 3535: epoch e126 → e127 (benchmark upgraded, incumbent re-baselined)
cycle 3538: epoch e127 → e128 (benchmark upgraded, incumbent re-baselined)
cycle 3541: epoch e128 → e129 (benchmark upgraded, incumbent re-baselined)
cycle 3544: epoch e129 → e130 (benchmark upgraded, incumbent re-baselined)
cycle 3547: epoch e130 → e131 (benchmark upgraded, incumbent re-baselined)
cycle 3562: question q48 → q49
cycle 3565: epoch e131 → e132 (benchmark upgraded, incumbent re-baselined)
```

## Final knowledge state

| question | status | saturation | best |
|---|---|---|---|
| iris-ood-q0 | partially_answered | 0.33 | 0.6444 |
| iris-ood-q1 | partially_answered | 0.00 | 0.5778 |
| iris-ood-q2 | partially_answered | 1.00 | 0.6667 |
| iris-ood-q3 | partially_answered | 1.00 | 0.6444 |
| iris-ood-q4 | partially_answered | 1.00 | 0.7111 |
| iris-ood-q5 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q6 | partially_answered | 0.18 | 0.9333 |
| iris-ood-q7 | partially_answered | 1.00 | 0.8444 |
| iris-ood-q8 | partially_answered | 0.00 | 0.8444 |
| iris-ood-q9 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q10 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q11 | partially_answered | 0.50 | 0.6000 |
| iris-ood-q12 | partially_answered | 0.00 | 0.6000 |
| iris-ood-q13 | partially_answered | 0.00 | 0.7556 |
| iris-ood-q14 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q15 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q16 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q17 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q18 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q19 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q20 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q21 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q22 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q23 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q24 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q25 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q26 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q27 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q28 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q29 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q30 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q31 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q32 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q33 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q34 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q35 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q36 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q37 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q38 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q39 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q40 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q41 | partially_answered | 0.55 | 0.9111 |
| iris-ood-q42 | partially_answered | 0.60 | 0.6222 |
| iris-ood-q43 | partially_answered | 1.00 | 0.6222 |
| iris-ood-q44 | partially_answered | 1.00 | 0.6222 |
| iris-ood-q45 | partially_answered | 1.00 | 0.6222 |
| iris-ood-q46 | partially_answered | 0.00 | 0.9333 |
| iris-ood-q47 | partially_answered | 0.00 | 0.5111 |
| iris-ood-q48 | partially_answered | 0.00 | 0.6000 |
| iris-ood-q49 | open | 0.00 | 0.8667 |

| hypothesis | status | confidence | support/refute |
|---|---|---|---|
| iris-ood-q0-h0 | refuted | 0.22 | 1/2 |
| iris-ood-q0-h1 | refuted | 0.45 | 1/2 |
| iris-ood-q0-h2 | refuted | 0.00 | 0/2 |
| iris-ood-q0-h3 | refuted | 0.00 | 0/2 |
| iris-ood-q0-h4 | refuted | 0.00 | 0/2 |
| iris-ood-q0-h5 | refuted | 0.00 | 0/2 |
| iris-ood-q0-h6 | refuted | 0.00 | 0/2 |
| iris-ood-q0-h7 | refuted | 0.00 | 0/2 |
| iris-ood-q0-h8 | refuted | 0.00 | 0/2 |
| iris-ood-q0-h9 | refuted | 0.00 | 0/2 |
| iris-ood-q0-h10 | refuted | 0.00 | 0/2 |
| iris-ood-q0-h11 | refuted | 0.00 | 0/2 |
| … (1645 more hypotheses) | | | |

## Strategy artifact (final state)

```python
"""Proposer strategy — the SELF-EDITABLE surface.

Everything between the sentinel markers may be rewritten by the
self-improvement loop (guard-checked: statement shape locked, scorer
untouchable). The proposer re-reads this file every cycle.
"""
# <<<SELF-IMPROVE-MUTABLE
# Knob exploration priority + per-knob candidate width. Values are plain
# python literals. PRIORITY entries that no longer match the task's knobs
# are ignored; remaining knobs follow file order.
PRIORITY = ['LR', 'CLIP', 'MOMENTUM']
EXPLORE_WIDTH = 6
# SELF-IMPROVE-MUTABLE>>>
```

---

*Generated by `scripts/report.py` (git `215e86d`) from sessions: main=sessions/naive-seed123; control=sessions/noop-seed123.*
