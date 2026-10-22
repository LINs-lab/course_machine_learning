# iris-ood — self-editing ablation report

- cycles: **3042** · seed: **42** · epochs: **235** · questions: **29** · hypotheses: **1342** · evidence records: **3042**
- arms: **self-editing ON** (naive priority-rotation improver) vs **self-editing OFF** (noop improver)
- baseline ood_accuracy: **0.4444** → final best (kept): **1.0000** (improvement +0.5556)
- keep-rate: **9.76%** · self-edits kept/reverted: **1/1**

![trajectory](figures/trajectory.pdf)

*Trajectory of ood_accuracy on the wall-clock axis (one single-variable experiment per cycle; seed 42). Purple line + stars: running best (kept runs); purple dots: keeps; gray crosses: discards; red triangles: fails. Background bands are research questions; dashed verticals are epoch boundaries (omitted and counted instead beyond 24); the dashed horizontal is the initial baseline. Baseline re-measurements appear at every epoch upgrade and question switch — the incumbent solution persists, its value re-baselines.*

![keeprate](figures/keeprate.pdf)

*Rolling keep rate (trailing window scaled to the run count) over wall-clock time. Self-edit decisions are marked at their application time: dotted vertical, star = kept, cross = reverted. The overall keep rate is annotated at the lower right.*

![exploration](figures/exploration.pdf)

*Rolling share of each knob over wall-clock time (trailing window 200 runs) — how the fast loop's exploration mix drifts. Legend labels carry each knob's trial count and keep rate (the per-knob payoff). Baseline re-measurements are bookkeeping at question/epoch switches and are excluded from the share.*

## Self-editing ablation

![compare](figures/compare.pdf)

*Running best on the wall-clock axis: self-editing ON (purple, solid, stars) vs self-editing OFF (red, dashed). Legend labels carry each series' final value.*

| | self-editing ON | self-editing OFF |
|---|---|---|
| final best ood_accuracy | **1.0000** | 0.9778 |
| keep-rate | 9.76% | 16.25% |
| mean running best (AUC/cycle) | **0.8700** | 0.7596 |
| cycles to match OFF final best | **2170** | — |
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
| edit 1 |  | mean run value | 0.45 (3) | 0.58 (6) | +0.130 | +0.000 [-0.126, +0.126] | n.s. |
| edit 1 |  | mean running best | 0.47 (3) | 0.60 (6) | +0.133 | +0.000 [-0.089, +0.089] | n.s. |
| **edit 2** | reverted (3-run window) | **no-change waste rate** | 1.00 (3) | 0.80 (60) | -0.200 | -0.067 [-0.200, +0.067] | **n.s.** |
| edit 2 |  | keep rate | 0.00 (3) | 0.07 (60) | +0.067 | -0.033 [-0.133, +0.067] | n.s. |
| edit 2 |  | mean run value | 0.64 (3) | 0.58 (60) | -0.065 | -0.002 [-0.036, +0.033] | n.s. |
| edit 2 |  | mean running best | 0.64 (3) | 0.66 (60) | +0.019 | +0.000 [-0.003, +0.003] | n.s. |
*edits numbered in application order; ledger ids: edit 1 = ed-c50ec327, edit 2 = ed-19fe9c13.*

**Decision audit (L2).** edit 1: arbitrator **kept** after a 3-run observation window; measured primary response n.s. (DiD -0.000) — not resolvable at this window. edit 2: arbitrator **reverted** after a 3-run observation window; measured primary response n.s. (DiD -0.067) — not resolvable at this window.

**Net-state contrast (L3, cycles 12–500).** After the last edit decision the arms differ by one persistent strategy state (the kept rotation):

| variable | ON | OFF | diff [95% CI] | verdict |
|---|---|---|---|---|
| **no-change waste rate** | 0.80 | 0.85 | -0.055 [-0.102, -0.008] | **gain** |
| keep rate | 0.18 | 0.15 | +0.033 [-0.014, +0.078] | n.s. |
| mean run value | 0.65 | 0.62 | +0.021 [+0.011, +0.031] | gain |
| mean running best | 0.70 | 0.67 | +0.030 [+0.028, +0.032] | gain |

Same contrast, **question-matched** — each question's first ≤100 runs in both arms, which removes the question-mix composition difference (running best is the within-question ratchet here):

| variable | ON | OFF | diff [95% CI] | verdict |
|---|---|---|---|---|
| **no-change waste rate** | 0.91 | 0.84 | +0.070 [+0.052, +0.090] | **harm** |
| keep rate | 0.07 | 0.14 | -0.071 [-0.088, -0.053] | harm |
| mean run value | 0.69 | 0.64 | +0.049 [+0.042, +0.055] | gain |
| mean running best | 0.78 | 0.68 | +0.091 [+0.087, +0.096] | gain |

*Caveat: the arms' question mixes differ over this tail (the ON arm left its first question after 69 runs vs 96 in the control — an early and possibly edit-caused reaction, but proposer temperature makes single-event attribution noisy), so the run-indexed table above bundles the edit with its downstream question-sequencing effects; the question-matched table is the like-for-like readout.*

*Structural note: the self-editing policy has been silent since run 12 — its only edit family (knob-priority rotation) has two distinct candidates for three knobs, both terminal in the content-addressed ledger, while its trigger (no-change share ≥ 30%) stayed satisfied (94% over the last 500 runs). Self-editing in this run = two early edits + one persistent rotation, not ongoing self-modification.*

## Key events

```
cycle 1: new best 0.4444 (baseline)
cycle 3: new best 0.5111 (model.REG)
cycle 3: self-edit 1 → kept (strategy priority rotation)
cycle 6: new best 0.6444 (model.CLIP)
cycle 9: self-edit 2 → reverted (strategy priority rotation)
cycle 19: epoch e001 → e002 (benchmark upgraded, incumbent re-baselined)
cycle 19: new best 0.6667 (baseline)
cycle 22: epoch e002 → e003 (benchmark upgraded, incumbent re-baselined)
cycle 70: question q0 → q1
cycle 79: new best 0.6889 (model.POWER)
cycle 142: epoch e003 → e004 (benchmark upgraded, incumbent re-baselined)
cycle 145: epoch e004 → e005 (benchmark upgraded, incumbent re-baselined)
cycle 148: epoch e005 → e006 (benchmark upgraded, incumbent re-baselined)
cycle 151: epoch e006 → e007 (benchmark upgraded, incumbent re-baselined)
cycle 154: epoch e007 → e008 (benchmark upgraded, incumbent re-baselined)
cycle 163: epoch e008 → e009 (benchmark upgraded, incumbent re-baselined)
cycle 166: epoch e009 → e010 (benchmark upgraded, incumbent re-baselined)
cycle 169: epoch e010 → e011 (benchmark upgraded, incumbent re-baselined)
cycle 181: epoch e011 → e012 (benchmark upgraded, incumbent re-baselined)
cycle 184: epoch e012 → e013 (benchmark upgraded, incumbent re-baselined)
cycle 199: epoch e013 → e014 (benchmark upgraded, incumbent re-baselined)
cycle 202: epoch e014 → e015 (benchmark upgraded, incumbent re-baselined)
cycle 205: epoch e015 → e016 (benchmark upgraded, incumbent re-baselined)
cycle 208: epoch e016 → e017 (benchmark upgraded, incumbent re-baselined)
cycle 211: epoch e017 → e018 (benchmark upgraded, incumbent re-baselined)
cycle 220: epoch e018 → e019 (benchmark upgraded, incumbent re-baselined)
cycle 223: epoch e019 → e020 (benchmark upgraded, incumbent re-baselined)
cycle 229: epoch e020 → e021 (benchmark upgraded, incumbent re-baselined)
cycle 235: epoch e021 → e022 (benchmark upgraded, incumbent re-baselined)
cycle 238: epoch e022 → e023 (benchmark upgraded, incumbent re-baselined)
cycle 241: epoch e023 → e024 (benchmark upgraded, incumbent re-baselined)
cycle 244: epoch e024 → e025 (benchmark upgraded, incumbent re-baselined)
cycle 247: epoch e025 → e026 (benchmark upgraded, incumbent re-baselined)
cycle 250: epoch e026 → e027 (benchmark upgraded, incumbent re-baselined)
cycle 253: epoch e027 → e028 (benchmark upgraded, incumbent re-baselined)
cycle 256: epoch e028 → e029 (benchmark upgraded, incumbent re-baselined)
cycle 259: epoch e029 → e030 (benchmark upgraded, incumbent re-baselined)
cycle 262: epoch e030 → e031 (benchmark upgraded, incumbent re-baselined)
cycle 271: epoch e031 → e032 (benchmark upgraded, incumbent re-baselined)
cycle 274: epoch e032 → e033 (benchmark upgraded, incumbent re-baselined)
cycle 277: epoch e033 → e034 (benchmark upgraded, incumbent re-baselined)
cycle 280: epoch e034 → e035 (benchmark upgraded, incumbent re-baselined)
cycle 283: epoch e035 → e036 (benchmark upgraded, incumbent re-baselined)
cycle 286: epoch e036 → e037 (benchmark upgraded, incumbent re-baselined)
cycle 289: epoch e037 → e038 (benchmark upgraded, incumbent re-baselined)
cycle 292: epoch e038 → e039 (benchmark upgraded, incumbent re-baselined)
cycle 295: epoch e039 → e040 (benchmark upgraded, incumbent re-baselined)
cycle 298: epoch e040 → e041 (benchmark upgraded, incumbent re-baselined)
cycle 301: epoch e041 → e042 (benchmark upgraded, incumbent re-baselined)
cycle 304: epoch e042 → e043 (benchmark upgraded, incumbent re-baselined)
cycle 307: epoch e043 → e044 (benchmark upgraded, incumbent re-baselined)
cycle 310: epoch e044 → e045 (benchmark upgraded, incumbent re-baselined)
cycle 313: epoch e045 → e046 (benchmark upgraded, incumbent re-baselined)
cycle 316: epoch e046 → e047 (benchmark upgraded, incumbent re-baselined)
cycle 319: epoch e047 → e048 (benchmark upgraded, incumbent re-baselined)
cycle 322: epoch e048 → e049 (benchmark upgraded, incumbent re-baselined)
cycle 325: epoch e049 → e050 (benchmark upgraded, incumbent re-baselined)
cycle 328: epoch e050 → e051 (benchmark upgraded, incumbent re-baselined)
cycle 331: epoch e051 → e052 (benchmark upgraded, incumbent re-baselined)
cycle 334: epoch e052 → e053 (benchmark upgraded, incumbent re-baselined)
cycle 337: epoch e053 → e054 (benchmark upgraded, incumbent re-baselined)
cycle 340: epoch e054 → e055 (benchmark upgraded, incumbent re-baselined)
cycle 343: epoch e055 → e056 (benchmark upgraded, incumbent re-baselined)
cycle 346: epoch e056 → e057 (benchmark upgraded, incumbent re-baselined)
cycle 349: epoch e057 → e058 (benchmark upgraded, incumbent re-baselined)
cycle 352: epoch e058 → e059 (benchmark upgraded, incumbent re-baselined)
cycle 355: epoch e059 → e060 (benchmark upgraded, incumbent re-baselined)
cycle 358: epoch e060 → e061 (benchmark upgraded, incumbent re-baselined)
cycle 361: epoch e061 → e062 (benchmark upgraded, incumbent re-baselined)
cycle 364: epoch e062 → e063 (benchmark upgraded, incumbent re-baselined)
cycle 367: epoch e063 → e064 (benchmark upgraded, incumbent re-baselined)
cycle 370: epoch e064 → e065 (benchmark upgraded, incumbent re-baselined)
cycle 373: epoch e065 → e066 (benchmark upgraded, incumbent re-baselined)
cycle 375: question q1 → q2
cycle 391: epoch e066 → e067 (benchmark upgraded, incumbent re-baselined)
cycle 394: epoch e067 → e068 (benchmark upgraded, incumbent re-baselined)
cycle 397: epoch e068 → e069 (benchmark upgraded, incumbent re-baselined)
cycle 400: epoch e069 → e070 (benchmark upgraded, incumbent re-baselined)
cycle 403: epoch e070 → e071 (benchmark upgraded, incumbent re-baselined)
cycle 406: epoch e071 → e072 (benchmark upgraded, incumbent re-baselined)
cycle 409: epoch e072 → e073 (benchmark upgraded, incumbent re-baselined)
cycle 412: epoch e073 → e074 (benchmark upgraded, incumbent re-baselined)
cycle 415: epoch e074 → e075 (benchmark upgraded, incumbent re-baselined)
cycle 418: epoch e075 → e076 (benchmark upgraded, incumbent re-baselined)
cycle 421: epoch e076 → e077 (benchmark upgraded, incumbent re-baselined)
cycle 423: new best 0.7556 (model.LR)
cycle 455: question q2 → q3
cycle 508: epoch e077 → e078 (benchmark upgraded, incumbent re-baselined)
cycle 511: epoch e078 → e079 (benchmark upgraded, incumbent re-baselined)
cycle 514: epoch e079 → e080 (benchmark upgraded, incumbent re-baselined)
cycle 517: epoch e080 → e081 (benchmark upgraded, incumbent re-baselined)
cycle 520: epoch e081 → e082 (benchmark upgraded, incumbent re-baselined)
cycle 523: epoch e082 → e083 (benchmark upgraded, incumbent re-baselined)
cycle 532: epoch e083 → e084 (benchmark upgraded, incumbent re-baselined)
cycle 535: epoch e084 → e085 (benchmark upgraded, incumbent re-baselined)
cycle 538: epoch e085 → e086 (benchmark upgraded, incumbent re-baselined)
cycle 541: epoch e086 → e087 (benchmark upgraded, incumbent re-baselined)
cycle 544: epoch e087 → e088 (benchmark upgraded, incumbent re-baselined)
cycle 547: epoch e088 → e089 (benchmark upgraded, incumbent re-baselined)
cycle 550: epoch e089 → e090 (benchmark upgraded, incumbent re-baselined)
cycle 553: epoch e090 → e091 (benchmark upgraded, incumbent re-baselined)
cycle 568: epoch e091 → e092 (benchmark upgraded, incumbent re-baselined)
cycle 571: epoch e092 → e093 (benchmark upgraded, incumbent re-baselined)
cycle 574: epoch e093 → e094 (benchmark upgraded, incumbent re-baselined)
cycle 577: question q3 → q4
cycle 580: epoch e094 → e095 (benchmark upgraded, incumbent re-baselined)
cycle 583: epoch e095 → e096 (benchmark upgraded, incumbent re-baselined)
cycle 645: question q4 → q5
cycle 736: epoch e096 → e097 (benchmark upgraded, incumbent re-baselined)
cycle 769: question q5 → q6
cycle 772: epoch e097 → e098 (benchmark upgraded, incumbent re-baselined)
cycle 775: epoch e098 → e099 (benchmark upgraded, incumbent re-baselined)
cycle 823: question q6 → q7
cycle 856: epoch e099 → e100 (benchmark upgraded, incumbent re-baselined)
cycle 859: epoch e100 → e101 (benchmark upgraded, incumbent re-baselined)
cycle 862: epoch e101 → e102 (benchmark upgraded, incumbent re-baselined)
cycle 862: new best 0.8222 (baseline)
cycle 865: epoch e102 → e103 (benchmark upgraded, incumbent re-baselined)
cycle 880: epoch e103 → e104 (benchmark upgraded, incumbent re-baselined)
cycle 891: new best 0.8444 (model.CLIP)
cycle 913: question q7 → q8
cycle 1009: question q8 → q9
cycle 1104: question q9 → q10
cycle 1135: epoch e104 → e105 (benchmark upgraded, incumbent re-baselined)
cycle 1138: epoch e105 → e106 (benchmark upgraded, incumbent re-baselined)
cycle 1168: epoch e106 → e107 (benchmark upgraded, incumbent re-baselined)
cycle 1171: epoch e107 → e108 (benchmark upgraded, incumbent re-baselined)
cycle 1171: new best 0.8667 (baseline)
cycle 1189: epoch e108 → e109 (benchmark upgraded, incumbent re-baselined)
cycle 1276: epoch e109 → e110 (benchmark upgraded, incumbent re-baselined)
cycle 1279: epoch e110 → e111 (benchmark upgraded, incumbent re-baselined)
cycle 1282: epoch e111 → e112 (benchmark upgraded, incumbent re-baselined)
cycle 1285: epoch e112 → e113 (benchmark upgraded, incumbent re-baselined)
cycle 1288: epoch e113 → e114 (benchmark upgraded, incumbent re-baselined)
cycle 1291: epoch e114 → e115 (benchmark upgraded, incumbent re-baselined)
cycle 1294: epoch e115 → e116 (benchmark upgraded, incumbent re-baselined)
cycle 1316: question q10 → q11
cycle 1393: epoch e116 → e117 (benchmark upgraded, incumbent re-baselined)
cycle 1396: epoch e117 → e118 (benchmark upgraded, incumbent re-baselined)
cycle 1399: epoch e118 → e119 (benchmark upgraded, incumbent re-baselined)
cycle 1402: epoch e119 → e120 (benchmark upgraded, incumbent re-baselined)
cycle 1405: epoch e120 → e121 (benchmark upgraded, incumbent re-baselined)
cycle 1411: epoch e121 → e122 (benchmark upgraded, incumbent re-baselined)
cycle 1414: epoch e122 → e123 (benchmark upgraded, incumbent re-baselined)
cycle 1417: epoch e123 → e124 (benchmark upgraded, incumbent re-baselined)
cycle 1438: epoch e124 → e125 (benchmark upgraded, incumbent re-baselined)
cycle 1441: question q11 → q12
cycle 1543: epoch e125 → e126 (benchmark upgraded, incumbent re-baselined)
cycle 1546: epoch e126 → e127 (benchmark upgraded, incumbent re-baselined)
cycle 1549: epoch e127 → e128 (benchmark upgraded, incumbent re-baselined)
cycle 1549: new best 0.8889 (baseline)
cycle 1552: epoch e128 → e129 (benchmark upgraded, incumbent re-baselined)
cycle 1555: epoch e129 → e130 (benchmark upgraded, incumbent re-baselined)
cycle 1569: question q12 → q13
cycle 1573: epoch e130 → e131 (benchmark upgraded, incumbent re-baselined)
cycle 1600: epoch e131 → e132 (benchmark upgraded, incumbent re-baselined)
cycle 1615: epoch e132 → e133 (benchmark upgraded, incumbent re-baselined)
cycle 1618: epoch e133 → e134 (benchmark upgraded, incumbent re-baselined)
cycle 1621: epoch e134 → e135 (benchmark upgraded, incumbent re-baselined)
cycle 1624: epoch e135 → e136 (benchmark upgraded, incumbent re-baselined)
cycle 1627: epoch e136 → e137 (benchmark upgraded, incumbent re-baselined)
cycle 1630: epoch e137 → e138 (benchmark upgraded, incumbent re-baselined)
cycle 1633: epoch e138 → e139 (benchmark upgraded, incumbent re-baselined)
cycle 1636: epoch e139 → e140 (benchmark upgraded, incumbent re-baselined)
cycle 1639: epoch e140 → e141 (benchmark upgraded, incumbent re-baselined)
cycle 1642: epoch e141 → e142 (benchmark upgraded, incumbent re-baselined)
cycle 1645: epoch e142 → e143 (benchmark upgraded, incumbent re-baselined)
cycle 1669: epoch e143 → e144 (benchmark upgraded, incumbent re-baselined)
cycle 1681: epoch e144 → e145 (benchmark upgraded, incumbent re-baselined)
cycle 1684: epoch e145 → e146 (benchmark upgraded, incumbent re-baselined)
cycle 1687: epoch e146 → e147 (benchmark upgraded, incumbent re-baselined)
cycle 1690: epoch e147 → e148 (benchmark upgraded, incumbent re-baselined)
cycle 1693: epoch e148 → e149 (benchmark upgraded, incumbent re-baselined)
cycle 1696: epoch e149 → e150 (benchmark upgraded, incumbent re-baselined)
cycle 1699: epoch e150 → e151 (benchmark upgraded, incumbent re-baselined)
cycle 1708: epoch e151 → e152 (benchmark upgraded, incumbent re-baselined)
cycle 1711: epoch e152 → e153 (benchmark upgraded, incumbent re-baselined)
cycle 1714: epoch e153 → e154 (benchmark upgraded, incumbent re-baselined)
cycle 1717: epoch e154 → e155 (benchmark upgraded, incumbent re-baselined)
cycle 1720: epoch e155 → e156 (benchmark upgraded, incumbent re-baselined)
cycle 1723: epoch e156 → e157 (benchmark upgraded, incumbent re-baselined)
cycle 1726: question q13 → q14
cycle 1729: epoch e157 → e158 (benchmark upgraded, incumbent re-baselined)
cycle 1732: epoch e158 → e159 (benchmark upgraded, incumbent re-baselined)
cycle 1735: epoch e159 → e160 (benchmark upgraded, incumbent re-baselined)
cycle 1738: epoch e160 → e161 (benchmark upgraded, incumbent re-baselined)
cycle 1738: new best 0.9333 (baseline)
cycle 1741: epoch e161 → e162 (benchmark upgraded, incumbent re-baselined)
cycle 1744: epoch e162 → e163 (benchmark upgraded, incumbent re-baselined)
cycle 1753: epoch e163 → e164 (benchmark upgraded, incumbent re-baselined)
cycle 1756: epoch e164 → e165 (benchmark upgraded, incumbent re-baselined)
cycle 1759: epoch e165 → e166 (benchmark upgraded, incumbent re-baselined)
cycle 1762: epoch e166 → e167 (benchmark upgraded, incumbent re-baselined)
cycle 1765: epoch e167 → e168 (benchmark upgraded, incumbent re-baselined)
cycle 1783: epoch e168 → e169 (benchmark upgraded, incumbent re-baselined)
cycle 1786: epoch e169 → e170 (benchmark upgraded, incumbent re-baselined)
cycle 1789: epoch e170 → e171 (benchmark upgraded, incumbent re-baselined)
cycle 1804: epoch e171 → e172 (benchmark upgraded, incumbent re-baselined)
cycle 1807: epoch e172 → e173 (benchmark upgraded, incumbent re-baselined)
cycle 1810: epoch e173 → e174 (benchmark upgraded, incumbent re-baselined)
cycle 1813: epoch e174 → e175 (benchmark upgraded, incumbent re-baselined)
cycle 1847: question q14 → q15
cycle 1948: question q15 → q16
cycle 1996: epoch e175 → e176 (benchmark upgraded, incumbent re-baselined)
cycle 1999: epoch e176 → e177 (benchmark upgraded, incumbent re-baselined)
cycle 2044: epoch e177 → e178 (benchmark upgraded, incumbent re-baselined)
cycle 2047: epoch e178 → e179 (benchmark upgraded, incumbent re-baselined)
cycle 2050: epoch e179 → e180 (benchmark upgraded, incumbent re-baselined)
cycle 2053: epoch e180 → e181 (benchmark upgraded, incumbent re-baselined)
cycle 2056: epoch e181 → e182 (benchmark upgraded, incumbent re-baselined)
cycle 2058: question q16 → q17
cycle 2071: epoch e182 → e183 (benchmark upgraded, incumbent re-baselined)
cycle 2074: epoch e183 → e184 (benchmark upgraded, incumbent re-baselined)
cycle 2143: epoch e184 → e185 (benchmark upgraded, incumbent re-baselined)
cycle 2146: epoch e185 → e186 (benchmark upgraded, incumbent re-baselined)
cycle 2149: epoch e186 → e187 (benchmark upgraded, incumbent re-baselined)
cycle 2152: epoch e187 → e188 (benchmark upgraded, incumbent re-baselined)
cycle 2155: epoch e188 → e189 (benchmark upgraded, incumbent re-baselined)
cycle 2158: epoch e189 → e190 (benchmark upgraded, incumbent re-baselined)
cycle 2161: epoch e190 → e191 (benchmark upgraded, incumbent re-baselined)
cycle 2164: epoch e191 → e192 (benchmark upgraded, incumbent re-baselined)
cycle 2167: epoch e192 → e193 (benchmark upgraded, incumbent re-baselined)
cycle 2170: epoch e193 → e194 (benchmark upgraded, incumbent re-baselined)
cycle 2170: new best 0.9778 (baseline)
cycle 2173: epoch e194 → e195 (benchmark upgraded, incumbent re-baselined)
cycle 2176: epoch e195 → e196 (benchmark upgraded, incumbent re-baselined)
cycle 2179: epoch e196 → e197 (benchmark upgraded, incumbent re-baselined)
cycle 2182: epoch e197 → e198 (benchmark upgraded, incumbent re-baselined)
cycle 2185: question q17 → q18
cycle 2188: epoch e198 → e199 (benchmark upgraded, incumbent re-baselined)
cycle 2209: question q18 → q19
cycle 2236: epoch e199 → e200 (benchmark upgraded, incumbent re-baselined)
cycle 2239: epoch e200 → e201 (benchmark upgraded, incumbent re-baselined)
cycle 2278: epoch e201 → e202 (benchmark upgraded, incumbent re-baselined)
cycle 2302: epoch e202 → e203 (benchmark upgraded, incumbent re-baselined)
cycle 2305: epoch e203 → e204 (benchmark upgraded, incumbent re-baselined)
cycle 2308: epoch e204 → e205 (benchmark upgraded, incumbent re-baselined)
cycle 2326: question q19 → q20
cycle 2392: question q20 → q21
cycle 2464: epoch e205 → e206 (benchmark upgraded, incumbent re-baselined)
cycle 2467: epoch e206 → e207 (benchmark upgraded, incumbent re-baselined)
cycle 2470: epoch e207 → e208 (benchmark upgraded, incumbent re-baselined)
cycle 2473: epoch e208 → e209 (benchmark upgraded, incumbent re-baselined)
cycle 2476: epoch e209 → e210 (benchmark upgraded, incumbent re-baselined)
cycle 2479: epoch e210 → e211 (benchmark upgraded, incumbent re-baselined)
cycle 2482: epoch e211 → e212 (benchmark upgraded, incumbent re-baselined)
cycle 2485: epoch e212 → e213 (benchmark upgraded, incumbent re-baselined)
cycle 2488: epoch e213 → e214 (benchmark upgraded, incumbent re-baselined)
cycle 2491: epoch e214 → e215 (benchmark upgraded, incumbent re-baselined)
cycle 2491: new best 1.0000 (baseline)
cycle 2494: epoch e215 → e216 (benchmark upgraded, incumbent re-baselined)
cycle 2497: epoch e216 → e217 (benchmark upgraded, incumbent re-baselined)
cycle 2503: question q21 → q22
cycle 2583: question q22 → q23
cycle 2668: question q23 → q24
cycle 2761: question q24 → q25
cycle 2856: question q25 → q26
cycle 2917: epoch e217 → e218 (benchmark upgraded, incumbent re-baselined)
cycle 2920: epoch e218 → e219 (benchmark upgraded, incumbent re-baselined)
cycle 2923: epoch e219 → e220 (benchmark upgraded, incumbent re-baselined)
cycle 2926: epoch e220 → e221 (benchmark upgraded, incumbent re-baselined)
cycle 2929: epoch e221 → e222 (benchmark upgraded, incumbent re-baselined)
cycle 2932: epoch e222 → e223 (benchmark upgraded, incumbent re-baselined)
cycle 2934: question q26 → q27
cycle 2947: epoch e223 → e224 (benchmark upgraded, incumbent re-baselined)
cycle 2971: epoch e224 → e225 (benchmark upgraded, incumbent re-baselined)
cycle 2986: epoch e225 → e226 (benchmark upgraded, incumbent re-baselined)
cycle 2989: epoch e226 → e227 (benchmark upgraded, incumbent re-baselined)
cycle 2992: epoch e227 → e228 (benchmark upgraded, incumbent re-baselined)
cycle 2995: epoch e228 → e229 (benchmark upgraded, incumbent re-baselined)
cycle 3009: question q27 → q28
cycle 3043: epoch e229 → e230 (benchmark upgraded, incumbent re-baselined)
cycle 3046: epoch e230 → e231 (benchmark upgraded, incumbent re-baselined)
cycle 3049: epoch e231 → e232 (benchmark upgraded, incumbent re-baselined)
cycle 3052: epoch e232 → e233 (benchmark upgraded, incumbent re-baselined)
cycle 3055: epoch e233 → e234 (benchmark upgraded, incumbent re-baselined)
```

## Final knowledge state

| question | status | saturation | best |
|---|---|---|---|
| iris-ood-q0 | partially_answered | 1.00 | 0.6444 |
| iris-ood-q1 | partially_answered | 0.00 | 0.6222 |
| iris-ood-q2 | partially_answered | 0.50 | 0.7556 |
| iris-ood-q3 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q4 | partially_answered | 0.40 | 0.6222 |
| iris-ood-q5 | partially_answered | 1.00 | 0.7556 |
| iris-ood-q6 | partially_answered | 1.00 | 0.7556 |
| iris-ood-q7 | partially_answered | 1.00 | 0.8444 |
| iris-ood-q8 | partially_answered | 0.00 | 0.8444 |
| iris-ood-q9 | partially_answered | 0.00 | 0.8444 |
| iris-ood-q10 | partially_answered | 0.60 | 0.7111 |
| iris-ood-q11 | partially_answered | 0.00 | 0.7556 |
| iris-ood-q12 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q13 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q14 | partially_answered | 0.00 | 0.7111 |
| iris-ood-q15 | partially_answered | 0.00 | 0.7111 |
| iris-ood-q16 | partially_answered | 0.00 | 0.7556 |
| iris-ood-q17 | partially_answered | 0.00 | 0.9111 |
| iris-ood-q18 | partially_answered | 0.80 | 0.7778 |
| iris-ood-q19 | partially_answered | 0.85 | 0.9556 |
| iris-ood-q20 | partially_answered | 0.00 | 0.9556 |
| iris-ood-q21 | partially_answered | 0.00 | 0.6444 |
| iris-ood-q22 | partially_answered | 0.50 | 0.7333 |
| iris-ood-q23 | partially_answered | 0.00 | 0.7333 |
| iris-ood-q24 | partially_answered | 1.00 | 0.7333 |
| iris-ood-q25 | partially_answered | 0.00 | 0.7333 |
| iris-ood-q26 | partially_answered | 0.00 | 0.6889 |
| iris-ood-q27 | partially_answered | 0.00 | 0.7556 |
| iris-ood-q28 | open | 0.00 | 0.7333 |

| hypothesis | status | confidence | support/refute |
|---|---|---|---|
| iris-ood-q0-h0 | refuted | 0.14 | 1/2 |
| iris-ood-q0-h1 | refuted | 0.23 | 1/2 |
| iris-ood-q0-h2 | refuted | 0.00 | 0/2 |
| iris-ood-q0-h3 | refuted | 0.00 | 0/2 |
| iris-ood-q0-h4 | refuted | 0.00 | 0/2 |
| iris-ood-q0-h5 | refuted | 0.00 | 0/2 |
| iris-ood-q0-h6 | refuted | 0.00 | 0/2 |
| iris-ood-q0-h7 | refuted | 0.00 | 0/2 |
| iris-ood-q0-h8 | refuted | 0.00 | 0/2 |
| iris-ood-q0-h9 | refuted | 0.00 | 0/2 |
| iris-ood-q0-h10 | refuted | 0.00 | 0/2 |
| iris-ood-q0-h11 | refuted | 0.16 | 1/2 |
| … (1330 more hypotheses) | | | |

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

*Generated by `scripts/report.py` (git `215e86d`) from sessions: main=sessions/naive-seed42; control=sessions/noop-seed42.*
