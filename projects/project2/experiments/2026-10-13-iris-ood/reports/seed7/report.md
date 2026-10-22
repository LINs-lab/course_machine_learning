# iris-ood — self-editing ablation report

- cycles: **4188** · seed: **7** · epochs: **567** · questions: **29** · hypotheses: **1746** · evidence records: **4188**
- arms: **self-editing ON** (naive priority-rotation improver) vs **self-editing OFF** (noop improver)
- baseline ood_accuracy: **0.3333** → final best (kept): **1.0000** (improvement +0.6667)
- keep-rate: **15.57%** · self-edits kept/reverted: **0/1**

![trajectory](figures/trajectory.pdf)

*Trajectory of ood_accuracy on the wall-clock axis (one single-variable experiment per cycle; seed 7). Purple line + stars: running best (kept runs); purple dots: keeps; gray crosses: discards; red triangles: fails. Background bands are research questions; dashed verticals are epoch boundaries (omitted and counted instead beyond 24); the dashed horizontal is the initial baseline. Baseline re-measurements appear at every epoch upgrade and question switch — the incumbent solution persists, its value re-baselines.*

![keeprate](figures/keeprate.pdf)

*Rolling keep rate (trailing window scaled to the run count) over wall-clock time. Self-edit decisions are marked at their application time: dotted vertical, star = kept, cross = reverted. The overall keep rate is annotated at the lower right.*

![exploration](figures/exploration.pdf)

*Rolling share of each knob over wall-clock time (trailing window 200 runs) — how the fast loop's exploration mix drifts. Legend labels carry each knob's trial count and keep rate (the per-knob payoff). Baseline re-measurements are bookkeeping at question/epoch switches and are excluded from the share.*

## Self-editing ablation

![compare](figures/compare.pdf)

*Running best on the wall-clock axis: self-editing ON (purple, solid, stars) vs self-editing OFF (red, dashed). Legend labels carry each series' final value.*

| | self-editing ON | self-editing OFF |
|---|---|---|
| final best ood_accuracy | **1.0000** | 0.9333 |
| keep-rate | 15.57% | 6.73% |
| mean running best (AUC/cycle) | **0.9694** | 0.8774 |
| cycles to match OFF final best | **442** | — |
| self-edits kept/reverted | **0/1** | 0/0 |

## Self-edit response — did an edit cause measurable gain?

![no-change response](figures/edit_response_nochange.pdf)

![keep-rate response](figures/edit_response_keeprate.pdf)

![run-value response](figures/edit_response_value.pdf)

![running-best response](figures/edit_response_best.pdf)

**Definition (L1).** For each edit applied at run index $c_e$: pre-window $P=[\max(\text{prev termination}, c_e-100)\dots c_e)$, post-window $Q=[c_e \dots \min(c_e+100, \text{next application}))$, both clipped to the question active at $c_e$; the control contributes position-matched windows from the *same* question (its question sequence desynchronizes from the ON arm's, so raw run-index matching would confound the value/best contrasts with question switches). Variables: no-change waste rate (**primary** — the edit's declared target), keep rate, mean run value, mean running best. **Gain on a variable ⇔ the difference-in-differences $\mathrm{DiD}=\Delta_{ON}-\Delta_{OFF}$ has a 95% bootstrap CI (2,000 within-window run resamples) lying strictly on the improvement side of zero.**

*Rolling (window 25 runs) response variables over the first 0.75 h: self-editing ON (purple) vs self-editing OFF (red, dashed). Dotted verticals mark edit application; shaded bands are the arbitrator's decision windows (≥2 runs); star/cross mark the kept/reverted call. Background bands show the ON arm's question spans.*

| edit | arbitrator decision | variable | pre (n) | post (n) | Δ ON | DiD [95% CI] | verdict |
|---|---|---|---|---|---|---|---|
| **edit 1** | reverted (3-run window) | **no-change waste rate** | 0.33 (3) | 0.81 (84) | +0.476 | -0.048 [-0.762, +0.690] | **n.s.** |
| edit 1 |  | keep rate | 0.67 (3) | 0.06 (84) | -0.607 | -0.048 [-0.774, +0.667] | n.s. |
| edit 1 |  | mean run value | 0.44 (3) | 0.57 (84) | +0.131 | -0.077 [-0.307, +0.147] | n.s. |
| edit 1 |  | mean running best | 0.44 (3) | 0.66 (84) | +0.226 | -0.092 [-0.310, +0.124] | n.s. |
*edits numbered in application order; ledger ids: edit 1 = ed-c50ec327.*

**Decision audit (L2).** edit 1: arbitrator **reverted** after a 3-run observation window; measured primary response n.s. (DiD -0.048) — not resolvable at this window.

**Net-state contrast (L3, cycles 6–500).** After the last edit decision the arms differ by one persistent strategy state (the kept rotation):

| variable | ON | OFF | diff [95% CI] | verdict |
|---|---|---|---|---|
| **no-change waste rate** | 0.78 | 0.88 | -0.099 [-0.144, -0.055] | **gain** |
| keep rate | 0.19 | 0.05 | +0.140 [+0.099, +0.178] | gain |
| mean run value | 0.66 | 0.65 | +0.005 [-0.011, +0.021] | n.s. |
| mean running best | 0.84 | 0.83 | +0.003 [-0.006, +0.011] | n.s. |

Same contrast, **question-matched** — each question's first ≤100 runs in both arms, which removes the question-mix composition difference (running best is the within-question ratchet here):

| variable | ON | OFF | diff [95% CI] | verdict |
|---|---|---|---|---|
| **no-change waste rate** | 0.87 | 0.92 | -0.057 [-0.084, -0.031] | **gain** |
| keep rate | 0.09 | 0.05 | +0.043 [+0.021, +0.066] | gain |
| mean run value | 0.70 | 0.68 | +0.021 [+0.008, +0.034] | gain |
| mean running best | 0.78 | 0.76 | +0.019 [+0.011, +0.028] | gain |

*Caveat: the arms' question mixes differ over this tail (the ON arm left its first question after 87 runs vs 139 in the control — an early and possibly edit-caused reaction, but proposer temperature makes single-event attribution noisy), so the run-indexed table above bundles the edit with its downstream question-sequencing effects; the question-matched table is the like-for-like readout.*

*Structural note: the self-editing policy has been silent since run 6 — its only edit family (knob-priority rotation) has two distinct candidates for three knobs, both terminal in the content-addressed ledger, while its trigger (no-change share ≥ 30%) stayed satisfied (88% over the last 500 runs). Self-editing in this run = two early edits + one persistent rotation, not ongoing self-modification.*

## Key events

```
cycle 1: new best 0.3333 (baseline)
cycle 3: new best 0.6444 (model.REG)
cycle 3: self-edit 1 → reverted (strategy priority rotation)
cycle 19: epoch e001 → e002 (benchmark upgraded, incumbent re-baselined)
cycle 19: new best 0.6667 (baseline)
cycle 49: epoch e002 → e003 (benchmark upgraded, incumbent re-baselined)
cycle 52: epoch e003 → e004 (benchmark upgraded, incumbent re-baselined)
cycle 82: epoch e004 → e005 (benchmark upgraded, incumbent re-baselined)
cycle 85: epoch e005 → e006 (benchmark upgraded, incumbent re-baselined)
cycle 89: question q0 → q1
cycle 97: new best 0.8444 (model.REG)
cycle 103: question q1 → q2
cycle 181: epoch e006 → e007 (benchmark upgraded, incumbent re-baselined)
cycle 190: epoch e007 → e008 (benchmark upgraded, incumbent re-baselined)
cycle 193: epoch e008 → e009 (benchmark upgraded, incumbent re-baselined)
cycle 196: epoch e009 → e010 (benchmark upgraded, incumbent re-baselined)
cycle 199: epoch e010 → e011 (benchmark upgraded, incumbent re-baselined)
cycle 202: epoch e011 → e012 (benchmark upgraded, incumbent re-baselined)
cycle 202: new best 0.8667 (baseline)
cycle 226: epoch e012 → e013 (benchmark upgraded, incumbent re-baselined)
cycle 229: epoch e013 → e014 (benchmark upgraded, incumbent re-baselined)
cycle 232: epoch e014 → e015 (benchmark upgraded, incumbent re-baselined)
cycle 235: epoch e015 → e016 (benchmark upgraded, incumbent re-baselined)
cycle 238: epoch e016 → e017 (benchmark upgraded, incumbent re-baselined)
cycle 241: epoch e017 → e018 (benchmark upgraded, incumbent re-baselined)
cycle 244: epoch e018 → e019 (benchmark upgraded, incumbent re-baselined)
cycle 247: epoch e019 → e020 (benchmark upgraded, incumbent re-baselined)
cycle 250: epoch e020 → e021 (benchmark upgraded, incumbent re-baselined)
cycle 259: epoch e021 → e022 (benchmark upgraded, incumbent re-baselined)
cycle 262: epoch e022 → e023 (benchmark upgraded, incumbent re-baselined)
cycle 265: epoch e023 → e024 (benchmark upgraded, incumbent re-baselined)
cycle 268: epoch e024 → e025 (benchmark upgraded, incumbent re-baselined)
cycle 271: epoch e025 → e026 (benchmark upgraded, incumbent re-baselined)
cycle 274: epoch e026 → e027 (benchmark upgraded, incumbent re-baselined)
cycle 277: epoch e027 → e028 (benchmark upgraded, incumbent re-baselined)
cycle 286: epoch e028 → e029 (benchmark upgraded, incumbent re-baselined)
cycle 289: epoch e029 → e030 (benchmark upgraded, incumbent re-baselined)
cycle 298: epoch e030 → e031 (benchmark upgraded, incumbent re-baselined)
cycle 301: epoch e031 → e032 (benchmark upgraded, incumbent re-baselined)
cycle 304: epoch e032 → e033 (benchmark upgraded, incumbent re-baselined)
cycle 307: epoch e033 → e034 (benchmark upgraded, incumbent re-baselined)
cycle 310: epoch e034 → e035 (benchmark upgraded, incumbent re-baselined)
cycle 313: epoch e035 → e036 (benchmark upgraded, incumbent re-baselined)
cycle 316: epoch e036 → e037 (benchmark upgraded, incumbent re-baselined)
cycle 319: epoch e037 → e038 (benchmark upgraded, incumbent re-baselined)
cycle 322: epoch e038 → e039 (benchmark upgraded, incumbent re-baselined)
cycle 325: epoch e039 → e040 (benchmark upgraded, incumbent re-baselined)
cycle 328: epoch e040 → e041 (benchmark upgraded, incumbent re-baselined)
cycle 331: epoch e041 → e042 (benchmark upgraded, incumbent re-baselined)
cycle 334: epoch e042 → e043 (benchmark upgraded, incumbent re-baselined)
cycle 337: epoch e043 → e044 (benchmark upgraded, incumbent re-baselined)
cycle 343: epoch e044 → e045 (benchmark upgraded, incumbent re-baselined)
cycle 346: epoch e045 → e046 (benchmark upgraded, incumbent re-baselined)
cycle 349: epoch e046 → e047 (benchmark upgraded, incumbent re-baselined)
cycle 352: epoch e047 → e048 (benchmark upgraded, incumbent re-baselined)
cycle 355: epoch e048 → e049 (benchmark upgraded, incumbent re-baselined)
cycle 358: epoch e049 → e050 (benchmark upgraded, incumbent re-baselined)
cycle 361: epoch e050 → e051 (benchmark upgraded, incumbent re-baselined)
cycle 394: epoch e051 → e052 (benchmark upgraded, incumbent re-baselined)
cycle 397: epoch e052 → e053 (benchmark upgraded, incumbent re-baselined)
cycle 400: epoch e053 → e054 (benchmark upgraded, incumbent re-baselined)
cycle 403: epoch e054 → e055 (benchmark upgraded, incumbent re-baselined)
cycle 406: epoch e055 → e056 (benchmark upgraded, incumbent re-baselined)
cycle 409: epoch e056 → e057 (benchmark upgraded, incumbent re-baselined)
cycle 412: epoch e057 → e058 (benchmark upgraded, incumbent re-baselined)
cycle 415: epoch e058 → e059 (benchmark upgraded, incumbent re-baselined)
cycle 418: epoch e059 → e060 (benchmark upgraded, incumbent re-baselined)
cycle 421: epoch e060 → e061 (benchmark upgraded, incumbent re-baselined)
cycle 424: epoch e061 → e062 (benchmark upgraded, incumbent re-baselined)
cycle 427: epoch e062 → e063 (benchmark upgraded, incumbent re-baselined)
cycle 430: epoch e063 → e064 (benchmark upgraded, incumbent re-baselined)
cycle 433: epoch e064 → e065 (benchmark upgraded, incumbent re-baselined)
cycle 436: epoch e065 → e066 (benchmark upgraded, incumbent re-baselined)
cycle 439: epoch e066 → e067 (benchmark upgraded, incumbent re-baselined)
cycle 442: epoch e067 → e068 (benchmark upgraded, incumbent re-baselined)
cycle 442: new best 0.9556 (baseline)
cycle 445: epoch e068 → e069 (benchmark upgraded, incumbent re-baselined)
cycle 448: epoch e069 → e070 (benchmark upgraded, incumbent re-baselined)
cycle 451: epoch e070 → e071 (benchmark upgraded, incumbent re-baselined)
cycle 454: epoch e071 → e072 (benchmark upgraded, incumbent re-baselined)
cycle 457: epoch e072 → e073 (benchmark upgraded, incumbent re-baselined)
cycle 460: epoch e073 → e074 (benchmark upgraded, incumbent re-baselined)
cycle 463: epoch e074 → e075 (benchmark upgraded, incumbent re-baselined)
cycle 466: epoch e075 → e076 (benchmark upgraded, incumbent re-baselined)
cycle 469: epoch e076 → e077 (benchmark upgraded, incumbent re-baselined)
cycle 472: epoch e077 → e078 (benchmark upgraded, incumbent re-baselined)
cycle 472: new best 0.9778 (baseline)
cycle 475: epoch e078 → e079 (benchmark upgraded, incumbent re-baselined)
cycle 478: epoch e079 → e080 (benchmark upgraded, incumbent re-baselined)
cycle 484: epoch e080 → e081 (benchmark upgraded, incumbent re-baselined)
cycle 487: epoch e081 → e082 (benchmark upgraded, incumbent re-baselined)
cycle 490: epoch e082 → e083 (benchmark upgraded, incumbent re-baselined)
cycle 493: epoch e083 → e084 (benchmark upgraded, incumbent re-baselined)
cycle 496: epoch e084 → e085 (benchmark upgraded, incumbent re-baselined)
cycle 499: epoch e085 → e086 (benchmark upgraded, incumbent re-baselined)
cycle 505: epoch e086 → e087 (benchmark upgraded, incumbent re-baselined)
cycle 508: epoch e087 → e088 (benchmark upgraded, incumbent re-baselined)
cycle 511: epoch e088 → e089 (benchmark upgraded, incumbent re-baselined)
cycle 520: epoch e089 → e090 (benchmark upgraded, incumbent re-baselined)
cycle 523: epoch e090 → e091 (benchmark upgraded, incumbent re-baselined)
cycle 526: epoch e091 → e092 (benchmark upgraded, incumbent re-baselined)
cycle 529: epoch e092 → e093 (benchmark upgraded, incumbent re-baselined)
cycle 532: epoch e093 → e094 (benchmark upgraded, incumbent re-baselined)
cycle 535: epoch e094 → e095 (benchmark upgraded, incumbent re-baselined)
cycle 538: epoch e095 → e096 (benchmark upgraded, incumbent re-baselined)
cycle 541: epoch e096 → e097 (benchmark upgraded, incumbent re-baselined)
cycle 544: epoch e097 → e098 (benchmark upgraded, incumbent re-baselined)
cycle 547: epoch e098 → e099 (benchmark upgraded, incumbent re-baselined)
cycle 550: epoch e099 → e100 (benchmark upgraded, incumbent re-baselined)
cycle 553: epoch e100 → e101 (benchmark upgraded, incumbent re-baselined)
cycle 556: epoch e101 → e102 (benchmark upgraded, incumbent re-baselined)
cycle 559: epoch e102 → e103 (benchmark upgraded, incumbent re-baselined)
cycle 562: epoch e103 → e104 (benchmark upgraded, incumbent re-baselined)
cycle 565: epoch e104 → e105 (benchmark upgraded, incumbent re-baselined)
cycle 568: epoch e105 → e106 (benchmark upgraded, incumbent re-baselined)
cycle 571: epoch e106 → e107 (benchmark upgraded, incumbent re-baselined)
cycle 574: epoch e107 → e108 (benchmark upgraded, incumbent re-baselined)
cycle 577: epoch e108 → e109 (benchmark upgraded, incumbent re-baselined)
cycle 580: epoch e109 → e110 (benchmark upgraded, incumbent re-baselined)
cycle 589: question q2 → q3
cycle 676: question q3 → q4
cycle 755: question q4 → q5
cycle 814: epoch e110 → e111 (benchmark upgraded, incumbent re-baselined)
cycle 828: question q5 → q6
cycle 859: epoch e111 → e112 (benchmark upgraded, incumbent re-baselined)
cycle 862: epoch e112 → e113 (benchmark upgraded, incumbent re-baselined)
cycle 880: epoch e113 → e114 (benchmark upgraded, incumbent re-baselined)
cycle 901: question q6 → q7
cycle 904: epoch e114 → e115 (benchmark upgraded, incumbent re-baselined)
cycle 907: epoch e115 → e116 (benchmark upgraded, incumbent re-baselined)
cycle 910: epoch e116 → e117 (benchmark upgraded, incumbent re-baselined)
cycle 913: epoch e117 → e118 (benchmark upgraded, incumbent re-baselined)
cycle 916: epoch e118 → e119 (benchmark upgraded, incumbent re-baselined)
cycle 919: epoch e119 → e120 (benchmark upgraded, incumbent re-baselined)
cycle 922: epoch e120 → e121 (benchmark upgraded, incumbent re-baselined)
cycle 925: epoch e121 → e122 (benchmark upgraded, incumbent re-baselined)
cycle 928: epoch e122 → e123 (benchmark upgraded, incumbent re-baselined)
cycle 931: epoch e123 → e124 (benchmark upgraded, incumbent re-baselined)
cycle 934: epoch e124 → e125 (benchmark upgraded, incumbent re-baselined)
cycle 937: epoch e125 → e126 (benchmark upgraded, incumbent re-baselined)
cycle 940: epoch e126 → e127 (benchmark upgraded, incumbent re-baselined)
cycle 943: epoch e127 → e128 (benchmark upgraded, incumbent re-baselined)
cycle 946: epoch e128 → e129 (benchmark upgraded, incumbent re-baselined)
cycle 949: epoch e129 → e130 (benchmark upgraded, incumbent re-baselined)
cycle 952: epoch e130 → e131 (benchmark upgraded, incumbent re-baselined)
cycle 955: epoch e131 → e132 (benchmark upgraded, incumbent re-baselined)
cycle 958: epoch e132 → e133 (benchmark upgraded, incumbent re-baselined)
cycle 961: epoch e133 → e134 (benchmark upgraded, incumbent re-baselined)
cycle 964: epoch e134 → e135 (benchmark upgraded, incumbent re-baselined)
cycle 967: epoch e135 → e136 (benchmark upgraded, incumbent re-baselined)
cycle 970: epoch e136 → e137 (benchmark upgraded, incumbent re-baselined)
cycle 973: epoch e137 → e138 (benchmark upgraded, incumbent re-baselined)
cycle 995: question q7 → q8
cycle 1015: epoch e138 → e139 (benchmark upgraded, incumbent re-baselined)
cycle 1018: epoch e139 → e140 (benchmark upgraded, incumbent re-baselined)
cycle 1027: epoch e140 → e141 (benchmark upgraded, incumbent re-baselined)
cycle 1030: epoch e141 → e142 (benchmark upgraded, incumbent re-baselined)
cycle 1033: epoch e142 → e143 (benchmark upgraded, incumbent re-baselined)
cycle 1060: epoch e143 → e144 (benchmark upgraded, incumbent re-baselined)
cycle 1072: epoch e144 → e145 (benchmark upgraded, incumbent re-baselined)
cycle 1078: question q8 → q9
cycle 1081: epoch e145 → e146 (benchmark upgraded, incumbent re-baselined)
cycle 1084: epoch e146 → e147 (benchmark upgraded, incumbent re-baselined)
cycle 1087: epoch e147 → e148 (benchmark upgraded, incumbent re-baselined)
cycle 1090: epoch e148 → e149 (benchmark upgraded, incumbent re-baselined)
cycle 1096: epoch e149 → e150 (benchmark upgraded, incumbent re-baselined)
cycle 1149: question q9 → q10
cycle 1219: epoch e150 → e151 (benchmark upgraded, incumbent re-baselined)
cycle 1225: question q10 → q11
cycle 1228: epoch e151 → e152 (benchmark upgraded, incumbent re-baselined)
cycle 1231: epoch e152 → e153 (benchmark upgraded, incumbent re-baselined)
cycle 1234: epoch e153 → e154 (benchmark upgraded, incumbent re-baselined)
cycle 1264: epoch e154 → e155 (benchmark upgraded, incumbent re-baselined)
cycle 1267: epoch e155 → e156 (benchmark upgraded, incumbent re-baselined)
cycle 1288: epoch e156 → e157 (benchmark upgraded, incumbent re-baselined)
cycle 1291: epoch e157 → e158 (benchmark upgraded, incumbent re-baselined)
cycle 1294: epoch e158 → e159 (benchmark upgraded, incumbent re-baselined)
cycle 1303: epoch e159 → e160 (benchmark upgraded, incumbent re-baselined)
cycle 1306: epoch e160 → e161 (benchmark upgraded, incumbent re-baselined)
cycle 1309: epoch e161 → e162 (benchmark upgraded, incumbent re-baselined)
cycle 1312: epoch e162 → e163 (benchmark upgraded, incumbent re-baselined)
cycle 1315: epoch e163 → e164 (benchmark upgraded, incumbent re-baselined)
cycle 1318: epoch e164 → e165 (benchmark upgraded, incumbent re-baselined)
cycle 1321: epoch e165 → e166 (benchmark upgraded, incumbent re-baselined)
cycle 1324: epoch e166 → e167 (benchmark upgraded, incumbent re-baselined)
cycle 1339: epoch e167 → e168 (benchmark upgraded, incumbent re-baselined)
cycle 1342: epoch e168 → e169 (benchmark upgraded, incumbent re-baselined)
cycle 1345: epoch e169 → e170 (benchmark upgraded, incumbent re-baselined)
cycle 1348: epoch e170 → e171 (benchmark upgraded, incumbent re-baselined)
cycle 1369: question q11 → q12
cycle 1426: epoch e171 → e172 (benchmark upgraded, incumbent re-baselined)
cycle 1429: epoch e172 → e173 (benchmark upgraded, incumbent re-baselined)
cycle 1432: epoch e173 → e174 (benchmark upgraded, incumbent re-baselined)
cycle 1435: epoch e174 → e175 (benchmark upgraded, incumbent re-baselined)
cycle 1438: epoch e175 → e176 (benchmark upgraded, incumbent re-baselined)
cycle 1441: epoch e176 → e177 (benchmark upgraded, incumbent re-baselined)
cycle 1444: epoch e177 → e178 (benchmark upgraded, incumbent re-baselined)
cycle 1447: epoch e178 → e179 (benchmark upgraded, incumbent re-baselined)
cycle 1450: epoch e179 → e180 (benchmark upgraded, incumbent re-baselined)
cycle 1453: epoch e180 → e181 (benchmark upgraded, incumbent re-baselined)
cycle 1456: epoch e181 → e182 (benchmark upgraded, incumbent re-baselined)
cycle 1459: epoch e182 → e183 (benchmark upgraded, incumbent re-baselined)
cycle 1465: epoch e183 → e184 (benchmark upgraded, incumbent re-baselined)
cycle 1468: epoch e184 → e185 (benchmark upgraded, incumbent re-baselined)
cycle 1471: epoch e185 → e186 (benchmark upgraded, incumbent re-baselined)
cycle 1474: epoch e186 → e187 (benchmark upgraded, incumbent re-baselined)
cycle 1477: epoch e187 → e188 (benchmark upgraded, incumbent re-baselined)
cycle 1480: epoch e188 → e189 (benchmark upgraded, incumbent re-baselined)
cycle 1483: epoch e189 → e190 (benchmark upgraded, incumbent re-baselined)
cycle 1486: epoch e190 → e191 (benchmark upgraded, incumbent re-baselined)
cycle 1489: epoch e191 → e192 (benchmark upgraded, incumbent re-baselined)
cycle 1492: epoch e192 → e193 (benchmark upgraded, incumbent re-baselined)
cycle 1495: epoch e193 → e194 (benchmark upgraded, incumbent re-baselined)
cycle 1498: epoch e194 → e195 (benchmark upgraded, incumbent re-baselined)
cycle 1519: epoch e195 → e196 (benchmark upgraded, incumbent re-baselined)
cycle 1522: epoch e196 → e197 (benchmark upgraded, incumbent re-baselined)
cycle 1525: epoch e197 → e198 (benchmark upgraded, incumbent re-baselined)
cycle 1528: epoch e198 → e199 (benchmark upgraded, incumbent re-baselined)
cycle 1531: epoch e199 → e200 (benchmark upgraded, incumbent re-baselined)
cycle 1534: epoch e200 → e201 (benchmark upgraded, incumbent re-baselined)
cycle 1546: epoch e201 → e202 (benchmark upgraded, incumbent re-baselined)
cycle 1549: epoch e202 → e203 (benchmark upgraded, incumbent re-baselined)
cycle 1555: question q12 → q13
cycle 1609: epoch e203 → e204 (benchmark upgraded, incumbent re-baselined)
cycle 1621: epoch e204 → e205 (benchmark upgraded, incumbent re-baselined)
cycle 1636: epoch e205 → e206 (benchmark upgraded, incumbent re-baselined)
cycle 1638: question q13 → q14
cycle 1666: epoch e206 → e207 (benchmark upgraded, incumbent re-baselined)
cycle 1669: epoch e207 → e208 (benchmark upgraded, incumbent re-baselined)
cycle 1672: epoch e208 → e209 (benchmark upgraded, incumbent re-baselined)
cycle 1699: epoch e209 → e210 (benchmark upgraded, incumbent re-baselined)
cycle 1702: epoch e210 → e211 (benchmark upgraded, incumbent re-baselined)
cycle 1732: epoch e211 → e212 (benchmark upgraded, incumbent re-baselined)
cycle 1735: epoch e212 → e213 (benchmark upgraded, incumbent re-baselined)
cycle 1738: epoch e213 → e214 (benchmark upgraded, incumbent re-baselined)
cycle 1741: epoch e214 → e215 (benchmark upgraded, incumbent re-baselined)
cycle 1744: epoch e215 → e216 (benchmark upgraded, incumbent re-baselined)
cycle 1753: epoch e216 → e217 (benchmark upgraded, incumbent re-baselined)
cycle 1756: epoch e217 → e218 (benchmark upgraded, incumbent re-baselined)
cycle 1759: epoch e218 → e219 (benchmark upgraded, incumbent re-baselined)
cycle 1762: epoch e219 → e220 (benchmark upgraded, incumbent re-baselined)
cycle 1765: epoch e220 → e221 (benchmark upgraded, incumbent re-baselined)
cycle 1768: epoch e221 → e222 (benchmark upgraded, incumbent re-baselined)
cycle 1771: epoch e222 → e223 (benchmark upgraded, incumbent re-baselined)
cycle 1774: epoch e223 → e224 (benchmark upgraded, incumbent re-baselined)
cycle 1782: question q14 → q15
cycle 1840: epoch e224 → e225 (benchmark upgraded, incumbent re-baselined)
cycle 1843: epoch e225 → e226 (benchmark upgraded, incumbent re-baselined)
cycle 1846: epoch e226 → e227 (benchmark upgraded, incumbent re-baselined)
cycle 1849: epoch e227 → e228 (benchmark upgraded, incumbent re-baselined)
cycle 1852: epoch e228 → e229 (benchmark upgraded, incumbent re-baselined)
cycle 1867: question q15 → q16
cycle 1903: epoch e229 → e230 (benchmark upgraded, incumbent re-baselined)
cycle 1906: epoch e230 → e231 (benchmark upgraded, incumbent re-baselined)
cycle 2002: epoch e231 → e232 (benchmark upgraded, incumbent re-baselined)
cycle 2005: epoch e232 → e233 (benchmark upgraded, incumbent re-baselined)
cycle 2011: epoch e233 → e234 (benchmark upgraded, incumbent re-baselined)
cycle 2014: epoch e234 → e235 (benchmark upgraded, incumbent re-baselined)
cycle 2023: epoch e235 → e236 (benchmark upgraded, incumbent re-baselined)
cycle 2026: epoch e236 → e237 (benchmark upgraded, incumbent re-baselined)
cycle 2029: epoch e237 → e238 (benchmark upgraded, incumbent re-baselined)
cycle 2032: epoch e238 → e239 (benchmark upgraded, incumbent re-baselined)
cycle 2035: epoch e239 → e240 (benchmark upgraded, incumbent re-baselined)
cycle 2041: epoch e240 → e241 (benchmark upgraded, incumbent re-baselined)
cycle 2044: epoch e241 → e242 (benchmark upgraded, incumbent re-baselined)
cycle 2050: epoch e242 → e243 (benchmark upgraded, incumbent re-baselined)
cycle 2053: epoch e243 → e244 (benchmark upgraded, incumbent re-baselined)
cycle 2056: epoch e244 → e245 (benchmark upgraded, incumbent re-baselined)
cycle 2059: epoch e245 → e246 (benchmark upgraded, incumbent re-baselined)
cycle 2065: epoch e246 → e247 (benchmark upgraded, incumbent re-baselined)
cycle 2089: epoch e247 → e248 (benchmark upgraded, incumbent re-baselined)
cycle 2092: epoch e248 → e249 (benchmark upgraded, incumbent re-baselined)
cycle 2095: epoch e249 → e250 (benchmark upgraded, incumbent re-baselined)
cycle 2098: epoch e250 → e251 (benchmark upgraded, incumbent re-baselined)
cycle 2101: epoch e251 → e252 (benchmark upgraded, incumbent re-baselined)
cycle 2107: epoch e252 → e253 (benchmark upgraded, incumbent re-baselined)
cycle 2110: epoch e253 → e254 (benchmark upgraded, incumbent re-baselined)
cycle 2123: question q16 → q17
cycle 2185: epoch e254 → e255 (benchmark upgraded, incumbent re-baselined)
cycle 2188: epoch e255 → e256 (benchmark upgraded, incumbent re-baselined)
cycle 2191: epoch e256 → e257 (benchmark upgraded, incumbent re-baselined)
cycle 2349: question q17 → q18
cycle 2404: epoch e257 → e258 (benchmark upgraded, incumbent re-baselined)
cycle 2461: epoch e258 → e259 (benchmark upgraded, incumbent re-baselined)
cycle 2488: epoch e259 → e260 (benchmark upgraded, incumbent re-baselined)
cycle 2491: epoch e260 → e261 (benchmark upgraded, incumbent re-baselined)
cycle 2494: epoch e261 → e262 (benchmark upgraded, incumbent re-baselined)
cycle 2527: epoch e262 → e263 (benchmark upgraded, incumbent re-baselined)
cycle 2530: epoch e263 → e264 (benchmark upgraded, incumbent re-baselined)
cycle 2533: epoch e264 → e265 (benchmark upgraded, incumbent re-baselined)
cycle 2536: epoch e265 → e266 (benchmark upgraded, incumbent re-baselined)
cycle 2539: epoch e266 → e267 (benchmark upgraded, incumbent re-baselined)
cycle 2542: epoch e267 → e268 (benchmark upgraded, incumbent re-baselined)
cycle 2545: epoch e268 → e269 (benchmark upgraded, incumbent re-baselined)
cycle 2548: epoch e269 → e270 (benchmark upgraded, incumbent re-baselined)
cycle 2551: epoch e270 → e271 (benchmark upgraded, incumbent re-baselined)
cycle 2551: new best 1.0000 (baseline)
cycle 2554: epoch e271 → e272 (benchmark upgraded, incumbent re-baselined)
cycle 2557: epoch e272 → e273 (benchmark upgraded, incumbent re-baselined)
cycle 2560: epoch e273 → e274 (benchmark upgraded, incumbent re-baselined)
cycle 2563: epoch e274 → e275 (benchmark upgraded, incumbent re-baselined)
cycle 2566: epoch e275 → e276 (benchmark upgraded, incumbent re-baselined)
cycle 2569: epoch e276 → e277 (benchmark upgraded, incumbent re-baselined)
cycle 2572: epoch e277 → e278 (benchmark upgraded, incumbent re-baselined)
cycle 2575: epoch e278 → e279 (benchmark upgraded, incumbent re-baselined)
cycle 2578: epoch e279 → e280 (benchmark upgraded, incumbent re-baselined)
cycle 2581: epoch e280 → e281 (benchmark upgraded, incumbent re-baselined)
cycle 2587: epoch e281 → e282 (benchmark upgraded, incumbent re-baselined)
cycle 2590: epoch e282 → e283 (benchmark upgraded, incumbent re-baselined)
cycle 2596: epoch e283 → e284 (benchmark upgraded, incumbent re-baselined)
cycle 2599: epoch e284 → e285 (benchmark upgraded, incumbent re-baselined)
cycle 2602: epoch e285 → e286 (benchmark upgraded, incumbent re-baselined)
cycle 2605: epoch e286 → e287 (benchmark upgraded, incumbent re-baselined)
cycle 2608: epoch e287 → e288 (benchmark upgraded, incumbent re-baselined)
cycle 2611: epoch e288 → e289 (benchmark upgraded, incumbent re-baselined)
cycle 2614: epoch e289 → e290 (benchmark upgraded, incumbent re-baselined)
cycle 2617: epoch e290 → e291 (benchmark upgraded, incumbent re-baselined)
cycle 2620: epoch e291 → e292 (benchmark upgraded, incumbent re-baselined)
cycle 2623: epoch e292 → e293 (benchmark upgraded, incumbent re-baselined)
cycle 2626: epoch e293 → e294 (benchmark upgraded, incumbent re-baselined)
cycle 2629: epoch e294 → e295 (benchmark upgraded, incumbent re-baselined)
cycle 2635: epoch e295 → e296 (benchmark upgraded, incumbent re-baselined)
cycle 2638: epoch e296 → e297 (benchmark upgraded, incumbent re-baselined)
cycle 2647: epoch e297 → e298 (benchmark upgraded, incumbent re-baselined)
cycle 2650: epoch e298 → e299 (benchmark upgraded, incumbent re-baselined)
cycle 2659: epoch e299 → e300 (benchmark upgraded, incumbent re-baselined)
cycle 2662: epoch e300 → e301 (benchmark upgraded, incumbent re-baselined)
cycle 2671: epoch e301 → e302 (benchmark upgraded, incumbent re-baselined)
cycle 2674: epoch e302 → e303 (benchmark upgraded, incumbent re-baselined)
cycle 2677: epoch e303 → e304 (benchmark upgraded, incumbent re-baselined)
cycle 2680: epoch e304 → e305 (benchmark upgraded, incumbent re-baselined)
cycle 2683: epoch e305 → e306 (benchmark upgraded, incumbent re-baselined)
cycle 2686: epoch e306 → e307 (benchmark upgraded, incumbent re-baselined)
cycle 2689: epoch e307 → e308 (benchmark upgraded, incumbent re-baselined)
cycle 2692: epoch e308 → e309 (benchmark upgraded, incumbent re-baselined)
cycle 2695: question q18 → q19
cycle 2698: epoch e309 → e310 (benchmark upgraded, incumbent re-baselined)
cycle 2701: epoch e310 → e311 (benchmark upgraded, incumbent re-baselined)
cycle 2704: epoch e311 → e312 (benchmark upgraded, incumbent re-baselined)
cycle 2707: epoch e312 → e313 (benchmark upgraded, incumbent re-baselined)
cycle 2752: epoch e313 → e314 (benchmark upgraded, incumbent re-baselined)
cycle 2755: epoch e314 → e315 (benchmark upgraded, incumbent re-baselined)
cycle 2758: epoch e315 → e316 (benchmark upgraded, incumbent re-baselined)
cycle 2761: epoch e316 → e317 (benchmark upgraded, incumbent re-baselined)
cycle 2773: epoch e317 → e318 (benchmark upgraded, incumbent re-baselined)
cycle 2776: epoch e318 → e319 (benchmark upgraded, incumbent re-baselined)
cycle 2779: epoch e319 → e320 (benchmark upgraded, incumbent re-baselined)
cycle 2782: epoch e320 → e321 (benchmark upgraded, incumbent re-baselined)
cycle 2785: epoch e321 → e322 (benchmark upgraded, incumbent re-baselined)
cycle 2788: epoch e322 → e323 (benchmark upgraded, incumbent re-baselined)
cycle 2797: epoch e323 → e324 (benchmark upgraded, incumbent re-baselined)
cycle 2803: epoch e324 → e325 (benchmark upgraded, incumbent re-baselined)
cycle 2806: epoch e325 → e326 (benchmark upgraded, incumbent re-baselined)
cycle 2809: epoch e326 → e327 (benchmark upgraded, incumbent re-baselined)
cycle 2812: epoch e327 → e328 (benchmark upgraded, incumbent re-baselined)
cycle 2815: epoch e328 → e329 (benchmark upgraded, incumbent re-baselined)
cycle 2818: epoch e329 → e330 (benchmark upgraded, incumbent re-baselined)
cycle 2833: question q19 → q20
cycle 2851: epoch e330 → e331 (benchmark upgraded, incumbent re-baselined)
cycle 2854: epoch e331 → e332 (benchmark upgraded, incumbent re-baselined)
cycle 2857: epoch e332 → e333 (benchmark upgraded, incumbent re-baselined)
cycle 2869: epoch e333 → e334 (benchmark upgraded, incumbent re-baselined)
cycle 2872: epoch e334 → e335 (benchmark upgraded, incumbent re-baselined)
cycle 2875: epoch e335 → e336 (benchmark upgraded, incumbent re-baselined)
cycle 2896: question q20 → q21
cycle 2908: epoch e336 → e337 (benchmark upgraded, incumbent re-baselined)
cycle 2917: epoch e337 → e338 (benchmark upgraded, incumbent re-baselined)
cycle 2920: epoch e338 → e339 (benchmark upgraded, incumbent re-baselined)
cycle 2923: epoch e339 → e340 (benchmark upgraded, incumbent re-baselined)
cycle 2935: epoch e340 → e341 (benchmark upgraded, incumbent re-baselined)
cycle 2959: epoch e341 → e342 (benchmark upgraded, incumbent re-baselined)
cycle 2962: epoch e342 → e343 (benchmark upgraded, incumbent re-baselined)
cycle 2965: epoch e343 → e344 (benchmark upgraded, incumbent re-baselined)
cycle 2968: epoch e344 → e345 (benchmark upgraded, incumbent re-baselined)
cycle 2971: epoch e345 → e346 (benchmark upgraded, incumbent re-baselined)
cycle 2974: epoch e346 → e347 (benchmark upgraded, incumbent re-baselined)
cycle 2977: epoch e347 → e348 (benchmark upgraded, incumbent re-baselined)
cycle 2980: epoch e348 → e349 (benchmark upgraded, incumbent re-baselined)
cycle 2983: epoch e349 → e350 (benchmark upgraded, incumbent re-baselined)
cycle 2986: epoch e350 → e351 (benchmark upgraded, incumbent re-baselined)
cycle 2989: epoch e351 → e352 (benchmark upgraded, incumbent re-baselined)
cycle 2992: epoch e352 → e353 (benchmark upgraded, incumbent re-baselined)
cycle 2995: epoch e353 → e354 (benchmark upgraded, incumbent re-baselined)
cycle 3004: epoch e354 → e355 (benchmark upgraded, incumbent re-baselined)
cycle 3007: epoch e355 → e356 (benchmark upgraded, incumbent re-baselined)
cycle 3016: epoch e356 → e357 (benchmark upgraded, incumbent re-baselined)
cycle 3019: epoch e357 → e358 (benchmark upgraded, incumbent re-baselined)
cycle 3022: epoch e358 → e359 (benchmark upgraded, incumbent re-baselined)
cycle 3025: epoch e359 → e360 (benchmark upgraded, incumbent re-baselined)
cycle 3034: epoch e360 → e361 (benchmark upgraded, incumbent re-baselined)
cycle 3037: epoch e361 → e362 (benchmark upgraded, incumbent re-baselined)
cycle 3040: epoch e362 → e363 (benchmark upgraded, incumbent re-baselined)
cycle 3043: epoch e363 → e364 (benchmark upgraded, incumbent re-baselined)
cycle 3046: epoch e364 → e365 (benchmark upgraded, incumbent re-baselined)
cycle 3049: epoch e365 → e366 (benchmark upgraded, incumbent re-baselined)
cycle 3052: epoch e366 → e367 (benchmark upgraded, incumbent re-baselined)
cycle 3055: epoch e367 → e368 (benchmark upgraded, incumbent re-baselined)
cycle 3058: epoch e368 → e369 (benchmark upgraded, incumbent re-baselined)
cycle 3061: epoch e369 → e370 (benchmark upgraded, incumbent re-baselined)
cycle 3064: epoch e370 → e371 (benchmark upgraded, incumbent re-baselined)
cycle 3067: epoch e371 → e372 (benchmark upgraded, incumbent re-baselined)
cycle 3070: epoch e372 → e373 (benchmark upgraded, incumbent re-baselined)
cycle 3073: epoch e373 → e374 (benchmark upgraded, incumbent re-baselined)
cycle 3076: epoch e374 → e375 (benchmark upgraded, incumbent re-baselined)
cycle 3079: epoch e375 → e376 (benchmark upgraded, incumbent re-baselined)
cycle 3082: epoch e376 → e377 (benchmark upgraded, incumbent re-baselined)
cycle 3085: epoch e377 → e378 (benchmark upgraded, incumbent re-baselined)
cycle 3088: epoch e378 → e379 (benchmark upgraded, incumbent re-baselined)
cycle 3091: epoch e379 → e380 (benchmark upgraded, incumbent re-baselined)
cycle 3094: epoch e380 → e381 (benchmark upgraded, incumbent re-baselined)
cycle 3097: epoch e381 → e382 (benchmark upgraded, incumbent re-baselined)
cycle 3100: epoch e382 → e383 (benchmark upgraded, incumbent re-baselined)
cycle 3103: epoch e383 → e384 (benchmark upgraded, incumbent re-baselined)
cycle 3106: epoch e384 → e385 (benchmark upgraded, incumbent re-baselined)
cycle 3109: epoch e385 → e386 (benchmark upgraded, incumbent re-baselined)
cycle 3112: epoch e386 → e387 (benchmark upgraded, incumbent re-baselined)
cycle 3115: epoch e387 → e388 (benchmark upgraded, incumbent re-baselined)
cycle 3118: epoch e388 → e389 (benchmark upgraded, incumbent re-baselined)
cycle 3121: epoch e389 → e390 (benchmark upgraded, incumbent re-baselined)
cycle 3124: epoch e390 → e391 (benchmark upgraded, incumbent re-baselined)
cycle 3127: epoch e391 → e392 (benchmark upgraded, incumbent re-baselined)
cycle 3130: epoch e392 → e393 (benchmark upgraded, incumbent re-baselined)
cycle 3142: epoch e393 → e394 (benchmark upgraded, incumbent re-baselined)
cycle 3145: epoch e394 → e395 (benchmark upgraded, incumbent re-baselined)
cycle 3148: epoch e395 → e396 (benchmark upgraded, incumbent re-baselined)
cycle 3151: epoch e396 → e397 (benchmark upgraded, incumbent re-baselined)
cycle 3154: epoch e397 → e398 (benchmark upgraded, incumbent re-baselined)
cycle 3157: epoch e398 → e399 (benchmark upgraded, incumbent re-baselined)
cycle 3160: epoch e399 → e400 (benchmark upgraded, incumbent re-baselined)
cycle 3163: epoch e400 → e401 (benchmark upgraded, incumbent re-baselined)
cycle 3166: epoch e401 → e402 (benchmark upgraded, incumbent re-baselined)
cycle 3169: epoch e402 → e403 (benchmark upgraded, incumbent re-baselined)
cycle 3172: epoch e403 → e404 (benchmark upgraded, incumbent re-baselined)
cycle 3181: epoch e404 → e405 (benchmark upgraded, incumbent re-baselined)
cycle 3184: epoch e405 → e406 (benchmark upgraded, incumbent re-baselined)
cycle 3187: epoch e406 → e407 (benchmark upgraded, incumbent re-baselined)
cycle 3190: epoch e407 → e408 (benchmark upgraded, incumbent re-baselined)
cycle 3193: epoch e408 → e409 (benchmark upgraded, incumbent re-baselined)
cycle 3196: epoch e409 → e410 (benchmark upgraded, incumbent re-baselined)
cycle 3199: epoch e410 → e411 (benchmark upgraded, incumbent re-baselined)
cycle 3202: epoch e411 → e412 (benchmark upgraded, incumbent re-baselined)
cycle 3205: epoch e412 → e413 (benchmark upgraded, incumbent re-baselined)
cycle 3208: epoch e413 → e414 (benchmark upgraded, incumbent re-baselined)
cycle 3211: epoch e414 → e415 (benchmark upgraded, incumbent re-baselined)
cycle 3220: epoch e415 → e416 (benchmark upgraded, incumbent re-baselined)
cycle 3223: epoch e416 → e417 (benchmark upgraded, incumbent re-baselined)
cycle 3226: epoch e417 → e418 (benchmark upgraded, incumbent re-baselined)
cycle 3229: epoch e418 → e419 (benchmark upgraded, incumbent re-baselined)
cycle 3232: epoch e419 → e420 (benchmark upgraded, incumbent re-baselined)
cycle 3235: epoch e420 → e421 (benchmark upgraded, incumbent re-baselined)
cycle 3238: epoch e421 → e422 (benchmark upgraded, incumbent re-baselined)
cycle 3241: epoch e422 → e423 (benchmark upgraded, incumbent re-baselined)
cycle 3244: epoch e423 → e424 (benchmark upgraded, incumbent re-baselined)
cycle 3247: epoch e424 → e425 (benchmark upgraded, incumbent re-baselined)
cycle 3250: epoch e425 → e426 (benchmark upgraded, incumbent re-baselined)
cycle 3253: epoch e426 → e427 (benchmark upgraded, incumbent re-baselined)
cycle 3256: epoch e427 → e428 (benchmark upgraded, incumbent re-baselined)
cycle 3259: epoch e428 → e429 (benchmark upgraded, incumbent re-baselined)
cycle 3262: epoch e429 → e430 (benchmark upgraded, incumbent re-baselined)
cycle 3265: epoch e430 → e431 (benchmark upgraded, incumbent re-baselined)
cycle 3268: epoch e431 → e432 (benchmark upgraded, incumbent re-baselined)
cycle 3271: epoch e432 → e433 (benchmark upgraded, incumbent re-baselined)
cycle 3274: epoch e433 → e434 (benchmark upgraded, incumbent re-baselined)
cycle 3277: epoch e434 → e435 (benchmark upgraded, incumbent re-baselined)
cycle 3280: epoch e435 → e436 (benchmark upgraded, incumbent re-baselined)
cycle 3286: epoch e436 → e437 (benchmark upgraded, incumbent re-baselined)
cycle 3289: epoch e437 → e438 (benchmark upgraded, incumbent re-baselined)
cycle 3292: epoch e438 → e439 (benchmark upgraded, incumbent re-baselined)
cycle 3295: epoch e439 → e440 (benchmark upgraded, incumbent re-baselined)
cycle 3298: epoch e440 → e441 (benchmark upgraded, incumbent re-baselined)
cycle 3301: epoch e441 → e442 (benchmark upgraded, incumbent re-baselined)
cycle 3304: epoch e442 → e443 (benchmark upgraded, incumbent re-baselined)
cycle 3307: epoch e443 → e444 (benchmark upgraded, incumbent re-baselined)
cycle 3310: epoch e444 → e445 (benchmark upgraded, incumbent re-baselined)
cycle 3313: epoch e445 → e446 (benchmark upgraded, incumbent re-baselined)
cycle 3316: epoch e446 → e447 (benchmark upgraded, incumbent re-baselined)
cycle 3319: epoch e447 → e448 (benchmark upgraded, incumbent re-baselined)
cycle 3322: epoch e448 → e449 (benchmark upgraded, incumbent re-baselined)
cycle 3325: epoch e449 → e450 (benchmark upgraded, incumbent re-baselined)
cycle 3328: epoch e450 → e451 (benchmark upgraded, incumbent re-baselined)
cycle 3331: epoch e451 → e452 (benchmark upgraded, incumbent re-baselined)
cycle 3334: epoch e452 → e453 (benchmark upgraded, incumbent re-baselined)
cycle 3337: epoch e453 → e454 (benchmark upgraded, incumbent re-baselined)
cycle 3340: epoch e454 → e455 (benchmark upgraded, incumbent re-baselined)
cycle 3343: epoch e455 → e456 (benchmark upgraded, incumbent re-baselined)
cycle 3346: epoch e456 → e457 (benchmark upgraded, incumbent re-baselined)
cycle 3349: epoch e457 → e458 (benchmark upgraded, incumbent re-baselined)
cycle 3352: epoch e458 → e459 (benchmark upgraded, incumbent re-baselined)
cycle 3355: epoch e459 → e460 (benchmark upgraded, incumbent re-baselined)
cycle 3358: epoch e460 → e461 (benchmark upgraded, incumbent re-baselined)
cycle 3361: epoch e461 → e462 (benchmark upgraded, incumbent re-baselined)
cycle 3364: epoch e462 → e463 (benchmark upgraded, incumbent re-baselined)
cycle 3367: epoch e463 → e464 (benchmark upgraded, incumbent re-baselined)
cycle 3370: epoch e464 → e465 (benchmark upgraded, incumbent re-baselined)
cycle 3373: epoch e465 → e466 (benchmark upgraded, incumbent re-baselined)
cycle 3376: epoch e466 → e467 (benchmark upgraded, incumbent re-baselined)
cycle 3379: epoch e467 → e468 (benchmark upgraded, incumbent re-baselined)
cycle 3382: epoch e468 → e469 (benchmark upgraded, incumbent re-baselined)
cycle 3385: epoch e469 → e470 (benchmark upgraded, incumbent re-baselined)
cycle 3388: epoch e470 → e471 (benchmark upgraded, incumbent re-baselined)
cycle 3391: epoch e471 → e472 (benchmark upgraded, incumbent re-baselined)
cycle 3394: epoch e472 → e473 (benchmark upgraded, incumbent re-baselined)
cycle 3397: epoch e473 → e474 (benchmark upgraded, incumbent re-baselined)
cycle 3400: epoch e474 → e475 (benchmark upgraded, incumbent re-baselined)
cycle 3403: epoch e475 → e476 (benchmark upgraded, incumbent re-baselined)
cycle 3406: epoch e476 → e477 (benchmark upgraded, incumbent re-baselined)
cycle 3409: epoch e477 → e478 (benchmark upgraded, incumbent re-baselined)
cycle 3412: epoch e478 → e479 (benchmark upgraded, incumbent re-baselined)
cycle 3415: epoch e479 → e480 (benchmark upgraded, incumbent re-baselined)
cycle 3418: epoch e480 → e481 (benchmark upgraded, incumbent re-baselined)
cycle 3421: epoch e481 → e482 (benchmark upgraded, incumbent re-baselined)
cycle 3424: epoch e482 → e483 (benchmark upgraded, incumbent re-baselined)
cycle 3427: epoch e483 → e484 (benchmark upgraded, incumbent re-baselined)
cycle 3430: epoch e484 → e485 (benchmark upgraded, incumbent re-baselined)
cycle 3433: epoch e485 → e486 (benchmark upgraded, incumbent re-baselined)
cycle 3436: epoch e486 → e487 (benchmark upgraded, incumbent re-baselined)
cycle 3439: epoch e487 → e488 (benchmark upgraded, incumbent re-baselined)
cycle 3442: epoch e488 → e489 (benchmark upgraded, incumbent re-baselined)
cycle 3445: epoch e489 → e490 (benchmark upgraded, incumbent re-baselined)
cycle 3448: epoch e490 → e491 (benchmark upgraded, incumbent re-baselined)
cycle 3451: epoch e491 → e492 (benchmark upgraded, incumbent re-baselined)
cycle 3454: epoch e492 → e493 (benchmark upgraded, incumbent re-baselined)
cycle 3457: epoch e493 → e494 (benchmark upgraded, incumbent re-baselined)
cycle 3460: epoch e494 → e495 (benchmark upgraded, incumbent re-baselined)
cycle 3463: epoch e495 → e496 (benchmark upgraded, incumbent re-baselined)
cycle 3466: epoch e496 → e497 (benchmark upgraded, incumbent re-baselined)
cycle 3469: epoch e497 → e498 (benchmark upgraded, incumbent re-baselined)
cycle 3472: epoch e498 → e499 (benchmark upgraded, incumbent re-baselined)
cycle 3475: epoch e499 → e500 (benchmark upgraded, incumbent re-baselined)
cycle 3478: epoch e500 → e501 (benchmark upgraded, incumbent re-baselined)
cycle 3481: epoch e501 → e502 (benchmark upgraded, incumbent re-baselined)
cycle 3484: epoch e502 → e503 (benchmark upgraded, incumbent re-baselined)
cycle 3487: epoch e503 → e504 (benchmark upgraded, incumbent re-baselined)
cycle 3490: epoch e504 → e505 (benchmark upgraded, incumbent re-baselined)
cycle 3493: epoch e505 → e506 (benchmark upgraded, incumbent re-baselined)
cycle 3505: epoch e506 → e507 (benchmark upgraded, incumbent re-baselined)
cycle 3508: epoch e507 → e508 (benchmark upgraded, incumbent re-baselined)
cycle 3511: epoch e508 → e509 (benchmark upgraded, incumbent re-baselined)
cycle 3514: epoch e509 → e510 (benchmark upgraded, incumbent re-baselined)
cycle 3517: epoch e510 → e511 (benchmark upgraded, incumbent re-baselined)
cycle 3520: epoch e511 → e512 (benchmark upgraded, incumbent re-baselined)
cycle 3523: epoch e512 → e513 (benchmark upgraded, incumbent re-baselined)
cycle 3525: question q21 → q22
cycle 3529: epoch e513 → e514 (benchmark upgraded, incumbent re-baselined)
cycle 3634: epoch e514 → e515 (benchmark upgraded, incumbent re-baselined)
cycle 3637: epoch e515 → e516 (benchmark upgraded, incumbent re-baselined)
cycle 3640: epoch e516 → e517 (benchmark upgraded, incumbent re-baselined)
cycle 3643: epoch e517 → e518 (benchmark upgraded, incumbent re-baselined)
cycle 3646: epoch e518 → e519 (benchmark upgraded, incumbent re-baselined)
cycle 3649: epoch e519 → e520 (benchmark upgraded, incumbent re-baselined)
cycle 3661: epoch e520 → e521 (benchmark upgraded, incumbent re-baselined)
cycle 3670: epoch e521 → e522 (benchmark upgraded, incumbent re-baselined)
cycle 3673: epoch e522 → e523 (benchmark upgraded, incumbent re-baselined)
cycle 3676: epoch e523 → e524 (benchmark upgraded, incumbent re-baselined)
cycle 3694: epoch e524 → e525 (benchmark upgraded, incumbent re-baselined)
cycle 3697: epoch e525 → e526 (benchmark upgraded, incumbent re-baselined)
cycle 3700: epoch e526 → e527 (benchmark upgraded, incumbent re-baselined)
cycle 3703: epoch e527 → e528 (benchmark upgraded, incumbent re-baselined)
cycle 3706: epoch e528 → e529 (benchmark upgraded, incumbent re-baselined)
cycle 3709: epoch e529 → e530 (benchmark upgraded, incumbent re-baselined)
cycle 3712: epoch e530 → e531 (benchmark upgraded, incumbent re-baselined)
cycle 3721: epoch e531 → e532 (benchmark upgraded, incumbent re-baselined)
cycle 3724: epoch e532 → e533 (benchmark upgraded, incumbent re-baselined)
cycle 3727: epoch e533 → e534 (benchmark upgraded, incumbent re-baselined)
cycle 3730: epoch e534 → e535 (benchmark upgraded, incumbent re-baselined)
cycle 3733: epoch e535 → e536 (benchmark upgraded, incumbent re-baselined)
cycle 3736: epoch e536 → e537 (benchmark upgraded, incumbent re-baselined)
cycle 3739: epoch e537 → e538 (benchmark upgraded, incumbent re-baselined)
cycle 3742: epoch e538 → e539 (benchmark upgraded, incumbent re-baselined)
cycle 3745: epoch e539 → e540 (benchmark upgraded, incumbent re-baselined)
cycle 3748: epoch e540 → e541 (benchmark upgraded, incumbent re-baselined)
cycle 3751: question q22 → q23
cycle 3835: question q23 → q24
cycle 3871: epoch e541 → e542 (benchmark upgraded, incumbent re-baselined)
cycle 3877: epoch e542 → e543 (benchmark upgraded, incumbent re-baselined)
cycle 3880: epoch e543 → e544 (benchmark upgraded, incumbent re-baselined)
cycle 3883: epoch e544 → e545 (benchmark upgraded, incumbent re-baselined)
cycle 3886: epoch e545 → e546 (benchmark upgraded, incumbent re-baselined)
cycle 3889: epoch e546 → e547 (benchmark upgraded, incumbent re-baselined)
cycle 3892: epoch e547 → e548 (benchmark upgraded, incumbent re-baselined)
cycle 3919: question q24 → q25
cycle 3922: epoch e548 → e549 (benchmark upgraded, incumbent re-baselined)
cycle 3925: epoch e549 → e550 (benchmark upgraded, incumbent re-baselined)
cycle 3928: epoch e550 → e551 (benchmark upgraded, incumbent re-baselined)
cycle 3931: epoch e551 → e552 (benchmark upgraded, incumbent re-baselined)
cycle 3934: epoch e552 → e553 (benchmark upgraded, incumbent re-baselined)
cycle 3937: epoch e553 → e554 (benchmark upgraded, incumbent re-baselined)
cycle 3952: epoch e554 → e555 (benchmark upgraded, incumbent re-baselined)
cycle 3955: epoch e555 → e556 (benchmark upgraded, incumbent re-baselined)
cycle 3958: epoch e556 → e557 (benchmark upgraded, incumbent re-baselined)
cycle 3961: epoch e557 → e558 (benchmark upgraded, incumbent re-baselined)
cycle 3964: epoch e558 → e559 (benchmark upgraded, incumbent re-baselined)
cycle 4009: question q25 → q26
cycle 4042: epoch e559 → e560 (benchmark upgraded, incumbent re-baselined)
cycle 4045: epoch e560 → e561 (benchmark upgraded, incumbent re-baselined)
cycle 4092: question q26 → q27
cycle 4123: epoch e561 → e562 (benchmark upgraded, incumbent re-baselined)
cycle 4153: epoch e562 → e563 (benchmark upgraded, incumbent re-baselined)
cycle 4156: epoch e563 → e564 (benchmark upgraded, incumbent re-baselined)
cycle 4159: epoch e564 → e565 (benchmark upgraded, incumbent re-baselined)
cycle 4177: epoch e565 → e566 (benchmark upgraded, incumbent re-baselined)
cycle 4180: epoch e566 → e567 (benchmark upgraded, incumbent re-baselined)
cycle 4186: question q27 → q28
```

## Final knowledge state

| question | status | saturation | best |
|---|---|---|---|
| iris-ood-q0 | partially_answered | 0.00 | 0.4222 |
| iris-ood-q1 | partially_answered | 0.85 | 0.8444 |
| iris-ood-q2 | partially_answered | 0.82 | 0.9111 |
| iris-ood-q3 | partially_answered | 0.00 | 0.9111 |
| iris-ood-q4 | partially_answered | 0.00 | 0.9111 |
| iris-ood-q5 | partially_answered | 0.00 | 0.8000 |
| iris-ood-q6 | partially_answered | 1.00 | 0.6667 |
| iris-ood-q7 | partially_answered | 0.33 | 0.6667 |
| iris-ood-q8 | partially_answered | 0.00 | 0.4889 |
| iris-ood-q9 | partially_answered | 0.33 | 0.7111 |
| iris-ood-q10 | partially_answered | 0.00 | 0.7333 |
| iris-ood-q11 | partially_answered | 1.00 | 0.7111 |
| iris-ood-q12 | partially_answered | 0.00 | 0.6000 |
| iris-ood-q13 | partially_answered | 0.00 | 0.6444 |
| iris-ood-q14 | partially_answered | 0.00 | 0.6222 |
| iris-ood-q15 | partially_answered | 1.00 | 0.9778 |
| iris-ood-q16 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q17 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q18 | partially_answered | 0.00 | 0.7111 |
| iris-ood-q19 | partially_answered | 0.71 | 0.8222 |
| iris-ood-q20 | partially_answered | 1.00 | 0.7333 |
| iris-ood-q21 | partially_answered | 0.00 | 0.9333 |
| iris-ood-q22 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q23 | partially_answered | 0.17 | 0.7333 |
| iris-ood-q24 | partially_answered | 1.00 | 0.9333 |
| iris-ood-q25 | partially_answered | 0.61 | 0.8667 |
| iris-ood-q26 | partially_answered | 0.36 | 0.6000 |
| iris-ood-q27 | partially_answered | 0.00 | 0.6667 |
| iris-ood-q28 | open | 0.00 | 0.6667 |

| hypothesis | status | confidence | support/refute |
|---|---|---|---|
| iris-ood-q0-h0 | refuted | 0.84 | 1/2 |
| iris-ood-q0-h1 | refuted | 0.00 | 0/2 |
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
| … (1734 more hypotheses) | | | |

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
PRIORITY = ["MOMENTUM", "LR", "CLIP"]
EXPLORE_WIDTH = 6
# SELF-IMPROVE-MUTABLE>>>
```

---

*Generated by `scripts/report.py` (git `215e86d`) from sessions: main=sessions/naive-seed7; control=sessions/noop-seed7.*
