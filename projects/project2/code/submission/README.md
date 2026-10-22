# Submission Template -- Course 2610EST3000, Project 2

A complete, compilable LaTeX skeleton for the Project 2 report, typeset in the official NeurIPS 2025 format (preprint variant) and adapted for course submission at Westlake University: non-anonymous author block with student IDs, course ID in the running header, Canvas link in the footer, and the first-page notice repurposed as the course notice. Every `.tex` file opens with an annotated header explaining what belongs in that section, how much space it gets, and what graders look for.

Course identity (edit in `main.tex` only if the course staff announces a change):

- Course: **2610EST3000**, Project 2 (Autoresearch-Base)
- Canvas: <https://canvas.westlake.edu.cn/courses/2063>
- Individual work: the author block carries the single author's **name + student ID + Westlake email** (edit it in `main.tex`)

The compiled `main.pdf` opens with a one-page **Assessment Guide** (the grading rubric) inserted by the course staff. It is not part of the report body, does not count toward the page budget, and must not be removed or altered.

## Layout

```
submission/
├── main.tex                  % entry point; style options, section order, page budget
├── neurips_2025.sty          % official NeurIPS 2025 style (do not modify)
├── sections/
│   ├── 00_abstract.tex       % 4-move skeleton, 150-250 words
│   ├── 01_introduction.tex   % 6-paragraph logic chain + contribution list
│   ├── 02_related_work.tex   % theme paragraphs + mandatory positioning
│   ├── 03_system.tex         % the given base system (keep brief)
│   ├── 04_method.tex         % YOUR strategies -- the core, one subsection per seam
│   ├── 05_experiments.tex    % setup checklist + main table conventions
│   ├── 06_analysis.tex       % ablation, promoted-edit anatomy, failure modes
│   ├── 07_discussion.tex     % limitations + broader impact
│   └── 08_conclusion.tex     % 3-4 sentences, no new claims
├── appendix/
│   ├── a_checklist.tex       % NeurIPS Paper Checklist (mandatory)
│   └── b_reproducibility.tex % commands, seeds, configs, self-edit ledger
├── figures/README.md         % figure conventions
├── references.bib            % seed bibliography (related-work threads pre-wired)
└── Makefile                  % `make` builds main.pdf
```

## Build

```bash
cd submission && make        # produces main.pdf
make clean                   # removes build artifacts
```

## The standards encoded in this template

| Rule | Where enforced |
|---|---|
| Non-anonymous: every author with name + student ID + email | `main.tex` `\author` block |
| Course ID / project in running header, Canvas link in footer | `main.tex` fancyhdr block |
| First-page course notice (repurposed preprint notice) | `main.tex` `\@noticestring` |
| 9-page content target; refs/checklist/appendix free | `main.tex` header |
| NeurIPS Paper Checklist as self-review rubric | `appendix/a_checklist.tex` |
| booktabs three-line tables; bold only on significant wins | `05_experiments.tex` |
| Vector PDF figures, variance bands on curves | `figures/README.md` |
| natbib `\citep` / `\citet`; every citation used by a sentence | `02_related_work.tex` |
| N >= 3 seeds, mean +/- std; equal-budget comparisons | `05_experiments.tex` |
| A claim-evidence map: every contribution cited to a table/figure | `01_introduction.tex` |
| LLM-use disclosure (methods AND writing assistance) | checklist item 16 |
| `\todo{...}` markers must reach zero before submission | `main.tex` macros |

## Suggested writing order

1. `05_experiments.tex` setup + main table -- fixes what you can claim.
2. `04_method.tex` -- the strategies, at re-implementable granularity.
3. `06_analysis.tex` -- ablation + the promoted-edit anatomy (assignment core).
4. `01_introduction.tex` + `00_abstract.tex` -- last, once results are known.
5. `02_related_work.tex`, `07_discussion.tex`, `08_conclusion.tex`, appendices.
6. Final pass: zero TODOs, all student IDs present, checklist honesty check, page count.
