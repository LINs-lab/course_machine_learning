# Figures

Conventions for every figure in this paper:

- **Vector PDF only.** No PNG/JPG screenshots; rasterized plots are an instant quality red flag at AI venues.
- **Readable at final size.** In-figure text no smaller than ~7pt when the figure is rendered at its printed width.
- **Uncertainty shown.** Trajectory curves carry a seed-variance band (mean +/- std or sem); bar charts carry error bars.
- **Self-contained captions.** A reader who only flips through figures should still get the main story.
- **Traceable.** Each figure maps to an experiment directory and a command; record that mapping here as you add figures.

Suggested starting set (already produced by the repo's report pipeline):

| File | Source | Purpose |
|---|---|---|
| `system.pdf` | hand-drawn | Fig 1: three loops + self-edit circuit (Section 3) |
| `trajectory.pdf` | `experiments/.../figures/` | best-result trajectories, naive vs yours, with variance band |
| `keeprate.pdf` | `experiments/.../figures/` | keep-rate curves (Section 5/6) |
| `selfedit_anatomy.pdf` | hand-assembled from run logs | the promoted-edit timeline (Section 6.2) |
| `question_graph.pdf` | `experiments/.../figures/` | question-graph evolution snapshot |
