"""Strategy — the ONLY file the researcher (or the RSI system) edits.

Baseline: an uninformed coin-flip strategy. It neither reads the opponent
nor tracks its adaptation — so it collects the mismatch payoff half the
time. The headroom is a tracking strategy. (v3: the knob surface carries
decoys — not every knob affects the game; evidence should tell.)
"""

EXPLOIT = 0.0        # probability of playing the opponent's tracked modal move
EXPLOIT_AFTER = 4.0  # rounds of history required before estimating the mode
TRACK_W = 6.0        # window length for the opponent-frequency estimate
OWN_BIAS_OPERA = 0.5  # otherwise: probability of playing O (0.5 = coin flip)
STICKINESS = 0.0     # probability of repeating own last move
JITTER = 0.0         # probability of a random flip after the modal estimate
EXPLORE_EVERY = 7.0  # rounds between forced random probes
EMA_ALPHA = 0.5      # smoothing on the opponent-frequency estimate


def init():
    return {}


def move(hist_own, hist_opp, rng, state, t):
    """Return "O" or "F" for round ``t``.

    ``hist_own``/``hist_opp`` are the moves played so far (this episode);
    ``rng`` is a seeded random generator owned by the harness; ``state`` is
    whatever ``init`` returned (per-episode scratch space).
    """
    if EXPLOIT > 0.0 and len(hist_opp) >= int(EXPLOIT_AFTER) and rng.random() < EXPLOIT:
        w = int(max(1.0, TRACK_W))
        recent = hist_opp[-w:]
        counts = {"O": recent.count("O"), "F": recent.count("F")}
        modal = "O" if counts["O"] >= counts["F"] else "F"
        return modal
    return "O" if rng.random() < OWN_BIAS_OPERA else "F"
