"""Strategy — the ONLY file the researcher (or the RSI system) edits.

Baseline: unconditional defection. Against the published trust dynamics this
collapses the opponent's trust after two defections and the episode decays
to (D,D). The headroom is a strategy that spends trust at a controlled rate
and repays it — a schedule, not a single switch. (v3: the knob surface
carries decoys — not every knob affects the game; evidence should tell.)
"""

COOP_RATE = 1.0     # probability of cooperating while in the coop phase
DEFECT_PERIOD = 1.0 # defect every P-th round (1 = always defect)
REPAY = 0.0         # rounds of pure cooperation after each defection
MIRROR_P = 0.0      # probability of copying the opponent's last move
OPENING = 1.0       # cooperation probability in the first two rounds
REBUILD_P = 0.0     # extra cooperation boost while trust is low


def init():
    return {"repay_left": 0}


def move(hist_own, hist_opp, rng, state, t):
    """Return "C" or "D" for round ``t`` (0-based)."""
    if state["repay_left"] > 0:
        state["repay_left"] -= 1
        return "C"
    p = max(1, int(DEFECT_PERIOD))
    if t % p == p - 1:
        state["repay_left"] = int(REPAY)
        return "D"
    return "C" if rng.random() < COOP_RATE else "D"
