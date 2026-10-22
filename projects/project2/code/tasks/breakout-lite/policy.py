"""Paddle controller — the ONLY file the researcher (or the RSI system) edits.

Baseline: a lazy tracker — the paddle chases the ball's CURRENT x position
with limited speed and no anticipation, so any descending ball with lateral
speed outruns it. Anticipating the landing point is the research headroom.
"""

LEAD_GAIN = 0.0    # weight on the extrapolated landing x (0 = no anticipation)
CENTER_PULL = 0.0  # pull toward field center when the ball moves away
GAIN = 0.3         # commanded paddle speed toward the target (<= 1 cell/step)
AIM_OFFSET = 0.0   # decoy: constant offset on the landing estimate
REACT_DELAY = 0.0  # decoy: act on a delayed observation
WALL_HUG = 0.0     # decoy: bias toward the wall the ball is nearest


def init():
    return {}


def paddle_target(ball_x, ball_y, ball_vx, ball_vy, paddle_cx, state, t):
    """Return the desired paddle center x for step ``t``.

    The harness moves the paddle at most 1 cell per step toward this target
    (scaled by the enforced speed cap in the physics; see run.py).
    """
    # dead-reckoned landing x: how far the ball drifts while falling to the
    # paddle line, ignoring wall bounces (a published simplification)
    fall = max(ball_y - 1.0, 0.0)
    vy = min(ball_vy, -1e-6) if ball_vy < 0 else -1e-6
    lead = ball_x + ball_vx / abs(vy) * fall * LEAD_GAIN
    target = lead + CENTER_PULL * (6.0 - lead if ball_vy > 0 else 0.0)
    # move toward the target at GAIN speed
    return paddle_cx + max(-1.0, min(1.0, GAIN)) * max(-1.0, min(1.0, target - paddle_cx))
