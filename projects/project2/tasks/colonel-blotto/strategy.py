"""Allocation strategy — the ONLY file the researcher (or the RSI system) edits.

Baseline: spread the budget evenly across all battlefields, no rotation.
Against the published countering opponent, a static concentration is read
and punished after the warmup — the headroom is a strategy that rotates
which fields it contests (or concentrates on fields the counter never
targets: the lowest-value ones the opponent is NOT told about).
"""

N_FIELDS = 12.0     # how many battlefields to contest
SPREAD = 0.0        # random jitter amplitude when splitting the budget (0 = even)
MARGIN = 1.0        # overshoot factor on contested fields (clipped at budget)
ROTATE_EVERY = 0.0  # games between rotating the contested block (0 = never)
ROTATE_BY = 3.0     # rotation stride (fields)
CENTER_BIAS = 0.0   # decoy: shift allocations toward middle fields
EVEN_TAIL = 0.0     # decoy: flatten tail fields
FIXED_SEED = 0.0    # decoy: reseed the rng each game (changes draws only)


def allocate(rng, game_idx):
    """Return a list of 12 non-negative allocations summing to <= 120."""
    k = max(1, min(12, int(N_FIELDS)))
    weights = [1.0 + SPREAD * rng.random() for _ in range(k)]
    scale = (120.0 * MARGIN) / sum(weights)
    if scale > 120.0 / sum(weights):  # budget cap
        scale = 120.0 / sum(weights)
    if ROTATE_EVERY >= 1.0:
        start = (int(game_idx // ROTATE_EVERY) * int(ROTATE_BY)) % 12
    else:
        start = 12 - k
    alloc = [0.0] * 12
    for i in range(k):
        alloc[(start + i) % 12] = weights[i] * scale
    return alloc
