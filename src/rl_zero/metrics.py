"""Statistics in units humans can inspect: task pass@k and paired percentage points."""

import math
import random
import statistics


def pass_at_k(n, correct, k):
    """Unbiased estimator 1 - C(n-c,k)/C(n,k), evaluated without huge factorials."""
    if not all(isinstance(v, int) for v in (n, correct, k)) or not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError("Require integers 0 <= correct <= n and 1 <= k <= n")
    if n - correct < k:
        return 1.0
    return 1 - math.prod((n - correct - i) / (n - i) for i in range(k))


def paired_changes(before, after, bootstrap_samples=2000, seed=42):
    if not before or before.keys() != after.keys():
        raise ValueError("Paired comparisons require the same nonempty example IDs")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    keys = sorted(before)
    values = [*before.values(), *after.values()]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Expected finite scores in [0, 1]")
    differences = [after[key] - before[key] for key in keys]
    rng = random.Random(seed)
    samples = sorted(
        statistics.mean(rng.choices(differences, k=len(keys))) * 100 for _ in range(bootstrap_samples)
    )
    return {
        "n": len(keys),
        "before_pct": statistics.mean(before.values()) * 100,
        "after_pct": statistics.mean(after.values()) * 100,
        "delta_pp": statistics.mean(differences) * 100,
        "delta_ci95_pp": [
            samples[int(0.025 * (bootstrap_samples - 1))],
            samples[int(0.975 * (bootstrap_samples - 1))],
        ],
        "regressed": sum(d < 0 for d in differences),
        "improved": sum(d > 0 for d in differences),
        "unchanged": sum(d == 0 for d in differences),
    }
