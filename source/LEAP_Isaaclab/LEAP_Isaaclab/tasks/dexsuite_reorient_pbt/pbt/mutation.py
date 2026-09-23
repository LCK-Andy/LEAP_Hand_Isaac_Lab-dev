# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import random
from collections.abc import Callable
from typing import Any


def mutate_float(x: float, change_min: float = 1.1, change_max: float = 1.5, rng=None) -> float:
    """Multiply or divide by a random factor in [change_min, change_max]."""
    rng = rng or random
    k = rng.uniform(change_min, change_max)
    return x / k if rng.random() < 0.5 else x * k


def mutate_discount(x: float, **kwargs) -> float:
    """Conservative change near 1.0 by mutating (1 - x) in [1.1, 1.2]."""
    inv = 1.0 - x
    new_inv = mutate_float(inv, change_min=1.1, change_max=1.2, rng=kwargs.get("rng"))
    return 1.0 - new_inv


MUTATION_FUNCS: dict[str, Callable[..., Any]] = {
    "mutate_float": mutate_float,
    "mutate_discount": mutate_discount,
}


def mutate(
    params: dict[str, Any],
    mutations: dict[str, str],
    mutation_rate: float,
    change_range: tuple[float, float],
    bounds: dict[str, tuple[float, float]],
    *,
    ensure_change: bool = True,
    rng=None,
) -> dict[str, Any]:
    import math

    rng = rng or random
    cmin, cmax = change_range
    out = dict(params)
    candidates = []
    for name, val in params.items():
        if name not in mutations:
            continue
        if mutations[name] not in MUTATION_FUNCS:
            raise ValueError(f"Unknown mutation function: {mutations[name]}")
        lo, hi = bounds[name]
        if not all(math.isfinite(float(v)) for v in (lo, hi, val)) or lo < 0 or lo >= hi:
            raise ValueError(f"Invalid bounds/value for {name}")
        # Clamp adaptive LR drift before evolving; zero-valued entropy stays zero.
        base = max(lo, min(hi, val))
        candidates.append((name, base, lo, hi))
        if rng.random() < mutation_rate:
            value = MUTATION_FUNCS[mutations[name]](base, change_min=cmin, change_max=cmax, rng=rng)
            out[name] = max(lo, min(hi, value))
        else:
            out[name] = base
    if ensure_change and out == params:
        movable = [item for item in candidates if item[1] > 0]
        if not movable:
            raise ValueError("No nonzero bounded parameter available for diversity")
        name, base, lo, hi = rng.choice(movable)
        value = base * cmin if base * cmin <= hi else base / cmin
        out[name] = max(lo, min(hi, value))
        if out[name] == params[name]:
            out[name] = lo if base != lo else hi
    return out
