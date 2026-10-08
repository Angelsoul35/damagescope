"""Estimate damage severity and a rough repair cost.

Severity comes from how much of the panel is damaged (the damaged-area fraction)
and how many separate regions there are. The cost is a simple synthetic model for
the demo, not a real estimate.
"""

from __future__ import annotations


def assess(
    total_damage_area: int,
    image_area: int,
    n_regions: int,
    medium_threshold: float = 0.03,
    high_threshold: float = 0.08,
) -> dict:
    fraction = total_damage_area / image_area if image_area else 0.0
    if fraction >= high_threshold:
        level = "high"
    elif fraction >= medium_threshold:
        level = "medium"
    elif n_regions > 0:
        level = "low"
    else:
        level = "none"

    # Synthetic cost model (COP): a base per damaged region plus an area term.
    estimated_cost = n_regions * 150_000 + fraction * image_area * 2_000
    return {
        "damage_area_fraction": round(fraction, 4),
        "n_regions": n_regions,
        "severity": level,
        "estimated_cost_cop": round(float(estimated_cost), 0),
    }
