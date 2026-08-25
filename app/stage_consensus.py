from __future__ import annotations

from collections.abc import Mapping


def choose_stage_by_count_then_confidence(
    counts: Mapping[int, int],
    confidence_sums: Mapping[int, float] | None = None,
) -> tuple[int, float, dict[int, float]]:
    """Choose count mode, then average confidence, then higher stage.

    Vehicle count is always authoritative when one stage has more vehicles.
    Confidence is consulted only among stages tied for the largest count.
    """
    normalised_counts = {
        level: max(0, int(counts.get(level, 0)))
        for level in range(5)
    }
    sums = confidence_sums or {}
    averages = {
        level: (
            max(0.0, float(sums.get(level, 0.0)))
            / float(normalised_counts[level])
            if normalised_counts[level] > 0
            else 0.0
        )
        for level in range(5)
    }

    max_count = max(normalised_counts.values(), default=0)
    if max_count <= 0:
        return 0, 0.0, averages

    candidates = [
        level
        for level, count in normalised_counts.items()
        if count == max_count
    ]
    winner = max(
        candidates,
        key=lambda level: (averages[level], level),
    )
    return int(winner), float(averages[winner]), averages

