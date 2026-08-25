from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
import time


def qualifies_stage_confidence(
    confidence: int | float | None,
    minimum: int | float = 0.70,
) -> bool:
    """True only when a model stage result meets the inclusive threshold."""
    try:
        value = float(confidence or 0.0)
        threshold = max(0.0, min(1.0, float(minimum)))
    except (TypeError, ValueError):
        return False
    return value >= threshold


def is_authoritative_stage_record(
    stage: int | float | None,
    details: dict | None,
) -> bool:
    """Reject every positive DB row that lacks explicit confirmation."""
    try:
        resolved = max(0, min(4, int(stage or 0)))
    except (TypeError, ValueError):
        return False
    return resolved == 0 or bool((details or {}).get("positive_confirmed"))


@dataclass
class _Candidate:
    stage: int
    hits: int
    first_at: float
    last_at: float


class PositiveFloodConfirmation:
    """Conservative multi-frame authority gate for public CCTV positives."""

    def __init__(
        self,
        *,
        required_hits: int = 5,
        minimum_duration_seconds: float = 3.0,
        maximum_gap_seconds: float = 10.0,
        minimum_positive_vehicles: int = 2,
        minimum_positive_ratio: float = 0.75,
        minimum_confidence: float = 0.70,
    ) -> None:
        self.required_hits = max(2, int(required_hits))
        self.minimum_duration_seconds = max(0.0, float(minimum_duration_seconds))
        self.maximum_gap_seconds = max(0.5, float(maximum_gap_seconds))
        self.minimum_positive_vehicles = max(1, int(minimum_positive_vehicles))
        self.minimum_positive_ratio = max(0.5, min(1.0, float(minimum_positive_ratio)))
        self.minimum_confidence = max(0.0, min(1.0, float(minimum_confidence)))
        self._states: dict[str, _Candidate] = {}
        self._lock = Lock()

    def evaluate(
        self,
        key: str,
        stage: int | None,
        confidence: float,
        *,
        positive_votes: int,
        total_votes: int,
        trusted_test: bool = False,
        now: float | None = None,
    ) -> dict:
        timestamp = time.monotonic() if now is None else float(now)
        resolved_stage = None if stage is None else max(0, min(4, int(stage)))
        votes = max(0, int(positive_votes))
        total = max(0, int(total_votes))
        ratio = votes / total if total else 0.0

        with self._lock:
            if resolved_stage is None:
                self._states.pop(key, None)
                return {"accepted": False, "pending": False, "hits": 0, "reason": "no_stage"}
            if float(confidence or 0.0) < self.minimum_confidence:
                self._states.pop(key, None)
                return {
                    "accepted": False, "pending": False, "hits": 0,
                    "reason": "below_minimum_confidence",
                }
            if resolved_stage == 0:
                self._states.pop(key, None)
                return {"accepted": True, "pending": False, "hits": 1, "reason": "normal_immediate"}
            if trusted_test:
                self._states.pop(key, None)
                return {"accepted": True, "pending": False, "hits": 1, "reason": "trusted_flood_test"}
            if votes < self.minimum_positive_vehicles or ratio < self.minimum_positive_ratio:
                self._states.pop(key, None)
                return {
                    "accepted": False, "pending": False, "hits": 0,
                    "reason": "insufficient_positive_vehicle_agreement",
                    "positive_ratio": round(ratio, 3),
                }

            candidate = self._states.get(key)
            if (
                candidate is None
                or candidate.stage != resolved_stage
                or timestamp - candidate.last_at > self.maximum_gap_seconds
            ):
                candidate = _Candidate(resolved_stage, 1, timestamp, timestamp)
            else:
                candidate.hits += 1
                candidate.last_at = timestamp
            self._states[key] = candidate

            duration = max(0.0, candidate.last_at - candidate.first_at)
            accepted = (
                candidate.hits >= self.required_hits
                and duration >= self.minimum_duration_seconds
            )
            return {
                "accepted": accepted,
                "pending": not accepted,
                "hits": candidate.hits,
                "required_hits": self.required_hits,
                "duration_seconds": round(duration, 3),
                "positive_ratio": round(ratio, 3),
                "reason": "confirmed" if accepted else "awaiting_repeated_confirmation",
            }
