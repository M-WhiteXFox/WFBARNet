from __future__ import annotations

from math import hypot
from typing import Mapping


class PlayerDistanceAccumulator:
    """Accumulates player travel distance from court-plane pose projections."""

    PLAYER_KEYS = ("top", "bottom")

    def __init__(
        self,
        *,
        min_step_cm: float = 2.0,
        max_step_cm: float = 180.0,
    ) -> None:
        self._min_step_cm = max(0.0, float(min_step_cm))
        self._max_step_cm = max(self._min_step_cm, float(max_step_cm))
        self._last_points: list[tuple[float, float] | None] = [None, None]
        self._totals_cm = [0.0, 0.0]

    def update(self, points_by_player: Mapping[int, tuple[float, float]]) -> dict[str, float]:
        seen = [False, False]
        for person_index, point in points_by_player.items():
            if person_index < 0 or person_index >= len(self.PLAYER_KEYS):
                continue
            if not self._valid_point(point):
                continue
            self._accumulate_person_step(person_index, point)
            seen[person_index] = True

        for index, is_seen in enumerate(seen):
            if not is_seen:
                self._last_points[index] = None
        return self.totals_m()

    def reset_tracking_points(self) -> None:
        self._last_points = [None, None]

    def totals_m(self) -> dict[str, float]:
        return {
            key: self._totals_cm[index] / 100.0
            for index, key in enumerate(self.PLAYER_KEYS)
        }

    def _accumulate_person_step(self, person_index: int, point: tuple[float, float]) -> None:
        last_point = self._last_points[person_index]
        if last_point is None:
            self._last_points[person_index] = point
            return

        step_cm = hypot(point[0] - last_point[0], point[1] - last_point[1])
        if step_cm > self._max_step_cm:
            self._last_points[person_index] = point
            return
        if step_cm >= self._min_step_cm:
            self._totals_cm[person_index] += step_cm
            self._last_points[person_index] = point

    @staticmethod
    def _valid_point(point: object) -> bool:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return False
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return False
        return _is_finite(x) and _is_finite(y)


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
