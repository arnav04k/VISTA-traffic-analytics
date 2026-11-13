"""Image-space speed estimates from a user-supplied pixel-to-meter scale.

A single global pixel-to-meter scale assumes a flat, roughly camera-facing road.
The result is a ground-plane proxy, not a calibrated speed measurement.
"""

from __future__ import annotations

from collections import defaultdict
from math import hypot


class SpeedEstimator:
    """Estimate per-track speed in km/h from a pixel-to-meter reference scale."""

    def __init__(self, px_per_meter: float, *, window_s: float = 0.5, min_samples: int = 2):
        if px_per_meter <= 0:
            raise ValueError("px_per_meter must be positive")
        self.px_per_meter = px_per_meter
        self.window_s = window_s
        self.min_samples = min_samples
        self._tracks: dict[int, list[tuple[float, float, float]]] = {}
        self.samples: list[dict] = []

    def update(self, track_id: int, time_s: float, x: float, y: float, vehicle_class: str) -> float | None:
        """Return instantaneous speed in km/h, or None when it cannot be estimated."""
        history = self._tracks.setdefault(track_id, [])
        history.append((time_s, x, y))
        while len(history) > 1 and time_s - history[0][0] > self.window_s:
            history.pop(0)
        if len(history) < self.min_samples:
            return None
        t0, x0, y0 = history[0]
        dt = time_s - t0
        if dt <= 0:
            return None
        dist_m = hypot(x - x0, y - y0) / self.px_per_meter
        speed_kmh = round(dist_m / dt * 3.6, 1)
        self.samples.append({
            "time_s": round(time_s, 3),
            "track_id": track_id,
            "vehicle_class": vehicle_class,
            "speed_kmh": speed_kmh,
        })
        return speed_kmh

    def result(self) -> dict:
        """Per-track aggregates and a 5-second speed series for charting."""
        by_track: dict[int, dict] = defaultdict(lambda: {"speeds": [], "class": None})
        for sample in self.samples:
            entry = by_track[sample["track_id"]]
            entry["speeds"].append(sample["speed_kmh"])
            entry["class"] = sample["vehicle_class"]
        per_track = [
            {
                "track_id": track_id,
                "vehicle_class": entry["class"],
                "mean_kmh": round(sum(entry["speeds"]) / len(entry["speeds"]), 1),
                "max_kmh": round(max(entry["speeds"]), 1),
            }
            for track_id, entry in by_track.items()
        ]
        per_track.sort(key=lambda row: row["max_kmh"], reverse=True)

        bins: dict[int, list[float]] = defaultdict(list)
        for sample in self.samples:
            bins[int(sample["time_s"] // 5) * 5].append(sample["speed_kmh"])
        duration = max((sample["time_s"] for sample in self.samples), default=0.0)
        series = []
        for second in range(0, max(5, int(duration) + 5), 5):
            values = bins.get(second, [])
            series.append({
                "start": second,
                "max_kmh": round(max(values), 1) if values else 0.0,
                "mean_kmh": round(sum(values) / len(values), 1) if values else 0.0,
            })
        return {"per_track": per_track, "series": series}
