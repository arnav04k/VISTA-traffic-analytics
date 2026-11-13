"""Camera-coordinate event rules for the VISTA Phase 2 prototype.

Input is a stream of tracked vehicle points. A detector/tracker is deliberately
outside this module so the line and waiting rules can be tested independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import hypot
from typing import Literal


Direction = Literal["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"]


@dataclass(frozen=True)
class TrackPoint:
    time_s: float
    track_id: int
    x: float
    y: float
    vehicle_class: str = "vehicle"


@dataclass(frozen=True)
class Zone:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def contains(self, point: TrackPoint) -> bool:
        return self.x_min <= point.x <= self.x_max and self.y_min <= point.y <= self.y_max


@dataclass(frozen=True)
class CrossingEvent:
    time_s: float
    track_id: int
    direction: Direction
    vehicle_class: str


@dataclass(frozen=True)
class StopEvent:
    time_s: float
    track_id: int
    duration_s: float
    vehicle_class: str


@dataclass(frozen=True)
class ODPair:
    time_s: float
    track_id: int
    origin: str
    destination: str
    vehicle_class: str


@dataclass
class _TrackState:
    last: TrackPoint | None = None
    stable_side: int | None = None
    counted_directions: set[Direction] = field(default_factory=set)
    stationary_since: float | None = None
    stop_reported: bool = False
    motion_history: list[TrackPoint] = field(default_factory=list)


class EventEngine:
    """Record one crossing per track/direction and visible stops in a zone.

    All distances are image pixels. The stop result is a visible waiting proxy,
    not physical speed, total queue length, or the cause of a stop.
    """

    def __init__(
        self,
        line_x: float | None,
        stop_zone: Zone,
        *,
        line_y: float | None = None,
        line_margin_px: float = 5.0,
        max_stationary_speed_px_s: float = 2.0,
        min_stop_s: float = 2.0,
        active_track_age_s: float = 1.5,
        motion_window_s: float = 1.0,
        max_track_gap_s: float = 1.5,
    ) -> None:
        if line_margin_px < 0 or max_stationary_speed_px_s < 0 or min_stop_s <= 0 or motion_window_s <= 0 or max_track_gap_s <= 0:
            raise ValueError("Margins and speed must be nonnegative; times must be positive")
        if (line_x is None) == (line_y is None):
            raise ValueError("Set exactly one of line_x or line_y")
        self.line_x = line_x
        self.line_y = line_y
        self.stop_zone = stop_zone
        self.line_margin_px = line_margin_px
        self.max_stationary_speed_px_s = max_stationary_speed_px_s
        self.min_stop_s = min_stop_s
        self.active_track_age_s = active_track_age_s
        self.motion_window_s = motion_window_s
        self.max_track_gap_s = max_track_gap_s
        self._tracks: dict[int, _TrackState] = {}
        self.crossings: list[CrossingEvent] = []
        self.stops: list[StopEvent] = []

    def _side(self, point: TrackPoint) -> int:
        position = point.x if self.line_x is not None else point.y
        line = self.line_x if self.line_x is not None else self.line_y
        assert line is not None
        if position < line - self.line_margin_px:
            return -1
        if position > line + self.line_margin_px:
            return 1
        return 0

    def update(self, point: TrackPoint) -> list[CrossingEvent | StopEvent]:
        state = self._tracks.setdefault(point.track_id, _TrackState())
        previous = state.last
        if previous is not None and point.time_s <= previous.time_s:
            raise ValueError("Timestamps must increase for each track")
        if previous is not None and point.time_s - previous.time_s > self.max_track_gap_s:
            state = self._tracks[point.track_id] = _TrackState()
            previous = None

        emitted: list[CrossingEvent | StopEvent] = []
        side = self._side(point)
        if side != 0:
            if state.stable_side is None:
                state.stable_side = side
            elif side != state.stable_side:
                if self.line_x is not None:
                    direction: Direction = "left_to_right" if side == 1 else "right_to_left"
                else:
                    direction = "top_to_bottom" if side == 1 else "bottom_to_top"
                if direction not in state.counted_directions:
                    event = CrossingEvent(point.time_s, point.track_id, direction, point.vehicle_class)
                    self.crossings.append(event)
                    emitted.append(event)
                    state.counted_directions.add(direction)
                state.stable_side = side

        if self.stop_zone.contains(point):
            state.motion_history.append(point)
            while len(state.motion_history) > 1 and point.time_s - state.motion_history[0].time_s > self.motion_window_s:
                state.motion_history.pop(0)
            start = state.motion_history[0]
            elapsed = point.time_s - start.time_s
            if elapsed >= self.motion_window_s * 0.8 and hypot(point.x - start.x, point.y - start.y) / elapsed <= self.max_stationary_speed_px_s:
                if state.stationary_since is None:
                    state.stationary_since = start.time_s
                duration = point.time_s - state.stationary_since
                if duration >= self.min_stop_s and not state.stop_reported:
                    event = StopEvent(point.time_s, point.track_id, duration, point.vehicle_class)
                    self.stops.append(event)
                    emitted.append(event)
                    state.stop_reported = True
            else:
                state.stationary_since = None
                state.stop_reported = False
        else:
            state.motion_history.clear()
            state.stationary_since = None
            state.stop_reported = False

        state.last = point
        return emitted

    def stopped_count(self, time_s: float) -> int:
        """Number of recently observed tracks that currently satisfy the stop rule."""
        return sum(
            state.stop_reported
            and state.last is not None
            and 0 <= time_s - state.last.time_s <= self.active_track_age_s
            for state in self._tracks.values()
        )

    def result(self) -> dict:
        return {
            "crossings": [asdict(event) for event in self.crossings],
            "stops": [asdict(event) for event in self.stops],
        }


class ODEngine:
    """Record origin->destination pairs for tracks that cross two configured lines.

    A line is identified as "line1" or "line2". When a track crosses two distinct
    lines, the first is the origin and the second is the destination.
    """

    def __init__(
        self,
        line1_axis: str,
        line1_pos: float,
        line2_axis: str,
        line2_pos: float,
        *,
        margin_px: float = 5.0,
    ) -> None:
        if margin_px < 0:
            raise ValueError("margin_px must be nonnegative")
        self.lines = {"line1": (line1_axis, line1_pos), "line2": (line2_axis, line2_pos)}
        self.margin_px = margin_px
        self._tracks: dict[int, dict] = {}
        self.pairs: list[ODPair] = []

    def _side(self, axis: str, position: float, x: float, y: float) -> int:
        coordinate = x if axis == "x" else y
        if coordinate < position - self.margin_px:
            return -1
        if coordinate > position + self.margin_px:
            return 1
        return 0

    def update(self, point: TrackPoint) -> list[ODPair]:
        track = self._tracks.setdefault(point.track_id, {"sides": {}, "order": [], "emitted": False})
        for name, (axis, position) in self.lines.items():
            side = self._side(axis, position, point.x, point.y)
            previous = track["sides"].get(name)
            if side != 0 and previous is not None and side != previous:
                if name not in track["order"]:
                    track["order"].append(name)
            if side != 0:
                track["sides"][name] = side
        if len(track["order"]) >= 2 and not track["emitted"]:
            pair = ODPair(point.time_s, point.track_id, track["order"][0], track["order"][1], point.vehicle_class)
            self.pairs.append(pair)
            track["emitted"] = True
            return [pair]
        return []

    def result(self) -> dict:
        matrix: dict[str, dict[str, int]] = {}
        for pair in self.pairs:
            matrix.setdefault(pair.origin, {}).setdefault(pair.destination, 0)
            matrix[pair.origin][pair.destination] += 1
        return {
            "od_pairs": [asdict(pair) for pair in self.pairs],
            "od_matrix": matrix,
        }


@dataclass(frozen=True)
class SlowdownEvent:
    time_s: float
    track_id: int
    vehicle_class: str
    approach_px_s: float
    min_px_s: float
    ratio: float


class SlowdownEngine:
    """Flag relative slowdowns through a band without a real-world scale.

    A track's free-flow image speed just before entering the band is compared
    to its minimum speed inside the band. Because both are pixel speeds in the
    same camera view, the scale cancels out — no calibration is required. A
    slowdown is reported when the in-band speed falls to ``ratio`` (or less) of
    the approach speed.
    """

    def __init__(
        self,
        axis: str,
        start: float,
        end: float,
        *,
        ratio: float = 0.5,
        min_approach_px_s: float = 5.0,
        window_s: float = 0.5,
        approach_window_s: float = 1.5,
        min_samples: int = 2,
    ) -> None:
        if axis not in ("x", "y"):
            raise ValueError("axis must be x or y")
        if start >= end:
            raise ValueError("start must be less than end")
        if not 0.0 < ratio <= 1.0:
            raise ValueError("ratio must be in (0, 1]")
        if min_approach_px_s < 0 or window_s <= 0 or approach_window_s <= 0 or min_samples < 2:
            raise ValueError("invalid speed/window parameters")
        self.axis = axis
        self.start = start
        self.end = end
        self.ratio = ratio
        self.min_approach_px_s = min_approach_px_s
        self.window_s = window_s
        self.approach_window_s = approach_window_s
        self.min_samples = min_samples
        self._tracks: dict[int, dict] = {}
        self.events: list[SlowdownEvent] = []

    def _position(self, point: TrackPoint) -> float:
        return point.x if self.axis == "x" else point.y

    def _inside(self, position: float) -> bool:
        return self.start <= position <= self.end

    def _speed(self, history: list[tuple[float, float, float]], point: TrackPoint) -> float | None:
        if len(history) < self.min_samples:
            return None
        t0, x0, y0 = history[0]
        dt = point.time_s - t0
        if dt <= 0:
            return None
        return hypot(point.x - x0, point.y - y0) / dt

    def update(self, point: TrackPoint) -> list[SlowdownEvent]:
        track = self._tracks.setdefault(point.track_id, {
            "history": [], "approach": [], "entered": False,
            "entry_speed": None, "min_speed": None,
        })
        history = track["history"]
        history.append((point.time_s, point.x, point.y))
        while len(history) > 1 and point.time_s - history[0][0] > self.window_s:
            history.pop(0)
        speed = self._speed(history, point)
        inside = self._inside(self._position(point))
        emitted: list[SlowdownEvent] = []

        if not track["entered"]:
            if inside:
                track["entered"] = True
                approach = [value for _, value in track["approach"]]
                track["entry_speed"] = max(approach) if approach else None
                track["min_speed"] = speed
                track["approach"] = []
            elif speed is not None:
                track["approach"].append((point.time_s, speed))
                while len(track["approach"]) > 1 and point.time_s - track["approach"][0][0] > self.approach_window_s:
                    track["approach"].pop(0)
        else:
            if inside and speed is not None:
                track["min_speed"] = speed if track["min_speed"] is None else min(track["min_speed"], speed)
            if not inside:
                entry = track["entry_speed"]
                low = track["min_speed"]
                if entry is not None and low is not None and entry >= self.min_approach_px_s and entry > 0:
                    ratio = low / entry
                    if ratio <= self.ratio:
                        event = SlowdownEvent(
                            point.time_s, point.track_id, point.vehicle_class,
                            round(entry, 1), round(low, 1), round(ratio, 3),
                        )
                        self.events.append(event)
                        emitted.append(event)
                track["entered"] = False
                track["entry_speed"] = None
                track["min_speed"] = None
                track["approach"] = []
        return emitted

    def result(self) -> dict:
        return {"slowdowns": [asdict(event) for event in self.events]}
