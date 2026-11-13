"""Run VISTA on a fixed-camera video and save auditable event records."""

from __future__ import annotations

import csv
import json
import sqlite3
import subprocess
from math import hypot
from time import perf_counter
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from vista_events import CrossingEvent, EventEngine, ODPair, ODEngine, SlowdownEngine, StopEvent, TrackPoint, Zone
from speed_estimator import SpeedEstimator


VEHICLE_CLASSES = (2, 3, 5, 7)  # COCO: car, motorcycle, bus, truck
CLASS_NAMES = ("car", "motorcycle", "bus", "truck")


def _class_series(events: list[dict], duration_s: float) -> list[dict]:
    """Per-class crossing counts in 5-second windows for charting."""
    buckets: dict[int, dict[str, int]] = {}
    for event in events:
        if event["type"] != "crossing":
            continue
        bucket = int(event["time_s"] // 5) * 5
        slot = buckets.setdefault(bucket, {name: 0 for name in CLASS_NAMES})
        cls = event.get("vehicle_class", "car")
        slot[cls] = slot.get(cls, 0) + 1
    stops = list(range(0, max(5, int(duration_s) + 5), 5))
    return [{"start": second, **{name: buckets.get(second, {}).get(name, 0) for name in CLASS_NAMES}} for second in stops]


@dataclass(frozen=True)
class RunConfig:
    line_axis: str = "y"
    line_at: float = 0.60
    zone: tuple[float, float, float, float] = (0.05, 0.05, 0.95, 0.90)
    model: str = "yolo26s.pt"
    tracker: str = "botsort.yaml"
    confidence: float = 0.25
    stop_seconds: float = 2.0
    max_stationary_speed_px_s: float = 8.0
    line_margin_px: float = 5.0
    max_frames: int | None = None
    device: str | None = None
    ref_p1: tuple[float, float] | None = None
    ref_p2: tuple[float, float] | None = None
    ref_length_m: float | None = None
    line2_axis: str | None = None
    line2_at: float | None = None
    slow_axis: str | None = None
    slow_start: float | None = None
    slow_end: float | None = None
    slow_ratio: float = 0.5
    slow_min_speed_px_s: float = 5.0

    def validate(self) -> None:
        if self.line_axis not in ("x", "y"):
            raise ValueError("line_axis must be x or y")
        if not 0 < self.line_at < 1:
            raise ValueError("line_at must be between 0 and 1")
        x1, y1, x2, y2 = self.zone
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("zone must be normalized x1,y1,x2,y2")
        if not 0 < self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        has_ref = self.ref_p1 is not None or self.ref_p2 is not None or self.ref_length_m is not None
        if has_ref:
            if self.ref_p1 is None or self.ref_p2 is None or self.ref_length_m is None:
                raise ValueError("Set ref_p1, ref_p2, and ref_length_m together")
            if self.ref_length_m <= 0:
                raise ValueError("ref_length_m must be positive")
            for point in (self.ref_p1, self.ref_p2):
                if len(point) != 2 or not (0 <= point[0] <= 1 and 0 <= point[1] <= 1):
                    raise ValueError("Reference points must be normalized x,y in 0..1")
            if self.ref_p1 == self.ref_p2:
                raise ValueError("Reference points must be distinct")
        has_line2 = self.line2_axis is not None or self.line2_at is not None
        if has_line2:
            if self.line2_axis is None or self.line2_at is None:
                raise ValueError("Set line2_axis and line2_at together")
            if self.line2_axis not in ("x", "y"):
                raise ValueError("line2_axis must be x or y")
            if not 0 < self.line2_at < 1:
                raise ValueError("line2_at must be between 0 and 1")
            if self.line2_axis == self.line_axis and abs(self.line2_at - self.line_at) < 1e-6:
                raise ValueError("The second line must differ from the counting line")
        has_slow = self.slow_axis is not None or self.slow_start is not None or self.slow_end is not None
        if has_slow:
            if self.slow_axis is None or self.slow_start is None or self.slow_end is None:
                raise ValueError("Set slow_axis, slow_start, and slow_end together")
            if self.slow_axis not in ("x", "y"):
                raise ValueError("slow_axis must be x or y")
            if not (0 < self.slow_start < self.slow_end < 1):
                raise ValueError("slow_start and slow_end must satisfy 0 < slow_start < slow_end < 1")
            if not 0 < self.slow_ratio <= 1:
                raise ValueError("slow_ratio must be between 0 and 1")
            if self.slow_min_speed_px_s < 0:
                raise ValueError("slow_min_speed_px_s must be nonnegative")


def _open_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY, source TEXT NOT NULL, model TEXT NOT NULL,
            tracker TEXT NOT NULL, fps REAL NOT NULL, width INTEGER NOT NULL,
            height INTEGER NOT NULL, frames INTEGER NOT NULL, config_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL REFERENCES runs(id),
            time_s REAL NOT NULL, track_id INTEGER NOT NULL, type TEXT NOT NULL,
            vehicle_class TEXT NOT NULL, direction TEXT, duration_s REAL
        );
        CREATE TABLE IF NOT EXISTS samples (
            run_id INTEGER NOT NULL REFERENCES runs(id), time_s REAL NOT NULL,
            stopped_count INTEGER NOT NULL, PRIMARY KEY (run_id, time_s)
        );
    """)
    return db


def _draw_overlay(frame, boxes, engine: EventEngine, config: RunConfig, zone: Zone, time_s: float):
    import cv2

    height, width = frame.shape[:2]
    tint = frame.copy()
    cv2.rectangle(tint, (int(zone.x_min), int(zone.y_min)),
                  (int(zone.x_max), int(zone.y_max)), (218, 190, 54), -1)
    cv2.addWeighted(tint, 0.08, frame, 0.92, 0, frame)
    cv2.rectangle(frame, (int(zone.x_min), int(zone.y_min)),
                  (int(zone.x_max), int(zone.y_max)), (218, 190, 54), 2)
    if config.line_axis == "x":
        x = round(config.line_at * width)
        cv2.line(frame, (x, 0), (x, height), (20, 210, 225), 3)
    else:
        y = round(config.line_at * height)
        cv2.line(frame, (0, y), (width, y), (20, 210, 225), 3)
    if config.slow_axis is not None and config.slow_start is not None and config.slow_end is not None:
        if config.slow_axis == "x":
            cv2.line(frame, (round(config.slow_start * width), 0), (round(config.slow_start * width), height), (0, 128, 255), 2)
            cv2.line(frame, (round(config.slow_end * width), 0), (round(config.slow_end * width), height), (0, 128, 255), 2)
        else:
            cv2.line(frame, (0, round(config.slow_start * height)), (width, round(config.slow_start * height)), (0, 128, 255), 2)
            cv2.line(frame, (0, round(config.slow_end * height)), (width, round(config.slow_end * height)), (0, 128, 255), 2)
    for box in boxes:
        x1, y1, x2, y2, track_id, label, confidence = box[:7]
        speed_kmh = box[7] if len(box) > 7 else None
        cv2.rectangle(frame, (x1, y1), (x2, y2), (8, 190, 158), 2)
        caption = f"{label} #{track_id} {confidence:.2f}"
        if speed_kmh is not None:
            caption += f"  {speed_kmh:.0f} km/h"
        cv2.putText(frame, caption, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 3)
        cv2.putText(frame, caption, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (14, 48, 57), 1)
        cv2.circle(frame, ((x1 + x2) // 2, y2), 3, (20, 210, 225), -1)
    header = f"VISTA  {time_s:05.1f}s  crossings {len(engine.crossings)}  visibly stopped {engine.stopped_count(time_s)}"
    cv2.rectangle(frame, (0, 0), (width, 36), (11, 31, 42), -1)
    cv2.putText(frame, header, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 255, 255), 2)
    return frame


def process_video(
    video_path: Path,
    output_dir: Path,
    config: RunConfig,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Analyze a clip. The video, database, CSV, JSON and HTML are written together."""
    import cv2
    from ultralytics import YOLO

    from dashboard import write_dashboard

    config.validate()
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Video has invalid frame rate or dimensions")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if config.max_frames is not None:
        total_frames = min(total_frames, config.max_frames) if total_frames > 0 else config.max_frames

    x1, y1, x2, y2 = config.zone
    zone = Zone(x1 * width, y1 * height, x2 * width, y2 * height)
    line = config.line_at * (width if config.line_axis == "x" else height)
    engine = EventEngine(
        line_x=line if config.line_axis == "x" else None,
        line_y=line if config.line_axis == "y" else None,
        stop_zone=zone,
        line_margin_px=config.line_margin_px,
        max_stationary_speed_px_s=config.max_stationary_speed_px_s,
        min_stop_s=config.stop_seconds,
    )

    speed_estimator: SpeedEstimator | None = None
    if config.ref_p1 is not None and config.ref_p2 is not None and config.ref_length_m is not None:
        ax, ay = config.ref_p1[0] * width, config.ref_p1[1] * height
        bx, by = config.ref_p2[0] * width, config.ref_p2[1] * height
        ref_px = hypot(bx - ax, by - ay)
        if ref_px <= 0:
            capture.release()
            raise ValueError("Reference points must be distinct")
        speed_estimator = SpeedEstimator(ref_px / config.ref_length_m)

    od_engine: ODEngine | None = None
    if config.line2_axis is not None and config.line2_at is not None:
        line2_pos = config.line2_at * (width if config.line2_axis == "x" else height)
        od_engine = ODEngine(config.line_axis, line, config.line2_axis, line2_pos, margin_px=config.line_margin_px)

    slow_engine: SlowdownEngine | None = None
    if config.slow_axis is not None:
        slow_axis_len = width if config.slow_axis == "x" else height
        slow_engine = SlowdownEngine(
            config.slow_axis, config.slow_start * slow_axis_len, config.slow_end * slow_axis_len,
            ratio=config.slow_ratio, min_approach_px_s=config.slow_min_speed_px_s,
        )

    raw_video = output_dir / "annotated_raw.mp4"
    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create output video")
    started = perf_counter()
    model = YOLO(config.model)
    db = _open_db(output_dir / "events.sqlite")
    cursor = db.execute(
        "INSERT INTO runs(source,model,tracker,fps,width,height,frames,config_json) VALUES(?,?,?,?,?,?,?,?)",
        (video_path.name, config.model, config.tracker, fps, width, height, 0, json.dumps(asdict(config))),
    )
    run_id = cursor.lastrowid
    frames = 0
    tracked_boxes = 0
    unique_tracks: set[int] = set()
    last_sample_second = -1
    tracks_file = (output_dir / "tracks.csv").open("w", newline="")
    tracks_writer = csv.DictWriter(
        tracks_file,
        fieldnames=["frame", "time_s", "track_id", "vehicle_class", "confidence", "x1", "y1", "x2", "y2"],
    )
    tracks_writer.writeheader()
    try:
        while config.max_frames is None or frames < config.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            time_s = frames / fps
            kwargs = dict(persist=True, tracker=config.tracker, classes=list(VEHICLE_CLASSES),
                          conf=config.confidence, verbose=False)
            if config.device:
                kwargs["device"] = config.device
            result = model.track(frame, **kwargs)[0]
            drawn_boxes = []
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                for index in range(len(boxes)):
                    track_id = int(boxes.id[index].item())
                    class_id = int(boxes.cls[index].item())
                    label = model.names[class_id]
                    confidence = float(boxes.conf[index].item())
                    x1b, y1b, x2b, y2b = [int(v) for v in boxes.xyxy[index].tolist()]
                    point = TrackPoint(time_s, track_id, (x1b + x2b) / 2, y2b, label)
                    speed_kmh = speed_estimator.update(track_id, time_s, point.x, point.y, label) if speed_estimator else None
                    drawn_boxes.append((x1b, y1b, x2b, y2b, track_id, label, confidence, speed_kmh))
                    tracks_writer.writerow({
                        "frame": frames + 1, "time_s": round(time_s, 3), "track_id": track_id,
                        "vehicle_class": label, "confidence": round(confidence, 3),
                        "x1": x1b, "y1": y1b, "x2": x2b, "y2": y2b,
                    })
                    tracked_boxes += 1
                    unique_tracks.add(track_id)
                    for event in engine.update(point):
                        event_type = "crossing" if isinstance(event, CrossingEvent) else "stop"
                        db.execute(
                            "INSERT INTO events(run_id,time_s,track_id,type,vehicle_class,direction,duration_s) "
                            "VALUES(?,?,?,?,?,?,?)",
                            (run_id, event.time_s, track_id, event_type, event.vehicle_class,
                             event.direction if event_type == "crossing" else None,
                             event.duration_s if isinstance(event, StopEvent) else None),
                        )
                    if od_engine:
                        od_engine.update(point)
                    if slow_engine:
                        slow_engine.update(point)
            second = int(time_s)
            if second > last_sample_second:
                db.execute("INSERT INTO samples(run_id,time_s,stopped_count) VALUES(?,?,?)",
                           (run_id, float(second), engine.stopped_count(time_s)))
                last_sample_second = second
            writer.write(_draw_overlay(frame, drawn_boxes, engine, config, zone, time_s))
            frames += 1
            if progress is not None and (frames == 1 or frames % 5 == 0 or frames == total_frames):
                progress(frames, total_frames)
    finally:
        capture.release()
        writer.release()
        tracks_file.close()
        db.execute("UPDATE runs SET frames=? WHERE id=?", (frames, run_id))
        db.commit()
        db.close()

    if frames == 0:
        raise ValueError("Video contained no decodable frames")
    playable_video = _make_browser_video(raw_video, output_dir / "annotated.mp4")
    processing_s = round(perf_counter() - started, 2)
    events = [dict(type="crossing", **asdict(e)) for e in engine.crossings]
    events += [dict(type="stop", **asdict(e)) for e in engine.stops]
    events.sort(key=lambda e: e["time_s"])
    counts = Counter(e["direction"] for e in events if e["type"] == "crossing")
    class_counts = Counter(e["vehicle_class"] for e in events if e["type"] == "crossing")
    stop_class_counts = Counter(e["vehicle_class"] for e in events if e["type"] == "stop")
    speed_result = speed_estimator.result() if speed_estimator else {"per_track": [], "series": []}
    od_result = od_engine.result() if od_engine else {"od_pairs": [], "od_matrix": {}}
    slow_result = slow_engine.result() if slow_engine else {"slowdowns": []}
    summary = {
        "source": video_path.name,
        "model": config.model,
        "tracker": config.tracker,
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frames,
        "tracked_boxes": tracked_boxes,
        "unique_tracks": len(unique_tracks),
        "duration_s": round(frames / fps, 2),
        "processing_s": processing_s,
        "processing_fps": round(frames / processing_s, 2) if processing_s > 0 else None,
        "video": playable_video.name,
        "crossings": len(engine.crossings),
        "stops": len(engine.stops),
        "by_direction": dict(counts),
        "by_class": dict(class_counts),
        "by_class_stops": dict(stop_class_counts),
        "class_series": _class_series(events, frames / fps),
        "speeds": speed_result["per_track"],
        "speed_series": speed_result["series"],
        "reference_m": config.ref_length_m,
        "od_pairs": od_result["od_pairs"],
        "od_matrix": od_result["od_matrix"],
        "slowdowns": slow_result["slowdowns"],
        "slowdown_count": len(slow_result["slowdowns"]),
        "slow_band": {"axis": config.slow_axis, "start": config.slow_start, "end": config.slow_end, "ratio": config.slow_ratio} if config.slow_axis else None,
        "config": asdict(config),
        "events": events,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "events.csv").open("w", newline="") as file:
        csv_writer = csv.DictWriter(file, fieldnames=["type", "time_s", "track_id", "vehicle_class", "direction", "duration_s"])
        csv_writer.writeheader()
        for event in events:
            csv_writer.writerow(event)
    write_dashboard(summary, output_dir / "dashboard.html")
    return summary


def _make_browser_video(raw_video: Path, output_video: Path) -> Path:
    """Transcode to H.264 when bundled ffmpeg is available; retain raw MP4 otherwise."""
    try:
        import imageio_ffmpeg

        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(raw_video),
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "25", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(output_video)]
        subprocess.run(command, check=True, capture_output=True)
        raw_video.unlink()
        return output_video
    except (ImportError, OSError, subprocess.CalledProcessError):
        return raw_video
