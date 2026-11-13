"""Near-real-time webcam/RTSP analysis for the VISTA prototype."""

from __future__ import annotations

import threading
import time

from vista_events import EventEngine, TrackPoint, Zone


def _is_youtube(url: str) -> bool:
    return "youtube.com/watch" in url or "youtu.be/" in url


def resolve_youtube_stream(url: str) -> str:
    """Resolve a YouTube watch URL to a direct HLS URL OpenCV can open.

    Picks the highest variant at or below 480p so CPU-only inference can
    keep up; live streams expose only HLS.
    """
    try:
        import yt_dlp
    except ImportError as error:
        raise RuntimeError("yt-dlp is not installed (pip install yt-dlp)") from error
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    formats = info.get("formats") or []
    video = [f for f in formats
             if f.get("protocol") in ("m3u8", "m3u8_native") and f.get("height")]
    if not video:
        direct = info.get("url") or (formats[-1].get("url") if formats else None)
        if not direct:
            raise RuntimeError("No playable stream found at the YouTube URL")
        return direct
    video.sort(key=lambda f: f["height"])
    best = next((f for f in reversed(video) if f["height"] <= 480), video[0])
    return best["url"]


class LiveSession:
    """Run the detector on a live source and hold the latest annotated frame."""

    def __init__(
        self,
        source,
        *,
        model: str,
        line_axis: str,
        line_at: float,
        zone: tuple[float, float, float, float],
        confidence: float = 0.25,
        stop_seconds: float = 2.0,
        max_stationary_speed_px_s: float = 8.0,
    ) -> None:
        self.source = source
        self.model_name = model
        self.line_axis = line_axis
        self.line_at = line_at
        self.zone_norm = zone
        self.confidence = confidence
        self.stop_seconds = stop_seconds
        self.max_stationary_speed_px_s = max_stationary_speed_px_s
        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._thread = None
        self._error = None
        self._engine: EventEngine | None = None
        self._width = 0
        self._height = 0
        self.started_at: float | None = None

    def _ensure_engine(self, width: int, height: int) -> None:
        x1, y1, x2, y2 = self.zone_norm
        zone = Zone(x1 * width, y1 * height, x2 * width, y2 * height)
        line = self.line_at * (width if self.line_axis == "x" else height)
        self._engine = EventEngine(
            line_x=line if self.line_axis == "x" else None,
            line_y=line if self.line_axis == "y" else None,
            stop_zone=zone,
            max_stationary_speed_px_s=self.max_stationary_speed_px_s,
            min_stop_s=self.stop_seconds,
        )

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _resolved_source(self):
        if isinstance(self.source, str) and _is_youtube(self.source):
            return resolve_youtube_stream(self.source)
        return self.source

    def _run(self) -> None:
        import cv2
        from ultralytics import YOLO

        try:
            model = YOLO(self.model_name)
            capture = cv2.VideoCapture(self._resolved_source())
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open source: {self.source}")
            while self._running:
                ok, frame = capture.read()
                if not ok:
                    if isinstance(self.source, str):
                        capture.release()
                        time.sleep(1.0)
                        capture = cv2.VideoCapture(self._resolved_source())
                        continue
                    break
                self._width = frame.shape[1]
                self._height = frame.shape[0]
                if self._engine is None:
                    self._ensure_engine(self._width, self._height)
                result = model.track(
                    frame, persist=True, tracker="botsort.yaml",
                    classes=[2, 3, 5, 7], conf=self.confidence, verbose=False,
                )[0]
                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    for index in range(len(boxes)):
                        track_id = int(boxes.id[index].item())
                        label = model.names[int(boxes.cls[index].item())]
                        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[index].tolist()]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (8, 190, 158), 2)
                        cv2.putText(frame, f"{label} #{track_id}", (x1, max(18, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        assert self.started_at is not None
                        self._engine.update(TrackPoint(
                            time.time() - self.started_at, track_id, (x1 + x2) / 2, y2, label))
                if self.line_axis == "x":
                    x = round(self.line_at * self._width)
                    cv2.line(frame, (x, 0), (x, self._height), (20, 210, 225), 3)
                else:
                    y = round(self.line_at * self._height)
                    cv2.line(frame, (0, y), (self._width, y), (20, 210, 225), 3)
                elapsed = time.time() - (self.started_at or time.time())
                header = f"VISTA live  crossings {len(self._engine.crossings)}  visibly stopped {self._engine.stopped_count(elapsed)}"
                cv2.rectangle(frame, (0, 0), (self._width, 30), (11, 31, 42), -1)
                cv2.putText(frame, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                with self._lock:
                    self._frame = frame
        except Exception as error:
            self._error = str(error)
        finally:
            self._running = False

    def summary(self) -> dict:
        engine = self._engine
        elapsed = (time.time() - self.started_at) if self.started_at else 0.0
        if engine is None:
            return {"running": self._running, "error": self._error, "source": str(self.source),
                    "crossings": 0, "stops": 0, "stopped": 0}
        return {
            "running": self._running,
            "error": self._error,
            "source": str(self.source),
            "crossings": len(engine.crossings),
            "stops": len(engine.stops),
            "stopped": engine.stopped_count(elapsed),
        }

    def frame_jpeg(self) -> bytes | None:
        import cv2

        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return encoded.tobytes() if ok else None

    def stop(self) -> None:
        self._running = False
