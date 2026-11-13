"""Command-line entry point for processing a fixed-camera traffic clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_pipeline import RunConfig, process_video


def main() -> None:
    parser = argparse.ArgumentParser(description="VISTA vehicle crossing and visible-stop analysis")
    parser.add_argument("video", type=Path, help="Local MP4/MOV/AVI traffic video")
    parser.add_argument("--output", type=Path, default=Path("runs/latest"), help="Output directory")
    parser.add_argument("--axis", choices=("x", "y"), default="y", help="x = vertical line; y = horizontal line")
    parser.add_argument("--line", type=float, default=0.60, help="Counting line position, 0-1 of frame")
    parser.add_argument("--zone", default="0.05,0.05,0.95,0.90", help="Stop zone as normalized x1,y1,x2,y2")
    parser.add_argument("--model", default="yolo26s.pt", help="Ultralytics checkpoint or path")
    parser.add_argument("--tracker", default="botsort.yaml", help="Ultralytics tracking config")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--stop-seconds", type=float, default=2.0)
    parser.add_argument("--stationary-px-s", type=float, default=8.0)
    parser.add_argument("--max-frames", type=int, help="Limit processing for a quick test")
    parser.add_argument("--device", help="Inference device, e.g. cpu or mps")
    parser.add_argument("--ref-p1", help="Reference line start, normalized x,y (e.g. 0.2,0.8)")
    parser.add_argument("--ref-p2", help="Reference line end, normalized x,y (e.g. 0.8,0.8)")
    parser.add_argument("--ref-length-m", type=float, help="Real-world length of the reference line in meters")
    parser.add_argument("--line2-axis", choices=("x", "y"), help="Optional second line axis for an OD matrix")
    parser.add_argument("--line2-at", type=float, help="Optional second line position, 0-1 of frame")
    parser.add_argument("--slow-axis", choices=("x", "y"), help="Optional slow band axis for relative slowdown detection")
    parser.add_argument("--slow-start", type=float, help="Slow band start, 0-1 of frame (must be < slow-end)")
    parser.add_argument("--slow-end", type=float, help="Slow band end, 0-1 of frame")
    parser.add_argument("--slow-ratio", type=float, default=0.5, help="Speed-kept threshold, 0-1 (default 0.5)")
    parser.add_argument("--slow-min-px-s", type=float, default=5.0, help="Ignore vehicles below this approach speed in px/s")
    args = parser.parse_args()

    def _point(text: str | None):
        if not text:
            return None
        parts = [float(part.strip()) for part in text.split(",")]
        if len(parts) != 2:
            parser.error("Reference points need two comma-separated numbers")
        return (parts[0], parts[1])
    try:
        zone = tuple(float(part.strip()) for part in args.zone.split(","))
        if len(zone) != 4:
            raise ValueError
    except ValueError:
        parser.error("--zone needs four comma-separated numbers")
    config = RunConfig(
        line_axis=args.axis, line_at=args.line, zone=zone, model=args.model,
        tracker=args.tracker, confidence=args.confidence, stop_seconds=args.stop_seconds,
        max_stationary_speed_px_s=args.stationary_px_s, max_frames=args.max_frames,
        device=args.device,
        ref_p1=_point(args.ref_p1), ref_p2=_point(args.ref_p2), ref_length_m=args.ref_length_m,
        line2_axis=args.line2_axis, line2_at=args.line2_at,
        slow_axis=args.slow_axis, slow_start=args.slow_start, slow_end=args.slow_end,
        slow_ratio=args.slow_ratio, slow_min_speed_px_s=args.slow_min_px_s,
    )
    summary = process_video(args.video, args.output, config)
    print(json.dumps({key: summary[key] for key in ("source", "frames", "duration_s", "crossings", "stops", "by_direction", "by_class")}, indent=2))
    print(f"Dashboard: {(args.output / 'dashboard.html').resolve()}")


if __name__ == "__main__":
    main()
