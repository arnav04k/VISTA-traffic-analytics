"""Export a run's tracked detections as a YOLO-format dataset for review/training.

The labels are auto-generated (weak) from the pretrained detector and tracking;
a human must review them before training.
"""

from __future__ import annotations

import csv
from pathlib import Path

from video_pipeline import CLASS_NAMES


CLASS_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


def build_yolo_dataset(video_path: Path, tracks_csv: Path, dataset_dir: Path) -> Path:
    """Extract annotated frames and YOLO-normalized labels into ``dataset_dir``."""
    import cv2

    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not tracks_csv.is_file():
        raise FileNotFoundError(tracks_csv)

    detections: dict[int, list[dict]] = {}
    with tracks_csv.open(newline="") as file:
        for row in csv.DictReader(file):
            cls = row["vehicle_class"]
            if cls not in CLASS_ID:
                continue
            detections.setdefault(int(row["frame"]), []).append({
                "class_id": CLASS_ID[cls],
                "class_name": cls,
                "confidence": float(row["confidence"]),
                "x1": float(row["x1"]), "y1": float(row["y1"]),
                "x2": float(row["x2"]), "y2": float(row["y2"]),
            })

    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meta_rows: list[dict] = []
    try:
        for frame in sorted(detections):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
            ok, image = capture.read()
            if not ok:
                continue
            name = f"frame_{frame:06d}"
            cv2.imwrite(str(images_dir / f"{name}.jpg"), image)
            lines = []
            for det in detections[frame]:
                cx = min(max((det["x1"] + det["x2"]) / 2 / width, 0.0), 1.0)
                cy = min(max((det["y1"] + det["y2"]) / 2 / height, 0.0), 1.0)
                w = min(max((det["x2"] - det["x1"]) / width, 0.0), 1.0)
                h = min(max((det["y2"] - det["y1"]) / height, 0.0), 1.0)
                lines.append(f"{det['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                meta_rows.append({
                    "image": f"images/{name}.jpg", "frame": frame,
                    "class": det["class_name"], "class_id": det["class_id"],
                    "confidence": det["confidence"],
                })
            (labels_dir / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    finally:
        capture.release()

    (dataset_dir / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    (dataset_dir / "dataset.yaml").write_text(
        "path: .\ntrain: images\nval: images\nnames:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )
    with (dataset_dir / "auto_labels.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "frame", "class", "class_id", "confidence"])
        writer.writeheader()
        writer.writerows(meta_rows)
    return dataset_dir
