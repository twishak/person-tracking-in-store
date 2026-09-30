"""Data prep, not part of the pipeline.

Saves every frame belonging to a shortlist of track IDs, so a candidate can be
reviewed across their full range of poses rather than from one crop. Produced
the verified staff reference set in data/persons_dense/.
"""

import time
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8m.pt"
VIDEOS = {
    "entrance": PROJECT_ROOT / "data" / "raw_videos" / "entrance.mp4",
    "interior": PROJECT_ROOT / "data" / "raw_videos" / "interior.mp4",
}
OUT_DIR = PROJECT_ROOT / "data" / "persons_dense"
PERSON_CLASS_ID = 0

TARGET_TRACKS = {
    "entrance": [
        2, 69, 79, 126, 131, 956, 979, 1044, 1045, 1066, 1084, 1206, 1210,
        1237, 1248, 1283, 1317, 1328, 1362, 1402, 1458, 1485, 1513, 1567,
        1712, 1719, 1720, 1756, 1854, 1857, 1895, 1918, 2385, 2531, 2714, 2731,
    ],
    "interior": [
        2, 117, 673, 880, 935, 959, 1036, 1053, 1248, 1251, 1254, 1264,
        1280, 1463, 1608, 1619,
    ],
}


def process_video(name: str, video_path: Path, target_ids: set) -> None:
    print(f"\n=== {name}: {len(target_ids)} target tracks ===", flush=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    model = YOLO(str(MODEL_PATH))

    frame_idx = 0
    saved = 0
    t0 = time.time()

    for result in model.track(
        source=str(video_path),
        classes=[PERSON_CLASS_ID],
        persist=True,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
    ):
        frame = result.orig_img
        frame_idx += 1

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                if tid not in target_ids:
                    continue
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                if x2i <= x1i or y2i <= y1i:
                    continue
                crop = frame[y1i:y2i, x1i:x2i]
                track_dir = OUT_DIR / name / f"track_{tid:04d}"
                track_dir.mkdir(parents=True, exist_ok=True)
                t = frame_idx / fps
                fname = f"frame_{frame_idx:06d}_t{t:.1f}s.jpg"
                cv2.imwrite(str(track_dir / fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1

        if frame_idx % 1000 == 0:
            elapsed = time.time() - t0
            rate = frame_idx / elapsed if elapsed > 0 else 0
            print(f"  frame {frame_idx}/{total_frames} ({rate:.1f} fps, saved: {saved})", flush=True)

    print(f"  done: {frame_idx} frames scanned, {saved} target-track crops saved", flush=True)


if __name__ == "__main__":
    for name, path in VIDEOS.items():
        process_video(name, path, set(TARGET_TRACKS[name]))
    print("\nAll videos processed.", flush=True)
