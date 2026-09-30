"""Data prep, not part of the pipeline.

Dumps every 10th frame for scene reference, and one best crop per tracked
person for manual staff/customer review. One crop per track, not per frame --
a person tracked for 8s is ~240 near-identical images.
"""

import csv
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
FRAMES_DIR = PROJECT_ROOT / "data" / "frames_full"
PERSONS_DIR = PROJECT_ROOT / "data" / "persons"

PERSON_CLASS_ID = 0  # COCO class 0 = person
FRAME_DUMP_SKIP = 10  # save 1 out of every 10 frames to frames_full/ (~3fps at 30fps source)


def process_video(name: str, video_path: Path) -> None:
    print(f"\n=== {name}: {video_path} ===", flush=True)

    frames_out = FRAMES_DIR / name
    persons_out = PERSONS_DIR / name
    frames_out.mkdir(parents=True, exist_ok=True)
    persons_out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"fps={fps:.2f} total_frames={total_frames}", flush=True)

    model = YOLO(str(MODEL_PATH))

    # best[track_id] = {"area": float, "crop": ndarray, "t": float, "bbox": (x1,y1,x2,y2)}
    best = {}

    frame_idx = 0
    saved_frames = 0
    t0 = time.time()
    results_gen = model.track(
        source=str(video_path),
        classes=[PERSON_CLASS_ID],
        persist=True,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
    )

    for result in results_gen:
        frame = result.orig_img  # BGR ndarray, full original frame
        frame_idx += 1

        # 1. sparse full-frame dump (reference/browsing only)
        if frame_idx % FRAME_DUMP_SKIP == 0:
            out_path = frames_out / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved_frames += 1

        # 2. best crop per track id, evaluated every frame
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                area = max(0, x2i - x1i) * max(0, y2i - y1i)
                if area <= 0:
                    continue
                prev = best.get(tid)
                if prev is None or area > prev["area"]:
                    crop = frame[y1i:y2i, x1i:x2i].copy()
                    best[tid] = {
                        "area": area,
                        "crop": crop,
                        "t": frame_idx / fps,
                        "bbox": (x1i, y1i, x2i, y2i),
                    }

        if frame_idx % 500 == 0:
            elapsed = time.time() - t0
            rate = frame_idx / elapsed if elapsed > 0 else 0
            print(
                f"  frame {frame_idx}/{total_frames} "
                f"({rate:.1f} fps, tracks so far: {len(best)})",
                flush=True,
            )

    print(
        f"  done: {frame_idx} frames processed, {saved_frames} saved to disk, "
        f"{len(best)} unique tracks",
        flush=True,
    )

    # write crops + log
    log_path = persons_out / "track_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "timestamp_s", "x1", "y1", "x2", "y2", "area_px", "crop_file"])
        for tid, info in sorted(best.items()):
            fname = f"track_{tid:04d}_t{info['t']:.1f}s.jpg"
            cv2.imwrite(str(persons_out / fname), info["crop"], [cv2.IMWRITE_JPEG_QUALITY, 95])
            x1, y1, x2, y2 = info["bbox"]
            writer.writerow([tid, f"{info['t']:.2f}", x1, y1, x2, y2, info["area"], fname])

    print(f"  crops + log written to {persons_out}", flush=True)


if __name__ == "__main__":
    for name, path in VIDEOS.items():
        process_video(name, path)
    print("\nAll videos processed.", flush=True)
