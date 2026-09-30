"""Benchmarks ByteTrack against BoT-SORT variants on both videos.

No identity labels exist, so fragmentation is proxied: count tracks that end
and are followed within 1s, within 100px, by a new track. Not an ID-switch
count, but comparable across configurations, which is what the choice needs.
"""

import time
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8m.pt"
VIDEOS = {
    "entrance": PROJECT_ROOT / "data" / "raw_videos" / "entrance.mp4",
    "interior": PROJECT_ROOT / "data" / "raw_videos" / "interior.mp4",
}
PERSON_CLASS_ID = 0

TRACKER_CONFIGS = {
    "ByteTrack": "bytetrack.yaml",
    "BoT-SORT (ReID=False)": "botsort.yaml",
    "BoT-SORT (ReID=True, static)": str(PROJECT_ROOT / "configs" / "botsort_reid_static.yaml"),
}

FRAG_MAX_GAP_FRAMES = 30  # 1s at 30fps
FRAG_MAX_DIST_PX = 100


def run_config_enumerated(video_path: Path, tracker_yaml: str, fps: float):
    model = YOLO(str(MODEL_PATH))
    tracks = {}

    t0 = time.time()
    frame_idx = 0
    for result in model.track(
        source=str(video_path),
        classes=[PERSON_CLASS_ID],
        persist=True,
        tracker=tracker_yaml,
        stream=True,
        verbose=False,
    ):
        frame_idx += 1
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), tid in zip(xyxy, ids):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if tid not in tracks:
                tracks[tid] = {"first_f": frame_idx, "last_f": frame_idx,
                                "first_c": (cx, cy), "last_c": (cx, cy), "n": 1}
            else:
                tracks[tid]["last_f"] = frame_idx
                tracks[tid]["last_c"] = (cx, cy)
                tracks[tid]["n"] += 1
    elapsed = time.time() - t0
    return tracks, elapsed


def analyze(tracks: dict, fps: float):
    total = len(tracks)
    durations = {tid: (t["last_f"] - t["first_f"] + 1) / fps for tid, t in tracks.items()}
    ge2s = sum(1 for d in durations.values() if d >= 2.0)
    ge5s = sum(1 for d in durations.values() if d >= 5.0)

    # candidate fragmentation pairs
    ended = sorted(tracks.items(), key=lambda kv: kv[1]["last_f"])
    started = sorted(tracks.items(), key=lambda kv: kv[1]["first_f"])
    frag_pairs = 0
    for tid_a, a in ended:
        ax, ay = a["last_c"]
        for tid_b, b in started:
            if tid_b == tid_a:
                continue
            gap = b["first_f"] - a["last_f"]
            if gap <= 0 or gap > FRAG_MAX_GAP_FRAMES:
                continue
            bx, by = b["first_c"]
            dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            if dist <= FRAG_MAX_DIST_PX:
                frag_pairs += 1
                break  # count each ended track at most once

    return {
        "total_ids": total,
        "tracks_ge2s": ge2s,
        "tracks_ge5s": ge5s,
        "candidate_frag_pairs": frag_pairs,
    }


if __name__ == "__main__":
    import cv2

    results_summary = {}

    for video_name, video_path in VIDEOS.items():
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        for cfg_name, cfg_path in TRACKER_CONFIGS.items():
            print(f"\n=== {video_name} | {cfg_name} ===", flush=True)
            tracks, elapsed = run_config_enumerated(video_path, cfg_path, fps)
            stats = analyze(tracks, fps)
            stats["runtime_s"] = round(elapsed, 1)
            results_summary[(video_name, cfg_name)] = stats
            print(f"  {stats}", flush=True)

    print("\n\n=== SUMMARY TABLE ===")
    header = f"{'video':10s} {'tracker':28s} {'total_ids':>10s} {'>=2s':>6s} {'>=5s':>6s} {'frag_pairs':>11s} {'runtime_s':>10s}"
    print(header)
    for (video_name, cfg_name), s in results_summary.items():
        print(f"{video_name:10s} {cfg_name:28s} {s['total_ids']:10d} {s['tracks_ge2s']:6d} "
              f"{s['tracks_ge5s']:6d} {s['candidate_frag_pairs']:11d} {s['runtime_s']:10.1f}")
