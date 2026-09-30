"""Picks the apron threshold using only verified positives.

No confirmed customer crops exist, but the ground-truth COUNT is known -- one
staff member in entrance.mp4 -- so any threshold labelling far more tracks than
one person can account for is refuted without needing a single negative label.
Sweeps threshold against staff-crop retention and resulting track count.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from build_cache import load_cache
from staff_id import apron_score, load_scores

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSE_MODEL = PROJECT_ROOT / "models" / "yolov8m-pose.pt"

# Plausible number of tracks for ONE person under this tracker's fragmentation,
# extrapolated from the user's manual count on the earlier run (36/513 tracks).
PLAUSIBLE_TRACK_RANGE = (10, 30)


def score_reference_crops(video_key: str, max_per_track: int = 6) -> np.ndarray:
    """Apron scores for verified staff crops, using pipeline-identical torso
    extraction (pose keypoints, not a bounding-box fraction)."""
    base = PROJECT_ROOT / "data" / "persons_dense" / video_key
    if not base.exists():
        return np.array([])

    model = YOLO(str(POSE_MODEL))
    scores = []
    for track_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        files = sorted(track_dir.glob("*.jpg"))
        if not files:
            continue
        picks = files[:: max(1, len(files) // max_per_track)][:max_per_track]
        for f in picks:
            img = cv2.imread(str(f))
            if img is None or img.shape[0] < 40:
                continue
            res = model.predict(img, verbose=False, classes=[0])
            if not res or res[0].keypoints is None or res[0].keypoints.data is None:
                continue
            kps = res[0].keypoints.data.cpu().numpy()
            if len(kps) == 0:
                continue
            # Largest detection in the crop is the subject.
            boxes = res[0].boxes.xyxy.cpu().numpy()
            best = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
            s = apron_score(img, kps[best], offset=(0, 0))
            if s is not None:
                scores.append(s)
    return np.array(scores)


def main(video_key: str):
    print(f"=== {video_key}: one-class apron calibration ===\n")

    ref = score_reference_crops(video_key)
    if len(ref) == 0:
        print("no reference crops scored")
        return
    print(f"verified STAFF crops scored: {len(ref)}")
    print("  percentiles:", np.round(np.percentile(ref, [5, 10, 25, 50, 75, 90]), 3))

    cache = load_cache(video_key)
    scores = load_scores(video_key)
    by_track = defaultdict(list)
    for row in range(len(scores)):
        if np.isfinite(scores[row]):
            by_track[int(cache["track_id"][row])].append(float(scores[row]))
    track_med = {t: float(np.median(v)) for t, v in by_track.items() if len(v) >= 5}
    print(f"video tracks with enough evidence to score: {len(track_med)}")

    print(f"\n{'thresh':>7} {'staff recall':>13} {'tracks labelled staff':>22} {'verdict':>10}")
    lo, hi = PLAUSIBLE_TRACK_RANGE
    workable = []
    for thr in np.arange(0.30, 1.00, 0.05):
        recall = float((ref >= thr).mean())
        n_tracks = sum(1 for m in track_med.values() if m >= thr)
        ok = (recall >= 0.70) and (lo <= n_tracks <= hi)
        if ok:
            workable.append((thr, recall, n_tracks))
        print(f"{thr:>7.2f} {recall:>12.0%} {n_tracks:>22d} {'OK' if ok else '':>10}")

    print()
    if workable:
        thr, recall, n = workable[len(workable) // 2]
        print(f"WORKABLE: threshold {thr:.2f} keeps {recall:.0%} of verified staff crops")
        print(f"          and labels {n} tracks staff (plausible for one person")
        print(f"          given this tracker's fragmentation).")
    else:
        print("NO WORKABLE THRESHOLD.")
        print("  At every level, the apron score either loses most verified staff")
        print("  crops or labels far more tracks than one person can account for.")
        print("  The colour signal does not separate staff from customers in this")
        print("  scene -- warm wood shelving and warm lighting put most torsos in")
        print("  the same hue range as the apron.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="entrance")
    args = parser.parse_args()
    main(args.video)
