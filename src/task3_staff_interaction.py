"""Task 3 -- Staff-Customer Interaction (entrance.mp4).

Reports the average number of interaction sessions per staff member.

Staff tracks that never overlap in time are merged into one logical instance --
that pattern is one person repeatedly re-acquired under new IDs, and reporting
22 instances would divide their sessions across 22 rows.

An interaction is proximity within a multiple of the staff member's box height
(a ratio, not pixels -- this camera has strong perspective), sustained long
enough to exclude brushing past, with brief lapses tolerated so fetching a shoe
box stays one session.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from build_cache import load_cache, rows_for_frame
from staff_id import classify_tracks
from episodes import EpisodeDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def foot_point(bbox):
    x1, _y1, x2, y2 = bbox
    return (float(x1) + float(x2)) / 2.0, float(y2)


def box_height(bbox):
    _x1, y1, _x2, y2 = bbox
    return float(y2) - float(y1)


def box_gap(a, b) -> float:
    """Edge-to-edge distance between two boxes; 0 when they overlap.

    Replaces the earlier foot-to-foot distance. Foot separation misreads the
    most important interaction in a shoe shop: staff kneel to fit a shoe, which
    puts their feet well away from a seated customer's feet while the two people
    are practically touching. Box proximity captures "these two are together"
    regardless of posture.
    """
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return float(np.hypot(dx, dy))


def run(video_key: str = "entrance") -> dict:
    cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "thresholds.yaml"))
    t3 = cfg["staff_interaction"]

    cache = load_cache(video_key)
    fps = float(cache["fps"])
    total_frames = int(cache["total_frames"])

    labels = classify_tracks(video_key)
    staff_tracks = {t for t, v in labels.items() if v["label"] == "staff"}

    # Verify the merge assumption rather than trusting it: if two staff tracks
    # genuinely overlap, they cannot be the same person and the merge would be
    # wrong. Reported so the assumption is visible, not buried.
    span = {}
    for row in range(len(cache["frame_idx"])):
        tid = int(cache["track_id"][row])
        if tid in staff_tracks:
            f = int(cache["frame_idx"][row])
            a, b = span.get(tid, (f, f))
            span[tid] = (min(a, f), max(b, f))
    iv = sorted((a, b) for a, b in span.values())
    overlaps = sum(1 for i in range(len(iv)) for j in range(i + 1, len(iv))
                   if iv[i][1] > iv[j][0] + fps * 0.5)

    # One detector per customer, tracking proximity to "the staff member"
    # (all staff tracks merged into one instance).
    detectors: dict[int, EpisodeDetector] = {}
    sessions = []
    last_seen: dict[int, tuple[int, float]] = {}

    for frame_idx in range(1, total_frames + 1):
        sl = rows_for_frame(cache, frame_idx)
        rows = list(range(sl.start, sl.stop))
        if not rows:
            continue
        t = frame_idx / fps

        srows = [r for r in rows if int(cache["track_id"][r]) in staff_tracks]
        crows = [r for r in rows if int(cache["track_id"][r]) not in staff_tracks]

        near_customers = set()
        for sr in srows:
            sb = cache["bbox"][sr]
            sh = box_height(sb)
            if sh <= 0:
                continue
            for cr in crows:
                # Normalised by the staff member's box height so the rule means
                # the same physical separation at any depth in the scene.
                if box_gap(sb, cache["bbox"][cr]) <= t3["max_box_gap_ratio"] * sh:
                    near_customers.add(int(cache["track_id"][cr]))

        # Update every customer seen this frame: near ones get a positive
        # signal, the rest an explicit negative so their session can close.
        for cr in crows:
            cid = int(cache["track_id"][cr])
            det = detectors.get(cid)
            if det is None:
                det = EpisodeDetector(t3["min_duration_s"], t3["max_gap_s"], source="wrist")
                detectors[cid] = det
            ep = det.update(frame_idx, t, "staff" if cid in near_customers else None)
            if ep:
                sessions.append((cid, ep))
            last_seen[cid] = (frame_idx, t)

    for cid, det in detectors.items():
        f, t = last_seen.get(cid, (total_frames, total_frames / fps))
        ep = det.flush(f, t)
        if ep:
            sessions.append((cid, ep))

    n_staff_instances = 1 if staff_tracks else 0
    total_sessions = len(sessions)
    average = total_sessions / n_staff_instances if n_staff_instances else 0.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "task3_staff_interaction.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["staff_instance", "interaction_sessions", "source_track_ids"])
        w.writerow(["Staff 1", total_sessions,
                    "|".join(str(t) for t in sorted(staff_tracks))])
        w.writerow([])
        w.writerow(["Average sessions per staff member", f"{average:.1f}"])

    with open(OUTPUT_DIR / "task3_interaction_detail.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session_id", "staff_instance", "customer_track_id",
                    "start_time_s", "end_time_s", "duration_s",
                    "start_frame", "end_frame", "observed_frames"])
        for i, (cid, ep) in enumerate(sorted(sessions, key=lambda s: s[1].start_time_s), 1):
            w.writerow([i, "Staff 1", cid, f"{ep.start_time_s:.2f}",
                        f"{ep.end_time_s:.2f}", f"{ep.duration_s:.2f}",
                        ep.start_frame, ep.end_frame, ep.observed_frames])

    print(f"Task 3 -- {video_key}.mp4")
    print(f"  staff tracks identified : {len(staff_tracks)}")
    print(f"  overlapping staff pairs : {overlaps}  (0 supports merging into one instance)")
    print(f"  staff instances         : {n_staff_instances}")
    print(f"  interaction sessions    : {total_sessions}")
    print(f"  AVERAGE sessions/staff  : {average:.1f}")
    if sessions:
        durs = [ep.duration_s for _c, ep in sessions]
        print(f"  session durations (s)   : min={min(durs):.1f} "
              f"median={float(np.median(durs)):.1f} max={max(durs):.1f}")
    return {"sessions": sessions, "staff_tracks": staff_tracks, "average": average}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="entrance")
    args = parser.parse_args()
    run(args.video)
