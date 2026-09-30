"""Reports the distributions every threshold in configs/thresholds.yaml came from.

Track durations, wrist-in-rack run lengths, gaps between runs, walking speeds,
plus a sensitivity sweep of event count against (min_duration, max_gap). Reads
the cache, so re-running after a zone edit is instant.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from build_cache import load_cache
from tracking import KEYPOINT_INDEX, MIN_KEYPOINT_CONF
from episodes import UNKNOWN, EpisodeDetector
from zones import load_zones

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WRIST_IDX = (KEYPOINT_INDEX["left_wrist"], KEYPOINT_INDEX["right_wrist"])


def percentiles(values, label, unit="s"):
    if len(values) == 0:
        print(f"  {label}: no data")
        return
    v = np.asarray(values, dtype=np.float64)
    qs = np.percentile(v, [10, 25, 50, 75, 90, 95])
    print(f"  {label}: n={len(v)}  mean={v.mean():.2f}{unit}")
    print(f"    p10={qs[0]:.2f}  p25={qs[1]:.2f}  p50={qs[2]:.2f}  "
          f"p75={qs[3]:.2f}  p90={qs[4]:.2f}  p95={qs[5]:.2f}")


def per_track_signals(cache, zones):
    """signals[track_id] = list of (frame_idx, timestamp, label-or-None-or-UNKNOWN)."""
    fps = float(cache["fps"])
    signals = defaultdict(list)
    kps, tids, frames = cache["keypoints"], cache["track_id"], cache["frame_idx"]

    for row in range(len(frames)):
        wrists = []
        for i in WRIST_IDX:
            x, y, c = kps[row][i]
            if c >= MIN_KEYPOINT_CONF:
                wrists.append((float(x), float(y)))

        if not wrists:
            label = UNKNOWN  # wrist not observed -> cannot evaluate
        else:
            hit = None
            for wx, wy in wrists:
                polys = zones.containing_polygons(wx, wy)
                if polys:
                    # Ambiguity rule: when a wrist falls inside overlapping
                    # polygons, attribute to the nearest centroid. Documented
                    # answer to the brief's "region between two shelves" case.
                    hit = min(polys, key=lambda p: (p.centroid[0] - wx) ** 2
                              + (p.centroid[1] - wy) ** 2).name
                    break
            label = hit  # None means "observed, not in any rack"

        signals[int(tids[row])].append((int(frames[row]), float(frames[row]) / fps, label))

    return signals


def run_lengths(signals):
    """Raw contiguous runs of the same rack label, and the gaps between them."""
    runs, gaps = [], []
    for _tid, seq in signals.items():
        current_label, run_start, last_positive = None, None, None
        for _frame, t, label in seq:
            if label == UNKNOWN:
                continue
            if label is not None and label == current_label:
                last_positive = t
            elif label is not None:
                if current_label is not None:
                    runs.append(last_positive - run_start)
                current_label, run_start, last_positive = label, t, t
            else:
                if current_label is not None:
                    runs.append(last_positive - run_start)
                    gap_start = last_positive
                    current_label = None
                    # find when (if ever) this track next goes positive
                    for _f2, t2, l2 in seq:
                        if t2 > gap_start and l2 not in (None, UNKNOWN):
                            gaps.append(t2 - gap_start)
                            break
        if current_label is not None:
            runs.append(last_positive - run_start)
    return runs, gaps


def sensitivity(signals, min_durations, max_gaps):
    """Event counts over a threshold grid -- shows how brittle the metric is."""
    print("\n5. Sensitivity of total event count to thresholds")
    header = "     min_dur \\ max_gap  " + "".join(f"{g:>8.1f}s" for g in max_gaps)
    print(header)
    for md in min_durations:
        row = [f"      {md:>9.1f}s      "]
        for mg in max_gaps:
            total = 0
            for _tid, seq in signals.items():
                det = EpisodeDetector(min_duration_s=md, max_gap_s=mg)
                for frame, t, label in seq:
                    if det.update(frame, t, label):
                        total += 1
                if det.flush(seq[-1][0], seq[-1][1]):
                    total += 1
            row.append(f"{total:>8d}")
        print("".join(row))


def speeds(cache):
    """Per-track ground-point speed samples, px/s, for the 'slowing down' cue."""
    fps = float(cache["fps"])
    bboxes, tids, frames = cache["bbox"], cache["track_id"], cache["frame_idx"]
    by_track = defaultdict(list)
    for row in range(len(frames)):
        x1, _y1, x2, y2 = bboxes[row]
        by_track[int(tids[row])].append((int(frames[row]), (x1 + x2) / 2.0, float(y2)))

    samples = []
    for _tid, pts in by_track.items():
        pts.sort()
        for i in range(1, len(pts)):
            df = pts[i][0] - pts[i - 1][0]
            if df <= 0:
                continue
            dist = np.hypot(pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2])
            samples.append(dist / (df / fps))
    return samples


def analyse(video_key: str):
    print(f"\n{'=' * 68}\n{video_key}.mp4\n{'=' * 68}")
    cache = load_cache(video_key)
    zones = load_zones(PROJECT_ROOT / "configs" / f"{video_key}_zones.json")
    fps = float(cache["fps"])

    # 1. track durations
    tids, frames = cache["track_id"], cache["frame_idx"]
    durations = []
    for tid in np.unique(tids):
        f = frames[tids == tid]
        durations.append((f.max() - f.min() + 1) / fps)
    print("\n1. Track durations")
    percentiles(durations, "all tracks")
    short = sum(1 for d in durations if d < 1.0)
    print(f"    tracks under 1s: {short}/{len(durations)} "
          f"({100 * short / len(durations):.0f}%) -- likely fragments/noise")

    # 2 & 3. wrist-in-rack runs and gaps
    signals = per_track_signals(cache, zones)
    runs, gaps = run_lengths(signals)
    print("\n2. Raw wrist-in-rack run lengths (before any thresholding)")
    percentiles(runs, "runs")
    for cutoff in (0.5, 1.0, 1.5, 2.0, 3.0):
        n = sum(1 for r in runs if r < cutoff)
        print(f"    runs under {cutoff:.1f}s: {n}/{len(runs)} ({100 * n / max(1, len(runs)):.0f}%)")

    print("\n3. Gaps between consecutive runs (same track)")
    percentiles(gaps, "gaps")
    for cutoff in (0.5, 1.0, 2.0, 3.0, 5.0):
        n = sum(1 for g in gaps if g < cutoff)
        print(f"    gaps under {cutoff:.1f}s: {n}/{len(gaps)} ({100 * n / max(1, len(gaps)):.0f}%)")

    # 4. speeds
    print("\n4. Person speed (ground point, px/s)")
    percentiles(speeds(cache), "speed samples", unit="px/s")

    # 5. sensitivity
    sensitivity(signals, [1.0, 1.5, 2.0, 3.0], [0.5, 1.0, 2.0, 3.0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="*", help="default: both")
    args = parser.parse_args()
    for key in (args.videos or ["entrance", "interior"]):
        analyse(key)
