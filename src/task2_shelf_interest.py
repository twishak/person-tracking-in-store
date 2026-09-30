"""Task 2 -- Per-Shelf Customer Interest.

Counts discrete interest events per annotated shelf region.

Evidence is a WRIST inside a region, not body proximity: shoppers stand beside
a rack and reach into it, so a foot-position test misses most engagement
(measured: wrist fires on 22.0% of person-observations vs 8.1% for feet). It
also settles the brief's "customer between two shelves" case -- the region
containing the hand is the shelf being engaged with.

Sustained body presence without a confirmed wrist is reported separately as an
"unsure" event rather than being discarded or counted as fact.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from build_cache import load_cache
from tracking import KEYPOINT_INDEX
from episodes import UNKNOWN, EpisodeDetector
from zones import load_zones

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
WRIST_IDX = (KEYPOINT_INDEX["left_wrist"], KEYPOINT_INDEX["right_wrist"])


# The brief designates shelves in ONE view only: Figure 1 labels four shelves
# (A-D) in interior.mp4, and Figure 2 marks none in the entrance view. So a
# region is part of the required deliverable only if it is a "Shelf_*" region
# in the interior clip. Entrance fixtures are annotated and analysed too, but
# they are supplementary by definition -- naming them Shelf_E..H keeps them
# distinct from the designated four without implying the brief asked for them.
DESIGNATED_VIDEO = "interior"


def is_designated(zone_name: str, video_key: str) -> bool:
    return video_key == DESIGNATED_VIDEO and zone_name.lower().startswith("shelf")


def load_thresholds() -> dict:
    with open(PROJECT_ROOT / "configs" / "thresholds.yaml") as f:
        return yaml.safe_load(f)


def frame_signal(keypoints: np.ndarray, zones, kp_conf: float):
    """Map one person-observation to a tri-state WRIST signal (sure evidence)."""
    wrists = []
    for i in WRIST_IDX:
        x, y, c = keypoints[i]
        if c >= kp_conf:
            wrists.append((float(x), float(y)))

    if not wrists:
        return UNKNOWN  # cannot evaluate -- never assume "not engaging"

    for wx, wy in wrists:
        polys = zones.containing_polygons(wx, wy)
        if polys:
            # Overlapping regions: attribute to the nearest centroid. This is
            # the documented tie-break for the brief's "between two shelves"
            # case -- the wrist's own position decides, not the body's.
            best = min(polys, key=lambda p: (p.centroid[0] - wx) ** 2 + (p.centroid[1] - wy) ** 2)
            return best.name

    return None  # observed, and not in any rack


def body_signal(bbox, zones):
    """Which shelf region the person is STANDING in, if any.

    Uses the bottom-centre of the box (where the person meets the floor)
    rather than the box centre: the regions are floor-plane polygons, and an
    overhead camera puts a standing person's box centre well above their
    actual floor position.

    This is the weaker, secondary signal behind "unsure" events -- presence
    without confirmed reaching.
    """
    x1, _y1, x2, y2 = bbox
    gx, gy = (float(x1) + float(x2)) / 2.0, float(y2)
    polys = zones.containing_polygons(gx, gy)
    if not polys:
        return None
    return min(polys, key=lambda p: (p.centroid[0] - gx) ** 2 + (p.centroid[1] - gy) ** 2).name


def run(video_key: str = "interior") -> dict:
    cfg = load_thresholds()
    si = cfg["shelf_interest"]
    kp_conf = cfg["keypoint_confidence"]

    cache = load_cache(video_key)
    zones = load_zones(PROJECT_ROOT / "configs" / f"{video_key}_zones.json")
    fps = float(cache["fps"])

    # Group observations by track, preserving frame order.
    by_track = defaultdict(list)
    for row in range(len(cache["frame_idx"])):
        by_track[int(cache["track_id"][row])].append(row)

    events = []
    for track_id, rows in by_track.items():
        # Two independent detectors per track:
        #   wrist -> SURE events (confirmed reaching into the shelf region)
        #   body  -> candidate UNSURE events (sustained presence, no confirmed
        #            reach). Filtered below so presence that already produced a
        #            sure event is not double counted.
        wrist_det = EpisodeDetector(
            min_duration_s=si["min_duration_s"], max_gap_s=si["max_gap_s"],
            source="wrist",
        )
        body_det = EpisodeDetector(
            min_duration_s=si["body_presence_min_s"], max_gap_s=si["max_gap_s"],
            source="body",
        )

        wrist_eps, body_eps = [], []
        last_frame, last_time = 0, 0.0
        for row in rows:
            frame_idx = int(cache["frame_idx"][row])
            t = frame_idx / fps

            ep = wrist_det.update(frame_idx, t, frame_signal(cache["keypoints"][row], zones, kp_conf))
            if ep:
                wrist_eps.append(ep)

            ep = body_det.update(frame_idx, t, body_signal(cache["bbox"][row], zones))
            if ep:
                body_eps.append(ep)

            last_frame, last_time = frame_idx, t

        ep = wrist_det.flush(last_frame, last_time)
        if ep:
            wrist_eps.append(ep)
        ep = body_det.flush(last_frame, last_time)
        if ep:
            body_eps.append(ep)

        events.extend((track_id, e) for e in wrist_eps)

        # A body episode only becomes an "unsure" event if it does NOT overlap
        # a sure event on the same shelf -- otherwise the same visit would be
        # counted twice, once as confirmed reaching and again as mere presence.
        for be in body_eps:
            overlaps = any(
                we.label == be.label
                and we.start_time_s <= be.end_time_s
                and be.start_time_s <= we.end_time_s
                for we in wrist_eps
            )
            if not overlaps:
                events.append((track_id, be))

    # Per-shelf tallies. Every annotated rack appears, including zero counts --
    # a shelf nobody touched is a real finding, not a missing row.
    counts = {p.name: {"sure": 0, "unsure": 0} for p in zones.polygons}
    for _tid, ep in events:
        counts[ep.label][ep.status] += 1

    # The brief designates specific shelves (Figure 1 labels four, A-D). Those
    # are the required deliverable and are reported alone in the headline CSV;
    # any other annotated fixture is supplementary and goes to its own file, so
    # a reviewer cannot mistake extra analysis for the required result.
    designated = {n: v for n, v in counts.items() if is_designated(n, video_key)}
    supplementary = {n: v for n, v in counts.items() if not is_designated(n, video_key)}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def write_summary(path, table, label):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            # interest_events (sure) is the headline metric the brief asks for;
            # unsure_events is reported alongside rather than folded in, so the
            # primary count rests only on confirmed evidence.
            w.writerow([label, "interest_events", "unsure_events"])
            for name in sorted(table):
                w.writerow([name, table[name]["sure"], table[name]["unsure"]])
            w.writerow(["TOTAL",
                        sum(v["sure"] for v in table.values()),
                        sum(v["unsure"] for v in table.values())])

    # Only write the designated-shelf CSV where designated shelves exist. For
    # the entrance clip the brief designates none, so emitting a near-empty
    # file there would imply a required result that was measured as zero.
    summary_path = None
    if designated:
        summary_path = OUTPUT_DIR / f"task2_{video_key}_shelf_interest.csv"
        write_summary(summary_path, designated, "shelf")

    supp_path = None
    if supplementary:
        supp_path = OUTPUT_DIR / f"task2_{video_key}_supplementary_zones.csv"
        write_summary(supp_path, supplementary, "zone")

    detail_path = OUTPUT_DIR / f"task2_{video_key}_shelf_events_detail.csv"
    with open(detail_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "event_id", "track_id", "shelf", "start_time_s", "end_time_s",
            "duration_s", "start_frame", "end_frame", "status", "evidence",
            "observed_frames", "bridged_frames", "longest_unknown_run",
            "observation_confidence",
        ])
        for i, (tid, ep) in enumerate(sorted(events, key=lambda e: e[1].start_time_s), 1):
            w.writerow([
                i, tid, ep.label, f"{ep.start_time_s:.2f}", f"{ep.end_time_s:.2f}",
                f"{ep.duration_s:.2f}", ep.start_frame, ep.end_frame,
                ep.status, ep.source, ep.observed_frames, ep.bridged_frames,
                ep.longest_unknown_run, f"{ep.confidence:.2f}",
            ])

    def totals(table):
        return (sum(v["sure"] for v in table.values()),
                sum(v["unsure"] for v in table.values()))

    print(f"Task 2 -- {video_key}.mp4")
    print(f"  thresholds: min_duration={si['min_duration_s']}s  "
          f"max_gap={si['max_gap_s']}s  body_presence={si['body_presence_min_s']}s")

    if designated:
        ds, du = totals(designated)
        print("\n  DESIGNATED SHELVES (the required deliverable)")
        print(f"  {'shelf':12s} {'sure':>6s} {'unsure':>7s}")
        for name in sorted(designated):
            print(f"  {name:12s} {designated[name]['sure']:6d} {designated[name]['unsure']:7d}")
        print(f"  {'TOTAL':12s} {ds:6d} {du:7d}")
    else:
        print("\n  (this view has no designated shelves -- the brief marks none)")

    if supplementary:
        ss, su = totals(supplementary)
        print(f"\n  supplementary zones (not required by the brief)")
        for name in sorted(supplementary):
            print(f"  {name:12s} {supplementary[name]['sure']:6d} "
                  f"{supplementary[name]['unsure']:7d}")
        print(f"  {'TOTAL':12s} {ss:6d} {su:7d}")

    written = ", ".join(p.name for p in (summary_path, supp_path) if p)
    print(f"\n  wrote {written} and {detail_path.name} to outputs/")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="interior")
    args = parser.parse_args()
    run(args.video)
