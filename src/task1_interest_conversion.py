"""Task 1 -- Store Interest and Conversion (entrance.mp4).

Reports Total Interested, Interested Entered, Interested Passed By.

Interest is evidenced by movement: slowing near the storefront, approaching it,
lingering, or reaching into a storefront display rack. A visible face only
corroborates -- the camera looks outward from inside the shop, so a detectable
face is a property of the geometry, not of attention.

Cues are only evaluated for tracks genuinely OUTSIDE the store, since the
metric is about passers-by deciding whether to come in.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from build_cache import load_cache
from tracking import KEYPOINT_INDEX
from zones import load_zones

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FACE_JOINTS = ("nose", "left_eye", "right_eye")


def storefront_axis(zones):
    """Best-fit line through every annotated Entry endpoint.

    Returns (point_on_line, unit_direction, unit_normal). Fitted by PCA rather
    than taking one segment, so all annotated openings contribute and no single
    short segment dominates the geometry.
    """
    pts = np.vstack([ln.points for ln in zones.lines]).astype(np.float64)
    centre = pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts - centre)
    direction = vt[0] / np.linalg.norm(vt[0])
    normal = np.array([-direction[1], direction[0]])
    return centre, direction, normal


def signed_distance(pt, centre, normal) -> float:
    """Perpendicular distance from the storefront axis; sign gives the side."""
    return float(np.dot(np.asarray(pt, dtype=np.float64) - centre, normal))


def has_face(keypoints, kp_conf: float) -> bool:
    """True when any face keypoint is confidently visible.

    The camera faces outward from inside the store, so a visible face means the
    head is oriented toward the shop front.
    """
    return any(keypoints[KEYPOINT_INDEX[j]][2] >= kp_conf for j in FACE_JOINTS)


def run(video_key: str = "entrance") -> dict:
    with open(PROJECT_ROOT / "configs" / "thresholds.yaml") as f:
        cfg = yaml.safe_load(f)
    t1 = cfg["store_interest"]
    kp_conf = cfg["keypoint_confidence"]

    cache = load_cache(video_key)
    zones = load_zones(PROJECT_ROOT / "configs" / f"{video_key}_zones.json")
    fps = float(cache["fps"])
    centre, _direction, normal = storefront_axis(zones)

    # Polarity: decide which sign of the normal means "inside the store" using a
    # reference point known to be indoors. Bottom-centre of the frame is deep
    # inside the shop floor in this camera's view.
    inside_sign = np.sign(signed_distance(t1["inside_reference_xy"], centre, normal))

    by_track = defaultdict(list)
    for row in range(len(cache["frame_idx"])):
        by_track[int(cache["track_id"][row])].append(row)

    results = {}
    for tid, rows in by_track.items():
        rows.sort(key=lambda r: int(cache["frame_idx"][r]))

        samples = []
        for row in rows:
            f = int(cache["frame_idx"][row])
            x1, _y1, x2, y2 = cache["bbox"][row]
            ground = ((float(x1) + float(x2)) / 2.0, float(y2))
            d = signed_distance(ground, centre, normal) * inside_sign
            # Is a wrist inside one of the storefront display racks? Several
            # annotated racks at this entrance face the walkway and are within
            # arm's reach of the corridor, so a passer-by can handle the
            # merchandise without stepping inside.
            touching = False
            kp = cache["keypoints"][row]
            for j in (KEYPOINT_INDEX["left_wrist"], KEYPOINT_INDEX["right_wrist"]):
                wx, wy, wc = kp[j]
                if wc >= kp_conf and zones.containing_polygons(float(wx), float(wy)):
                    touching = True
                    break
            samples.append({
                "frame": f, "t": f / fps, "pt": ground, "dist": d,
                "face": has_face(kp, kp_conf), "touching": touching,
            })

        if len(samples) < 2:
            continue

        # --- entry detection -------------------------------------------------
        # Trajectory-level rather than instantaneous. An earlier version
        # required a sign flip across the fitted storefront axis AND a segment
        # intersection on the SAME frame pair; measured on this footage that
        # never once co-occurred (6 tracks flipped sign, 7 crossed a segment,
        # zero did both), because the global best-fit axis and the individual
        # short openings are not close enough for both tests to fire together.
        #
        # Instead: the person must genuinely start outside, genuinely end up
        # inside, and have passed through one of the annotated openings at some
        # point. Averaging over a settling window rather than trusting single
        # frames also absorbs bounding-box jitter at the threshold.
        settle_frames = max(1, int(t1["settle_s"] * fps))

        def mean_dist(seq):
            return float(np.mean([s["dist"] for s in seq])) if seq else 0.0

        started_outside = mean_dist(samples[:settle_frames]) < 0
        ended_inside = mean_dist(samples[-settle_frames:]) > 0

        crossing_t = None
        for i in range(1, len(samples)):
            if zones.crossed_any_boundary(samples[i - 1]["pt"], samples[i]["pt"]):
                crossing_t = samples[i]["t"]
                break

        entered_at = crossing_t if (started_outside and ended_inside and crossing_t) else None

        # --- passer-by gate --------------------------------------------------
        # Interest cues describe someone OUTSIDE deciding whether to come in, so
        # they may only be evaluated for a track that is genuinely a passer-by.
        #
        # Without this gate a customer already inside the shop gets counted:
        # observed track 367 sat inside for 4.6 minutes (median distance +180px)
        # yet 1.74% of its frames dipped marginally outside -- bounding-box
        # jitter near the threshold -- which was enough to fire "slowing" and
        # label a seated customer an interested passer-by.
        #
        # A margin absorbs that jitter, and the track must be PREDOMINANTLY
        # outside to qualify at all. Anyone who actually entered is exempt:
        # walking in is conclusive evidence of interest, and such a track is
        # legitimately outside then inside.
        margin = t1["outside_margin_px"]
        outside_frames = [s for s in samples if s["dist"] < -margin]
        frac_outside = len(outside_frames) / len(samples)
        time_outside = (outside_frames[-1]["t"] - outside_frames[0]["t"]) if outside_frames else 0.0

        is_passer_by = (
            frac_outside >= t1["min_outside_fraction"]
            and time_outside >= t1["min_outside_s"]
        )
        if not (is_passer_by or entered_at is not None):
            continue

        # --- interest cues (evaluated only while OUTSIDE and near) -----------
        outside_near = [s for s in samples
                        if s["dist"] < -margin and abs(s["dist"]) <= t1["attention_band_px"]]

        cues = set()
        if len(outside_near) >= 2:
            # Per-sample speeds while outside and near the store.
            speeds = []
            for i in range(1, len(outside_near)):
                a, b = outside_near[i - 1], outside_near[i]
                dt = b["t"] - a["t"]
                if dt > 0:
                    speeds.append(float(np.hypot(b["pt"][0] - a["pt"][0],
                                                 b["pt"][1] - a["pt"][1])) / dt)
            speeds = np.array(speeds) if speeds else np.array([0.0])

            faced = sum(1 for s in outside_near if s["face"]) >= t1["face_min_frames"]

            # SLOWING -- was clearly walking, then clearly slowed. Requiring
            # both ends of the transition avoids flagging someone who merely
            # happened to be stationary the whole time (e.g. waiting for a lift
            # across the corridor).
            if speeds.max() >= t1["walking_speed_px_s"] and speeds.min() <= t1["slow_speed_px_s"]:
                cues.add("slowing")

            # LINGERING -- spent a sustained period genuinely slow, not merely
            # present in the band. An earlier version counted any 2s inside the
            # band, which flagged 61% of all passers-by: simply walking through
            # the band takes that long. Requiring the MEDIAN speed to be low
            # means the person actually paused, rather than transited.
            time_near = outside_near[-1]["t"] - outside_near[0]["t"]
            if time_near >= t1["dwell_min_s"] and float(np.median(speeds)) <= t1["slow_speed_px_s"]:
                cues.add("lingering")

            # APPROACHING -- moved substantially closer to the storefront and
            # finished close to it. Directly matches the brief's "approaching
            # the entrance" cue, and is direction-aware rather than distance-
            # agnostic.
            d_start = abs(outside_near[0]["dist"])
            d_end = abs(outside_near[-1]["dist"])
            if d_start - d_end >= t1["approach_min_px"] and d_end <= t1["approach_final_px"]:
                cues.add("approaching")

            # TOUCHING_DISPLAY -- reached into a storefront display rack while
            # still outside the shop. This is the strongest interest evidence
            # available: the person physically handled the merchandise without
            # stepping in. Several racks at this entrance face the walkway and
            # are within arm's reach of the corridor, so this is a real
            # behaviour rather than an artefact -- observed on track 1139,
            # standing at median -64px and reaching into rack_K.
            #
            # Requires display_touch_min_frames of contact rather than a single
            # frame: the Task 2 run-length analysis showed 75% of raw
            # wrist-in-rack runs last under 0.5s and are polygon-edge clipping
            # as someone walks past, not deliberate handling.
            if sum(1 for s in outside_near if s["touching"]) >= t1["display_touch_min_frames"]:
                cues.add("touching_display")

            # FACING is deliberately NOT sufficient on its own. The camera looks
            # outward from inside the shop, so anyone walking down the corridor
            # toward it presents a detectable face -- that is a consequence of
            # the camera geometry, not evidence of attention. It is kept only as
            # corroboration of a movement-based cue.
            if faced and cues:
                cues.add("facing")

        # Entering is itself conclusive evidence of interest.
        if entered_at is not None:
            cues.add("entered")

        if cues:
            results[tid] = {
                "cues": sorted(cues),
                "entered": entered_at is not None,
                "entered_at": entered_at,
                "first_t": samples[0]["t"],
                "last_t": samples[-1]["t"],
                "duration_s": samples[-1]["t"] - samples[0]["t"],
                "min_dist_px": min(abs(s["dist"]) for s in samples),
            }

    # Tracks too short to be a real person-visit are excluded: the tracker
    # produces brief fragments in a busy mall walkway, and counting each as a
    # distinct interested passer-by would badly inflate the total.
    interested = {tid: r for tid, r in results.items()
                  if r["duration_s"] >= t1["min_track_duration_s"]}
    entered = {tid: r for tid, r in interested.items() if r["entered"]}
    passed_by = {tid: r for tid, r in interested.items() if not r["entered"]}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "task1_store_interest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "count"])
        w.writerow(["Total Interested", len(interested)])
        w.writerow(["Interested Entered", len(entered)])
        w.writerow(["Interested Passed By", len(passed_by)])

    with open(OUTPUT_DIR / "task1_interest_detail.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "outcome", "interest_cues", "first_seen_s",
                    "last_seen_s", "duration_s", "entered_at_s", "closest_approach_px"])
        for tid, r in sorted(interested.items(), key=lambda kv: kv[1]["first_t"]):
            w.writerow([
                tid, "entered" if r["entered"] else "passed_by",
                "|".join(r["cues"]), f"{r['first_t']:.2f}", f"{r['last_t']:.2f}",
                f"{r['duration_s']:.2f}",
                f"{r['entered_at']:.2f}" if r["entered_at"] is not None else "",
                f"{r['min_dist_px']:.0f}",
            ])

    print(f"Task 1 -- {video_key}.mp4")
    print(f"  Total Interested      {len(interested)}")
    print(f"  Interested Entered    {len(entered)}")
    print(f"  Interested Passed By  {len(passed_by)}")
    cue_counts = defaultdict(int)
    for r in interested.values():
        for c in r["cues"]:
            cue_counts[c] += 1
    print(f"  cue breakdown: {dict(cue_counts)}")
    return {"interested": interested, "entered": entered, "passed_by": passed_by}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="entrance")
    args = parser.parse_args()
    run(args.video)
