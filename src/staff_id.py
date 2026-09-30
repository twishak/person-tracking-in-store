"""Staff vs customer classification from the red apron.

The torso is taken from the shoulder/hip keypoints, not a slice of the bounding
box: a box is mostly background, and this store's warm wood shelving occupies
the same hue range as the uniform (measured: 56% red for staff vs 49% for the
general population -- useless).

Classification is per TRACK, never per frame, so one bad viewing angle cannot
flip the label mid-track.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from build_cache import CACHE_DIR, load_cache

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Red apron gate. Hue is circular in OpenCV (0-179) and red straddles the seam,
# so it is expressed as two arcs either side of the wrap point.
# Gate bounds measured from the verified staff crops in data/persons_dense/
# against a random sample of torsos from the footage, rather than guessed.
#
# Hue alone is nearly useless here -- the store's warm wood and warm lighting
# put a quarter of ALL torsos in the red arc. The uniform separates on
# SATURATION and VALUE: staff torso pixels measure median S=93 / V=72, against
# population medians of S=48 / V=111. The uniform is a deep, saturated colour;
# most other clothing in shot is pale and washed out.
#
# Sweeping the three bounds for the best staff-vs-population contrast:
#   H+-12, S>=60,  V<=255  (previous)  staff 63.4%  pop 25.7%   2.5x
#   H+-8,  S>=100, V<=150              staff 38.5%  pop  4.8%   8.1x
#   H+-8,  S>=120, V<=150  (chosen)    staff 32.3%  pop  2.8%  11.4x
#
# The chosen gate accepts a smaller share of the true uniform in exchange for
# rejecting almost all of the population -- the right trade when a false STAFF
# label on a customer is the failure that matters.
# Gates are PER VIEW. The two cameras see the same uniform under different
# lighting and at different distances, and a single gate cannot serve both:
# tuned for the entrance it rejects the interior uniform entirely, and relaxed
# enough for the interior it re-admits a quarter of the entrance population.
#
# Measured staff-vs-population contrast for the chosen bounds:
#   entrance  H+-8,  S>=120, V<=150   staff 32.3%  pop 2.8%   11.4x
#   interior  H+-10, S>=60,  V<=120   staff 32.1%  pop 7.5%    4.3x
#
# The interior gate is materially weaker, and is treated accordingly: interior
# staff labelling is presentational only (no metric consumes it), whereas Task
# 3 -- the only staff metric -- runs on the entrance view where separation is
# strong.
# `threshold` is the per-view cut on the aggregated track score. It differs by
# view for the same reason the gate does -- the score distributions are not on a
# common scale. Entrance shows a clean cliff (9 tracks at 0.65-0.96, next 0.18),
# so 0.40 sits in open space. Interior declines gradually with no cliff, so 0.25
# is a judgement call rather than a gap, and is documented as such.
GATES = {
    "entrance": dict(hue=8, min_sat=120, min_val=30, max_val=150, threshold=0.40),
    "interior": dict(hue=10, min_sat=60, min_val=30, max_val=120, threshold=0.25),
}
DEFAULT_GATE = GATES["entrance"]

# Legacy module-level names, kept so the calibration/analysis scripts that
# import them keep working; they mirror the entrance gate.
HUE_LOW_MAX = DEFAULT_GATE["hue"]
HUE_HIGH_MIN = 180 - DEFAULT_GATE["hue"]
MIN_SAT = DEFAULT_GATE["min_sat"]
MIN_VAL, MAX_VAL = DEFAULT_GATE["min_val"], DEFAULT_GATE["max_val"]

MIN_TORSO_PIXELS = 40  # too few pixels -> score is unreliable, return None

from tracking import KEYPOINT_INDEX, MIN_KEYPOINT_CONF  # noqa: E402

_TORSO_JOINTS = ("left_shoulder", "right_shoulder", "right_hip", "left_hip")


def torso_mask_from_keypoints(shape, keypoints: np.ndarray, offset=(0, 0)):
    """Polygon mask of the actual torso, from shoulder+hip keypoints.

    Replaces the earlier "middle slice of the bounding box" heuristic, which
    was measured to be useless here: a person's box is mostly background, and
    this store's background (warm wood shelving, skin, warm lighting) occupies
    the same hue range as the apron. Sampling the shoulder-to-hip quadrilateral
    measures the GARMENT rather than the scene behind it.

    Returns None when the four torso joints are not confidently observed --
    "unknown", never "not staff".
    """
    pts = []
    for name in _TORSO_JOINTS:
        x, y, c = keypoints[KEYPOINT_INDEX[name]]
        if c < MIN_KEYPOINT_CONF:
            return None
        pts.append([x - offset[0], y - offset[1]])

    poly = np.array(pts, dtype=np.int32)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask.astype(bool)


def apron_score(crop: np.ndarray, keypoints: np.ndarray = None, offset=(0, 0),
                gate: dict = None):
    """Fraction of torso pixels matching the uniform colour, under `gate`.

    Returns None when the torso cannot be located or is too small to judge.
    None means 'unknown', never 'not staff'.
    """
    if crop.size == 0 or keypoints is None:
        return None

    g = gate or DEFAULT_GATE
    torso = torso_mask_from_keypoints(crop.shape, keypoints, offset)
    if torso is None or torso.sum() < MIN_TORSO_PIXELS:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    is_red = (((H <= g["hue"]) | (H >= 180 - g["hue"]))
              & torso & (S >= g["min_sat"])
              & (V >= g["min_val"]) & (V <= g["max_val"]))

    # Denominator is the WHOLE torso, not just its saturated pixels.
    #
    # Dividing by saturated pixels only was measured to be badly wrong: a dark,
    # desaturated garment has almost nothing clearing the saturation floor, so
    # the ratio was computed over a handful of warm-tinted pixels and returned
    # ~1.0. Observed on this footage -- customer tracks scoring 1.000 while
    # only 2-15% of their torso was saturated at all, against genuine aprons
    # covering 50%+ of the torso. Dividing by torso area asks the question the
    # signal is actually meant to answer: how much of this person's torso is
    # apron-coloured?
    return float(is_red.sum()) / float(torso.sum())


def build_scores(video_key: str) -> Path:
    """One decode pass over the video, scoring every cached person-observation.

    Decode-only (no model inference), so this is far cheaper than the tracking
    pass and is kept separate from it.
    """
    gate = GATES.get(video_key, DEFAULT_GATE)
    cache = load_cache(video_key)
    video_path = PROJECT_ROOT / "data" / "raw_videos" / f"{video_key}.mp4"
    cap = cv2.VideoCapture(str(video_path))

    frame_idx_arr = cache["frame_idx"]
    bboxes = cache["bbox"]
    scores = np.full(len(frame_idx_arr), np.nan, dtype=np.float32)

    row = 0
    current_frame = 0
    total = int(cache["total_frames"])
    while row < len(frame_idx_arr):
        target = int(frame_idx_arr[row])
        while current_frame < target:
            ok, frame = cap.read()
            current_frame += 1
            if not ok:
                break
        if not ok:
            break

        # score every observation belonging to this frame
        while row < len(frame_idx_arr) and int(frame_idx_arr[row]) == target:
            kp = cache["keypoints"][row]
            # Crop tightly around the torso joints (not the whole person box)
            # and shift the keypoints into crop coordinates.
            joints = [kp[KEYPOINT_INDEX[n]] for n in _TORSO_JOINTS]
            if all(j[2] >= MIN_KEYPOINT_CONF for j in joints):
                xs = [j[0] for j in joints]
                ys = [j[1] for j in joints]
                x1 = max(0, int(min(xs)) - 2)
                y1 = max(0, int(min(ys)) - 2)
                x2 = min(frame.shape[1], int(max(xs)) + 3)
                y2 = min(frame.shape[0], int(max(ys)) + 3)
                crop = frame[y1:y2, x1:x2]
                if crop.size:
                    s = apron_score(crop, kp, offset=(x1, y1), gate=gate)
                    if s is not None:
                        scores[row] = s
            row += 1

        if target % 3000 == 0:
            print(f"  {target}/{total}", flush=True)

    cap.release()

    out = CACHE_DIR / f"{video_key}_apron.npz"
    np.savez_compressed(out, apron_score=scores)
    valid = np.isfinite(scores)
    print(f"  -> {out}  ({valid.sum()}/{len(scores)} observations scored)")
    return out


def load_scores(video_key: str) -> np.ndarray:
    path = CACHE_DIR / f"{video_key}_apron.npz"
    if not path.exists():
        raise FileNotFoundError(f"run: python src/staff_id.py {video_key} --build")
    return np.load(path)["apron_score"]


# Videos where exactly one staff member is known to appear, so the
# single-staff constraint may be applied. Verified by manual review for
# entrance.mp4. NOT applied to interior.mp4: that view includes the service
# counter, and tracks scoring 0.98-0.99 for 20+ seconds appear there
# concurrently -- near-total apron coverage on two people at once, i.e. more
# than one staff member is genuinely present. Forcing a single instance there
# demoted those tracks to "customer" and visibly mislabelled staff on screen.
SINGLE_STAFF_VIDEOS = {"entrance"}


def classify_tracks(video_key: str, staff_threshold: float | None = None,
                    min_scored_frames: int = 5, percentile: float | None = None,
                    enforce_single_staff: bool | None = None) -> dict[int, dict]:
    """Aggregate per-frame apron scores to a per-track STAFF/CUSTOMER label.

    Aggregated at TRACK level, never per frame, so one bad viewing angle cannot
    flip the label mid-track and make the overlay flicker.

    The statistic is an upper percentile, not the median. The apron is only
    clearly visible for part of any given track -- the staff member turns,
    bends to fit a shoe, and is partly occluded by the customer they are
    serving -- so a median is dragged below threshold by frames in which the
    garment simply is not in view, and the track is wrongly called a customer.
    Measured effect: the median rule recognised only 32s of staff screen time
    against a manually verified ground truth of ~148s (4,427 frames across the
    36 hand-checked staff tracks). Asking instead whether the top ~30% of a
    track's frames show the apron recovers 161s, without admitting tracks that
    never show it at all.
    """
    cfg = {}
    if staff_threshold is None or percentile is None:
        with open(PROJECT_ROOT / "configs" / "thresholds.yaml") as f:
            cfg = yaml.safe_load(f).get("staff", {})
    if staff_threshold is None:
        staff_threshold = GATES.get(video_key, DEFAULT_GATE).get("threshold", 0.40)
    if percentile is None:
        percentile = cfg.get("aggregation_percentile", 70)

    cache = load_cache(video_key)
    scores = load_scores(video_key)
    tids = cache["track_id"]

    by_track = defaultdict(list)
    for row in range(len(tids)):
        if np.isfinite(scores[row]):
            by_track[int(tids[row])].append(float(scores[row]))

    result = {}
    for tid in np.unique(tids):
        vals = by_track.get(int(tid), [])
        if len(vals) < min_scored_frames:
            # Too little evidence to judge -- explicitly unknown rather than
            # silently defaulting to "customer".
            result[int(tid)] = {"label": "unknown", "median": None, "n": len(vals)}
            continue
        score = float(np.percentile(vals, percentile))
        result[int(tid)] = {
            "label": "staff" if score >= staff_threshold else "customer",
            "median": score,  # the aggregated statistic, kept under this key
            "n": len(vals),
        }

    if enforce_single_staff is None:
        enforce_single_staff = video_key in SINGLE_STAFF_VIDEOS
    if enforce_single_staff:
        result = _resolve_overlapping_staff(result, cache)
    return result


def _resolve_overlapping_staff(result: dict, cache) -> dict:
    """Enforce the domain constraint that only ONE staff member exists.

    Why this is needed: sweeping the threshold showed no setting that achieves
    both adequate coverage and internal consistency. Settings recovering a
    realistic ~150-180s of staff screen time also produced 7-14 pairs of
    staff-labelled tracks that overlap in TIME -- and two staff cannot be on
    screen simultaneously when there is only one. Settings with no overlaps
    recovered barely a quarter of the staff's actual screen time.

    Rather than accept either a contradictory or a crippled result, the
    constraint is applied directly: among staff candidates that overlap in
    time, the highest-scoring one is kept and the others are demoted to
    customer. Score order is the right tie-break because the score IS the
    evidence -- the track showing the apron most clearly is the real staff
    member; the overlapping lower-scoring track is a customer whose clothing
    happened to pass the colour gate.

    This uses ground truth we genuinely have (one staff member, confirmed by
    manual review) rather than a tuning parameter, so it adds no unjustified
    freedom to the model.
    """
    frames = cache["frame_idx"]
    tids = cache["track_id"]
    span: dict[int, list[int]] = {}
    for row in range(len(frames)):
        tid = int(tids[row])
        f = int(frames[row])
        if tid in span:
            a, b = span[tid]
            span[tid] = [min(a, f), max(b, f)]
        else:
            span[tid] = [f, f]

    fps = float(cache["fps"])
    tol = fps * 0.5  # ignore half-second brushes at track boundaries

    # Rank by TOTAL apron evidence (score x observed frames), not by score
    # alone. A 0.2s track scoring 0.99 is far weaker evidence than a 30s track
    # scoring 0.85, yet score-only ranking let the former evict the latter and
    # cost 88s of correctly-identified staff time.
    candidates = sorted(
        (t for t, v in result.items() if v["label"] == "staff"),
        key=lambda t: -(result[t]["median"] * result[t]["n"]),
    )

    kept: list[int] = []
    for tid in candidates:
        a, b = span[tid]
        clash = any(not (b < span[k][0] + tol or a > span[k][1] - tol) for k in kept)
        if clash:
            result[tid] = {**result[tid], "label": "customer", "demoted_overlap": True}
        else:
            kept.append(tid)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="*", help="default: both")
    parser.add_argument("--build", action="store_true", help="compute and cache apron scores")
    args = parser.parse_args()

    for key in (args.videos or ["entrance", "interior"]):
        print(f"\n=== {key} ===")
        if args.build:
            build_scores(key)

        labels = classify_tracks(key)
        scored = {k: v for k, v in labels.items() if v["median"] is not None}
        staff = {k: v for k, v in scored.items() if v["label"] == "staff"}

        print(f"  tracks: {len(labels)}  scored: {len(scored)}  "
              f"staff: {len(staff)}  customer: {len(scored) - len(staff)}  "
              f"unknown: {len(labels) - len(scored)}")

        if scored:
            meds = np.array([v["median"] for v in scored.values()])
            qs = np.percentile(meds, [10, 25, 50, 75, 90, 95, 99])
            print("  median-apron-score distribution across tracks:")
            print(f"    p10={qs[0]:.3f} p25={qs[1]:.3f} p50={qs[2]:.3f} "
                  f"p75={qs[3]:.3f} p90={qs[4]:.3f} p95={qs[5]:.3f} p99={qs[6]:.3f}")

        top = sorted(scored.items(), key=lambda kv: -kv[1]["median"])[:15]
        print("  highest-scoring tracks:")
        for tid, info in top:
            print(f"    track {tid:5d}  median={info['median']:.3f}  "
                  f"frames={info['n']:5d}  -> {info['label']}")
