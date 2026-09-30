"""Interior deliverable video for Task 2.

Shows the shelf each customer is engaging with, a live dwell timer, wrist
keypoints (the evidence the count rests on), and cumulative per-shelf counts.
Replays the same episode logic as the CSV, so the two cannot disagree.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

from build_cache import load_cache, rows_for_frame
from staff_id import classify_tracks
from episodes import EpisodeDetector
from task2_shelf_interest import body_signal, frame_signal
from tracking import KEYPOINT_INDEX
from zones import load_zones

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

COL_IDLE = (120, 120, 120)
COL_ACTIVE = (0, 230, 0)
COL_UNSURE = (0, 200, 255)
COL_STAFF = (255, 0, 255)  # magenta -- apron-wearing staff
COL_BOX = (200, 160, 0)
COL_PANEL_BG = (0, 0, 0)
COL_TEXT = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Closing seconds are marked FINAL, so the reported totals are clearly shown
# without the panel announcing them from the first frame.
FINAL_BANNER_S = 8.0


def draw_panel(frame, counts, active_count, elapsed_s, total_events, unsure_total,
               is_final=False):
    """Cumulative per-shelf tally, top-left (as required by the brief).

    Counts accumulate as their evidence appears; the final figure is never
    displayed in advance. The closing frames are marked FINAL, at which point
    the running values are the reported results.
    """
    # Only racks that have registered something are listed. Printing all eight
    # (five of which stay at zero) made the panel 254px tall and 250px wide,
    # which measurably covered rack_G and clipped an Entry line endpoint at
    # (244,243) -- i.e. the panel was hiding the very evidence it reports on.
    # Zero-count racks are still reported in full in the CSV.
    names = [n for n in sorted(counts) if counts[n]["sure"] or counts[n]["unsure"]]
    line_h = 21
    height = line_h * (len(names) + 3) + 10
    width = 228
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), COL_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = 24
    cv2.putText(frame, "SHELF INTEREST EVENTS" + ("  -- FINAL" if is_final else ""),
                (10, y), FONT, 0.5, COL_ACTIVE if is_final else COL_TEXT, 1, cv2.LINE_AA)
    y += line_h
    for name in names:
        s, u = counts[name]["sure"], counts[name]["unsure"]
        label = f"{name:<9s} {s:>3d}"
        if u:
            label += f"   +{u}?"
        colour = COL_ACTIVE if s else (COL_UNSURE if u else COL_IDLE)
        cv2.putText(frame, label, (10, y), FONT, 0.5, colour, 1, cv2.LINE_AA)
        y += line_h

    y += 4
    cv2.putText(frame, f"TOTAL {total_events}   unsure +{unsure_total}",
                (10, y), FONT, 0.5, COL_TEXT, 1, cv2.LINE_AA)
    y += line_h
    cv2.putText(frame, f"active now: {active_count}   t={elapsed_s:6.1f}s",
                (10, y), FONT, 0.45, COL_TEXT, 1, cv2.LINE_AA)


def render(video_key: str = "interior") -> Path:
    with open(PROJECT_ROOT / "configs" / "thresholds.yaml") as f:
        cfg = yaml.safe_load(f)
    si = cfg["shelf_interest"]
    kp_conf = cfg["keypoint_confidence"]

    cache = load_cache(video_key)
    zones = load_zones(PROJECT_ROOT / "configs" / f"{video_key}_zones.json")
    # Staff are labelled in this video too. Task 2 does not need the
    # distinction, but calling an apron-wearing employee a "customer" on screen
    # would be plainly wrong.
    staff_tracks = {t for t, v in classify_tracks(video_key).items()
                    if v["label"] == "staff"}
    fps = float(cache["fps"])
    total_frames = int(cache["total_frames"])

    video_path = PROJECT_ROOT / "data" / "raw_videos" / f"{video_key}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"task2_{video_key}_annotated.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # One wrist detector (sure evidence) and one body detector (presence only)
    # per track, mirroring task2_shelf_interest.py exactly so the video tally
    # and the CSV cannot diverge.
    wrist_dets: dict[int, EpisodeDetector] = {}
    body_dets: dict[int, EpisodeDetector] = {}
    wrist_eps_by_track: dict[int, list] = defaultdict(list)
    pending_body: list[tuple[int, object]] = []
    counts = {p.name: {"sure": 0, "unsure": 0} for p in zones.polygons}
    last_seen: dict[int, tuple[int, float]] = {}
    total_events = unsure_total = 0

    for frame_idx in range(1, total_frames + 1):
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        rows = rows_for_frame(cache, frame_idx)

        active_zone_names: set[str] = set()
        active_count = 0

        for row in range(rows.start, rows.stop):
            tid = int(cache["track_id"][row])
            x1, y1, x2, y2 = (int(v) for v in cache["bbox"][row])
            signal = frame_signal(cache["keypoints"][row], zones, kp_conf)

            det = wrist_dets.get(tid)
            if det is None:
                det = EpisodeDetector(min_duration_s=si["min_duration_s"],
                                      max_gap_s=si["max_gap_s"], source="wrist")
                wrist_dets[tid] = det
                body_dets[tid] = EpisodeDetector(
                    min_duration_s=si["body_presence_min_s"],
                    max_gap_s=si["max_gap_s"], source="body")

            episode = det.update(frame_idx, t, signal)
            if episode:
                wrist_eps_by_track[tid].append(episode)
                counts[episode.label]["sure"] += 1
                total_events += 1

            # Body-presence episodes are held back and only counted as unsure
            # if no sure event on the same shelf overlaps them.
            bep = body_dets[tid].update(frame_idx, t, body_signal(cache["bbox"][row], zones))
            if bep:
                pending_body.append((tid, bep))

            last_seen[tid] = (frame_idx, t)

            open_ep = det.open_episode
            body_open = body_dets[tid].open_episode
            in_episode = open_ep is not None
            # Amber = present in the region but no confirmed reach yet.
            colour = COL_ACTIVE if in_episode else (COL_UNSURE if body_open else COL_BOX)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2 if in_episode else 1)

            # Role rather than track ID -- a reviewer cares who this is, not
            # what the tracker numbered them. Staff are labelled here too even
            # though Task 2 does not require the distinction, because silently
            # calling an apron-wearing employee a "customer" would be wrong.
            role = "STAFF" if tid in staff_tracks else "CUSTOMER"
            role_colour = COL_STAFF if tid in staff_tracks else colour
            cv2.putText(frame, role, (x1, y1 - 6), FONT, 0.5, role_colour,
                        2 if tid in staff_tracks else 1, cv2.LINE_AA)

            if in_episode:
                label, start_t = open_ep
                active_count += 1
                active_zone_names.add(label)
                dur = t - start_t
                # Shelf identity + live dwell time, on the customer's own box.
                cv2.putText(frame, f"{label}  {dur:4.1f}s", (x1, y2 + 18),
                            FONT, 0.55, COL_ACTIVE, 2, cv2.LINE_AA)
            elif body_open:
                label, start_t = body_open
                active_zone_names.add(label)
                cv2.putText(frame, f"{label}? {t - start_t:4.1f}s", (x1, y2 + 18),
                            FONT, 0.55, COL_UNSURE, 2, cv2.LINE_AA)

            # Hand-pose visibility is the evidence this whole task rests on, so
            # it is always drawn: filled green = wrist observed inside a rack,
            # hollow white = observed but outside, "no wrist" = not observed
            # (the "unsure" state, which can sustain but never start an episode).
            kp = cache["keypoints"][row]
            drew_wrist = False
            for side in ("left_wrist", "right_wrist"):
                wx, wy, wc = kp[KEYPOINT_INDEX[side]]
                if wc < kp_conf:
                    continue
                drew_wrist = True
                inside = bool(zones.containing_polygons(float(wx), float(wy)))
                cv2.circle(frame, (int(wx), int(wy)), 7,
                           COL_ACTIVE if inside else (255, 255, 255), -1)
                cv2.circle(frame, (int(wx), int(wy)), 8, (0, 0, 0), 1)
            if not drew_wrist:
                cv2.putText(frame, "no wrist", (x1, y1 - 22),
                            FONT, 0.45, COL_UNSURE, 1, cv2.LINE_AA)

        # Shelf polygons. The engaged one lights up -- that highlight, plus the
        # shelf name printed on the customer's own box, is the customer<->shelf
        # association the brief asks for.
        for poly in zones.polygons:
            active = poly.name in active_zone_names
            cv2.polylines(frame, [poly.points], True,
                          COL_ACTIVE if active else COL_IDLE, 3 if active else 1)

        is_final = frame_idx >= total_frames - int(FINAL_BANNER_S * fps)
        draw_panel(frame, counts, active_count, t, total_events, unsure_total, is_final)
        writer.write(frame)

        if frame_idx % 2000 == 0:
            print(f"  {frame_idx}/{total_frames}", flush=True)

    # Close any episodes still open at end of video so the video's final tally
    # matches the CSV exactly.
    for tid, det in wrist_dets.items():
        f, t = last_seen.get(tid, (total_frames, total_frames / fps))
        episode = det.flush(f, t)
        if episode:
            wrist_eps_by_track[tid].append(episode)
            counts[episode.label]["sure"] += 1
            total_events += 1
        bep = body_dets[tid].flush(f, t)
        if bep:
            pending_body.append((tid, bep))

    # Resolve held-back body episodes: an unsure event only stands where no
    # sure event on the same shelf overlaps it.
    for tid, be in pending_body:
        overlaps = any(
            we.label == be.label
            and we.start_time_s <= be.end_time_s
            and be.start_time_s <= we.end_time_s
            for we in wrist_eps_by_track.get(tid, [])
        )
        if not overlaps:
            counts[be.label]["unsure"] += 1
            unsure_total += 1

    writer.release()
    cap.release()
    print(f"wrote {out_path}")
    print(f"  final tally: {total_events} sure events, {unsure_total} unsure")
    for name in sorted(counts):
        print(f"    {name}: {counts[name]['sure']} sure, {counts[name]['unsure']} unsure")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="interior")
    args = parser.parse_args()
    render(args.video)
