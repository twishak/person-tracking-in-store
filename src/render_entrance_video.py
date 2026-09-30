"""Entrance deliverable video: Tasks 1 and 3 in one file, as the brief requires.

Shelf interaction overlays are included too, so both output videos read
consistently. Replays the task logic rather than recomputing it, so the video
and the CSVs cannot disagree.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from build_cache import load_cache, rows_for_frame
from task1_interest_conversion import run as run_task1, storefront_axis, signed_distance
from task2_shelf_interest import body_signal, frame_signal
from task3_staff_interaction import box_gap, box_height, foot_point
from task3_staff_interaction import run as run_task3
from episodes import EpisodeDetector
from tracking import KEYPOINT_INDEX
from zones import load_zones

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

COL_PLAIN = (150, 150, 150)
COL_INTEREST = (0, 200, 255)
COL_ENTERED = (0, 230, 0)
COL_ENTRYLINE = (0, 0, 255)
COL_RACK_IDLE = (110, 110, 110)
COL_RACK_ACTIVE = (0, 230, 0)
COL_UNSURE = (0, 200, 255)
COL_STAFF = (255, 0, 255)  # magenta -- distinct from every Task 1/2 colour
COL_SESSION = (0, 255, 255)  # yellow -- active staff-customer interaction
COL_TEXT = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# How long the "ENTERED" banner stays on screen after the crossing. Entering is
# an event, not a state; the green box carries the classification afterwards.
ENTERED_LABEL_HOLD_S = 4.0

# The last N seconds are marked FINAL, so the closing frames clearly present the
# reported totals without the panels announcing them from the start.
FINAL_BANNER_S = 8.0


def draw_task1_panel(frame, running, t, is_final):
    """Three headline counts, top-left as the brief specifies.

    Shows the LIVE count only. An earlier version printed "running / final",
    which announced the answer from the first frame and reduced the overlay to
    watching a counter climb to a number already given away. The counts here
    accumulate as the evidence for them appears on screen, and the closing
    frames are marked FINAL -- at which point the running value IS the reported
    result.
    """
    w, h = 330, 128
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    header = "STORE INTEREST & CONVERSION" + ("  -- FINAL" if is_final else "")
    cv2.putText(frame, header, (10, 22), FONT, 0.48,
                COL_ENTERED if is_final else COL_TEXT, 1, cv2.LINE_AA)
    labels = [("Total Interested", COL_INTEREST), ("Interested Entered", COL_ENTERED),
              ("Interested Passed By", COL_INTEREST)]
    y = 48
    for (label, colour), run_v in zip(labels, running):
        cv2.putText(frame, f"{label:<21s} {run_v:>3d}", (10, y),
                    FONT, 0.48, colour, 2 if is_final else 1, cv2.LINE_AA)
        y += 23
    cv2.putText(frame, f"counted so far      t={t:6.1f}s", (10, y), FONT, 0.40,
                COL_TEXT, 1, cv2.LINE_AA)


def draw_shelf_panel(frame, counts, width, is_final=False):
    """Cumulative per-rack engagement, top-right.

    Only racks with a non-zero count are listed: entrance.mp4 has 13 annotated
    racks and printing every zero row would fill the frame without adding
    information. The total line reports across all racks regardless.
    """
    active = {k: v for k, v in counts.items() if v["sure"] or v["unsure"]}
    rows = sorted(active)
    w, h = 250, 26 * (len(rows) + 2) + 8
    x0 = width - w
    ov = frame.copy()
    cv2.rectangle(ov, (x0, 0), (width, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "SHELF INTERACTIONS" + ("  -- FINAL" if is_final else ""),
                (x0 + 10, 22), FONT, 0.48,
                COL_ENTERED if is_final else COL_TEXT, 1, cv2.LINE_AA)
    y = 46
    for name in rows:
        s, u = counts[name]["sure"], counts[name]["unsure"]
        txt = f"{name:<8s} {s:>2d}" + (f"  +{u}?" if u else "")
        cv2.putText(frame, txt, (x0 + 10, y), FONT, 0.46,
                    COL_RACK_ACTIVE if s else COL_UNSURE, 1, cv2.LINE_AA)
        y += 26
    ts = sum(v["sure"] for v in counts.values())
    tu = sum(v["unsure"] for v in counts.values())
    cv2.putText(frame, f"TOTAL {ts}   unsure +{tu}", (x0 + 10, y), FONT, 0.46,
                COL_TEXT, 1, cv2.LINE_AA)


def draw_staff_panel(frame, sessions_so_far, n_instances, height, is_final):
    """Task 3 results, bottom-left.

    Like the Task 1 panel, this counts up rather than displaying the final
    figure from the outset. The average is derived from the live session count
    (sessions / instances), so it too is genuinely computed as the video runs
    and only equals the reported result once all evidence has been seen.
    """
    w, h = 340, 92
    y0 = height - h
    ov = frame.copy()
    cv2.rectangle(ov, (0, y0), (w, height), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    live_avg = sessions_so_far / n_instances if n_instances else 0.0

    y = y0 + 24
    header = "STAFF-CUSTOMER INTERACTION" + ("  -- FINAL" if is_final else "")
    cv2.putText(frame, header, (10, y), FONT, 0.48,
                COL_ENTERED if is_final else COL_TEXT, 1, cv2.LINE_AA)
    y += 23
    cv2.putText(frame, f"Staff 1 sessions      {sessions_so_far:>3d}", (10, y),
                FONT, 0.48, COL_SESSION, 2 if is_final else 1, cv2.LINE_AA)
    y += 23
    cv2.putText(frame, f"Avg sessions / staff  {live_avg:>5.1f}", (10, y), FONT, 0.48,
                COL_STAFF, 2 if is_final else 1, cv2.LINE_AA)


def render(video_key: str = "entrance") -> Path:
    task1 = run_task1(video_key)
    interested = task1["interested"]
    entered_ids = set(task1["entered"])
    totals = (len(interested), len(entered_ids), len(interested) - len(entered_ids))

    task3 = run_task3(video_key)
    staff_tracks = task3["staff_tracks"]
    task3_sessions = task3["sessions"]
    # All staff tracks are one person (verified non-overlapping in task3), so
    # the instance count is 1 whenever any staff was detected.
    n_staff_instances = 1 if staff_tracks else 0

    cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "thresholds.yaml"))
    t1, si = cfg["store_interest"], cfg["shelf_interest"]
    t3 = cfg["staff_interaction"]
    kp_conf = cfg["keypoint_confidence"]

    cache = load_cache(video_key)
    zones = load_zones(PROJECT_ROOT / "configs" / f"{video_key}_zones.json")
    fps = float(cache["fps"])
    total_frames = int(cache["total_frames"])
    centre, _d, normal = storefront_axis(zones)
    inside_sign = np.sign(signed_distance(t1["inside_reference_xy"], centre, normal))

    first_frame = {}
    for row in range(len(cache["frame_idx"])):
        tid, f = int(cache["track_id"][row]), int(cache["frame_idx"][row])
        if tid not in first_frame or f < first_frame[tid]:
            first_frame[tid] = f
    appear = defaultdict(list)
    for tid in interested:
        appear[first_frame[tid]].append(tid)

    wrist_dets: dict[int, EpisodeDetector] = {}
    body_dets: dict[int, EpisodeDetector] = {}
    wrist_eps_by_track = defaultdict(list)
    pending_body = []
    shelf_counts = {p.name: {"sure": 0, "unsure": 0} for p in zones.polygons}
    last_seen = {}

    video_path = PROJECT_ROOT / "data" / "raw_videos" / f"{video_key}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{video_key}_annotated.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    seen = [0, 0, 0]

    for frame_idx in range(1, total_frames + 1):
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps

        for tid in appear.get(frame_idx, []):
            seen[0] += 1
            seen[1 if tid in entered_ids else 2] += 1

        rows = rows_for_frame(cache, frame_idx)
        active_racks: set[str] = set()

        # --- Task 3: who is currently in an interaction? --------------------
        # Recomputed here with the same proximity rule as
        # task3_staff_interaction.py so the overlay and the CSV agree.
        row_list = list(range(rows.start, rows.stop))
        staff_rows = [r for r in row_list if int(cache["track_id"][r]) in staff_tracks]
        interacting: set[int] = set()
        session_links: list[tuple[tuple, tuple]] = []
        for sr in staff_rows:
            sb = cache["bbox"][sr]
            sp, sh = foot_point(sb), box_height(sb)
            if sh <= 0:
                continue
            for cr in row_list:
                cid = int(cache["track_id"][cr])
                if cid in staff_tracks:
                    continue
                cp = foot_point(cache["bbox"][cr])
                # Same box-gap rule as task3_staff_interaction.py, imported
                # rather than reimplemented -- this overlay previously carried
                # its own copy of the proximity test, which silently kept using
                # foot distance after the rule changed to box proximity.
                if box_gap(sb, cache["bbox"][cr]) <= t3["max_box_gap_ratio"] * sh:
                    # Only draw pairs that belong to a COUNTED session, so the
                    # video never shows an interaction the CSV does not contain.
                    if any(c == cid and ep.start_time_s <= t <= ep.end_time_s
                           for c, ep in task3_sessions):
                        interacting.add(cid)
                        session_links.append((sp, cp))

        sessions_so_far = sum(1 for _c, ep in task3_sessions if ep.start_time_s <= t)

        for row in range(rows.start, rows.stop):
            tid = int(cache["track_id"][row])
            x1, y1, x2, y2 = (int(v) for v in cache["bbox"][row])
            info = interested.get(tid)

            # --- shelf engagement (Task 2 logic) ---------------------------
            det = wrist_dets.get(tid)
            if det is None:
                det = EpisodeDetector(si["min_duration_s"], si["max_gap_s"], source="wrist")
                wrist_dets[tid] = det
                body_dets[tid] = EpisodeDetector(
                    si["body_presence_min_s"], si["max_gap_s"], source="body")

            ep = det.update(frame_idx, t, frame_signal(cache["keypoints"][row], zones, kp_conf))
            if ep:
                wrist_eps_by_track[tid].append(ep)
                shelf_counts[ep.label]["sure"] += 1
            bep = body_dets[tid].update(frame_idx, t, body_signal(cache["bbox"][row], zones))
            if bep:
                pending_body.append((tid, bep))
            last_seen[tid] = (frame_idx, t)

            open_ep = det.open_episode
            body_open = body_dets[tid].open_episode

            # --- person box ------------------------------------------------
            # Staff identity takes visual precedence over the Task 1 verdict:
            # the brief requires detected staff instances to be clearly shown,
            # and staff are not passers-by so the two never really compete.
            is_staff = tid in staff_tracks
            if is_staff:
                colour, thickness = COL_STAFF, 3
            elif info is None:
                colour, thickness = COL_PLAIN, 1
            elif tid in entered_ids:
                colour, thickness = COL_ENTERED, 3
            else:
                colour, thickness = COL_INTEREST, 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

            # Role, not track ID: a reviewer cares whether this is staff or a
            # customer, and the raw tracker ID means nothing to them. IDs remain
            # in the CSVs so any box can still be traced back to its event rows.
            tag = "STAFF" if is_staff else "CUSTOMER"
            cv2.putText(frame, tag, (x1, y1 - 6), FONT, 0.45, colour,
                        2 if is_staff else 1, cv2.LINE_AA)

            # Highlight a customer currently in a counted interaction session.
            if tid in interacting:
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), COL_SESSION, 2)
                cv2.putText(frame, "INTERACTING", (x1, y1 - 24), FONT, 0.45,
                            COL_SESSION, 2, cv2.LINE_AA)

            label_y = y2 + 16
            if info is not None:
                if tid in entered_ids:
                    # ENTERED marks a MOMENT, not a lasting state -- the person
                    # is inside for the rest of their track, so a permanent
                    # banner reads as if they are perpetually entering. Show it
                    # only around the crossing; the green box still identifies
                    # them as a counted entry for the remainder.
                    ent_t = info.get("entered_at")
                    if ent_t is not None and 0 <= t - ent_t <= ENTERED_LABEL_HOLD_S:
                        cv2.putText(frame, f"ENTERED @{ent_t:.0f}s", (x1, label_y),
                                    FONT, 0.5, colour, 2, cv2.LINE_AA)
                        label_y += 18
                else:
                    cv2.putText(frame, "INTERESTED", (x1, label_y), FONT, 0.5,
                                colour, 2, cv2.LINE_AA)
                    label_y += 18
                    cues = "+".join(c for c in info["cues"] if c != "entered")
                    if cues:
                        cv2.putText(frame, cues, (x1, label_y), FONT, 0.4, colour, 1, cv2.LINE_AA)
                        label_y += 16

            # shelf engagement label sits under the Task 1 labels
            if open_ep is not None:
                rack, start_t = open_ep
                active_racks.add(rack)
                cv2.putText(frame, f"{rack}  {t - start_t:4.1f}s", (x1, label_y),
                            FONT, 0.5, COL_RACK_ACTIVE, 2, cv2.LINE_AA)
            elif body_open is not None:
                rack, start_t = body_open
                active_racks.add(rack)
                cv2.putText(frame, f"{rack}? {t - start_t:4.1f}s", (x1, label_y),
                            FONT, 0.5, COL_UNSURE, 2, cv2.LINE_AA)

            # --- wrists: the evidence behind shelf engagement --------------
            for side in ("left_wrist", "right_wrist"):
                wx, wy, wc = cache["keypoints"][row][KEYPOINT_INDEX[side]]
                if wc < kp_conf:
                    continue
                inside = bool(zones.containing_polygons(float(wx), float(wy)))
                cv2.circle(frame, (int(wx), int(wy)), 6,
                           COL_RACK_ACTIVE if inside else (255, 255, 255), -1)
                cv2.circle(frame, (int(wx), int(wy)), 7, (0, 0, 0), 1)

            # inside/outside marker at the feet
            gx, gy = (x1 + x2) // 2, y2
            d = signed_distance((gx, gy), centre, normal) * inside_sign
            cv2.circle(frame, (gx, gy), 4, (0, 255, 0) if d > 0 else (0, 165, 255), -1)

        # --- scene geometry ------------------------------------------------
        for poly in zones.polygons:
            act = poly.name in active_racks
            cv2.polylines(frame, [poly.points], True,
                          COL_RACK_ACTIVE if act else COL_RACK_IDLE, 3 if act else 1)
            if act:
                cx, cy = poly.centroid
                cv2.putText(frame, poly.name, (int(cx) - 26, int(cy)), FONT, 0.5,
                            COL_RACK_ACTIVE, 2, cv2.LINE_AA)
        for line in zones.lines:
            cv2.polylines(frame, [line.points], False, COL_ENTRYLINE, 3)

        # link staff to the customer they are engaging with
        for sp, cp in session_links:
            cv2.line(frame, (int(sp[0]), int(sp[1])), (int(cp[0]), int(cp[1])),
                     COL_SESSION, 2, cv2.LINE_AA)

        # The closing seconds are marked FINAL: by then every count has
        # accumulated its full evidence, so the live values are the reported
        # results and the brief's requirement to display the final counts is
        # met without pre-announcing them.
        is_final = t >= (total_frames / fps) - FINAL_BANNER_S

        draw_task1_panel(frame, seen, t, is_final)
        draw_shelf_panel(frame, shelf_counts, width, is_final)
        draw_staff_panel(frame, sessions_so_far, n_staff_instances, height, is_final)
        writer.write(frame)

        if frame_idx % 2000 == 0:
            print(f"  {frame_idx}/{total_frames}", flush=True)

    # close open episodes so the video tally matches the CSV
    for tid, det in wrist_dets.items():
        f, t = last_seen.get(tid, (total_frames, total_frames / fps))
        ep = det.flush(f, t)
        if ep:
            wrist_eps_by_track[tid].append(ep)
            shelf_counts[ep.label]["sure"] += 1
        bep = body_dets[tid].flush(f, t)
        if bep:
            pending_body.append((tid, bep))
    for tid, be in pending_body:
        if not any(we.label == be.label and we.start_time_s <= be.end_time_s
                   and be.start_time_s <= we.end_time_s
                   for we in wrist_eps_by_track.get(tid, [])):
            shelf_counts[be.label]["unsure"] += 1

    writer.release()
    cap.release()
    print(f"wrote {out_path}")
    print(f"  Task 1: interested {totals[0]}, entered {totals[1]}, passed by {totals[2]}")
    ts = sum(v["sure"] for v in shelf_counts.values())
    tu = sum(v["unsure"] for v in shelf_counts.values())
    print(f"  Shelves: {ts} sure, {tu} unsure")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="entrance")
    args = parser.parse_args()
    render(args.video)
