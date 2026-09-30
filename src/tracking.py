"""Detection and tracking -- the single source of person observations.

YOLOv8m-pose gives boxes and 17 keypoints in one pass, so the wrist, torso and
face signals used downstream cost no extra inference. Tracker config is chosen
by benchmark (see compare_trackers.py).

This layer does no interpretation: it reports what was seen, not what it means.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSE_MODEL_PATH = PROJECT_ROOT / "models" / "yolov8m-pose.pt"
TRACKER_CONFIG = PROJECT_ROOT / "configs" / "botsort_static.yaml"

PERSON_CLASS_ID = 0  # COCO class 0 = person

# COCO-17 keypoint order, as emitted by the ultralytics pose models.
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
KEYPOINT_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# Below this per-keypoint confidence the joint is treated as "not observed"
# rather than trusted at its predicted coordinates. Overhead CCTV frequently
# occludes wrists (a shopper's arm behind their body, or behind a rack), and a
# hallucinated wrist position would silently create false shelf interactions.
# 0.5 is the ultralytics convention for "reasonably confident" and is the value
# their own visualisers use to decide whether to draw a joint at all.
MIN_KEYPOINT_CONF = 0.5


@dataclass
class PersonObservation:
    """One tracked person, as seen in one frame. Raw observation, no judgement."""

    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    keypoints: Optional[np.ndarray] = None  # (17, 3) -> x, y, conf

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def ground_point(self) -> tuple[float, float]:
        """Bottom-centre of the box ~= where the person meets the floor.

        Preferred over the centroid for any "which zone is this person in?"
        test: the annotated zones are floor regions, and with an overhead
        camera a standing person's box centre sits well above their actual
        floor position, which would place a tall person in the wrong zone.
        """
        x1, _, x2, y2 = self.bbox
        return (x1 + x2) / 2, y2

    @property
    def bbox_area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def keypoint(self, name: str) -> Optional[tuple[float, float]]:
        """Return (x, y) for a named joint, or None when it is not confidently
        observed. None is meaningful -- callers should treat it as "unknown",
        never as "not present"."""
        if self.keypoints is None:
            return None
        idx = KEYPOINT_INDEX[name]
        x, y, conf = self.keypoints[idx]
        if conf < MIN_KEYPOINT_CONF:
            return None
        return float(x), float(y)

    def wrists(self) -> list[tuple[float, float]]:
        """Confidently-observed wrist positions (0, 1 or 2 of them)."""
        found = []
        for side in ("left_wrist", "right_wrist"):
            kp = self.keypoint(side)
            if kp is not None:
                found.append(kp)
        return found


@dataclass
class FrameObservation:
    """Everything observed in a single frame."""

    frame_idx: int  # 1-based
    timestamp_s: float
    frame: np.ndarray  # BGR, original resolution
    persons: list[PersonObservation]


def video_properties(video_path: str | Path) -> tuple[float, int, int, int]:
    """(fps, total_frames, width, height) without decoding the whole file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total, width, height


def track_video(
    video_path: str | Path,
    model_path: str | Path = POSE_MODEL_PATH,
    tracker_config: str | Path = TRACKER_CONFIG,
    conf: float = 0.25,
    device: Optional[str] = None,
) -> Iterator[FrameObservation]:
    """Stream per-frame person observations for a video.

    Yields one FrameObservation per frame, in order, including frames where
    nobody was detected (an empty `persons` list) -- downstream state machines
    need to see those frames to time out stale tracks correctly.
    """
    fps, _, _, _ = video_properties(video_path)
    model = YOLO(str(model_path))

    frame_idx = 0
    for result in model.track(
        source=str(video_path),
        classes=[PERSON_CLASS_ID],
        conf=conf,
        persist=True,
        tracker=str(tracker_config),
        stream=True,
        verbose=False,
        device=device,
    ):
        frame_idx += 1
        persons: list[PersonObservation] = []

        boxes = result.boxes
        # boxes.id is None when the tracker produced no confirmed tracks this
        # frame; untracked detections are dropped on purpose, since every
        # downstream metric is defined per-track.
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            kp_all = None
            if result.keypoints is not None and result.keypoints.data is not None:
                kp_all = result.keypoints.data.cpu().numpy()  # (n_persons, 17, 3)

            for i, (box, tid, c) in enumerate(zip(xyxy, ids, confs)):
                kp = kp_all[i] if kp_all is not None and i < len(kp_all) else None
                persons.append(
                    PersonObservation(
                        track_id=int(tid),
                        bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        confidence=float(c),
                        keypoints=kp,
                    )
                )

        yield FrameObservation(
            frame_idx=frame_idx,
            timestamp_s=frame_idx / fps,
            frame=result.orig_img,
            persons=persons,
        )


if __name__ == "__main__":
    # Smoke test: first 150 frames of the entrance clip.
    import itertools

    video = PROJECT_ROOT / "data" / "raw_videos" / "entrance.mp4"
    fps, total, w, h = video_properties(video)
    print(f"{video.name}: {w}x{h} @ {fps:.1f}fps, {total} frames")

    seen_tracks, wrist_frames, person_frames = set(), 0, 0
    for obs in itertools.islice(track_video(video), 150):
        for p in obs.persons:
            seen_tracks.add(p.track_id)
            if p.wrists():
                wrist_frames += 1
            person_frames += 1

    print(f"150 frames -> {len(seen_tracks)} tracks, {person_frames} person-observations")
    pct = 100 * wrist_frames / person_frames if person_frames else 0
    print(f"at least one confident wrist in {wrist_frames}/{person_frames} ({pct:.1f}%)")
