"""Runs detection + tracking once per video and persists every observation.

Tracking is the expensive stage (~8 min for both clips) and needs every
preceding frame for track IDs to be meaningful, so all analysis and rendering
reads data/cache/<video>_tracks.npz instead of re-running it.
"""

import argparse
from pathlib import Path

import numpy as np

from tracking import track_video, video_properties

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"


def build(video_key: str) -> Path:
    video_path = PROJECT_ROOT / "data" / "raw_videos" / f"{video_key}.mp4"
    fps, total, _, _ = video_properties(video_path)
    print(f"{video_key}: {total} frames @ {fps:.1f}fps")

    frame_idx_l, track_id_l, bbox_l, conf_l, kp_l = [], [], [], [], []
    empty_kp = np.zeros((17, 3), dtype=np.float32)

    for obs in track_video(video_path):
        for p in obs.persons:
            frame_idx_l.append(obs.frame_idx)
            track_id_l.append(p.track_id)
            bbox_l.append(p.bbox)
            conf_l.append(p.confidence)
            kp_l.append(p.keypoints if p.keypoints is not None else empty_kp)
        if obs.frame_idx % 2000 == 0:
            print(f"  {obs.frame_idx}/{total}  ({len(frame_idx_l)} observations)", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{video_key}_tracks.npz"
    np.savez_compressed(
        out,
        frame_idx=np.array(frame_idx_l, dtype=np.int32),
        track_id=np.array(track_id_l, dtype=np.int32),
        bbox=np.array(bbox_l, dtype=np.float32),
        conf=np.array(conf_l, dtype=np.float32),
        keypoints=np.array(kp_l, dtype=np.float32),
        fps=np.float32(fps),
        total_frames=np.int32(total),
    )
    n_tracks = len(set(track_id_l))
    print(f"  -> {out}  ({len(frame_idx_l)} observations, {n_tracks} tracks)")
    return out


def load_cache(video_key: str) -> dict:
    """Load a cache into a dict of arrays, plus a frame->rows index."""
    path = CACHE_DIR / f"{video_key}_tracks.npz"
    if not path.exists():
        raise FileNotFoundError(f"no cache for {video_key!r}; run: python src/build_cache.py {video_key}")
    data = dict(np.load(path))
    # Rows are emitted in frame order, so a frame's rows form a contiguous
    # slice -- searchsorted gives O(log n) lookup without building a dict.
    data["_frame_starts"] = np.searchsorted(data["frame_idx"], np.arange(1, int(data["total_frames"]) + 2))
    return data


def rows_for_frame(cache: dict, frame_idx: int) -> slice:
    starts = cache["_frame_starts"]
    if frame_idx < 1 or frame_idx + 1 >= len(starts):
        return slice(0, 0)
    return slice(int(starts[frame_idx - 1]), int(starts[frame_idx]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # NOTE: no argparse `choices=` here -- with nargs="*" argparse validates the
    # empty list itself against choices and rejects it, so the no-argument
    # "do both" default is impossible to express that way. Validated manually.
    parser.add_argument("videos", nargs="*", help="videos to cache (default: both)")
    args = parser.parse_args()

    valid = {"entrance", "interior"}
    keys = args.videos or sorted(valid)
    unknown = set(keys) - valid
    if unknown:
        parser.error(f"unknown video(s): {sorted(unknown)}; choose from {sorted(valid)}")

    for key in keys:
        build(key)
