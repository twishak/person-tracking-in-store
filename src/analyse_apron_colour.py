"""Measures the apron's torso hue across the curated staff crops.

Reports per-track medians so an inconsistent track stands out as an outlier
rather than being averaged into the signature -- the staff/customer split rests
on these crops all showing the same uniformed person.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DENSE_DIR = PROJECT_ROOT / "data" / "persons_dense"

# Torso band as a fraction of box height. Starts below the head and stops above
# the legs, so it captures the apron/chest area for both upright and bent poses.
TORSO_TOP, TORSO_BOTTOM = 0.20, 0.65
# Pixels below this saturation are skin/hair/floor/white-wall, not garment.
MIN_SATURATION = 60
# Ignore near-black and blown-out pixels, whose hue is meaningless.
MIN_VALUE, MAX_VALUE = 40, 250


def torso_hsv(image: np.ndarray):
    """Median (H, S, V) of saturated torso pixels, or None if too few."""
    h, w = image.shape[:2]
    if h < 20 or w < 10:
        return None
    band = image[int(h * TORSO_TOP): int(h * TORSO_BOTTOM), :]
    if band.size == 0:
        return None

    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (S >= MIN_SATURATION) & (V >= MIN_VALUE) & (V <= MAX_VALUE)
    if mask.sum() < 30:
        return None

    # Hue is circular (0 and 179 are adjacent in OpenCV's scale), so take the
    # median on the unit circle rather than naively -- a naive median of reds
    # spanning the 0/179 wrap would land on cyan, the opposite colour.
    hues = H[mask].astype(np.float64) * (2 * np.pi / 180.0)
    mean_vec = np.array([np.cos(hues).mean(), np.sin(hues).mean()])
    hue = (np.arctan2(mean_vec[1], mean_vec[0]) * 180.0 / (2 * np.pi)) % 180
    concentration = float(np.linalg.norm(mean_vec))  # 1 = single hue, 0 = spread
    return hue, float(np.median(S[mask])), float(np.median(V[mask])), concentration, int(mask.sum())


def analyse_track(track_dir: Path, sample: int = 25):
    files = sorted(track_dir.glob("*.jpg"))
    if not files:
        return None
    # Even sample across the track so one pose does not dominate.
    step = max(1, len(files) // sample)
    picks = files[::step][:sample]

    rows = []
    for f in picks:
        img = cv2.imread(str(f))
        if img is None:
            continue
        res = torso_hsv(img)
        if res:
            rows.append(res)
    if not rows:
        return None

    arr = np.array([r[:3] for r in rows])
    hues = arr[:, 0] * (2 * np.pi / 180.0)
    mv = np.array([np.cos(hues).mean(), np.sin(hues).mean()])
    hue = (np.arctan2(mv[1], mv[0]) * 180.0 / (2 * np.pi)) % 180
    return {
        "track": track_dir.name,
        "n_crops": len(files),
        "n_used": len(rows),
        "hue": hue,
        "sat": float(np.median(arr[:, 1])),
        "val": float(np.median(arr[:, 2])),
        "hue_consistency": float(np.linalg.norm(mv)),
    }


def hue_name(h):
    """OpenCV hue (0-179) -> rough colour name, for readability."""
    bands = [(0, 8, "red"), (8, 20, "orange"), (20, 33, "yellow"),
             (33, 78, "green"), (78, 100, "cyan"), (100, 130, "blue"),
             (130, 160, "purple"), (160, 180, "red")]
    for lo, hi, name in bands:
        if lo <= h < hi:
            return name
    return "?"


def main(video_key: str):
    base = DENSE_DIR / video_key
    if not base.exists():
        print(f"no crops at {base}")
        return

    results = []
    for track_dir in sorted(base.iterdir()):
        if track_dir.is_dir():
            r = analyse_track(track_dir)
            if r:
                results.append(r)

    print(f"\n=== {video_key}: torso colour per reference track ===")
    print(f"{'track':16s} {'crops':>6s} {'hue':>6s} {'colour':>8s} {'sat':>5s} {'val':>5s} {'consist':>8s}")
    for r in sorted(results, key=lambda r: r["hue"]):
        print(f"{r['track']:16s} {r['n_crops']:6d} {r['hue']:6.1f} {hue_name(r['hue']):>8s} "
              f"{r['sat']:5.0f} {r['val']:5.0f} {r['hue_consistency']:8.2f}")

    if results:
        hues = np.array([r["hue"] for r in results]) * (2 * np.pi / 180.0)
        mv = np.array([np.cos(hues).mean(), np.sin(hues).mean()])
        overall = (np.arctan2(mv[1], mv[0]) * 180.0 / (2 * np.pi)) % 180
        print(f"\n  pooled hue = {overall:.1f} ({hue_name(overall)}), "
              f"agreement across tracks = {np.linalg.norm(mv):.2f}")
        print("  (agreement near 1.0 = all tracks share one garment colour;")
        print("   markedly lower = the reference set contains more than one person)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="*", help="default: both")
    args = parser.parse_args()
    for key in (args.videos or ["entrance", "interior"]):
        main(key)
