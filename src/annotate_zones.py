"""Interactive click tool for scene geometry: rack polygons and entry lines.

The camera is fixed, so geometry is annotated once per video on one reference
frame. Shapes are free-form -- you name them and decide what to draw.

Usage:
    python src/annotate_zones.py data/raw_videos/entrance.mp4 --frame 100

Existing annotations are loaded on start, so a saved file can be edited rather
than redrawn.

Controls
    left-click empty space  add a point to the active shape
    left-click a vertex     select that shape and drag the vertex to move it
    right-click a vertex    delete just that vertex
    n / t                   new shape / switch active shape (prompts in TERMINAL)
    u                       undo last point      d  delete the whole active shape
    s                       save and quit        q  quit without saving
"""

import argparse
import json
from pathlib import Path

import cv2

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 255, 0), (0, 128, 255),
]


def grab_frame(video_path: Path, frame_number: int):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_number} from {video_path}")
    return frame


class FreeAnnotator:
    def __init__(self, frame, initial_shapes=None):
        self.base_frame = frame
        self.shapes = initial_shapes if initial_shapes is not None else []  # [{"name": str, "points": [[x,y], ...]}, ...]
        self.active_idx = None
        self.drag_point = None  # index of the vertex currently being dragged
        self.window = "Annotator"

    # Click within this many pixels of a vertex to grab it instead of adding a
    # new point. Large enough to hit comfortably, small enough that adding a
    # point next to an existing one still works.
    HIT_RADIUS = 9

    def _find_vertex(self, x, y):
        """(shape_idx, point_idx) of the vertex nearest the cursor, or None."""
        best, best_d = None, self.HIT_RADIUS ** 2
        for si, s in enumerate(self.shapes):
            for pi, (px, py) in enumerate(s["points"]):
                d = (px - x) ** 2 + (py - y) ** 2
                if d <= best_d:
                    best, best_d = (si, pi), d
        return best

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = self._find_vertex(x, y)
            if hit:
                # Grab an existing vertex: select its shape and start dragging.
                self.active_idx, self.drag_point = hit[0], hit[1]
                return
            if self.active_idx is None:
                print("No active shape yet -- press 'n' to start one, or click an existing point.")
                return
            self.shapes[self.active_idx]["points"].append([x, y])

        elif event == cv2.EVENT_MOUSEMOVE and self.drag_point is not None:
            self.shapes[self.active_idx]["points"][self.drag_point] = [x, y]

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_point = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Delete a single vertex, so one misplaced click does not mean
            # redrawing the whole polygon.
            hit = self._find_vertex(x, y)
            if hit is None:
                return
            si, pi = hit
            del self.shapes[si]["points"][pi]
            print(f"deleted point {pi} of '{self.shapes[si]['name']}' "
                  f"({len(self.shapes[si]['points'])} left)")

    def _new_shape(self):
        name = input("Label for new shape (any name you like): ").strip()
        if not name:
            print("Empty name, cancelled.")
            return
        self.shapes.append({"name": name, "points": []})
        self.active_idx = len(self.shapes) - 1
        print(f"Now annotating '{name}'. Click points in the image window.")

    def _switch_active(self):
        if not self.shapes:
            print("No shapes yet -- press 'n' to create one first.")
            return
        print("Existing shapes:")
        for i, s in enumerate(self.shapes):
            print(f"  [{i}] {s['name']} ({len(s['points'])} pts)")
        raw = input("Switch to which index?: ").strip()
        try:
            idx = int(raw)
            if 0 <= idx < len(self.shapes):
                self.active_idx = idx
                print(f"Active shape is now '{self.shapes[idx]['name']}'.")
            else:
                print("Invalid index.")
        except ValueError:
            print("Invalid input.")

    def _delete_active(self):
        if self.active_idx is None:
            print("No active shape.")
            return
        name = self.shapes[self.active_idx]["name"]
        del self.shapes[self.active_idx]
        self.active_idx = None
        print(f"Deleted '{name}'. Press 'n' or 't' to pick/create the next active shape.")

    def _redraw(self):
        img = self.base_frame.copy()
        for i, s in enumerate(self.shapes):
            color = COLORS[i % len(COLORS)]
            active = i == self.active_idx
            pts = s["points"]
            # The active shape is drawn thicker with larger, ringed vertices, so
            # it is obvious which one an edit will land on.
            thick = 3 if active else 2
            for j, p in enumerate(pts):
                cv2.circle(img, tuple(p), 6 if active else 4, color, -1)
                if active:
                    cv2.circle(img, tuple(p), 8, (255, 255, 255), 1)
                if j > 0:
                    cv2.line(img, tuple(pts[j - 1]), tuple(p), color, thick)
            if len(pts) > 2 and not active:
                cv2.line(img, tuple(pts[-1]), tuple(pts[0]), color, thick)
            if pts:
                label = f"* {s['name']}" if active else s["name"]
                cv2.putText(img, label, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        active_name = self.shapes[self.active_idx]["name"] if self.active_idx is not None else "(none)"
        status = (
            f"active: {active_name}  |  L-click=add/grab point  drag=move  "
            "R-click=delete point  n=new  t=switch  u=undo  d=delete-shape  "
            "s=save+quit  q=quit"
        )
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(img, status, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imshow(self.window, img)

    def run(self):
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._mouse_cb)
        saved = False
        while True:
            self._redraw()
            key = cv2.waitKey(20) & 0xFF
            if key == ord("n"):
                self._new_shape()
            elif key == ord("t"):
                self._switch_active()
            elif key == ord("u"):
                if self.active_idx is not None and self.shapes[self.active_idx]["points"]:
                    self.shapes[self.active_idx]["points"].pop()
            elif key == ord("d"):
                self._delete_active()
            elif key == ord("s"):
                saved = True
                break
            elif key == ord("q"):
                break
        cv2.destroyAllWindows()
        return self.shapes if saved else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="path to a video file, e.g. raw_videos/entrance.mp4")
    parser.add_argument("--frame", type=int, default=100, help="frame number to annotate on")
    parser.add_argument("--out", help="output json path (default: configs/<video_stem>_zones.json)")
    args = parser.parse_args()

    video_path = Path(args.video)
    frame = grab_frame(video_path, args.frame)

    out_path = Path(args.out) if args.out else Path("configs") / f"{video_path.stem}_zones.json"
    initial_shapes = []
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        initial_shapes = existing.get("shapes", [])
        print(f"Loaded {len(initial_shapes)} existing shape(s) from {out_path}:")
        for s in initial_shapes:
            print(f"  {s['name']}: {len(s['points'])} points")
        print("Press 't' to keep editing one of these, or 'n' to add a new shape.")
    else:
        print("Press 'n' to name and start your first shape. Nothing is fixed in advance --")
        print("annotate whatever you want, call it whatever you want.")

    annotator = FreeAnnotator(frame, initial_shapes=initial_shapes)
    result = annotator.run()

    if result is None:
        print("Quit without saving.")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"video": str(video_path), "frame_number": args.frame, "shapes": result}, f, indent=2)
        print(f"Saved to {out_path}")
        for s in result:
            print(f"  {s['name']}: {len(s['points'])} points")
