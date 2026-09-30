"""Scene geometry: the hand-annotated rack polygons and entry boundaries.

Provides the two spatial predicates the event logic needs -- point-in-polygon
for region membership, and segment intersection for boundary crossing. The
camera is fixed, so this geometry is annotated once and valid for a whole clip.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _orientation(a, b, c) -> float:
    """Cross product sign: >0 counter-clockwise, <0 clockwise, 0 collinear."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(p1, p2, q1, q2) -> bool:
    """True when segment p1-p2 properly intersects segment q1-q2.

    Standard orientation test. Collinear/touching cases are treated as NOT
    crossing: a tracked point that merely grazes the threshold without passing
    through it should not register an entry.
    """
    d1 = _orientation(q1, q2, p1)
    d2 = _orientation(q1, q2, p2)
    d3 = _orientation(p1, p2, q1)
    d4 = _orientation(p1, p2, q2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


@dataclass
class Polygon:
    """A closed region of interest, e.g. one rack."""

    name: str
    points: np.ndarray  # (N, 2) int32, in pixel coordinates

    def contains(self, x: float, y: float) -> bool:
        """Ray-casting point-in-polygon test (even-odd rule)."""
        pts = self.points
        n = len(pts)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            # does the horizontal ray at height y cross edge (j -> i)?
            if (yi > y) != (yj > y):
                x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < x_cross:
                    inside = not inside
            j = i
        return inside

    @property
    def centroid(self) -> tuple[float, float]:
        return float(self.points[:, 0].mean()), float(self.points[:, 1].mean())


@dataclass
class BoundaryLine:
    """An open boundary (the storefront threshold), defined by 2+ points.

    Sign convention: `side()` returns the sign of the 2D cross product of the
    line direction with the vector to the query point. It is stable but
    arbitrary -- which sign means "inside the store" is resolved once, in the
    Task 1 config, by checking a point known to be inside; see
    `side_of_reference()`.
    """

    name: str
    points: np.ndarray  # (N, 2)

    @property
    def start(self) -> np.ndarray:
        return self.points[0]

    @property
    def end(self) -> np.ndarray:
        return self.points[-1]

    def side(self, x: float, y: float) -> float:
        """Signed perpendicular offset from the line. Sign = which side."""
        (x1, y1), (x2, y2) = self.start, self.end
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    def side_of_reference(self, x: float, y: float, ref_x: float, ref_y: float) -> bool:
        """True when (x, y) lies on the same side as a known reference point."""
        a, b = self.side(x, y), self.side(ref_x, ref_y)
        return (a > 0) == (b > 0)

    def crossed(self, prev_xy: tuple[float, float], curr_xy: tuple[float, float]) -> bool:
        """True when movement from prev to curr intersects this boundary.

        Tests intersection with the drawn SEGMENT(s), not the infinite line.
        The storefront is open-plan with several separate walkable openings, so
        each threshold is a short local segment; extending one infinitely would
        slice across the whole frame and fire on people nowhere near it.

        Multi-point boundaries are treated as a polyline and every sub-segment
        is tested, so a threshold can bend around a corner.
        """
        for i in range(len(self.points) - 1):
            if _segments_intersect(prev_xy, curr_xy, self.points[i], self.points[i + 1]):
                return True
        return False


@dataclass
class SceneZones:
    """All annotated geometry for one video."""

    video: str
    frame_number: int
    polygons: list[Polygon]
    lines: list[BoundaryLine]

    def polygon(self, name: str) -> Polygon:
        for p in self.polygons:
            if p.name == name:
                return p
        raise KeyError(f"no polygon named {name!r} (have: {[p.name for p in self.polygons]})")

    def line(self, name: str) -> BoundaryLine:
        for ln in self.lines:
            if ln.name == name:
                return ln
        raise KeyError(f"no line named {name!r} (have: {[ln.name for ln in self.lines]})")

    def crossed_any_boundary(self, prev_xy: tuple[float, float],
                             curr_xy: tuple[float, float]) -> BoundaryLine | None:
        """The first annotated boundary crossed between two positions, if any.

        The storefront is open-plan with several walkable openings, so entry is
        tested against every annotated threshold rather than a single gate.
        Returns the line that was crossed (useful for reporting WHICH opening
        someone used), or None.
        """
        for line in self.lines:
            if line.crossed(prev_xy, curr_xy):
                return line
        return None

    def containing_polygons(self, x: float, y: float) -> list[Polygon]:
        """Every polygon containing the point. Usually 0 or 1, but the
        annotated racks may overlap slightly, so callers must decide how to
        disambiguate rather than assuming uniqueness."""
        return [p for p in self.polygons if p.contains(x, y)]


# Name prefixes that denote an open boundary rather than a closed region.
# Matched as a case-insensitive PREFIX so an arbitrary number of thresholds can
# be annotated -- "Entry", "Entry_2", "gate_left" all register as boundaries.
# The storefront is open-plan with several walkable openings, so there is no
# single gate to hard-code.
BOUNDARY_PREFIXES = ("gate", "entry", "entrance")


def _is_boundary(name: str, prefixes: tuple[str, ...]) -> bool:
    lowered = name.strip().lower()
    return any(lowered.startswith(p) for p in prefixes)


def load_zones(config_path: str | Path,
               line_names: tuple[str, ...] = BOUNDARY_PREFIXES) -> SceneZones:
    """Read a *_zones.json written by src/annotate_zones.py.

    Shapes are split into polygons vs. lines by name (`line_names`), with a
    safety net: any shape with fewer than 3 points cannot bound an area and is
    treated as a line regardless. Empty shapes (an accidental keypress during
    annotation) are skipped.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = json.load(f)

    polygons: list[Polygon] = []
    lines: list[BoundaryLine] = []

    for shape in cfg["shapes"]:
        pts = np.array(shape["points"], dtype=np.int32)
        if len(pts) == 0:
            continue
        if _is_boundary(shape["name"], line_names) or len(pts) < 3:
            lines.append(BoundaryLine(name=shape["name"], points=pts))
        else:
            polygons.append(Polygon(name=shape["name"], points=pts))

    return SceneZones(
        video=cfg["video"],
        frame_number=cfg["frame_number"],
        polygons=polygons,
        lines=lines,
    )


if __name__ == "__main__":
    for name in ("entrance", "interior"):
        zones = load_zones(PROJECT_ROOT / "configs" / f"{name}_zones.json")
        print(f"{name}: {len(zones.polygons)} polygons, {len(zones.lines)} lines")
        for p in zones.polygons:
            cx, cy = p.centroid
            print(f"   polygon {p.name:10s} {len(p.points):2d} pts  centroid=({cx:.0f}, {cy:.0f})")
        for ln in zones.lines:
            print(f"   line    {ln.name:10s} {tuple(ln.start)} -> {tuple(ln.end)}")
