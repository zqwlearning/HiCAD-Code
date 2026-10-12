"""Render visible STEP edges as sketches; --jitter enables hand-drawn noise."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hicad_data import relative_path

LOGGER = logging.getLogger(__name__)


def load_step(path: Path):
    """Read and transfer all STEP roots into one OpenCascade shape."""
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file {path}: status {status}.")
    if reader.TransferRoots() == 0:
        raise RuntimeError(f"No transferable STEP roots in {path}.")
    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError(f"Empty STEP shape: {path}")
    return shape


def get_eye(shape):
    """Get the nominal camera direction from an OpenGL viewer."""
    from OCC.Display.OCCViewer import Viewer3d
    from OCC.Extend.TopologyUtils import TopologyExplorer

    viewer = Viewer3d()
    viewer.Create(phong_shading=True, create_default_lights=True)
    viewer.SetModeShaded()
    viewer.hide_triedron()
    topology = TopologyExplorer(shape)
    for face in topology.faces():
        viewer.DisplayShape(face)
    for edge in topology.edges():
        viewer.DisplayShape(edge)
    viewer.SetSize(496, 473)
    viewer.FitAll()
    return viewer.View.Eye()


def get_hlr_edges(shape, view_direction=None):
    """Project visible and hidden edges; hidden edges only determine framing."""
    from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCC.Core.HLRAlgo import HLRAlgo_Projector
    from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    direction = get_eye(shape) if view_direction is None else view_direction
    projector = HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*direction)))
    algorithm = HLRBRep_Algo()
    algorithm.Add(shape)
    algorithm.Projector(projector)
    algorithm.Update()
    algorithm.Hide()
    result = HLRBRep_HLRToShape(algorithm)

    def collect(compound):
        if compound.IsNull():
            return []
        edges = []
        explorer = TopExp_Explorer(compound, TopAbs_EDGE)
        while explorer.More():
            edges.append(topods.Edge(explorer.Current()))
            explorer.Next()
        return edges

    return collect(result.VCompound()), collect(result.HCompound())


def extract_edges_points(edges, max_samples=200, min_samples=3, curve_factor=2.0):
    """Sample projected curves using length-dependent point counts."""
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GCPnts import GCPnts_UniformAbscissa
    from OCC.Core.GeomAbs import GeomAbs_Line

    polylines = []
    for edge in edges:
        curve = BRepAdaptor_Curve(edge)
        first, last = curve.FirstParameter(), curve.LastParameter()
        try:
            estimate = GCPnts_UniformAbscissa(curve, 100)
            if estimate.IsDone():
                length = sum(
                    curve.Value(estimate.Parameter(i + 1)).Distance(
                        curve.Value(estimate.Parameter(i))
                    )
                    for i in range(1, estimate.NbPoints())
                )
            else:
                length = last - first
        except RuntimeError:
            length = last - first
        factor = 1.0 if curve.GetType() == GeomAbs_Line else curve_factor
        count = min(max_samples, max(min_samples, int(factor * length / 10.0)))
        sampling = GCPnts_UniformAbscissa(curve, count)
        parameters = (
            [sampling.Parameter(i) for i in range(1, sampling.NbPoints() + 1)]
            if sampling.IsDone()
            else np.linspace(first, last, count)
        )
        points = [curve.Value(parameter) for parameter in parameters]
        polylines.append([(point.X(), point.Y()) for point in points])
    return polylines


def map_to_image(polylines, size, min_xy, max_xy):
    """Fit a shared bounding box with 5% margin, preserve aspect ratio, and flip Y."""
    scale = (
        min(size[0] / (max_xy[0] - min_xy[0] + 1e-6), size[1] / (max_xy[1] - min_xy[1] + 1e-6))
        * 0.95
    )
    offset = np.array(size) / 2 - scale * (min_xy + max_xy) / 2
    return [
        [(x * scale + offset[0], size[1] - (y * scale + offset[1])) for x, y in points]
        for points in polylines
    ]


class HandDrawnRenderer:
    """Draw only visible edges, optionally applying the paper's sketch augmentation."""

    def __init__(self, line_width=2, jitter_scale=1.5, seed=None, jitter=False):
        self.line_width = line_width
        self.jitter_scale = jitter_scale
        self.jitter = jitter
        # A local RNG avoids changing the random state of other dataset operations.
        self.rng = np.random.RandomState(seed)

    @staticmethod
    def _subdivide_line(points, max_segment_length=30.0):
        subdivided = []
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            count = max(1, int(np.hypot(x1 - x0, y1 - y0) / max_segment_length))
            for index in range(count):
                fraction = index / count
                subdivided.append((x0 + fraction * (x1 - x0), y0 + fraction * (y1 - y0)))
        return subdivided + [points[-1]]

    def draw_handdrawn(self, image, polylines):
        draw = ImageDraw.Draw(image)
        for points in polylines:
            if len(points) < 2:
                continue
            if self.jitter:
                points = self._subdivide_line(points)
                points = [
                    (
                        x + self.rng.uniform(-self.jitter_scale, self.jitter_scale),
                        y + self.rng.uniform(-self.jitter_scale, self.jitter_scale),
                    )
                    for x, y in points
                ]
            draw.line(points, fill=(0, 0, 0), width=self.line_width)


def run(
    step_path: Path,
    image_path: Path,
    *,
    image_size=512,
    linear_deflection=0.5,
    samples_per_edge=200,
    line_width=2,
    jitter=False,
    jitter_scale=1.5,
    seed=None,
    view_direction=None,
):
    """Render one STEP file to PNG."""
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh

    shape = load_step(step_path)
    BRepMesh_IncrementalMesh(shape, linear_deflection)
    visible_edges, hidden_edges = get_hlr_edges(shape, view_direction)
    visible_points = extract_edges_points(visible_edges, samples_per_edge)
    hidden_points = extract_edges_points(hidden_edges, samples_per_edge)
    all_points = np.array([point for points in visible_points + hidden_points for point in points])
    if not visible_points or all_points.size == 0:
        raise ValueError(f"No drawable visible edges in {step_path}.")
    size = (image_size, image_size)
    mapped = map_to_image(visible_points, size, all_points.min(axis=0), all_points.max(axis=0))
    image = Image.new("RGB", size, "white")
    renderer = HandDrawnRenderer(line_width, jitter_scale, seed, jitter)
    renderer.draw_handdrawn(image, mapped)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    image.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--step-file", type=relative_path, help="Render a single STEP file.")
    source.add_argument(
        "--step-dir", type=relative_path, help="Render all .step/.stp files in a directory."
    )
    parser.add_argument("--output-dir", type=relative_path, default=Path("HiCAD_Dataset/handdraw"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--samples-per-edge", type=int, default=200)
    parser.add_argument("--linear-deflection", type=float, default=0.5)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument(
        "--jitter",
        action="store_true",
        help="Enable subdivision and jitter from paper Algorithm 1.",
    )
    parser.add_argument("--jitter-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--view-direction", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.image_size <= 0 or args.samples_per_edge < 3 or args.line_width <= 0:
        parser.error(
            "Image size and line width must be positive; samples per edge must be at least 3."
        )
    if args.linear_deflection <= 0 or args.jitter_scale < 0:
        parser.error("Linear deflection must be positive and jitter scale must be non-negative.")
    if args.view_direction is not None and not any(args.view_direction):
        parser.error("The view direction must be non-zero.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.step_file:
        paths = [args.step_file]
    else:
        directory = args.step_dir or Path("HiCAD_Dataset/step")
        if not directory.is_dir():
            parser.exit(1, f"Error: STEP directory not found: {directory}\n")
        paths = sorted(
            path for path in directory.iterdir() if path.suffix.lower() in (".step", ".stp")
        )
    if not paths:
        parser.exit(1, "Error: No STEP files found.\n")
    if len({path.stem for path in paths}) != len(paths):
        parser.error("STEP filenames have duplicate stems; output PNG names would collide.")
    failures = 0
    for path in paths:
        output = args.output_dir / f"{path.stem}.png"
        if output.exists() and not args.overwrite:
            LOGGER.info("Skipping existing sketch: %s", output)
            continue
        try:
            if not path.is_file():
                raise FileNotFoundError(f"STEP file not found: {path}")
            run(
                path,
                output,
                image_size=args.image_size,
                linear_deflection=args.linear_deflection,
                samples_per_edge=args.samples_per_edge,
                line_width=args.line_width,
                jitter=args.jitter,
                jitter_scale=args.jitter_scale,
                seed=args.seed,
                view_direction=args.view_direction,
            )
        except ImportError as exc:
            parser.exit(1, f"Error: Install pythonocc-core in the sketch environment: {exc}\n")
        except (OSError, RuntimeError, ValueError) as exc:
            failures += 1
            LOGGER.error("%s: %s", path, exc)
        else:
            LOGGER.info("Saved %s", output)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
