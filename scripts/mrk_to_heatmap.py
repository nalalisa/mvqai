from __future__ import annotations

import argparse
import json
from pathlib import Path

import nrrd
import numpy as np
from scipy.interpolate import splprep, splev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 3D Slicer markup points in voxel space to a soft Gaussian ring heatmap."
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to mrk.json file.")
    parser.add_argument("--output", required=True, type=Path, help="Path to output .nrrd heatmap.")
    parser.add_argument(
        "--shape",
        required=True,
        nargs=3,
        type=int,
        metavar=("D", "H", "W"),
        help="Target heatmap shape in voxel coordinates.",
    )
    parser.add_argument("--sigma", default=2.0, type=float, help="Gaussian sigma in voxels.")
    parser.add_argument(
        "--num-samples",
        default=256,
        type=int,
        help="Number of spline samples to rasterize along the closed ring.",
    )
    return parser.parse_args()


def load_control_points(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    markup = payload["markups"][0]
    points = [cp["position"] for cp in markup["controlPoints"]]
    if len(points) < 4:
        raise ValueError("At least 4 control points are required for a closed spline.")
    return np.asarray(points, dtype=np.float32)


def interpolate_closed_curve(points: np.ndarray, num_samples: int) -> np.ndarray:
    points = np.concatenate([points, points[:1]], axis=0)
    tck, _ = splprep(points.T, s=0.0, per=True)
    u = np.linspace(0.0, 1.0, num=num_samples, endpoint=False)
    sampled = np.stack(splev(u, tck), axis=1)
    return sampled.astype(np.float32)


def draw_heatmap(shape: tuple[int, int, int], samples: np.ndarray, sigma: float) -> np.ndarray:
    heatmap = np.zeros(shape, dtype=np.float32)
    radius = max(1, int(np.ceil(3.0 * sigma)))

    for point in samples:
        center = np.round(point).astype(int)
        z0, y0, x0 = np.maximum(center - radius, 0)
        z1, y1, x1 = np.minimum(center + radius + 1, np.asarray(shape))

        zz, yy, xx = np.meshgrid(
            np.arange(z0, z1),
            np.arange(y0, y1),
            np.arange(x0, x1),
            indexing="ij",
        )
        dist2 = (zz - point[0]) ** 2 + (yy - point[1]) ** 2 + (xx - point[2]) ** 2
        patch = np.exp(-dist2 / (2.0 * sigma**2))
        heatmap[z0:z1, y0:y1, x0:x1] = np.maximum(heatmap[z0:z1, y0:y1, x0:x1], patch)

    return heatmap


def main() -> None:
    args = parse_args()
    points = load_control_points(args.input)
    samples = interpolate_closed_curve(points, num_samples=args.num_samples)
    heatmap = draw_heatmap(tuple(args.shape), samples, sigma=args.sigma)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nrrd.write(str(args.output), heatmap)


if __name__ == "__main__":
    main()

