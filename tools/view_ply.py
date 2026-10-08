import argparse
import time

import numpy as np
import open3d as o3d
import viser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View a PLY point cloud in Viser.")
    parser.add_argument("--pcd_path", type=str, required=True, help="Path to input PLY file.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Viser server host.")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port.")
    parser.add_argument(
        "--point_size",
        type=float,
        default=0.01,
        help="Rendered point size in Viser.",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=1.0,
        help="Fraction of points to keep for visualization (0 < r <= 1).",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=0,
        help="Hard cap on number of displayed points (0 disables cap).",
    )
    return parser.parse_args()


def maybe_subsample(points: np.ndarray, colors: np.ndarray, sample_ratio: float, max_points: int):
    n = points.shape[0]
    if n == 0:
        return points, colors

    sample_ratio = float(np.clip(sample_ratio, 1e-6, 1.0))
    keep = int(n * sample_ratio)
    keep = max(1, keep)

    if max_points > 0:
        keep = min(keep, max_points)

    if keep >= n:
        return points, colors

    idx = np.random.choice(n, size=keep, replace=False)
    return points[idx], colors[idx]


def main() -> None:
    args = parse_args()

    pcd = o3d.io.read_point_cloud(args.pcd_path)
    if pcd.is_empty():
        raise ValueError(f"Point cloud is empty: {args.pcd_path}")

    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)

    if colors.shape[0] != points.shape[0]:
        colors = np.full((points.shape[0], 3), 0.8, dtype=np.float32)

    points, colors = maybe_subsample(points, colors, args.sample_ratio, args.max_points)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_point_cloud(
        name="pcd",
        points=points,
        colors=colors,
        point_size=args.point_size,
    )

    print(f"Loaded PLY: {args.pcd_path}")
    print(f"Showing {points.shape[0]} points on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
