import numpy as np
import struct
import argparse
import json
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


# ======================
# COLMAP LOADER
# ======================
def load_colmap_points(path, max_points=5_000_000):
    pts = []
    colors = []

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]

        for _ in range(n):
            f.read(8)
            x, y, z = struct.unpack("<ddd", f.read(24))

            r, g, b = struct.unpack("<BBB", f.read(3))
            f.read(8)

            track_len = struct.unpack("<Q", f.read(8))[0]
            f.seek(track_len * 8, 1)

            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                pts.append([x, y, z])
                colors.append([r, g, b])

            if len(pts) >= max_points:
                break

    return np.asarray(pts, np.float32), np.asarray(colors, np.float32)


# ======================
# TRANSFORM COLMAP → NERFSTUDIO SPACE
# ======================
def apply_transform(pts, transform=None, scale=None):
    """
    Applies Nerfstudio dataparser transform if provided.

    X_ns = scale * (R X + t)
    """

    if transform is None:
        return pts

    R = np.array([row[:3] for row in transform], dtype=np.float32)
    t = np.array([row[3] for row in transform], dtype=np.float32)

    pts = pts @ R.T + t

    if scale is not None:
        pts *= scale

    return pts


# ======================
# LOCAL SCALE FROM COLMAP DENSITY
# ======================
def compute_local_scales(colmap_pts, factor=0.25):
    tree = cKDTree(colmap_pts)
    dists, _ = tree.query(colmap_pts, k=2)

    nn_dist = dists[:, 1]
    median = np.median(nn_dist)

    scale = nn_dist * factor
    scale = np.clip(scale, median * 0.05, median * 0.5)

    return scale


# ======================
# SPLATS
# ======================
def build_splats(pts, colors, scales):
    n = len(pts)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),

        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),

        ("opacity", "f4"),

        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),

        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]

    data = np.zeros(n, dtype=dtype)

    data["x"] = pts[:, 0]
    data["y"] = pts[:, 1]
    data["z"] = pts[:, 2]

    data["f_dc_0"] = colors[:, 0] / 255.0
    data["f_dc_1"] = colors[:, 1] / 255.0
    data["f_dc_2"] = colors[:, 2] / 255.0

    data["opacity"] = 1.0

    log_scale = np.log(scales + 1e-8)

    data["scale_0"] = log_scale
    data["scale_1"] = log_scale
    data["scale_2"] = log_scale

    data["rot_0"] = 1.0
    data["rot_1"] = 0.0
    data["rot_2"] = 0.0
    data["rot_3"] = 0.0

    return data


# ======================
# WRITE PLY
# ======================
def write_ply(data, out_path):
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(out_path)


# ======================
# LOAD TRANSFORM (OPTIONAL)
# ======================
def load_transform(path):
    if path is None:
        return None, None

    with open(path, "r") as f:
        data = json.load(f)

    transform = data.get("transform", None)
    scale = data.get("scale", None)

    return transform, scale


# ======================
# MAIN
# ======================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--points", required=True)
    parser.add_argument("--out-ply", required=True)
    parser.add_argument("--transform", default=None, help="dataparser json (optional)")

    args = parser.parse_args()

    print("Loading COLMAP...")
    pts, colors = load_colmap_points(args.points)

    print(f"Points: {len(pts)}")

    # ======================
    # APPLY NERFSTUDIO SPACE TRANSFORM
    # ======================
    transform, scale = load_transform(args.transform)

    if transform is not None:
        print("Applying Nerfstudio transform...")
        pts = apply_transform(pts, transform, scale)
    else:
        print("No transform applied (raw COLMAP space)")

    # ======================
    # LOCAL SCALE (IN FINAL SPACE)
    # ======================
    print("Computing local density scales...")
    scales = compute_local_scales(pts)

    print(f"Scale stats:")
    print(f"  min: {scales.min():.6f}")
    print(f"  max: {scales.max():.6f}")
    print(f"  median: {np.median(scales):.6f}")

    # ======================
    # BUILD SPLATS
    # ======================
    print("Building pseudo splats...")
    splats = build_splats(pts, colors, scales)

    print("Writing PLY...")
    write_ply(splats, args.out_ply)

    print(f"Saved: {args.out_ply}")


if __name__ == "__main__":
    main()
