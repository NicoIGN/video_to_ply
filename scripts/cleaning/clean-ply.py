import numpy as np
import argparse
import os
import struct
import json
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement


# ======================
# LOG
# ======================
def log(msg):
    print(f"[INFO] {msg}")


def warn(msg):
    print(f"[WARN] {msg}")


# ======================
# SAFE READ
# ======================
def safe_read(f, size):
    data = f.read(size)
    if len(data) != size:
        raise EOFError(f"Expected {size} bytes, got {len(data)}")
    return data


# ======================
# TRANSFORM
# ======================
def apply_transform(pts, transform, scale=1.0):
    R = np.array(transform[:3, :3], dtype=np.float64)
    t = np.array(transform[:3, 3], dtype=np.float64)

    pts = (pts @ R.T) + t
    pts = pts * scale

    return pts


# ======================
# COLMAP LOADER
# ======================
def load_colmap_points(path, transform, scale, max_points=4_000_000):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    pts = np.empty((max_points, 3), dtype=np.float32)
    idx = 0

    with open(path, "rb") as f:
        n = struct.unpack("<Q", safe_read(f, 8))[0]
        log(f"Raw COLMAP points: {n}")

        for i in range(n):
            try:
                safe_read(f, 8)
                x, y, z = struct.unpack("<ddd", safe_read(f, 24))
                safe_read(f, 3)
                safe_read(f, 8)

                track_len = struct.unpack("<Q", safe_read(f, 8))[0]
                f.seek(track_len * 8, 1)

                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    pts[idx] = (x, y, z)
                    idx += 1

                if idx >= max_points:
                    break

            except Exception as e:
                warn(f"Skipping point {i}: {e}")
                break

    pts = pts[:idx]

    if len(pts) == 0:
        raise ValueError("No valid COLMAP points")

    # 🔥 APPLY TRANSFORM + SCALE
    pts = apply_transform(pts, transform, scale)

    log(f"Valid COLMAP points (transformed): {len(pts)}")

    return pts


# ======================
# RADIUS
# ======================
def estimate_radius(pts, factor=3.0):
    n = min(5000, len(pts))
    sample = pts[np.random.choice(len(pts), n, replace=False)]

    tree = cKDTree(sample)
    d, _ = tree.query(sample, k=2)

    return float(np.median(d[:, 1]) * factor)


# ======================
# LOAD PLY
# ======================
def load_ply(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data

    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)

    mask = np.isfinite(xyz).all(axis=1)

    return xyz[mask], ply, mask


# ======================
# FILTER
# ======================
def filter_gaussians(gaussians, colmap_pts, dist, batch=200000):
    tree = cKDTree(colmap_pts)

    mask = np.zeros(len(gaussians), dtype=bool)

    for i in range(0, len(gaussians), batch):
        g = gaussians[i:i+batch]

        valid = np.isfinite(g).all(axis=1)
        if not np.any(valid):
            continue

        d, _ = tree.query(g[valid], k=1)

        idx_valid = np.where(valid)[0]
        mask[i:i+batch][valid] = d < dist

    return mask


# ======================
# WRITE PLY
# ======================
def write_ply(template_ply, mask, out_path):
    vertex = template_ply["vertex"].data
    new_vertex = vertex[mask]

    PlyData([PlyElement.describe(new_vertex, "vertex")], text=False).write(out_path)


# ======================
# MAIN
# ======================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in-ply", required=True)
    parser.add_argument("--points", required=True)
    parser.add_argument("--transform", required=True)  # JSON now
    parser.add_argument("--dist", type=float, default=None)
    parser.add_argument("--out-ply", required=True)

    args = parser.parse_args()

    # ======================
    # LOAD TRANSFORM JSON
    # ======================
    log(f"Loading transform: {args.transform}")
    with open(args.transform, "r") as f:
        data = json.load(f)

    transform = np.array(data["transform"], dtype=np.float64)
    scale = float(data.get("scale", 1.0))

    # ======================
    # LOAD COLMAP
    # ======================
    log(f"Loading COLMAP points from: {args.points}")
    colmap_pts = load_colmap_points(args.points, transform, scale)

    # ======================
    # LOAD PLY
    # ======================
    log(f"Loading PLY from: {args.in_ply}")
    gaussians, ply, ply_mask = load_ply(args.in_ply)

    log(f"Gaussians: {len(gaussians)}")

    bbox_min = gaussians.min(axis=0)
    bbox_max = gaussians.max(axis=0)
    log(f"PLY bbox: {bbox_min} → {bbox_max}")

    # ======================
    # RADIUS
    # ======================
    dist = args.dist or estimate_radius(colmap_pts)
    log(f"Radius: {dist:.4f}")

    # ======================
    # FILTER
    # ======================
    log("Filtering...")
    mask = filter_gaussians(gaussians, colmap_pts, dist)

    log(f"Remaining: {mask.sum()}")

    # ======================
    # WRITE
    # ======================
    write_ply(ply, mask, args.out_ply)

    log(f"Saved: {args.out_ply}")


if __name__ == "__main__":
    main()
