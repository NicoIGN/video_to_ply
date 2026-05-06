import numpy as np
import argparse
import os
import struct
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
# COLMAP LOADER (CLEAN + FINITE ONLY)
# ======================
def load_colmap_points(path, max_points=4_000_000):
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

                # HARD CLEAN
                if (
                    np.isfinite(x) and
                    np.isfinite(y) and
                    np.isfinite(z)
                ):
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

    # FINAL CLEAN (IMPORTANT FIX FOR KDTree)
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]

    log(f"Valid COLMAP points: {len(pts)}")

    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)

    log("📦 COLMAP BBOX:")
    log(f"  min: {bbox_min}")
    log(f"  max: {bbox_max}")
    log(f"  size: {bbox_max - bbox_min}")

    return pts


# ======================
# RADIUS ESTIMATION
# ======================
def estimate_radius(pts, factor=3.0):
    n = min(5000, len(pts))
    sample = pts[np.random.choice(len(pts), n, replace=False)]

    tree = cKDTree(sample)
    d, _ = tree.query(sample, k=2)

    return float(np.median(d[:, 1]) * factor)


# ======================
# LOAD PLY (PLYFILE SAFE)
# ======================
def load_ply(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data

    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)

    mask = np.isfinite(xyz).all(axis=1)

    return xyz[mask], ply, mask


# ======================
# FILTER (FAST KDTree)
# ======================
def filter_gaussians(gaussians, colmap_pts, dist, batch=200000):
    tree = cKDTree(colmap_pts)

    mask = np.zeros(len(gaussians), dtype=bool)

    for i in range(0, len(gaussians), batch):
        g = gaussians[i:i+batch]

        # safety
        valid = np.isfinite(g).all(axis=1)
        if not np.any(valid):
            continue

        d, _ = tree.query(g[valid], k=1)

        mask_idx = np.where(valid)[0]
        mask[i:i+batch][valid] = d < dist

    return mask


# ======================
# WRITE PLY PROPERLY
# ======================
def write_ply(template_ply, mask, out_path):
    vertex = template_ply["vertex"].data
    new_vertex = vertex[mask]

    PlyData(
        [PlyElement.describe(new_vertex, "vertex")],
        text=False
    ).write(out_path)


# ======================
# MAIN
# ======================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in-ply", required=True)
    parser.add_argument("--points", required=True)
    parser.add_argument("--dist", type=float, default=None)
    parser.add_argument("--out-ply", required=True)

    args = parser.parse_args()

    log("Loading COLMAP...")
    colmap_pts = load_colmap_points(args.points)

    log("Loading PLY...")
    gaussians, ply, ply_mask = load_ply(args.in_ply)
    
# ======================
# PLY BBOX (DEBUG)
# ======================
    bbox_min = np.min(gaussians, axis=0)
    bbox_max = np.max(gaussians, axis=0)
    bbox_size = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) * 0.5

    log("📦 PLY BBOX:")
    log(f"  min: {bbox_min}")
    log(f"  max: {bbox_max}")
    log(f"  size: {bbox_size}")
    log(f"  center: {bbox_center}")

    log(f"Gaussians: {len(gaussians)}")

    dist = args.dist or estimate_radius(colmap_pts)
    log(f"Radius: {dist:.4f}")

    log("Filtering...")
    mask = filter_gaussians(gaussians, colmap_pts, dist)

    log(f"Remaining: {mask.sum()}")

    write_ply(ply, mask, args.out_ply)

    log(f"Saved: {args.out_ply}")


if __name__ == "__main__":
    main()
