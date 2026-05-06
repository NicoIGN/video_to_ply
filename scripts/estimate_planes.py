import numpy as np
import struct
import os
import sys
import argparse


MAX_POINTS = 200000
MAX_CAMERAS = 200


# ======================
# LOGGING
# ======================
def log(msg):
    print(f"[INFO] {msg}")


def err(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)


# ======================
# HELP
# ======================
def print_help():
    print("""
COLMAP Scene Inspector

Usage:
  python estimate_planes.py --input /path/to/colmap/sparse/0

Input:
  --input    Path to COLMAP sparse model folder (must contain images.bin & points3D.bin)

Output:
  NEAR=...
  FAR=...

Example:
  python estimate_planes.py --input ./sparse/0
""")
    sys.exit(0)


# ======================
# SAFETY
# ======================
def safe_normalize(q):
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-8:
        return None
    return q / norm


def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    if os.path.getsize(path) < 100:
        raise ValueError(f"File too small / corrupted: {path}")


# ======================
# COLMAP LOADING
# ======================
def read_points3d_bin(path):
    check_file(path)

    pts = []
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]

        if num_points > 50_000_000 or num_points <= 0:
            raise ValueError(f"Invalid num_points3D: {num_points}")

        for _ in range(num_points):
            f.read(8)
            xyz = struct.unpack("<ddd", f.read(24))
            f.read(3)
            f.read(8)

            track_len = struct.unpack("<Q", f.read(8))[0]

            # safety: avoid insane reads (corrupted COLMAP)
            if track_len > 1_000_000:
                raise ValueError(f"Corrupted track_len: {track_len}")

            f.read(8 * track_len)

            if np.all(np.isfinite(xyz)):
                pts.append(xyz)

            if len(pts) >= MAX_POINTS:
                break

    pts = np.array(pts, dtype=np.float64)
    log(f"Loaded points3D: {len(pts)}")
    return pts


def read_images_bin(path):
    check_file(path)

    cams = []

    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]

        if num_images > 1_000_000 or num_images <= 0:
            raise ValueError(f"Invalid num_images: {num_images}")

        log(f"Images found: {num_images}")

        for _ in range(num_images):
            f.read(4)
            qvec = np.array(struct.unpack("<dddd", f.read(32)))
            tvec = np.array(struct.unpack("<ddd", f.read(24)))
            f.read(4)

            # image name
            while True:
                if f.read(1) == b"\x00":
                    break

            num_pts2D = struct.unpack("<Q", f.read(8))[0]

            if num_pts2D > 5_000_000:
                raise ValueError(f"Corrupted num_pts2D: {num_pts2D}")

            f.read(num_pts2D * 24)

            q = safe_normalize(qvec)
            if q is None:
                continue

            qw, qx, qy, qz = q

            R = np.array([
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
            ])

            if not np.all(np.isfinite(R)):
                continue

            C = -R.T @ tvec

            if np.all(np.isfinite(C)):
                cams.append(C)

            if len(cams) >= MAX_CAMERAS:
                break

    cams = np.array(cams, dtype=np.float64)
    log(f"Valid cameras: {len(cams)}")
    return cams


# ======================
# ESTIMATION
# ======================
def estimate_planes(colmap_dir):
    pts = read_points3d_bin(os.path.join(colmap_dir, "points3D.bin"))
    cams = read_images_bin(os.path.join(colmap_dir, "images.bin"))

    if len(pts) < 1000 or len(cams) < 2:
        raise RuntimeError("Invalid COLMAP data")

    # ======================
    # 1. CLEAN BBOX (ignore outliers)
    # ======================
    bbox_min = np.percentile(pts, 5, axis=0)
    bbox_max = np.percentile(pts, 95, axis=0)

    scene_size = np.linalg.norm(bbox_max - bbox_min)

    # ======================
    # 2. CAMERA CENTER STATS
    # ======================
    cam_center = np.mean(cams, axis=0)

    cam_dist = np.linalg.norm(cams - cam_center, axis=1)
    cam_scale = np.median(cam_dist)

    # ======================
    # 3. NEAR (stable)
    # ======================
    near = cam_scale * 0.3

    # ======================
    # 4. FAR (IMPORTANT FIX)
    # ======================
    # clamp scene depth instead of raw point distance
    far = min(scene_size * 0.6, cam_scale * 2.5)

    # safety constraints
    near = max(near, 0.05)
    far = max(far, near * 3)

    if not np.isfinite(near) or not np.isfinite(far):
        raise RuntimeError("Invalid result")

    return float(near), float(far)


# ======================
# MAIN
# ======================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input", type=str)

    args, unknown = parser.parse_known_args()

    if "--help" in sys.argv or args.input is None:
        print_help()

    colmap_dir = args.input

    if not os.path.isdir(colmap_dir):
        err(f"Invalid input directory: {colmap_dir}")
        sys.exit(1)

    try:
        log("📦 Loading COLMAP scene...")
        near, far = estimate_planes(colmap_dir)

        log("📏 Estimated planes:")
        print(f"NEAR={near:.4f}")
        print(f"FAR={far:.4f}")

    except Exception as e:
        err(str(e))
        sys.exit(1)
