import numpy as np
import struct
import os

MAX_POINTS = 200000   # safety cap
MAX_CAMERAS = 200     # safety cap

def safe_normalize(q):
    norm = np.linalg.norm(q)
    if norm < 1e-8 or not np.isfinite(norm):
        return None
    return q / norm

def read_points3d_bin(path):
    pts = []
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]

        for i in range(num_points):
            f.read(8)  # id
            xyz = struct.unpack("<ddd", f.read(24))
            f.read(3)  # rgb
            f.read(8)  # error
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * track_len)

            if np.all(np.isfinite(xyz)):
                pts.append(xyz)

            if len(pts) >= MAX_POINTS:
                break

    return np.array(pts, dtype=np.float64)

def read_images_bin(path):
    cams = []

    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]

        for i in range(num_images):
            f.read(8)  # image id
            qvec = np.array(struct.unpack("<dddd", f.read(32)))
            tvec = np.array(struct.unpack("<ddd", f.read(24)))
            f.read(4)

            # skip name
            while True:
                if f.read(1) == b"\x00":
                    break

            num_pts2D = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts2D * 24)

            q = safe_normalize(qvec)
            if q is None:
                continue

            qw, qx, qy, qz = q

            # rotation matrix (stable)
            R = np.array([
                [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw),         1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw),         2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]
            ])

            if not np.all(np.isfinite(R)):
                continue

            C = -R.T @ tvec

            if np.all(np.isfinite(C)):
                cams.append(C)

            if len(cams) >= MAX_CAMERAS:
                break

    return np.array(cams, dtype=np.float64)

def estimate_planes(colmap_dir):
    pts = read_points3d_bin(os.path.join(colmap_dir, "points3D.bin"))
    cams = read_images_bin(os.path.join(colmap_dir, "images.bin"))

    if len(pts) == 0 or len(cams) == 0:
        raise RuntimeError("No valid COLMAP data")

    dists = []

    # sample to avoid explosion
    pts_sample = pts[np.random.choice(len(pts), min(len(pts), 50000), replace=False)]

    for c in cams:
        d = np.linalg.norm(pts_sample - c, axis=1)
        d = d[np.isfinite(d)]
        dists.append(d)

    dists = np.concatenate(dists)

    if len(dists) == 0:
        raise RuntimeError("No valid distances")

    near = np.percentile(dists, 5)
    far  = np.percentile(dists, 95)

    # safety margins
    near *= 0.8
    far  *= 1.2

    # hard safety clamps
    near = max(near, 0.05)
    far  = min(far, 50.0)

    if not np.isfinite(near) or not np.isfinite(far):
        raise RuntimeError("NaN detected in result")

    return near, far

if __name__ == "__main__":
    import sys

    try:
        near, far = estimate_planes(sys.argv[1])
        print(f"NEAR={near:.4f}")
        print(f"FAR={far:.4f}")
    except Exception as e:
        print("ERROR:", str(e))
        print("NEAR=0.1")
        print("FAR=10.0")
