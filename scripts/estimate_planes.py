import numpy as np
import struct
import os

def read_points3d_bin(path):
    points = []
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            f.read(8)  # id
            xyz = struct.unpack("<ddd", f.read(24))
            f.read(3)  # rgb
            f.read(8)  # error
            track_length = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * track_length)  # track
            points.append(xyz)
    return np.array(points)

def read_images_bin(path):
    cameras = []
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            f.read(8)  # image id
            qvec = struct.unpack("<dddd", f.read(32))
            tvec = struct.unpack("<ddd", f.read(24))
            f.read(4)  # camera id

            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c

            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2D * 24)

            # compute camera center
            q = np.array(qvec)
            t = np.array(tvec)

            # rotation matrix from quaternion
            qw, qx, qy, qz = q
            R = np.array([
                [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
                [2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
                [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx*qx-2*qy*qy]
            ])

            C = -R.T @ t
            cameras.append(C)

    return np.array(cameras)

def estimate_planes(colmap_dir):
    pts = read_points3d_bin(os.path.join(colmap_dir, "points3D.bin"))
    cams = read_images_bin(os.path.join(colmap_dir, "images.bin"))

    dists = []

    for c in cams:
        diff = pts - c
        d = np.linalg.norm(diff, axis=1)
        dists.append(d)

    dists = np.concatenate(dists)

    near = np.percentile(dists, 5)
    far = np.percentile(dists, 95)

    near *= 0.8
    far *= 1.2

    return near, far

if __name__ == "__main__":
    import sys
    colmap_dir = sys.argv[1]

    near, far = estimate_planes(colmap_dir)

    print(f"NEAR={near:.4f}")
    print(f"FAR={far:.4f}")
