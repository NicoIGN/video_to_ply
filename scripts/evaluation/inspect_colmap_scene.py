#!/usr/bin/env python3
import argparse
import struct
import os
import sys
import numpy as np


# ======================
# LOGGING
# ======================
def log(msg):
    print(f"[INFO] {msg}")


def warn(msg):
    print(f"[WARN] {msg}")


def err(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)


# ======================
# SAFE READ
# ======================
def safe_read(f, size):
    data = f.read(size)
    if len(data) != size:
        raise EOFError(f"Expected {size} bytes, got {len(data)}")
    return data


# ======================
# READ IMAGES.BIN (SAFE)
# ======================
def read_images_binary(path):
    if not os.path.isfile(path):
        err(f"images.bin not found: {path}")

    images = {}

    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", safe_read(f, 8))[0]
        log(f"Number of images: {num_images}")

        if num_images > 1e6:
            err("Unrealistic number of images → corrupted file")

        for i in range(num_images):
            try:
                image_id = struct.unpack("<i", safe_read(f, 4))[0]

                qw, qx, qy, qz = struct.unpack("<dddd", safe_read(f, 32))
                tx, ty, tz = struct.unpack("<ddd", safe_read(f, 24))
                camera_id = struct.unpack("<i", safe_read(f, 4))[0]

                # read name
                name_bytes = b""
                while True:
                    c = safe_read(f, 1)
                    if c == b"\x00":
                        break
                    name_bytes += c
                name = name_bytes.decode("utf-8", errors="ignore")

                num_points2D = struct.unpack("<Q", safe_read(f, 8))[0]

                # 🔒 SAFETY CHECK
                if num_points2D > 10_000_000:
                    warn(f"Image {image_id}: absurd num_points2D={num_points2D}, skipping")
                    continue

                # skip points safely
                skip_bytes = num_points2D * (8 + 8 + 8)  # x,y + id
                f.seek(skip_bytes, 1)

                images[image_id] = {
                    "q": np.array([qw, qx, qy, qz], dtype=np.float64),
                    "t": np.array([tx, ty, tz], dtype=np.float64),
                    "name": name,
                }

            except Exception as e:
                warn(f"Failed parsing image {i}: {e}")
                break

    return images


# ======================
# QUAT → ROT (SAFE)
# ======================
def qvec2rotmat(q):
    if np.any(np.isnan(q)) or np.linalg.norm(q) == 0:
        return None

    q = q / np.linalg.norm(q)
    w, x, y, z = q

    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])

    if np.any(np.isnan(R)) or np.any(np.isinf(R)):
        return None

    return R


# ======================
# MAIN
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="COLMAP sparse dir")
    args = parser.parse_args()

    colmap_dir = args.input
    images_path = os.path.join(colmap_dir, "images.bin")

    log("📦 Loading COLMAP data...")
    images = read_images_binary(images_path)

    if len(images) == 0:
        err("No valid images parsed")

    log(f"Valid cameras: {len(images)}")

    centers = []

    for img_id, data in images.items():
        R = qvec2rotmat(data["q"])
        t = data["t"]

        if R is None:
            warn(f"Invalid rotation for image {img_id}, skipping")
            continue

        C = -R.T @ t

        if np.any(np.isnan(C)) or np.any(np.isinf(C)):
            warn(f"Invalid camera center for image {img_id}, skipping")
            continue

        centers.append(C)

    centers = np.array(centers)

    if len(centers) == 0:
        err("No valid camera centers")

    # ======================
    # STATS
    # ======================
    bbox_min = centers.min(axis=0)
    bbox_max = centers.max(axis=0)
    bbox_size = bbox_max - bbox_min

    log("📐 Scene stats:")
    print(f"  Cameras: {len(centers)}")
    print(f"  BBox min: {bbox_min}")
    print(f"  BBox max: {bbox_max}")
    print(f"  BBox size: {bbox_size}")

    # distances
    dists = np.linalg.norm(centers - centers.mean(axis=0), axis=1)

    near = np.percentile(dists, 5)
    far = np.percentile(dists, 95)

    print("")
    print("📏 Suggested planes:")
    print(f"NEAR={near:.4f}")
    print(f"FAR={far:.4f}")


if __name__ == "__main__":
    main()
