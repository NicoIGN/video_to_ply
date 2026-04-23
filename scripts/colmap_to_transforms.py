import json
import numpy as np
from pathlib import Path
from struct import unpack
import sys

# ======================
# ARGS
# ======================
if len(sys.argv) < 2:
    raise ValueError("Usage: python colmap_to_transforms.py <output_dir>")

output_dir = Path(sys.argv[1])
colmap_dir = output_dir / "colmap" / "sparse" / "0"
out_path = output_dir / "transforms.json"

# ======================
# UTILS
# ======================
def read_next_bytes(fid, num_bytes, fmt, endian="<"):
    return unpack(endian + fmt, fid.read(num_bytes))

def qvec2rotmat(qvec):
    q0, q1, q2, q3 = qvec
    return np.array([
        [1 - 2*q2*q2 - 2*q3*q3,
         2*q1*q2 - 2*q0*q3,
         2*q1*q3 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3,
         1 - 2*q1*q1 - 2*q3*q3,
         2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2,
         2*q2*q3 + 2*q0*q1,
         1 - 2*q1*q1 - 2*q2*q2]
    ])

# ======================
# READ CAMERAS (FIXED COLMAP BINARY)
# ======================
def read_cameras(path):
    cameras = {}

    with open(path, "rb") as f:
        num_cams = read_next_bytes(f, 8, "Q")[0]

        for _ in range(num_cams):
            cam_id = read_next_bytes(f, 4, "I")[0]
            model_id = read_next_bytes(f, 4, "I")[0]

            width = read_next_bytes(f, 8, "Q")[0]
            height = read_next_bytes(f, 8, "Q")[0]

            # COLMAP params are float64 but variable size → read remaining camera block safely
            # We assume typical pinhole model (4 params)
            params = np.frombuffer(f.read(4 * 8), dtype=np.float64)

            cameras[cam_id] = {
                "w": int(width),
                "h": int(height),
                "fx": float(params[0]),
                "fy": float(params[1]),
                "cx": float(params[2]),
                "cy": float(params[3]),
            }

    return cameras

# ======================
# READ IMAGES (FIXED)
# ======================
def read_images(path):
    images = {}

    with open(path, "rb") as f:
        num_images = read_next_bytes(f, 8, "Q")[0]

        for _ in range(num_images):
            image_id = read_next_bytes(f, 4, "I")[0]

            qvec = np.array(read_next_bytes(f, 32, "dddd"), dtype=np.float64)
            tvec = np.array(read_next_bytes(f, 24, "ddd"), dtype=np.float64)

            cam_id = read_next_bytes(f, 4, "I")[0]

            # image name (null terminated string)
            name = ""
            while True:
                c = f.read(1).decode("utf-8", errors="ignore")
                if c == "\x00":
                    break
                name += c

            num_points2D = read_next_bytes(f, 8, "Q")[0]

            # each point2D = (x, y, point3D_id) -> 8+8+8 bytes
            f.read(num_points2D * 24)

            images[name] = (qvec, tvec, cam_id)

    return images

# ======================
# LOAD DATA
# ======================
cameras = read_cameras(colmap_dir / "cameras.bin")
images = read_images(colmap_dir / "images.bin")

cam = list(cameras.values())[0]

# ======================
# BUILD TRANSFORMS
# ======================
frames = []

flip = np.diag([1, -1, -1, 1])  # COLMAP → NeRF

for name, (qvec, tvec, cam_id) in images.items():

    qvec = qvec / (np.linalg.norm(qvec) + 1e-12)

    R = qvec2rotmat(qvec)
    t = tvec.reshape(3, 1)

    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t[:, 0]

    c2w = np.linalg.inv(w2c)
    c2w = c2w @ flip

    if np.isnan(c2w).any():
        print(f"⚠️ Skipping invalid pose: {name}")
        continue

    frames.append({
        "file_path": f"images/{name}",
        "transform_matrix": c2w.tolist()
    })

# ======================
# OUTPUT JSON
# ======================
out = {
    "fl_x": float(cam["fx"]),
    "fl_y": float(cam["fy"]),
    "cx": float(cam["cx"]),
    "cy": float(cam["cy"]),
    "w": int(cam["w"]),
    "h": int(cam["h"]),
    "frames": frames
}

with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"✅ transforms.json written to {out_path}")
