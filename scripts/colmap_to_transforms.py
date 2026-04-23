import json
import numpy as np
from pathlib import Path
from struct import unpack
import sys

# ======================
# ARGS
# ======================
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
# READ CAMERAS (IMPORTANT FIX)
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
            params = np.array(read_next_bytes(f, 8*4, "dddd"))

            cameras[cam_id] = {
                "w": width,
                "h": height,
                "fx": params[0],
                "fy": params[1],
                "cx": params[2],
                "cy": params[3],
            }

    return cameras

# ======================
# READ IMAGES
# ======================
def read_images(path):
    images = {}
    with open(path, "rb") as f:
        num_images = read_next_bytes(f, 8, "Q")[0]

        for _ in range(num_images):
            image_id = read_next_bytes(f, 4, "I")[0]
            qvec = np.array(read_next_bytes(f, 32, "dddd"))
            tvec = np.array(read_next_bytes(f, 24, "ddd"))
            cam_id = read_next_bytes(f, 4, "I")[0]

            name = ""
            while True:
                c = f.read(1).decode("utf-8")
                if c == "\x00":
                    break
                name += c

            num_points2D = read_next_bytes(f, 8, "Q")[0]
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

for name, (qvec, tvec, cam_id) in images.items():

    R = qvec2rotmat(qvec)
    t = tvec.reshape(3, 1)

    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t[:, 0]

    c2w = np.linalg.inv(w2c)

    frames.append({
        "file_path": f"images/{name}",
        "transform_matrix": c2w.tolist()
    })

# ======================
# FINAL JSON (NERF COMPATIBLE)
# ======================
out = {
    "fl_x": cam["fx"],
    "fl_y": cam["fy"],
    "cx": cam["cx"],
    "cy": cam["cy"],
    "w": cam["w"],
    "h": cam["h"],
    "frames": frames
}

with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"✅ transforms.json written to {out_path}")
