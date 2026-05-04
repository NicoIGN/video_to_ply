import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import imageio

import gsplat


# -----------------------------
# ARGPARSE
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ply", required=True)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)

    return parser.parse_args()


args = get_args()

PLY_PATH = os.path.abspath(args.ply)
TRANSFORMS_PATH = os.path.abspath(args.transforms)
OUTPUT_DIR = os.path.abspath(args.output_dir)

FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
VIDEO_PATH = os.path.join(OUTPUT_DIR, "video.mp4")

os.makedirs(FRAMES_DIR, exist_ok=True)


# -----------------------------
# VALIDATION
# -----------------------------
if not os.path.isfile(PLY_PATH):
    raise FileNotFoundError(PLY_PATH)

if not os.path.isfile(TRANSFORMS_PATH):
    raise FileNotFoundError(TRANSFORMS_PATH)


print("[INFO] Loading transforms:", TRANSFORMS_PATH)


# -----------------------------
# LOAD TRANSFORMS
# -----------------------------
with open(TRANSFORMS_PATH, "r") as f:
    data = json.load(f)

frames = data["frames"]
W = args.width
H = args.height


# -----------------------------
# LOAD GSPLAT SCENE (CORRECT WAY)
# -----------------------------
print("[INFO] Loading PLY:", PLY_PATH)

scene = gsplat.Scene.from_ply(PLY_PATH)


# -----------------------------
# RENDER LOOP
# -----------------------------
print("[INFO] Rendering frames...")

for i, frame in enumerate(tqdm(frames)):

    cam = np.array(frame["transform_matrix"])

    img = gsplat.render(
        scene,
        cam,
        width=W,
        height=H
    )

    out_path = os.path.join(FRAMES_DIR, f"frame_{i:04d}.png")
    imageio.imwrite(out_path, img)


# -----------------------------
# ENCODE VIDEO
# -----------------------------
print("[INFO] Encoding video...")

os.system(f"""
ffmpeg -y -framerate {args.fps} \
-i {FRAMES_DIR}/frame_%04d.png \
-c:v libx264 -pix_fmt yuv420p \
{VIDEO_PATH}
""")

print("[DONE] Video saved to:", VIDEO_PATH)
