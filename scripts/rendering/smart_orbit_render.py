#!/usr/bin/env python3
"""
smart_orbit_render.py

Usage:
    python smart_orbit_render.py \
        --config /path/to/config.yml \
        --ply /path/to/model.ply \
        --output /path/to/output.mp4

What it does:
- Loads the PLY point cloud
- Estimates scene center + scale from bounding box
- Builds an intelligent orbital camera path around the object
- Slightly elevates camera for better viewing angle
- Exports Nerfstudio camera path JSON
- Launches ns-render automatically
"""

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from plyfile import PlyData


def load_ply_points(ply_path):
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    pts = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    return pts


def compute_scene_stats(points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = maxs - mins
    radius = np.linalg.norm(extent) * 0.8
    return center, extent, radius


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def look_at(camera_pos, target, up=np.array([0, 0, 1], dtype=np.float32)):
    forward = normalize(target - camera_pos)
    right = normalize(np.cross(forward, up))
    true_up = normalize(np.cross(right, forward))

    rot = np.eye(4, dtype=np.float32)
    rot[:3, 0] = right
    rot[:3, 1] = true_up
    rot[:3, 2] = -forward
    rot[:3, 3] = camera_pos
    return rot


def generate_orbit(center, radius, extent, num_frames=240):
    frames = []

    vertical_offset = max(extent[2] * 0.3, radius * 0.15)
    cam_height = center[2] + vertical_offset

    for i in range(num_frames):
        theta = 2 * math.pi * i / num_frames

        # Elliptical orbit for more cinematic motion
        x = center[0] + radius * math.cos(theta)
        y = center[1] + radius * 0.85 * math.sin(theta)
        z = cam_height + math.sin(theta * 2) * radius * 0.05

        cam_pos = np.array([x, y, z], dtype=np.float32)

        mat = look_at(cam_pos, center)

        frames.append({
            "camera_to_world": mat.tolist(),
            "fov": 60.0
        })

    return frames


def save_camera_path(frames, out_json, fps=30):
    data = {
        "camera_type": "perspective",
        "render_height": 1080,
        "render_width": 1920,
        "fps": fps,
        "seconds": len(frames) / fps,
        "camera_path": frames,
    }

    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)


def render_video(config_path, camera_path_json, output_mp4):
    cmd = [
    "ns-render", "camera-path",
        "--load-config", str(config_path),
        "--camera-path-filename", str(camera_path_json),
        "--output-path", str(output_mp4),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to Nerfstudio config.yml")
    parser.add_argument("--ply", required=True, help="Path to .ply point cloud")
    parser.add_argument("--output", required=True, help="Output mp4 path")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)

    args = parser.parse_args()

    print("Loading PLY...")
    points = load_ply_points(args.ply)

    print("Analyzing scene...")
    center, extent, radius = compute_scene_stats(points)

    print("Center:", center)
    print("Extent:", extent)
    print("Orbit radius:", radius)

    frames = generate_orbit(
        center=center,
        radius=radius,
        extent=extent,
        num_frames=args.frames
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        cam_json = Path(tmpdir) / "camera_path.json"

        save_camera_path(frames, cam_json, fps=args.fps)

        print("Rendering video...")
        render_video(
            config_path=args.config,
            camera_path_json=cam_json,
            output_mp4=args.output
        )

    print("Done:", args.output)


if __name__ == "__main__":
    main()
