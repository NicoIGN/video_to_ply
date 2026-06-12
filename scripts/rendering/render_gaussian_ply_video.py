#!/usr/bin/env python3
import argparse
import math
import os
import struct
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm


# ============================================================
# Utils
# ============================================================

def normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy()
    return v / n


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def quat_to_rotmat(q):
    """
    q format: [w, x, y, z]
    """
    q = np.asarray(q, dtype=np.float32)
    q = q / max(np.linalg.norm(q), 1e-8)
    w, x, y, z = q

    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float32)


def look_at(camera_pos, target, up):
    forward = normalize(target - camera_pos)
    right = normalize(np.cross(forward, up))
    true_up = normalize(np.cross(right, forward))

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = forward
    c2w[:3, 3] = camera_pos
    return c2w


# ============================================================
# PLY reader
# ============================================================

PLY_DTYPE_MAP = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "i2",
    "ushort": "u2",
    "int16": "i2",
    "uint16": "u2",
    "int": "i4",
    "uint": "u4",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


@dataclass
class PlyData:
    vertex: np.ndarray
    property_names: list


def read_ply_header(fp):
    line = fp.readline().decode("utf-8").strip()
    if line != "ply":
        raise ValueError("Fichier non PLY")

    fmt = None
    vertex_count = None
    properties = []
    current_element = None

    while True:
        line = fp.readline().decode("utf-8")
        if not line:
            raise ValueError("Header PLY incomplet")
        line = line.strip()
        if line == "end_header":
            break

        parts = line.split()
        if not parts:
            continue

        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if parts[1] == "list":
                raise NotImplementedError("Les propriétés list ne sont pas supportées")
            properties.append((parts[2], parts[1]))

    if fmt is None or vertex_count is None:
        raise ValueError("Header PLY invalide")

    return fmt, vertex_count, properties


def load_ply_vertex_data(path):
    with open(path, "rb") as fp:
        fmt, vertex_count, properties = read_ply_header(fp)

        dtype = np.dtype([(name, PLY_DTYPE_MAP[typ]) for name, typ in properties])

        if fmt == "binary_little_endian":
            data = np.fromfile(fp, dtype=dtype, count=vertex_count)
        elif fmt == "ascii":
            rows = []
            for _ in range(vertex_count):
                line = fp.readline().decode("utf-8").strip()
                vals = line.split()
                row = []
                for (name, typ), v in zip(properties, vals):
                    dt = np.dtype(PLY_DTYPE_MAP[typ])
                    row.append(np.array(v, dtype=dt))
                rows.append(tuple(row))
            data = np.array(rows, dtype=dtype)
        else:
            raise NotImplementedError(f"Format PLY non supporté: {fmt}")

    return PlyData(vertex=data, property_names=[p[0] for p in properties])


# ============================================================
# Scene model
# ============================================================

@dataclass
class GaussianCloud:
    xyz: np.ndarray          # [N,3]
    opacity: np.ndarray      # [N]
    scale: np.ndarray        # [N,3]
    quat: np.ndarray         # [N,4] [w,x,y,z]
    color: np.ndarray        # [N,3] in [0,1]


def extract_gaussian_cloud(ply: PlyData) -> GaussianCloud:
    names = set(ply.property_names)
    v = ply.vertex

    required_xyz = ["x", "y", "z"]
    for k in required_xyz:
        if k not in names:
            raise ValueError(f"Champ requis manquant: {k}")

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    # opacity
    if "opacity" in names:
        opacity = sigmoid(v["opacity"].astype(np.float32))
    elif "alpha" in names:
        opacity = np.clip(v["alpha"].astype(np.float32), 0.0, 1.0)
    else:
        opacity = np.ones((len(v),), dtype=np.float32) * 0.8

    # scale
    if all(k in names for k in ["scale_0", "scale_1", "scale_2"]):
        # la plupart des exports stockent log-scale
        scale = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32))
    elif all(k in names for k in ["scale_x", "scale_y", "scale_z"]):
        scale = np.stack([v["scale_x"], v["scale_y"], v["scale_z"]], axis=1).astype(np.float32)
    else:
        scale = np.ones((len(v), 3), dtype=np.float32) * 0.01

    # rotation
    if all(k in names for k in ["rot_0", "rot_1", "rot_2", "rot_3"]):
        quat = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)
    elif all(k in names for k in ["qw", "qx", "qy", "qz"]):
        quat = np.stack([v["qw"], v["qx"], v["qy"], v["qz"]], axis=1).astype(np.float32)
    else:
        quat = np.zeros((len(v), 4), dtype=np.float32)
        quat[:, 0] = 1.0

    # color
    if all(k in names for k in ["f_dc_0", "f_dc_1", "f_dc_2"]):
        # approximation classique des DC SH
        color = 0.5 + 0.28209479177 * np.stack(
            [v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1
        ).astype(np.float32)
        color = np.clip(color, 0.0, 1.0)
    elif all(k in names for k in ["red", "green", "blue"]):
        color = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32)
        if color.max() > 1.5:
            color = color / 255.0
    else:
        color = np.ones((len(v), 3), dtype=np.float32) * 0.8

    return GaussianCloud(
        xyz=xyz,
        opacity=opacity,
        scale=scale,
        quat=quat,
        color=color,
    )


# ============================================================
# Stats + trajectory
# ============================================================

@dataclass
class SceneStats:
    center: np.ndarray
    radius: float
    up: np.ndarray


def compute_scene_stats(xyz):
    q01 = np.quantile(xyz, 0.01, axis=0)
    q99 = np.quantile(xyz, 0.99, axis=0)
    center = 0.5 * (q01 + q99)
    extent = q99 - q01
    radius = max(1e-3, 0.5 * np.linalg.norm(extent))

    centered = xyz - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    global_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    candidates = [eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2], global_up]

    up = max(candidates, key=lambda c: abs(np.dot(normalize(c), global_up)))
    up = normalize(up)
    if np.dot(up, global_up) < 0:
        up = -up

    return SceneStats(center=center.astype(np.float32), radius=float(radius), up=up.astype(np.float32))


def smart_orbit_trajectory(stats, n_frames):
    center = stats.center
    up = stats.up
    base_r = stats.radius * 1.8

    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(np.dot(tmp, up)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    right0 = normalize(np.cross(tmp, up))
    fwd0 = normalize(np.cross(up, right0))

    poses = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)

        theta = 2.0 * math.pi * 1.1 * t
        elev = math.radians(18.0 + 6.0 * math.sin(2.0 * math.pi * t))
        r = base_r * (1.0 + 0.12 * math.sin(2.0 * math.pi * t + 0.4))

        circle_dir = math.cos(theta) * right0 + math.sin(theta) * fwd0
        cam_pos = center + circle_dir * (r * math.cos(elev)) + up * (r * math.sin(elev))
        target = center + 0.04 * stats.radius * math.sin(2.0 * math.pi * t) * right0
        poses.append(look_at(cam_pos, target, up))

    return poses


# ============================================================
# Rendering
# ============================================================

def project_points(xyz_world, c2w, width, height, fov_deg):
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    xyz_cam = (xyz_world - t) @ R

    z = xyz_cam[:, 2]
    valid = z > 1e-4
    xyz_cam = xyz_cam[valid]
    z = z[valid]
    valid_idx = np.where(valid)[0]

    fx = 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)
    fy = 0.5 * height / math.tan(math.radians(fov_deg) * 0.5)
    cx = width / 2.0
    cy = height / 2.0

    u = fx * (xyz_cam[:, 0] / z) + cx
    v = fy * (-xyz_cam[:, 1] / z) + cy

    in_img = (u >= -32) & (u < width + 32) & (v >= -32) & (v < height + 32)
    return u[in_img], v[in_img], z[in_img], valid_idx[in_img], xyz_cam[in_img], fx, fy


def render_frame_gaussians(
    cloud: GaussianCloud,
    c2w: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    bg_white: bool = True,
    max_splats: int = 40000,
):
    bg = 1.0 if bg_white else 0.0
    img = np.full((height, width, 3), bg, dtype=np.float32)

    alpha_acc = np.zeros((height, width), dtype=np.float32)

    xyz = cloud.xyz
    opacity = cloud.opacity
    scale = cloud.scale
    quat = cloud.quat
    color = cloud.color

    # sous-échantillonnage si énorme scène
    n = len(xyz)
    if n > max_splats:
        idx = np.random.choice(n, max_splats, replace=False)
        xyz = xyz[idx]
        opacity = opacity[idx]
        scale = scale[idx]
        quat = quat[idx]
        color = color[idx]

    u, v, z, idx_valid, xyz_cam, fx, fy = project_points(xyz, c2w, width, height, fov_deg)

    if len(u) == 0:
        return (img * 255).astype(np.uint8)

    opacity = opacity[idx_valid]
    scale = scale[idx_valid]
    quat = quat[idx_valid]
    color = color[idx_valid]

    order = np.argsort(z)[::-1]  # far -> near
    u = u[order]
    v = v[order]
    z = z[order]
    opacity = opacity[order]
    scale = scale[order]
    quat = quat[order]
    color = color[order]

    R_cam = c2w[:3, :3].T

    for ui, vi, zi, oi, si, qi, ci in zip(u, v, z, opacity, scale, quat, color):
        Rg = quat_to_rotmat(qi)
        # axes principaux monde
        ax0 = Rg[:, 0] * si[0]
        ax1 = Rg[:, 1] * si[1]

        # passage caméra
        ax0c = R_cam @ ax0
        ax1c = R_cam @ ax1

        # taille projetée approx
        ru = fx * np.linalg.norm(ax0c[:2]) / max(zi, 1e-4)
        rv = fy * np.linalg.norm(ax1c[:2]) / max(zi, 1e-4)

        ru = float(np.clip(ru * 3.0, 1.0, 80.0))
        rv = float(np.clip(rv * 3.0, 1.0, 80.0))

        angle = math.degrees(math.atan2(ax0c[1], ax0c[0] + 1e-8))

        cx = int(round(ui))
        cy = int(round(vi))

        rx = int(math.ceil(max(ru, rv))) + 2
        x0 = max(0, cx - rx)
        x1 = min(width - 1, cx + rx)
        y0 = max(0, cy - rx)
        y1 = min(height - 1, cy + rx)

        if x1 < x0 or y1 < y0:
            continue

        xs = np.arange(x0, x1 + 1, dtype=np.float32)
        ys = np.arange(y0, y1 + 1, dtype=np.float32)
        XX, YY = np.meshgrid(xs, ys)

        # rotation 2D
        a = math.radians(angle)
        ca = math.cos(a)
        sa = math.sin(a)

        dx = XX - ui
        dy = YY - vi

        xr = ca * dx + sa * dy
        yr = -sa * dx + ca * dy

        g = np.exp(-0.5 * ((xr / max(ru, 1e-4)) ** 2 + (yr / max(rv, 1e-4)) ** 2))
        a_local = np.clip(oi * g, 0.0, 0.95)

        patch_alpha = alpha_acc[y0:y1+1, x0:x1+1]
        patch_img = img[y0:y1+1, x0:x1+1]

        contrib = (1.0 - patch_alpha) * a_local
        patch_img[:] = patch_img * (1.0 - contrib[..., None]) + ci[None, None, :] * contrib[..., None]
        patch_alpha[:] = np.clip(patch_alpha + contrib, 0.0, 1.0)

    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


# ============================================================
# Video
# ============================================================

def render_video(ply_path, output_mp4, fps, width, height, duration, fov_deg, bg, max_splats):
    print("Lecture du PLY gaussian splat...")
    ply = load_ply_vertex_data(ply_path)
    cloud = extract_gaussian_cloud(ply)

    print("Champs détectés :", ", ".join(ply.property_names))
    print("Nombre de splats :", len(cloud.xyz))

    stats = compute_scene_stats(cloud.xyz)
    n_frames = max(1, int(round(fps * duration)))

    print(f"Centre : {stats.center}")
    print(f"Rayon  : {stats.radius:.4f}")
    print(f"Frames : {n_frames}")

    poses = smart_orbit_trajectory(stats, n_frames)

    os.makedirs(os.path.dirname(os.path.abspath(output_mp4)) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        output_mp4,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir {output_mp4}")

    bg_white = (bg == "white")

    try:
        for pose in tqdm(poses):
            rgb = render_frame_gaussians(
                cloud=cloud,
                c2w=pose,
                width=width,
                height=height,
                fov_deg=fov_deg,
                bg_white=bg_white,
                max_splats=max_splats,
            )
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()

    print(f"Vidéo écrite dans: {output_mp4}")


def main():
    parser = argparse.ArgumentParser(description="Rendu CPU d'un vrai PLY Gaussian Splat vers MP4.")
    parser.add_argument("ply", type=str)
    parser.add_argument("-o", "--output", type=str, default="output.mp4")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fov", type=float, default=50.0)
    parser.add_argument("--bg", choices=["white", "black"], default="white")
    parser.add_argument("--max-splats", type=int, default=25000)
    args = parser.parse_args()

    render_video(
        ply_path=args.ply,
        output_mp4=args.output,
        fps=args.fps,
        width=args.width,
        height=args.height,
        duration=args.duration,
        fov_deg=args.fov,
        bg=args.bg,
        max_splats=args.max_splats,
    )


if __name__ == "__main__":
    main()
