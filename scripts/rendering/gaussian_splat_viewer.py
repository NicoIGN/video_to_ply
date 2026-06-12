#!/usr/bin/env python3
import argparse
import math
import os
from dataclasses import dataclass

import cv2
import numpy as np
import pyglet
from pyglet import gl
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
# PLY loader
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
                vals = fp.readline().decode("utf-8").strip().split()
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
# Gaussian extraction
# ============================================================

@dataclass
class GaussianCloud:
    xyz: np.ndarray
    opacity: np.ndarray
    scale: np.ndarray
    quat: np.ndarray
    color: np.ndarray


def extract_gaussian_cloud(ply: PlyData) -> GaussianCloud:
    v = ply.vertex
    names = set(ply.property_names)

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    if "opacity" in names:
        opacity = sigmoid(v["opacity"].astype(np.float32))
    else:
        opacity = np.ones((len(v),), dtype=np.float32) * 0.8

    if all(k in names for k in ["scale_0", "scale_1", "scale_2"]):
        scale = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32))
    else:
        scale = np.ones((len(v), 3), dtype=np.float32) * 0.01

    if all(k in names for k in ["rot_0", "rot_1", "rot_2", "rot_3"]):
        quat = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)
    else:
        quat = np.zeros((len(v), 4), dtype=np.float32)
        quat[:, 0] = 1.0

    if all(k in names for k in ["f_dc_0", "f_dc_1", "f_dc_2"]):
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

    return GaussianCloud(xyz=xyz, opacity=opacity, scale=scale, quat=quat, color=color)


# ============================================================
# Scene stats / camera
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


class OrbitCamera:
    def __init__(self, center, radius, up):
        self.center = center.astype(np.float32)
        self.scene_radius = float(radius)
        self.up = up.astype(np.float32)

        self.distance = radius * 1.8
        self.yaw = 0.0
        self.pitch = math.radians(18.0)
        self.fov_deg = 50.0

    def position(self):
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)

        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(np.dot(tmp, self.up)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        right0 = normalize(np.cross(tmp, self.up))
        fwd0 = normalize(np.cross(self.up, right0))

        dir_h = cy * right0 + sy * fwd0
        pos = self.center + dir_h * (self.distance * cp) + self.up * (self.distance * sp)
        return pos.astype(np.float32)

    def c2w(self):
        return look_at(self.position(), self.center, self.up)


# ============================================================
# Renderer
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


def render_frame_gaussians(cloud, c2w, width, height, fov_deg, bg_white=True, max_splats=20000):
    bg = 1.0 if bg_white else 0.0
    img = np.full((height, width, 3), bg, dtype=np.float32)
    alpha_acc = np.zeros((height, width), dtype=np.float32)

    xyz = cloud.xyz
    opacity = cloud.opacity
    scale = cloud.scale
    quat = cloud.quat
    color = cloud.color

    if len(xyz) > max_splats:
        idx = np.random.choice(len(xyz), max_splats, replace=False)
        xyz = xyz[idx]
        opacity = opacity[idx]
        scale = scale[idx]
        quat = quat[idx]
        color = color[idx]

    R_wc = c2w[:3, :3]
    t_wc = c2w[:3, 3]

    # world -> camera
    xyz_cam = (xyz - t_wc) @ R_wc
    X = xyz_cam[:, 0]
    Y = xyz_cam[:, 1]
    Z = xyz_cam[:, 2]

    valid = Z > 1e-4
    xyz_cam = xyz_cam[valid]
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]
    opacity = opacity[valid]
    scale = scale[valid]
    quat = quat[valid]
    color = color[valid]

    fx = 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)
    fy = 0.5 * height / math.tan(math.radians(fov_deg) * 0.5)
    cx = width / 2.0
    cy = height / 2.0

    u = fx * (X / Z) + cx
    v = fy * (-Y / Z) + cy

    in_img = (u >= -128) & (u < width + 128) & (v >= -128) & (v < height + 128)
    u = u[in_img]
    v = v[in_img]
    X = X[in_img]
    Y = Y[in_img]
    Z = Z[in_img]
    opacity = opacity[in_img]
    scale = scale[in_img]
    quat = quat[in_img]
    color = color[in_img]

    if len(u) == 0:
        return (img * 255).astype(np.uint8)

    order = np.argsort(Z)[::-1]  # far -> near
    u = u[order]
    v = v[order]
    X = X[order]
    Y = Y[order]
    Z = Z[order]
    opacity = opacity[order]
    scale = scale[order]
    quat = quat[order]
    color = color[order]

    R_cam = R_wc.T

    for ui, vi, Xi, Yi, Zi, oi, si, qi, ci in zip(u, v, X, Y, Z, opacity, scale, quat, color):
        Rg = quat_to_rotmat(qi)

        # covariance 3D monde
        S = np.diag(np.square(si.astype(np.float32)))
        Sigma3 = Rg @ S @ Rg.T

        # covariance 3D caméra
        SigmaC = R_cam @ Sigma3 @ R_cam.T

        # jacobien projection perspective
        J = np.array([
            [fx / Zi, 0.0, -fx * Xi / (Zi * Zi)],
            [0.0, -fy / Zi, fy * Yi / (Zi * Zi)],
        ], dtype=np.float32)

        Sigma2 = J @ SigmaC @ J.T

        # stabilisation numérique
        Sigma2 += np.eye(2, dtype=np.float32) * 1e-6

        # eigendecomposition pour axes ellipse
        evals, evecs = np.linalg.eigh(Sigma2)
        evals = np.clip(evals, 1e-8, None)

        # rayon à ~3 sigma
        r1 = 3.0 * math.sqrt(float(evals[1]))
        r0 = 3.0 * math.sqrt(float(evals[0]))

        # vecteur principal
        major = evecs[:, 1]
        angle = math.atan2(major[1], major[0])

        rad = int(math.ceil(max(r0, r1))) + 2
        cxi = int(round(ui))
        cyi = int(round(vi))

        x0 = max(0, cxi - rad)
        x1 = min(width - 1, cxi + rad)
        y0 = max(0, cyi - rad)
        y1 = min(height - 1, cyi + rad)
        if x1 < x0 or y1 < y0:
            continue

        xs = np.arange(x0, x1 + 1, dtype=np.float32)
        ys = np.arange(y0, y1 + 1, dtype=np.float32)
        XX, YY = np.meshgrid(xs, ys)

        d = np.stack([XX - ui, YY - vi], axis=-1)  # [H,W,2]

        try:
            Sigma2_inv = np.linalg.inv(Sigma2)
        except np.linalg.LinAlgError:
            continue

        m = np.einsum("...i,ij,...j->...", d, Sigma2_inv, d)
        g = np.exp(-0.5 * m)

        a_local = np.clip(oi * g, 0.0, 0.99)

        patch_alpha = alpha_acc[y0:y1+1, x0:x1+1]
        patch_img = img[y0:y1+1, x0:x1+1]

        contrib = (1.0 - patch_alpha) * a_local
        patch_img[:] = patch_img * (1.0 - contrib[..., None]) + ci[None, None, :] * contrib[..., None]
        patch_alpha[:] = np.clip(patch_alpha + contrib, 0.0, 1.0)

    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


# ============================================================
# App
# ============================================================

class GaussianSplatViewer(pyglet.window.Window):
    def __init__(self, cloud, stats, width=1280, height=720, bg_white=True, max_splats=25000):
        config = gl.Config(double_buffer=True, depth_size=24)
        super().__init__(width=width, height=height, caption="Gaussian Splat Viewer", resizable=True, config=config)

        self.cloud = cloud
        self.stats = stats
        self.bg_white = bg_white
        self.max_splats = max_splats

        self.camera = OrbitCamera(stats.center, stats.radius, stats.up)
        self.dragging = False
        self.last_mouse = None

        self.current_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        pyglet.clock.schedule_interval(self.update, 1 / 30.0)

    def update(self, dt):
        pass

    def on_draw(self):
        self.clear()
        rgb = render_frame_gaussians(
            cloud=self.cloud,
            c2w=self.camera.c2w(),
            width=self.width,
            height=self.height,
            fov_deg=self.camera.fov_deg,
            bg_white=self.bg_white,
            max_splats=self.max_splats,
        )
        self.current_rgb = rgb

        img_data = pyglet.image.ImageData(
            self.width,
            self.height,
            "RGB",
            rgb[::-1].tobytes(),
            pitch=self.width * 3,
        )
        img_data.blit(0, 0)

    def on_mouse_press(self, x, y, button, modifiers):
        self.dragging = True
        self.last_mouse = (x, y)

    def on_mouse_release(self, x, y, button, modifiers):
        self.dragging = False
        self.last_mouse = None

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.camera.yaw += dx * 0.008
        self.camera.pitch += dy * 0.005
        self.camera.pitch = np.clip(self.camera.pitch, math.radians(-80), math.radians(80))

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.camera.distance *= (0.92 ** scroll_y)
        self.camera.distance = np.clip(self.camera.distance, self.stats.radius * 0.2, self.stats.radius * 10.0)

    def export_video(self, output_path, fps=30, duration=5.0):
        n_frames = max(1, int(round(fps * duration)))
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (self.width, self.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir {output_path}")

        base_distance = self.stats.radius * 1.8

        try:
            for i in tqdm(range(n_frames), desc="Export vidéo"):
                t = i / max(1, n_frames - 1)
                self.camera.yaw = 2.0 * math.pi * 1.1 * t
                self.camera.pitch = math.radians(18.0 + 6.0 * math.sin(2.0 * math.pi * t))
                self.camera.distance = base_distance * (1.0 + 0.12 * math.sin(2.0 * math.pi * t + 0.4))

                self.dispatch_events()
                self.on_draw()
                self.flip()

                frame = self.current_rgb
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
        finally:
            writer.release()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Viewer Python pour Gaussian Splats PLY avec export vidéo.")
    parser.add_argument("ply", type=str)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--bg", choices=["white", "black"], default="white")
    parser.add_argument("--max-splats", type=int, default=25000)
    parser.add_argument("--export", type=str, default=None, help="Chemin mp4 de sortie")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()

    print("Chargement du PLY...")
    ply = load_ply_vertex_data(args.ply)
    cloud = extract_gaussian_cloud(ply)
    stats = compute_scene_stats(cloud.xyz)

    print("Champs détectés :", ", ".join(ply.property_names))
    print("Nb splats       :", len(cloud.xyz))
    print("Centre          :", stats.center)
    print("Rayon           :", stats.radius)

    viewer = GaussianSplatViewer(
        cloud=cloud,
        stats=stats,
        width=args.width,
        height=args.height,
        bg_white=(args.bg == "white"),
        max_splats=args.max_splats,
    )

    if args.export:
        def do_export(dt):
            print("Export vidéo...")
            viewer.export_video(args.export, fps=args.fps, duration=args.duration)
            print(f"Vidéo écrite dans: {args.export}")
        pyglet.clock.schedule_once(do_export, 0.5)

    pyglet.app.run()


if __name__ == "__main__":
    main()
