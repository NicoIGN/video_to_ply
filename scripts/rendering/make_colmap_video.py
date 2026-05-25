#!/usr/bin/env python3

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from plyfile import PlyData
import pycolmap
import yaml
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R, Slerp

try:
    import torch
except Exception:
    torch = None


# ============================================================
# Utils
# ============================================================

def normalize_quaternions(q):
    q = np.asarray(q, dtype=float)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.clip(norms, 1e-12, None)


def arc_length_param(points):
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.array([], dtype=float)
    if len(points) == 1:
        return np.array([0.0], dtype=float)

    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]

    if total < 1e-12:
        return np.linspace(0.0, 1.0, len(points))
    return s / total


def resample_by_arclength(points, n, smoothness=0.0):
    points = np.asarray(points, dtype=float)

    if len(points) == 0:
        raise ValueError("Empty point array")
    if len(points) == 1:
        return np.repeat(points, n, axis=0)
    if len(points) == 2:
        t = np.linspace(0.0, 1.0, n)[:, None]
        return (1.0 - t) * points[0] + t * points[1]

    s = arc_length_param(points)
    t = np.linspace(0.0, 1.0, n)
    k = min(3, len(points) - 1)

    tck, _ = splprep(
        [points[:, 0], points[:, 1], points[:, 2]],
        u=s,
        s=max(0.0, float(smoothness)) * len(points),
        k=k
    )
    x, y, z = splev(t, tck)
    return np.stack([x, y, z], axis=1)


def make_loop(points, quats):
    return np.vstack([points, points[:1]]), np.vstack([quats, quats[:1]])


def make_even(x, multiple=16):
    return int(np.ceil(int(x) / multiple) * multiple)


def pad_frame_to_size(frame, out_w, out_h):
    h, w = frame.shape[:2]
    if h == out_h and w == out_w:
        return frame
    padded = np.zeros((out_h, out_w, 3), dtype=frame.dtype)
    padded[:h, :w] = frame
    return padded


def resolve_device(device_arg):
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if torch is None:
            raise RuntimeError("Requested --device cuda but PyTorch is not installed.")
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return "cuda"
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ============================================================
# Path helpers
# ============================================================

def candidate_search_roots(colmap_path, output_path, ply_path=None, checkpoint=None, load_config=None):
    roots = []

    def add_with_parents(p):
        if p is None:
            return
        p = Path(p).expanduser()
        if p.is_file():
            p = p.parent
        elif not p.exists():
            p = p.parent
        roots.append(p)
        roots.extend(list(p.parents))

    add_with_parents(colmap_path)
    add_with_parents(output_path)
    add_with_parents(ply_path)
    add_with_parents(checkpoint)
    add_with_parents(load_config)
    add_with_parents(Path.cwd())

    seen = set()
    unique = []
    for r in roots:
        try:
            key = str(r.resolve(strict=False))
        except Exception:
            key = str(r)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def resolve_moved_path(path_like, colmap_path, output_path, ply_path=None, checkpoint=None, load_config=None):
    if path_like is None:
        return None

    p = Path(path_like).expanduser()

    if p.exists():
        return p.resolve()

    cwd_try = Path.cwd() / p
    if cwd_try.exists():
        return cwd_try.resolve()

    roots = candidate_search_roots(colmap_path, output_path, ply_path, checkpoint, load_config)
    parts = p.parts

    for root in roots:
        for n in range(min(len(parts), 8), 0, -1):
            cand = root / Path(*parts[-n:])
            if cand.exists():
                return cand.resolve()

    if p.name:
        for root in roots:
            matches = list(root.rglob(p.name))
            if len(matches) == 1:
                return matches[0].resolve()
            if len(matches) > 1:
                for m in matches:
                    if str(m).endswith("config.yml") or str(m).endswith(".ckpt"):
                        return m.resolve()
                return matches[0].resolve()

    return p


def auto_find_config(load_config, colmap_path, checkpoint, output_path, ply_path=None):
    if load_config is not None:
        p = resolve_moved_path(load_config, colmap_path, output_path, ply_path, checkpoint, load_config)
        if p.exists():
            return p

    if checkpoint is not None:
        ckpt = resolve_moved_path(checkpoint, colmap_path, output_path, ply_path, checkpoint, load_config)
        ckpt_cfg = ckpt.parent.parent / "config.yml"
        if ckpt_cfg.exists():
            return ckpt_cfg.resolve()

    roots = candidate_search_roots(colmap_path, output_path, ply_path, checkpoint, load_config)
    for root in roots:
        matches = list(root.rglob("config.yml"))
        if matches:
            for m in matches:
                if (m.parent / "nerfstudio_models").exists():
                    return m.resolve()
            return matches[0].resolve()

    return None


def auto_find_checkpoint(config_path, explicit_checkpoint=None):
    if explicit_checkpoint is not None:
        ckpt = Path(explicit_checkpoint).expanduser()
        if ckpt.exists():
            return ckpt.resolve()

    cfg_dir = Path(config_path).resolve().parent
    candidates = sorted(cfg_dir.rglob("*.ckpt"))
    if candidates:
        return candidates[-1].resolve()

    return None


def auto_find_sparse_pc(dataset_root):
    dataset_root = Path(dataset_root).resolve()
    candidates = list(dataset_root.rglob("sparse_pc.ply"))
    if candidates:
        return candidates[0].resolve()
    return None


# ============================================================
# COLMAP pose extraction
# ============================================================

def extract_rigid3d_pose(T):
    q_xyzw = None
    t = None

    for q_attr in ["rotation_xyzw", "quat", "qvec"]:
        if hasattr(T, q_attr):
            obj = getattr(T, q_attr)
            try:
                val = obj() if callable(obj) else obj
                arr = np.asarray(val, dtype=float).reshape(-1)
                if arr.size == 4:
                    q_xyzw = arr
                    break
            except Exception:
                pass

    for t_attr in ["translation", "tvec", "translation_vector", "t"]:
        if hasattr(T, t_attr):
            obj = getattr(T, t_attr)
            try:
                val = obj() if callable(obj) else obj
                arr = np.asarray(val, dtype=float).reshape(-1)
                if arr.size == 3:
                    t = arr
                    break
            except Exception:
                pass

    if (q_xyzw is None or t is None) and hasattr(T, "matrix"):
        try:
            M = T.matrix() if callable(T.matrix) else T.matrix
            M = np.asarray(M, dtype=float)
            if M.shape == (3, 4):
                R_cw = M[:, :3]
                t = M[:, 3]
                q_xyzw = R.from_matrix(R_cw).as_quat()
            elif M.shape == (4, 4):
                R_cw = M[:3, :3]
                t = M[:3, 3]
                q_xyzw = R.from_matrix(R_cw).as_quat()
        except Exception:
            pass

    if q_xyzw is None or t is None:
        text = repr(T)
        q_match = re.search(r"rotation_xyzw=\[([^\]]+)\]", text)
        t_match = re.search(r"translation=\[([^\]]+)\]", text)
        if q_match:
            q_xyzw = np.asarray([float(x.strip()) for x in q_match.group(1).split(",")], dtype=float)
        if t_match:
            t = np.asarray([float(x.strip()) for x in t_match.group(1).split(",")], dtype=float)

    if q_xyzw is None or t is None:
        raise RuntimeError(
            "Could not extract pose from pycolmap Rigid3d.\n"
            f"type(T)={type(T)}\n"
            f"repr(T)={repr(T)}"
        )

    return np.asarray(q_xyzw, dtype=float).reshape(4,), np.asarray(t, dtype=float).reshape(3,)


def extract_intrinsics(camera, default_w, default_h):
    if hasattr(camera, "focal_length_x"):
        fx = float(camera.focal_length_x)
        fy = float(camera.focal_length_y)
        cx = float(camera.principal_point_x)
        cy = float(camera.principal_point_y)
        w = int(getattr(camera, "width", default_w))
        h = int(getattr(camera, "height", default_h))
        return fx, fy, cx, cy, w, h

    if hasattr(camera, "params"):
        p = np.asarray(camera.params, dtype=float)
        w = int(getattr(camera, "width", default_w))
        h = int(getattr(camera, "height", default_h))
        if len(p) >= 4:
            return float(p[0]), float(p[1]), float(p[2]), float(p[3]), w, h

    return float(default_w), float(default_w), default_w / 2.0, default_h / 2.0, int(default_w), int(default_h)


def load_colmap_poses_and_camera(colmap_path, out_width, out_height):
    recon = pycolmap.Reconstruction(str(colmap_path))
    images = sorted(recon.images.values(), key=lambda im: im.image_id)

    if len(images) == 0:
        raise RuntimeError(f"No images found in COLMAP reconstruction: {colmap_path}")

    cam_centers = []
    quats_xyzw = []

    ref_camera = images[0].camera
    fx, fy, cx, cy, cam_w, cam_h = extract_intrinsics(ref_camera, out_width, out_height)

    for img in images:
        T = img.cam_from_world()
        q_xyzw, t = extract_rigid3d_pose(T)
        q_xyzw = q_xyzw / max(np.linalg.norm(q_xyzw), 1e-12)

        R_cw = R.from_quat(q_xyzw).as_matrix()
        c_from_pose = -R_cw.T @ t

        if hasattr(img, "projection_center"):
            try:
                c_world = np.asarray(img.projection_center(), dtype=float).reshape(3,)
            except Exception:
                c_world = c_from_pose
        else:
            c_world = c_from_pose

        cam_centers.append(c_world)
        quats_xyzw.append(q_xyzw)

    return (
        np.asarray(cam_centers),
        np.asarray(quats_xyzw),
        {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "cam_w": cam_w,
            "cam_h": cam_h,
        }
    )


def interpolate_rotations(rotations, n):
    rotations = normalize_quaternions(rotations)
    if len(rotations) == 0:
        raise ValueError("Empty rotation array")
    if len(rotations) == 1:
        return np.repeat(rotations, n, axis=0)

    key_times = np.linspace(0.0, 1.0, len(rotations))
    target_times = np.linspace(0.0, 1.0, n)
    slerp = Slerp(key_times, R.from_quat(rotations))
    return slerp(target_times).as_quat()


# ============================================================
# Nerfstudio helpers
# ============================================================

def infer_local_dataset_root(config_path, colmap_path):
    config_dir = Path(config_path).resolve().parent
    colmap_path = Path(colmap_path).resolve()

    candidates = [
        config_dir,
        colmap_path.parent,
        colmap_path.parent.parent,
        colmap_path.parent.parent.parent,
        colmap_path.parent.parent.parent.parent,
    ]

    for c in candidates:
        if c.exists():
            return c.resolve()

    return config_dir.resolve()


def validate_transforms_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "frames" not in data:
        raise RuntimeError(f"{path} is missing 'frames'")
    return data


def find_best_transforms_json(dataset_root):
    dataset_root = Path(dataset_root).resolve()
    candidates = [
        dataset_root / "transforms.json",
        dataset_root / "colmap_table_clem_quality" / "transforms.json",
        dataset_root / "dataparser_transforms.json",
    ]

    for cand in candidates:
        if cand.exists():
            try:
                validate_transforms_json(cand)
                return cand.resolve()
            except Exception:
                continue

    existing = [c for c in candidates if c.exists()]
    if existing:
        raise RuntimeError(
            "Found transform files, but none are valid Nerfstudio transforms.json with a 'frames' key:\n"
            + "\n".join(str(x) for x in existing)
        )

    raise RuntimeError(f"Could not find a valid transforms.json under {dataset_root}")


def prepare_nerfstudio_data_dir(dataset_root, tmpdir):
    dataset_root = Path(dataset_root).resolve()
    work_dir = Path(tmpdir) / "ns_data"
    work_dir.mkdir(parents=True, exist_ok=True)

    transforms_src = find_best_transforms_json(dataset_root)
    shutil.copy2(transforms_src, work_dir / "transforms.json")

    sparse_pc = auto_find_sparse_pc(dataset_root)
    if sparse_pc is not None:
        try:
            (work_dir / "sparse_pc.ply").symlink_to(sparse_pc)
        except Exception:
            shutil.copy2(sparse_pc, work_dir / "sparse_pc.ply")

    for name in ["images", "imgs", "ori", "rgb", "masks", "depth"]:
        src = dataset_root / name
        if src.exists():
            dst = work_dir / name
            try:
                dst.symlink_to(src, target_is_directory=src.is_dir())
            except Exception:
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)

    for name in ["colmap", "sparse", "colmap_table_clem_quality"]:
        src = dataset_root / name
        if src.exists():
            dst = work_dir / name
            try:
                dst.symlink_to(src, target_is_directory=src.is_dir())
            except Exception:
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)

    return work_dir, transforms_src, sparse_pc


def patch_nerfstudio_config_struct(load_config, prepared_data_dir, temp_output_root):
    load_config = Path(load_config).resolve()
    config = yaml.load(load_config.read_text(encoding="utf-8"), Loader=yaml.Loader)

    prepared_data_dir = Path(prepared_data_dir).resolve()
    temp_output_root = Path(temp_output_root).resolve()

    if hasattr(config, "output_dir"):
        config.output_dir = temp_output_root

    if hasattr(config, "pipeline") and hasattr(config.pipeline, "datamanager"):
        dm = config.pipeline.datamanager
        if hasattr(dm, "data"):
            dm.data = prepared_data_dir
        if hasattr(dm, "dataparser") and hasattr(dm.dataparser, "data"):
            dm.dataparser.data = prepared_data_dir

    return config


def materialize_expected_checkpoint_tree(config, checkpoint_path, tmpdir):
    checkpoint_path = Path(checkpoint_path).resolve()
    tmpdir = Path(tmpdir).resolve()

    experiment_name = getattr(config, "experiment_name", "model3d")
    method_name = getattr(config, "method_name", "splatfacto")
    timestamp = getattr(config, "timestamp", "default")
    relative_model_dir = getattr(config, "relative_model_dir", Path("nerfstudio_models"))

    expected_dir = tmpdir / experiment_name / method_name / timestamp / Path(relative_model_dir)
    expected_dir.mkdir(parents=True, exist_ok=True)

    dst_ckpt = expected_dir / checkpoint_path.name
    try:
        dst_ckpt.symlink_to(checkpoint_path)
    except Exception:
        shutil.copy2(checkpoint_path, dst_ckpt)

    return expected_dir, dst_ckpt


def scale_intrinsics_to_output(intr, out_w, out_h):
    sx = out_w / float(intr["cam_w"])
    sy = out_h / float(intr["cam_h"])
    return {
        "fx": intr["fx"] * sx,
        "fy": intr["fy"] * sy,
        "cx": intr["cx"] * sx,
        "cy": intr["cy"] * sy,
    }


def c2w_from_center_quat(center, quat_xyzw):
    R_cw = R.from_quat(quat_xyzw).as_matrix()
    R_wc = R_cw.T

    c2w = np.eye(4, dtype=float)
    c2w[:3, :3] = R_wc
    c2w[:3, 3] = center
    return c2w


def build_nerfstudio_camera_path(poses, out_w, out_h, intrinsics, fps):
    intr = scale_intrinsics_to_output(intrinsics, out_w, out_h)
    camera_path = []

    for pose in poses:
        c2w = c2w_from_center_quat(pose["pos"], pose["rot"])
        camera_path.append({
            "camera_to_world": c2w.tolist(),
            "fov": None,
            "aspect": float(out_w) / float(out_h),
        })

    return {
        "camera_type": "perspective",
        "render_height": int(out_h),
        "render_width": int(out_w),
        "fps": float(fps),
        "seconds": float(len(poses) / fps) if fps > 0 else 0.0,
        "smoothness_value": 0.0,
        "is_cycle": False,
        "camera_path": camera_path,
        "keyframes": [],
        "camera_intrinsics": {
            "fx": float(intr["fx"]),
            "fy": float(intr["fy"]),
            "cx": float(intr["cx"]),
            "cy": float(intr["cy"]),
        },
    }


def find_ns_render():
    exe = shutil.which("ns-render")
    if exe is None:
        raise RuntimeError("Could not find 'ns-render' in PATH.")
    return exe


def render_with_nerfstudio(load_config, camera_path_json, output_path, colmap_path, checkpoint_path):
    exe = find_ns_render()
    dataset_root = infer_local_dataset_root(load_config, colmap_path)

    with tempfile.TemporaryDirectory(prefix="ns_cfg_") as tmpdir:
        tmpdir = Path(tmpdir).resolve()

        prepared_data_dir, transforms_src, sparse_pc = prepare_nerfstudio_data_dir(dataset_root, tmpdir)
        original_config = yaml.load(Path(load_config).read_text(encoding="utf-8"), Loader=yaml.Loader)

        expected_ckpt_dir = None
        expected_ckpt_file = None
        if checkpoint_path is not None:
            expected_ckpt_dir, expected_ckpt_file = materialize_expected_checkpoint_tree(
                config=original_config,
                checkpoint_path=checkpoint_path,
                tmpdir=tmpdir,
            )

        patched_config = patch_nerfstudio_config_struct(
            load_config=load_config,
            prepared_data_dir=prepared_data_dir,
            temp_output_root=tmpdir,
        )

        patched_cfg = tmpdir / "config.patched.yml"
        patched_cfg.write_text(yaml.dump(patched_config), encoding="utf-8")

        print(f"[INFO] Patched Nerfstudio config: {patched_cfg}")
        print(f"[INFO] Using prepared data dir: {prepared_data_dir}")
        print(f"[INFO] Using transforms source: {transforms_src}")
        print(f"[INFO] Using sparse_pc: {sparse_pc}")
        print(f"[INFO] Using checkpoint: {checkpoint_path}")
        print(f"[INFO] Materialized checkpoint dir: {expected_ckpt_dir}")
        print(f"[INFO] Materialized checkpoint file: {expected_ckpt_file}")
        print(f"[INFO] Temporary output root: {tmpdir}")

        cmd = [
            exe, "camera-path",
            "--load-config", str(patched_cfg),
            "--camera-path-filename", str(camera_path_json),
            "--output-path", str(output_path),
        ]
        print("[INFO] Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


# ============================================================
# PLY fallback renderer
# ============================================================

def load_gaussian_ply(ply_path):
    ply = PlyData.read(ply_path)
    vertex = ply["vertex"]
    names = vertex.data.dtype.names

    def pick(*candidates, required=True, default=None):
        for c in candidates:
            if c in names:
                return np.asarray(vertex[c])
        if required:
            raise RuntimeError(f"Missing PLY field. Tried: {candidates}. Available: {names}")
        return default

    xyz = np.stack([pick("x"), pick("y"), pick("z")], axis=1).astype(np.float32)

    if all(k in names for k in ["red", "green", "blue"]):
        rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float32) / 255.0
    elif all(k in names for k in ["f_dc_0", "f_dc_1", "f_dc_2"]):
        rgb = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)
        rgb = np.clip(0.5 + rgb, 0.0, 1.0)
    else:
        rgb = np.ones((xyz.shape[0], 3), dtype=np.float32) * 0.8

    if "opacity" in names:
        opacity = np.asarray(vertex["opacity"]).astype(np.float32)
        opacity = 1.0 / (1.0 + np.exp(-opacity))
    else:
        opacity = np.ones((xyz.shape[0],), dtype=np.float32) * 0.7

    if all(k in names for k in ["scale_0", "scale_1", "scale_2"]):
        scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1).astype(np.float32)
        scales = np.exp(scales)
        radius = np.mean(scales, axis=1)
    else:
        radius = np.ones((xyz.shape[0],), dtype=np.float32) * 0.01

    return {"xyz": xyz, "rgb": rgb, "opacity": opacity, "radius": radius}


def render_frame_cpu(gs, cam_pos, cam_quat_xyzw, fx, fy, cx, cy, width, height,
                     znear=0.01, zfar=1e6, max_points=120000, radius_scale=120.0):
    xyz = gs["xyz"]
    rgb = gs["rgb"]
    opacity = gs["opacity"]
    radius = gs["radius"]

    if xyz.shape[0] > max_points:
        idx = np.linspace(0, xyz.shape[0] - 1, max_points).astype(int)
        xyz = xyz[idx]
        rgb = rgb[idx]
        opacity = opacity[idx]
        radius = radius[idx]

    R_cw = R.from_quat(cam_quat_xyzw).as_matrix()
    t_cw = -R_cw @ cam_pos
    pts_cam = (R_cw @ xyz.T).T + t_cw

    z = pts_cam[:, 2]
    valid = (z > znear) & (z < zfar)
    if not np.any(valid):
        return np.zeros((height, width, 3), dtype=np.uint8)

    pts_cam = pts_cam[valid]
    z = z[valid]
    col = rgb[valid]
    alp = opacity[valid]
    rad = radius[valid]

    u = fx * (pts_cam[:, 0] / z) + cx
    v = fy * (pts_cam[:, 1] / z) + cy

    inside = (u >= -100) & (u < width + 100) & (v >= -100) & (v < height + 100)
    if not np.any(inside):
        return np.zeros((height, width, 3), dtype=np.uint8)

    u = u[inside]
    v = v[inside]
    z = z[inside]
    col = col[inside]
    alp = alp[inside]
    rad = rad[inside]

    order = np.argsort(z)[::-1]
    u, v, col, alp, rad, z = u[order], v[order], col[order], alp[order], rad[order], z[order]

    img = np.zeros((height, width, 3), dtype=np.float32)
    trans = np.ones((height, width), dtype=np.float32)
    screen_r = np.clip(radius_scale * rad / np.maximum(z, 1e-6), 1.0, 25.0)

    for i in range(len(u)):
        cx_i = int(round(u[i]))
        cy_i = int(round(v[i]))
        r = int(np.ceil(screen_r[i]))

        x0, x1 = max(0, cx_i - r), min(width, cx_i + r + 1)
        y0, y1 = max(0, cy_i - r), min(height, cy_i + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        ys, xs = np.mgrid[y0:y1, x0:x1]
        dx = (xs - u[i]) / max(screen_r[i], 1e-6)
        dy = (ys - v[i]) / max(screen_r[i], 1e-6)
        w = np.exp(-0.5 * (dx * dx + dy * dy))
        a = np.clip(alp[i] * w, 0.0, 0.99)

        img[y0:y1, x0:x1] += trans[y0:y1, x0:x1, None] * a[..., None] * col[i][None, None, :]
        trans[y0:y1, x0:x1] *= (1.0 - a)

    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def render_frame_torch(gs_t, cam_pos, cam_quat_xyzw, fx, fy, cx, cy, width, height,
                       znear=0.01, zfar=1e6, max_points=120000, radius_scale=120.0):
    device = gs_t["xyz"].device
    xyz = gs_t["xyz"]
    rgb = gs_t["rgb"]
    opacity = gs_t["opacity"]
    radius = gs_t["radius"]

    if xyz.shape[0] > max_points:
        idx = torch.linspace(0, xyz.shape[0] - 1, max_points, device=device).long()
        xyz = xyz[idx]
        rgb = rgb[idx]
        opacity = opacity[idx]
        radius = radius[idx]

    R_cw = torch.tensor(R.from_quat(cam_quat_xyzw).as_matrix(), dtype=torch.float32, device=device)
    cam_pos_t = torch.tensor(cam_pos, dtype=torch.float32, device=device)
    t_cw = -R_cw @ cam_pos_t
    pts_cam = (R_cw @ xyz.T).T + t_cw

    z = pts_cam[:, 2]
    valid = (z > znear) & (z < zfar)
    if valid.sum().item() == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    pts_cam = pts_cam[valid]
    z = z[valid]
    col = rgb[valid]
    alp = opacity[valid]
    rad = radius[valid]

    u = fx * (pts_cam[:, 0] / z) + cx
    v = fy * (pts_cam[:, 1] / z) + cy

    inside = (u >= -100) & (u < width + 100) & (v >= -100) & (v < height + 100)
    if inside.sum().item() == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    u = u[inside]
    v = v[inside]
    z = z[inside]
    col = col[inside]
    alp = alp[inside]
    rad = rad[inside]

    order = torch.argsort(z, descending=True)
    u, v, col, alp, rad, z = u[order], v[order], col[order], alp[order], rad[order], z[order]

    img = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
    trans = torch.ones((height, width), dtype=torch.float32, device=device)
    screen_r = torch.clamp(radius_scale * rad / torch.clamp(z, min=1e-6), 1.0, 25.0)

    for i in range(u.shape[0]):
        cx_i = int(torch.round(u[i]).item())
        cy_i = int(torch.round(v[i]).item())
        r = int(torch.ceil(screen_r[i]).item())

        x0, x1 = max(0, cx_i - r), min(width, cx_i + r + 1)
        y0, y1 = max(0, cy_i - r), min(height, cy_i + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        ys, xs = torch.meshgrid(
            torch.arange(y0, y1, device=device),
            torch.arange(x0, x1, device=device),
            indexing="ij"
        )
        dx = (xs - u[i]) / torch.clamp(screen_r[i], min=1e-6)
        dy = (ys - v[i]) / torch.clamp(screen_r[i], min=1e-6)
        w = torch.exp(-0.5 * (dx * dx + dy * dy))
        a = torch.clamp(alp[i] * w, 0.0, 0.99)

        img[y0:y1, x0:x1] += trans[y0:y1, x0:x1, None] * a[..., None] * col[i][None, None, :]
        trans[y0:y1, x0:x1] *= (1.0 - a)

    return (torch.clamp(img, 0.0, 1.0) * 255.0).byte().cpu().numpy()


def render_ply_fallback(ply_path, poses, width, height, intrinsics, device="cpu"):
    gs = load_gaussian_ply(ply_path)
    intr = scale_intrinsics_to_output(intrinsics, width, height)
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]

    print(f"[INFO] Using fallback PLY renderer on {device}")
    print(f"[INFO] Rendering {len(poses)} frames")

    gs_t = None
    if device == "cuda":
        gs_t = {
            "xyz": torch.tensor(gs["xyz"], dtype=torch.float32, device="cuda"),
            "rgb": torch.tensor(gs["rgb"], dtype=torch.float32, device="cuda"),
            "opacity": torch.tensor(gs["opacity"], dtype=torch.float32, device="cuda"),
            "radius": torch.tensor(gs["radius"], dtype=torch.float32, device="cuda"),
        }

    frames = []
    for i, pose in enumerate(poses):
        if i % 10 == 0 or i == len(poses) - 1:
            print(f"[INFO] Frame {i+1}/{len(poses)}")

        if device == "cuda":
            frame = render_frame_torch(gs_t, pose["pos"], pose["rot"], fx, fy, cx, cy, width, height)
        else:
            frame = render_frame_cpu(gs, pose["pos"], pose["rot"], fx, fy, cx, cy, width, height)
        frames.append(frame)

    return frames


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Render a video from COLMAP camera path using Nerfstudio or PLY fallback."
    )
    parser.add_argument("--colmap", required=True, help="Path to COLMAP sparse model folder")
    parser.add_argument("--output", required=True, help="Output video path (.mp4)")
    parser.add_argument("--load-config", default=None, help="Path to Nerfstudio config.yml")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path")
    parser.add_argument("--ply", default=None, help="Optional PLY fallback renderer input")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--smoothness", type=float, default=0.2)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"[INFO] Selected device: {device}")

    output_path = Path(args.output).expanduser()
    colmap_path = resolve_moved_path(args.colmap, args.colmap, output_path, args.ply, args.checkpoint, args.load_config)
    ply_path = resolve_moved_path(args.ply, colmap_path, output_path, args.ply, args.checkpoint, args.load_config) if args.ply else None
    checkpoint_path = resolve_moved_path(args.checkpoint, colmap_path, output_path, args.ply, args.checkpoint, args.load_config) if args.checkpoint else None

    config_path = auto_find_config(
        args.load_config,
        colmap_path=colmap_path,
        checkpoint=checkpoint_path,
        output_path=output_path,
        ply_path=ply_path,
    )

    use_nerfstudio = config_path is not None

    if use_nerfstudio:
        print(f"[INFO] Found Nerfstudio config: {config_path}")
        checkpoint_path = auto_find_checkpoint(config_path, checkpoint_path)
        print(f"[INFO] Found checkpoint: {checkpoint_path}")

        # Splatfacto in this environment is CUDA-only.
        if device != "cuda":
            if ply_path is not None:
                print("[WARN] CUDA unavailable; forcing PLY fallback instead of Nerfstudio.")
                use_nerfstudio = False
            else:
                raise RuntimeError(
                    "Nerfstudio Splatfacto requires CUDA in this environment, "
                    "but PyTorch CUDA is not available. "
                    "Provide --ply for CPU fallback or run on a CUDA machine."
                )
    elif ply_path is not None:
        print("[WARN] No Nerfstudio config found, falling back to PLY renderer.")
    else:
        raise RuntimeError("No Nerfstudio config found and no --ply provided.")

    cam_centers, quats, intrinsics = load_colmap_poses_and_camera(
        colmap_path, args.width, args.height
    )

    if len(cam_centers) < 2:
        raise RuntimeError("Not enough COLMAP poses")

    if args.loop:
        cam_centers, quats = make_loop(cam_centers, quats)

    n_frames = max(1, int(args.fps * args.duration))
    traj = resample_by_arclength(cam_centers, n_frames, smoothness=args.smoothness)
    rot = interpolate_rotations(quats, n_frames)
    poses = [{"pos": traj[i], "rot": rot[i]} for i in range(n_frames)]

    video_w = make_even(args.width, 16)
    video_h = make_even(args.height, 16)
    if (video_w, video_h) != (args.width, args.height):
        print(f"[INFO] Adjusting video size from {args.width}x{args.height} to {video_w}x{video_h}")

    if use_nerfstudio:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = build_nerfstudio_camera_path(
            poses=poses,
            out_w=video_w,
            out_h=video_h,
            intrinsics=intrinsics,
            fps=args.fps,
        )

        with tempfile.TemporaryDirectory(prefix="ns_camera_path_") as tmpdir:
            camera_json = Path(tmpdir) / "camera_path.json"
            with open(camera_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            render_with_nerfstudio(
                load_config=config_path,
                camera_path_json=camera_json,
                output_path=output_path,
                colmap_path=colmap_path,
                checkpoint_path=checkpoint_path,
            )

        print(f"[OK] Saved: {output_path}")
        return

    frames = render_ply_fallback(
        ply_path=ply_path,
        poses=poses,
        width=args.width,
        height=args.height,
        intrinsics=intrinsics,
        device=device,
    )
    frames = [pad_frame_to_size(f, video_w, video_h) for f in frames]
    imageio.mimsave(str(output_path), frames, fps=args.fps, macro_block_size=16)
    print(f"[OK] Saved: {output_path}")


if __name__ == "__main__":
    main()
