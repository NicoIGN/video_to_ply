#!/usr/bin/env python3

import argparse
from pathlib import Path
from collections import deque
import sys
import traceback

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


# ==========================================================
# Helpers
# ==========================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_opacity(raw_opacity):
    return sigmoid(raw_opacity)


def decode_scales(vertices):
    names = vertices.dtype.names

    if all(n in names for n in ("scale_0", "scale_1", "scale_2")):
        scales_log = np.column_stack([
            vertices["scale_0"],
            vertices["scale_1"],
            vertices["scale_2"]
        ])
        return np.exp(scales_log)

    if all(n in names for n in ("sx", "sy", "sz")):
        return np.column_stack([
            np.abs(vertices["sx"]),
            np.abs(vertices["sy"]),
            np.abs(vertices["sz"]),
        ])

    return None


def compute_scale_metric(scales, metric):
    if metric == "max":
        return np.max(scales, axis=1)
    elif metric == "mean":
        return np.mean(scales, axis=1)
    elif metric == "norm":
        return np.linalg.norm(scales, axis=1)
    raise ValueError(metric)


def get_xyz(vertices):
    return np.column_stack([
        vertices["x"],
        vertices["y"],
        vertices["z"]
    ])


def robust_center(xyz):
    return np.median(xyz, axis=0)


def robust_sigma(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad + 1e-12


def print_stage(title):
    print(f"\n[{title}]")


# ==========================================================
# Validation
# ==========================================================

def validate_args(args):
    if args.min_opacity is not None and not (0.0 <= args.min_opacity <= 1.0):
        raise ValueError("--min-opacity must be in [0, 1]")

    if args.max_scale_percentile is not None:
        if not (0.0 < args.max_scale_percentile <= 100.0):
            raise ValueError("--max-scale-percentile must be in (0, 100]")

    if args.center_percentile is not None:
        if not (0.0 < args.center_percentile <= 100.0):
            raise ValueError("--center-percentile must be in (0, 100]")

    if args.spatial_k < 1:
        raise ValueError("--spatial-k must be >= 1")

    if args.min_neighbors < 0:
        raise ValueError("--min-neighbors must be >= 0")

    if args.voxel_grid_size < 8:
        raise ValueError("--voxel-grid-size must be >= 8")

    if args.voxel_min_points < 1:
        raise ValueError("--voxel-min-points must be >= 1")

    if args.voxel_connectivity not in (6, 18, 26):
        raise ValueError("--voxel-connectivity must be one of: 6, 18, 26")

    if args.voxel_center_sigma <= 0:
        raise ValueError("--voxel-center-sigma must be > 0")

    if not (0.0 < args.voxel_max_fraction <= 1.0):
        raise ValueError("--voxel-max-fraction must be in (0, 1]")

    if args.attach_layers < 0:
        raise ValueError("--attach-layers must be >= 0")

    if args.attach_min_points < 0:
        raise ValueError("--attach-min-points must be >= 0")

    if args.center_scale_percentile <= 0 or args.center_scale_percentile > 100:
        raise ValueError("--center-scale-percentile must be in (0, 100]")

    if args.center_opacity_percentile <= 0 or args.center_opacity_percentile > 100:
        raise ValueError("--center-opacity-percentile must be in (0, 100]")

    if args.center_keep_percentile <= 0 or args.center_keep_percentile > 100:
        raise ValueError("--center-keep-percentile must be in (0, 100]")

    if args.voxel_radial_density_bias < 0:
        raise ValueError("--voxel-radial-density-bias must be >= 0")

    if args.voxel_scale_bias < 0:
        raise ValueError("--voxel-scale-bias must be >= 0")

    if args.voxel_scale_percentile <= 0 or args.voxel_scale_percentile > 100:
        raise ValueError("--voxel-scale-percentile must be in (0, 100]")

    if args.attach_radial_density_bias < 0:
        raise ValueError("--attach-radial-density-bias must be >= 0")


# ==========================================================
# Cheap point filters
# ==========================================================

def opacity_filter(vertices, min_opacity):
    opacity = decode_opacity(vertices["opacity"])
    return opacity >= min_opacity


def global_scale_filter(vertices, percentile, metric):
    scales = decode_scales(vertices)
    if scales is None:
        return None, None

    scale_metric = compute_scale_metric(scales, metric)
    threshold = np.percentile(scale_metric, percentile)
    keep = scale_metric <= threshold
    return keep, threshold


def center_filter(vertices, percentile, scene_center=None):
    xyz = get_xyz(vertices)

    if scene_center is None:
        center = robust_center(xyz)
    else:
        center = scene_center

    dist = np.linalg.norm(xyz - center, axis=1)
    threshold = np.percentile(dist, percentile)
    keep = dist <= threshold
    return keep, threshold, center


# ==========================================================
# Optional spatial filter
# ==========================================================

def spatial_filter(vertices, spatial_k, min_neighbors):
    xyz = get_xyz(vertices)
    n = len(xyz)

    if n == 0:
        return np.zeros(0, dtype=bool), 0.0, 0.0

    k_eff = min(spatial_k + 1, n)
    tree = cKDTree(xyz)
    distances, _ = tree.query(xyz, k=k_eff)

    if k_eff == 1:
        kth = np.zeros(n, dtype=np.float64)
    else:
        kth = distances[:, -1]

    median_distance = np.median(kth)
    radius = median_distance * 5.0

    counts = np.sum(distances <= radius, axis=1) - 1
    keep = counts >= min_neighbors

    return keep, radius, median_distance


# ==========================================================
# Scene center estimation
# ==========================================================

def estimate_scene_center(vertices, scale_metric_name,
                          center_scale_percentile,
                          center_opacity_percentile,
                          center_keep_percentile):
    """
    Estimate scene center from points that are:
    - among the smaller splats
    - among the more opaque splats
    - and relatively central geometrically

    This helps favor the object core over blurry borders.
    """
    xyz = get_xyz(vertices)
    n = len(xyz)

    if n == 0:
        return np.zeros(3, dtype=np.float64), {}

    base_center = robust_center(xyz)
    base_dist = np.linalg.norm(xyz - base_center, axis=1)

    # Opacity
    if "opacity" in vertices.dtype.names:
        opacity = decode_opacity(vertices["opacity"])
    else:
        opacity = np.ones(n, dtype=np.float64)

    # Scale
    scales = decode_scales(vertices)
    if scales is not None:
        scale_metric = compute_scale_metric(scales, scale_metric_name)
    else:
        scale_metric = np.ones(n, dtype=np.float64)

    scale_thr = np.percentile(scale_metric, center_scale_percentile)
    opacity_thr = np.percentile(opacity, center_opacity_percentile)
    dist_thr = np.percentile(base_dist, center_keep_percentile)

    keep = (
        (scale_metric <= scale_thr) &
        (opacity >= opacity_thr) &
        (base_dist <= dist_thr)
    )

    # Fallbacks if intersection too small
    if np.sum(keep) < max(128, int(0.001 * n)):
        keep = (
            (scale_metric <= np.percentile(scale_metric, min(center_scale_percentile + 10, 100))) &
            (opacity >= np.percentile(opacity, max(center_opacity_percentile - 10, 0))) &
            (base_dist <= np.percentile(base_dist, min(center_keep_percentile + 5, 100)))
        )

    if np.sum(keep) == 0:
        center = base_center
    else:
        center = np.median(xyz[keep], axis=0)

    info = {
        "base_center": base_center,
        "center": center,
        "selected_points": int(np.sum(keep)),
        "scale_threshold": float(scale_thr),
        "opacity_threshold": float(opacity_thr),
        "distance_threshold": float(dist_thr),
    }

    return center, info


# ==========================================================
# Voxel clustering with adaptive radial penalties
# ==========================================================

def compute_voxel_indices(xyz, mins, voxel_size, grid_size):
    rel = (xyz - mins) / voxel_size
    idx = np.floor(rel).astype(np.int32)
    idx = np.clip(idx, 0, grid_size - 1)
    return idx


def make_neighbor_offsets(connectivity):
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue

                manhattan = abs(dx) + abs(dy) + abs(dz)

                if connectivity == 6 and manhattan == 1:
                    offsets.append((dx, dy, dz))
                elif connectivity == 18 and manhattan <= 2:
                    offsets.append((dx, dy, dz))
                elif connectivity == 26:
                    offsets.append((dx, dy, dz))
    return offsets


def voxel_cluster_filter(
    vertices,
    scene_center,
    scale_metric_name,
    voxel_grid_size,
    voxel_min_points,
    voxel_connectivity,
    voxel_center_sigma,
    voxel_max_fraction,
    attach_layers,
    attach_min_points,
    voxel_radial_density_bias,
    voxel_scale_bias,
    voxel_scale_percentile,
    attach_radial_density_bias
):
    xyz = get_xyz(vertices)
    n = len(xyz)

    if n == 0:
        return np.zeros(0, dtype=bool), {}

    dist_to_center = np.linalg.norm(xyz - scene_center, axis=1)
    dist_med = np.median(dist_to_center)
    dist_sig = robust_sigma(dist_to_center)
    radial_limit = dist_med + voxel_center_sigma * dist_sig

    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-9)
    voxel_size = extent / float(voxel_grid_size)

    voxel_idx = compute_voxel_indices(
        xyz,
        mins=mins,
        voxel_size=voxel_size,
        grid_size=voxel_grid_size
    )

    voxel_keys, inverse, counts = np.unique(
        voxel_idx,
        axis=0,
        return_inverse=True,
        return_counts=True
    )

    voxel_centers = mins + (voxel_keys.astype(np.float64) + 0.5) * voxel_size
    voxel_center_dist = np.linalg.norm(voxel_centers - scene_center, axis=1)
    voxel_radial_norm = voxel_center_dist / (radial_limit + 1e-12)

    # Per-voxel median scale
    scales = decode_scales(vertices)
    if scales is not None:
        point_scale_metric = compute_scale_metric(scales, scale_metric_name)
        voxel_scale = np.zeros(len(voxel_keys), dtype=np.float64)
        for i in range(len(voxel_keys)):
            voxel_scale[i] = np.median(point_scale_metric[inverse == i])
        voxel_scale_limit = np.percentile(voxel_scale, voxel_scale_percentile)
    else:
        voxel_scale = np.ones(len(voxel_keys), dtype=np.float64)
        voxel_scale_limit = np.inf

    dense_required = voxel_min_points * (
        1.0 + voxel_radial_density_bias * (voxel_radial_norm ** 2)
    )

    # Scale criterion gets stricter with distance:
    # far voxels tolerate less oversize
    if np.isfinite(voxel_scale_limit):
        scale_allowed = voxel_scale_limit / (
            1.0 + voxel_scale_bias * np.maximum(voxel_radial_norm - 0.5, 0.0) ** 2
        )
        scale_ok = voxel_scale <= scale_allowed
    else:
        scale_ok = np.ones(len(voxel_keys), dtype=bool)

    radial_ok = voxel_center_dist <= radial_limit
    dense_ok = counts >= dense_required

    active_mask = dense_ok & radial_ok & scale_ok

    active_voxel_keys = voxel_keys[active_mask]
    active_center_dist = voxel_center_dist[active_mask]

    if len(active_voxel_keys) == 0:
        info = {
            "num_occupied_voxels": len(voxel_keys),
            "num_active_voxels": 0,
            "selected_voxels": 0,
            "selected_points": 0,
            "dist_median": dist_med,
            "dist_sigma": dist_sig,
            "radial_limit": radial_limit,
            "voxel_scale_limit": voxel_scale_limit,
            "stop_reason": "no_active_voxels",
        }
        return np.zeros(n, dtype=bool), info

    active_dict = {
        tuple(v.tolist()): i
        for i, v in enumerate(active_voxel_keys)
    }

    seed_idx = int(np.argmin(active_center_dist))
    seed_voxel = tuple(active_voxel_keys[seed_idx].tolist())

    offsets = make_neighbor_offsets(voxel_connectivity)

    selected = set()
    q = deque([seed_voxel])
    selected.add(seed_voxel)

    max_voxels = max(1, int(np.floor(voxel_max_fraction * len(active_voxel_keys))))
    stop_reason = "frontier_exhausted"

    while q:
        if len(selected) >= max_voxels:
            stop_reason = "max_fraction_reached"
            break

        v = q.popleft()
        x, y, z = v

        for dx, dy, dz in offsets:
            nb = (x + dx, y + dy, z + dz)
            if nb in active_dict and nb not in selected:
                selected.add(nb)
                q.append(nb)
                if len(selected) >= max_voxels:
                    stop_reason = "max_fraction_reached"
                    break

        if len(selected) >= max_voxels:
            break

    # Optional attachment: more permissive than active_mask,
    # but still with radial-density bias and scale penalty.
    if attach_layers > 0:
        selected_expanded = set(selected)

        occupied_dict = {
            tuple(v.tolist()): i
            for i, v in enumerate(voxel_keys)
        }

        frontier = set(selected)

        for _ in range(attach_layers):
            new_frontier = set()

            for v in frontier:
                x, y, z = v

                for dx, dy, dz in offsets:
                    nb = (x + dx, y + dy, z + dz)
                    if nb in selected_expanded:
                        continue
                    if nb not in occupied_dict:
                        continue

                    occ_idx = occupied_dict[nb]

                    radial_norm = voxel_radial_norm[occ_idx]
                    required_attach = attach_min_points * (
                        1.0 + attach_radial_density_bias * (radial_norm ** 2)
                    )

                    if counts[occ_idx] < required_attach:
                        continue
                    if voxel_center_dist[occ_idx] > radial_limit:
                        continue
                    if not scale_ok[occ_idx]:
                        continue

                    selected_expanded.add(nb)
                    new_frontier.add(nb)

            if not new_frontier:
                break

            frontier = new_frontier

        selected = selected_expanded

    selected_struct = set(selected)

    keep_points = np.array(
        [tuple(v.tolist()) in selected_struct for v in voxel_idx],
        dtype=bool
    )

    info = {
        "num_occupied_voxels": len(voxel_keys),
        "num_active_voxels": len(active_voxel_keys),
        "selected_voxels": len(selected),
        "selected_points": int(np.sum(keep_points)),
        "dist_median": dist_med,
        "dist_sigma": dist_sig,
        "radial_limit": radial_limit,
        "seed_voxel": seed_voxel,
        "voxel_size": voxel_size,
        "voxel_scale_limit": voxel_scale_limit,
        "stop_reason": stop_reason,
    }

    return keep_points, info


# ==========================================================
# Main
# ==========================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
Gaussian splat filtering with:
- cheap point cleanup
- intelligent scene center estimation
- adaptive voxel clustering

Scene center is estimated from points that are:
- small
- opaque
- relatively central

Far from center:
- density requirement increases
- large splats are rejected more aggressively
"""
    )

    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)

    p.add_argument("--min-opacity", type=float, default=0.005)
    p.add_argument("--max-scale-percentile", type=float, default=99.0)
    p.add_argument("--scale-metric", choices=["max", "mean", "norm"], default="max")
    p.add_argument("--center-percentile", type=float, default=98.0)

    p.add_argument("--spatial-k", type=int, default=8)
    p.add_argument("--min-neighbors", type=int, default=0)

    # intelligent scene center
    p.add_argument("--center-scale-percentile", type=float, default=70.0)
    p.add_argument("--center-opacity-percentile", type=float, default=70.0)
    p.add_argument("--center-keep-percentile", type=float, default=80.0)

    # voxel clustering
    p.add_argument("--voxel-grid-size", type=int, default=100)
    p.add_argument("--voxel-min-points", type=int, default=3)
    p.add_argument("--voxel-connectivity", type=int, default=26, choices=[6, 18, 26])
    p.add_argument("--voxel-center-sigma", type=float, default=2.5)
    p.add_argument("--voxel-max-fraction", type=float, default=0.35)
    p.add_argument("--attach-layers", type=int, default=1)
    p.add_argument("--attach-min-points", type=int, default=2)

    # adaptive penalties
    p.add_argument(
        "--voxel-radial-density-bias",
        type=float,
        default=2.0,
        help="Increase required voxel density with distance from scene center"
    )
    p.add_argument(
        "--voxel-scale-bias",
        type=float,
        default=2.0,
        help="Reject large splats more strongly near scene border"
    )
    p.add_argument(
        "--voxel-scale-percentile",
        type=float,
        default=95.0,
        help="Reference percentile for allowed voxel median scale"
    )
    p.add_argument(
        "--attach-radial-density-bias",
        type=float,
        default=1.5,
        help="Radial density bias for attached voxels"
    )

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--recenter", action="store_true")
    p.add_argument("--supersplat", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    validate_args(args)

    if args.recenter and args.supersplat:
        raise ValueError("--recenter and --supersplat are mutually exclusive")

    print(f"\nLoading PLY:\n{args.input}")

    ply = PlyData.read(str(args.input))
    if "vertex" not in ply:
        raise RuntimeError("No vertex element found")

    vertices = ply["vertex"].data
    total = len(vertices)

    print(f"Input splats: {total:,}")

    keep_mask = np.ones(total, dtype=bool)

    # ------------------------------------------------------
    # Opacity
    # ------------------------------------------------------
    if args.min_opacity is not None and args.min_opacity > 0:
        keep = opacity_filter(vertices, args.min_opacity)
        removed = int(np.sum(keep_mask & ~keep))
        keep_mask &= keep

        print_stage("Opacity filter")
        print(f"min opacity : {args.min_opacity}")
        print(f"removed     : {removed:,}")
        print(f"remaining   : {np.sum(keep_mask):,}")

    # ------------------------------------------------------
    # Global scale
    # ------------------------------------------------------
    if args.max_scale_percentile is not None and args.max_scale_percentile > 0:
        keep, threshold = global_scale_filter(
            vertices,
            args.max_scale_percentile,
            args.scale_metric
        )

        print_stage("Global scale filter")
        if keep is None:
            print("Skipped: no scale fields found")
        else:
            removed = int(np.sum(keep_mask & ~keep))
            keep_mask &= keep
            print(f"metric      : {args.scale_metric}")
            print(f"percentile  : {args.max_scale_percentile}")
            print(f"threshold   : {threshold:.6f}")
            print(f"removed     : {removed:,}")
            print(f"remaining   : {np.sum(keep_mask):,}")

    # ------------------------------------------------------
    # Intelligent scene center
    # ------------------------------------------------------
    current = vertices[keep_mask]
    scene_center, center_info = estimate_scene_center(
        current,
        scale_metric_name=args.scale_metric,
        center_scale_percentile=args.center_scale_percentile,
        center_opacity_percentile=args.center_opacity_percentile,
        center_keep_percentile=args.center_keep_percentile
    )

    print_stage("Scene center estimation")
    print(
        f"scene center     : "
        f"[{scene_center[0]:.6f}, {scene_center[1]:.6f}, {scene_center[2]:.6f}]"
    )
    print(f"selected points  : {center_info['selected_points']:,}")
    print(f"scale thr        : {center_info['scale_threshold']:.6f}")
    print(f"opacity thr      : {center_info['opacity_threshold']:.6f}")
    print(f"distance thr     : {center_info['distance_threshold']:.6f}")

    # ------------------------------------------------------
    # Center trim
    # ------------------------------------------------------
    if args.center_percentile is not None and args.center_percentile > 0:
        current = vertices[keep_mask]
        keep_local, threshold, center = center_filter(
            current,
            args.center_percentile,
            scene_center=scene_center
        )

        global_idx = np.flatnonzero(keep_mask)
        new_mask = np.zeros_like(keep_mask)
        new_mask[global_idx[keep_local]] = True

        removed = int(np.sum(keep_mask) - np.sum(new_mask))
        keep_mask = new_mask

        print_stage("Center filter")
        print(f"percentile  : {args.center_percentile}")
        print(f"threshold   : {threshold:.6f}")
        print(f"center      : [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]")
        print(f"removed     : {removed:,}")
        print(f"remaining   : {np.sum(keep_mask):,}")

    # ------------------------------------------------------
    # Optional spatial filter
    # ------------------------------------------------------
    if args.min_neighbors > 0:
        current = vertices[keep_mask]
        keep_local, radius, median_distance = spatial_filter(
            current,
            spatial_k=args.spatial_k,
            min_neighbors=args.min_neighbors
        )

        global_idx = np.flatnonzero(keep_mask)
        new_mask = np.zeros_like(keep_mask)
        new_mask[global_idx[keep_local]] = True

        removed = int(np.sum(keep_mask) - np.sum(new_mask))
        keep_mask = new_mask

        print_stage("Spatial filter")
        print(f"spatial_k        : {args.spatial_k}")
        print(f"median kNN dist  : {median_distance:.6f}")
        print(f"auto radius      : {radius:.6f}")
        print(f"min neighbors    : {args.min_neighbors}")
        print(f"removed          : {removed:,}")
        print(f"remaining        : {np.sum(keep_mask):,}")
    else:
        print_stage("Spatial filter")
        print("Skipped: min-neighbors == 0")

    # ------------------------------------------------------
    # Adaptive voxel cluster filter
    # ------------------------------------------------------
    current = vertices[keep_mask]
    keep_local, info = voxel_cluster_filter(
        current,
        scene_center=scene_center,
        scale_metric_name=args.scale_metric,
        voxel_grid_size=args.voxel_grid_size,
        voxel_min_points=args.voxel_min_points,
        voxel_connectivity=args.voxel_connectivity,
        voxel_center_sigma=args.voxel_center_sigma,
        voxel_max_fraction=args.voxel_max_fraction,
        attach_layers=args.attach_layers,
        attach_min_points=args.attach_min_points,
        voxel_radial_density_bias=args.voxel_radial_density_bias,
        voxel_scale_bias=args.voxel_scale_bias,
        voxel_scale_percentile=args.voxel_scale_percentile,
        attach_radial_density_bias=args.attach_radial_density_bias
    )

    global_idx = np.flatnonzero(keep_mask)
    new_mask = np.zeros_like(keep_mask)
    new_mask[global_idx[keep_local]] = True

    removed = int(np.sum(keep_mask) - np.sum(new_mask))
    keep_mask = new_mask

    vs = info["voxel_size"]

    print_stage("Adaptive voxel cluster filter")
    print(f"grid size        : {args.voxel_grid_size}")
    print(f"voxel min points : {args.voxel_min_points}")
    print(f"connectivity     : {args.voxel_connectivity}")
    print(f"center sigma     : {args.voxel_center_sigma}")
    print(f"max fraction     : {args.voxel_max_fraction:.3f}")
    print(f"attach layers    : {args.attach_layers}")
    print(f"attach min pts   : {args.attach_min_points}")
    print(f"radial dens bias : {args.voxel_radial_density_bias}")
    print(f"scale bias       : {args.voxel_scale_bias}")
    print(f"scale percentile : {args.voxel_scale_percentile}")
    print(f"occupied voxels  : {info['num_occupied_voxels']:,}")
    print(f"active voxels    : {info['num_active_voxels']:,}")
    print(f"selected voxels  : {info['selected_voxels']:,}")
    print(f"selected points  : {info['selected_points']:,}")
    print(f"voxel size       : [{vs[0]:.6f}, {vs[1]:.6f}, {vs[2]:.6f}]")
    print(f"radial limit     : {info['radial_limit']:.6f}")
    print(f"voxel scale lim  : {info['voxel_scale_limit']:.6f}")
    print(f"stop reason      : {info['stop_reason']}")
    print(f"removed          : {removed:,}")
    print(f"remaining        : {np.sum(keep_mask):,}")

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------
    kept = int(np.sum(keep_mask))
    removed = total - kept

    print("\n========== SUMMARY ==========")
    print(f"Total splats     : {total:,}")
    print(f"Final kept       : {kept:,}")
    print(f"Final removed    : {removed:,}")
    print(f"Retention        : {100.0 * kept / max(total, 1):.2f}%")

    if args.dry_run:
        print("\nDry run enabled.")
        return

    filtered_vertices = vertices[keep_mask].copy()
    xyz_new = get_xyz(filtered_vertices)

    if args.recenter:
        xyz_new -= xyz_new.mean(axis=0)
    elif args.supersplat:
        center = xyz_new.mean(axis=0)
        xyz_new -= center
        scale = np.max(np.linalg.norm(xyz_new, axis=1))
        if scale > 0:
            xyz_new /= scale

    filtered_vertices["x"] = xyz_new[:, 0]
    filtered_vertices["y"] = xyz_new[:, 1]
    filtered_vertices["z"] = xyz_new[:, 2]

    PlyData(
        [PlyElement.describe(filtered_vertices, "vertex")],
        text=False
    ).write(str(args.output))

    print(f"\nSaved:\n{args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nERROR:")
        traceback.print_exc()
        sys.exit(1)
