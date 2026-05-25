#!/usr/bin/env python3

from pathlib import Path
import argparse
import sys
import traceback

import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)

    # neighborhood
    p.add_argument("--k", type=int, default=24)

    # global trimming
    p.add_argument("--center-percentile", type=float, default=98.0)

    # structure filtering
    p.add_argument("--keep-percentile", type=float, default=60.0)

    # isolation filtering
    p.add_argument("--radius-mult", type=float, default=1.5)
    p.add_argument("--min-neighbors", type=int, default=8)

    # splat size filtering (0 = disabled)
    p.add_argument(
        "--max-relative-scale",
        type=float,
        default=0.0,
        help="Reject splats larger than local_median * factor (0 disables)"
    )

    # output modes
    p.add_argument("--recenter", action="store_true")
    p.add_argument("--supersplat", action="store_true")

    return p.parse_args()


# ============================================================
# STRUCTURE SCORE
# ============================================================

def structure_score(neighbors):
    cov = np.cov(neighbors.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]

    s = eigvals.sum()
    if s <= 1e-12:
        return 0.0

    l1, l2, l3 = eigvals / s

    linearity = l1 - l2
    planarity = l2 - l3

    return linearity + planarity


def connectivity_score(distances):
    mean_d = np.mean(distances)
    std_d = np.std(distances)

    if mean_d <= 1e-12:
        return 0.0

    cv = std_d / mean_d
    return 1.0 / (cv + 1e-6)


# ============================================================
# SPLAT SIZE
# ============================================================

def extract_splat_scale(vertex):
    """
    Try common Gaussian Splat scale formats.

    Returns:
        per-point scalar size
        or None if unavailable.
    """
    names = vertex.dtype.names

    # Standard 3DGS log-scales
    if all(n in names for n in ("scale_0", "scale_1", "scale_2")):
        s0 = np.exp(vertex["scale_0"])
        s1 = np.exp(vertex["scale_1"])
        s2 = np.exp(vertex["scale_2"])
        return (s0 * s1 * s2) ** (1.0 / 3.0)

    # Alternative naming
    if all(n in names for n in ("sx", "sy", "sz")):
        sx = np.abs(vertex["sx"])
        sy = np.abs(vertex["sy"])
        sz = np.abs(vertex["sz"])
        return (sx * sy * sz) ** (1.0 / 3.0)

    return None


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    print(f"📥 Loading: {args.input}")

    ply = PlyData.read(str(args.input))
    vertex = ply["vertex"].data

    xyz = np.vstack([
        vertex["x"],
        vertex["y"],
        vertex["z"]
    ]).T

    splat_scale = extract_splat_scale(vertex)

    print(f"📊 Input points: {len(xyz):,}")

    # ========================================================
    # 1. GLOBAL TRIM
    # ========================================================

    if args.center_percentile > 0:
        center = np.median(xyz, axis=0)
        dist = np.linalg.norm(xyz - center, axis=1)

        mask = dist < np.percentile(dist, args.center_percentile)

        xyz_f = xyz[mask]
        idx_map = np.where(mask)[0]

        print(f"📊 After center trim: {len(xyz_f):,}")
    else:
        xyz_f = xyz
        idx_map = np.arange(len(xyz))
        print("📊 Center trim disabled")

    # ========================================================
    # 2. KNN
    # ========================================================

    nn = NearestNeighbors(n_neighbors=args.k)
    nn.fit(xyz_f)

    distances, indices = nn.kneighbors(xyz_f)

    # ========================================================
    # 3. STRUCTURE SCORE
    # ========================================================

    scores = np.zeros(len(xyz_f))

    for i in range(len(xyz_f)):
        neigh = xyz_f[indices[i]]

        s = structure_score(neigh)
        c = connectivity_score(distances[i])

        scores[i] = s * c

    # ========================================================
    # 4. STRUCTURE FILTER
    # ========================================================

    if args.keep_percentile > 0:
        thresh = np.percentile(scores, args.keep_percentile)
        mask_clean = scores >= thresh

        xyz_f = xyz_f[mask_clean]
        idx_map = idx_map[mask_clean]

        print(f"📊 After structure filter: {len(xyz_f):,}")
    else:
        print("📊 Structure filter disabled")

    # ========================================================
    # 5. SPLAT SIZE FILTER
    # ========================================================

    if args.max_relative_scale > 0 and splat_scale is not None:
        local_scales = splat_scale[idx_map]

        nn_scale = NearestNeighbors(n_neighbors=args.k)
        nn_scale.fit(xyz_f)

        _, idx_scale = nn_scale.kneighbors(xyz_f)

        neighbor_scales = local_scales[idx_scale]
        local_median = np.median(neighbor_scales, axis=1)

        ratio = local_scales / (local_median + 1e-9)

        mask_size = ratio <= args.max_relative_scale

        before = len(xyz_f)

        xyz_f = xyz_f[mask_size]
        idx_map = idx_map[mask_size]

        removed = before - len(xyz_f)

        print(
            f"📊 After splat-size filter: "
            f"{len(xyz_f):,} (removed {removed:,})"
        )

    elif args.max_relative_scale > 0:
        print("⚠️ Splat-size filter requested, but no scale fields found")
    else:
        print("📊 Splat-size filter disabled")

    # ========================================================
    # 6. RADIAL ISOLATION FILTER
    # ========================================================

    if args.radius_mult > 0 and args.min_neighbors > 0:
        scene_center = np.median(xyz_f, axis=0)

        # Normalized radial distance
        radial_dist = np.linalg.norm(
            xyz_f - scene_center,
            axis=1
        )

        radial_scale = np.percentile(radial_dist, 95) + 1e-9
        radial_norm = radial_dist / radial_scale

        # Base neighborhood radius
        nn_iso = NearestNeighbors(
            n_neighbors=min(args.k, len(xyz_f))
        )
        nn_iso.fit(xyz_f)

        d_iso, _ = nn_iso.kneighbors(xyz_f)
        avg_dist = np.mean(d_iso[:, 1:])

        radius = avg_dist * args.radius_mult

        nbrs = NearestNeighbors(radius=radius)
        nbrs.fit(xyz_f)

        neighbors = nbrs.radius_neighbors(
            xyz_f,
            return_distance=False
        )

        density = np.array([len(n) for n in neighbors])

        # ====================================================
        # Distance-aware density requirement
        #
        # Near center: standard threshold
        # Far away   : increasingly strict
        # ====================================================

        required_neighbors = (
            args.min_neighbors *
            (1.0 + radial_norm ** 2)
        )

        mask_iso = density >= required_neighbors

        before = len(xyz_f)

        xyz_f = xyz_f[mask_iso]
        idx_map = idx_map[mask_iso]

        removed = before - len(xyz_f)

        print(
            f"📊 After radial isolation filter: "
            f"{len(xyz_f):,} (removed {removed:,})"
        )

    else:
        print("📊 Isolation filter disabled")

    # ========================================================
    # REBUILD
    # ========================================================

    new_vertex = vertex[idx_map]

    xyz_new = np.vstack([
        new_vertex["x"],
        new_vertex["y"],
        new_vertex["z"]
    ]).T

    if args.recenter:
        xyz_new -= xyz_new.mean(axis=0)

    elif args.supersplat:
        center = xyz_new.mean(axis=0)
        xyz_new -= center

        scale = np.max(np.linalg.norm(xyz_new, axis=1))
        if scale > 0:
            xyz_new /= scale

    new_vertex["x"] = xyz_new[:, 0]
    new_vertex["y"] = xyz_new[:, 1]
    new_vertex["z"] = xyz_new[:, 2]

    PlyData(
        [PlyElement.describe(new_vertex, "vertex")],
        text=False
    ).write(str(args.output))

    print(f"✅ Saved: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n💥 ERROR:")
        traceback.print_exc()
        sys.exit(1)
