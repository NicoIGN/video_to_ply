#!/usr/bin/env python3

from pathlib import Path
import argparse
import sys
import traceback

import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)

    p.add_argument("--nb-neighbors", type=int, default=32)

    # SOR control
    p.add_argument("--sor-percentile", type=float, default=85.0)

    p.add_argument("--dbscan-min-points", type=int, default=50)

    p.add_argument("--center-percentile", type=float, default=95.0)

    # global aggressivity control
    p.add_argument("--clean-level", type=float, default=1.0)

    # 🔥 NEW PARAMETER (IMPORTANT)
    p.add_argument("--edge-percentile", type=float, default=20.0)
    # 10 = very aggressive
    # 20 = balanced
    # 35 = conservative

    p.add_argument("--recenter", action="store_true")
    p.add_argument("--supersplat", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    print(f"📥 Loading: {args.input}")

    ply = PlyData.read(str(args.input))
    vertex = ply["vertex"].data

    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T
    print(f"📊 Points: {len(xyz):,}")

    # =========================
    # CENTER FILTER
    # =========================
    center = np.median(xyz, axis=0)
    dist_center = np.linalg.norm(xyz - center, axis=1)

    threshold = np.percentile(dist_center, args.center_percentile)
    mask = dist_center < threshold

    xyz_f = xyz[mask]
    idx_map = np.where(mask)[0]

    print(f"📊 After center filter: {len(xyz_f):,}")

    # =========================
    # SOR FILTER
    # =========================
    nn = NearestNeighbors(n_neighbors=args.nb_neighbors).fit(xyz_f)
    dists, _ = nn.kneighbors(xyz_f)

    mean_dist = dists.mean(axis=1)

    sor_thresh = np.percentile(
        mean_dist,
        80 + (10 / args.clean_level)
    )

    mask_sor = mean_dist < sor_thresh
    xyz_f = xyz_f[mask_sor]
    idx_map = idx_map[mask_sor]

    print(f"📊 After SOR: {len(xyz_f):,}")

    # =========================
    # EDGE REMOVAL (IMPROVED + PARAMETERIZED)
    # =========================

    global_center = np.mean(xyz_f, axis=0)
    dist_global = np.linalg.norm(xyz_f - global_center, axis=1)

    density = 1.0 / (mean_dist[mask_sor] + 1e-8)

    score = density / (dist_global + 1e-8)

    # 🔥 now fully controllable
    edge_thresh = np.percentile(score, args.edge_percentile)

    mask_edge = score > edge_thresh

    xyz_f = xyz_f[mask_edge]
    idx_map = idx_map[mask_edge]

    print(f"📊 After EDGE removal: {len(xyz_f):,}")

    # =========================
    # DBSCAN (scale-free)
    # =========================
    print("🔗 DBSCAN clustering...")

    local_scale = np.median(mean_dist)
    eps = local_scale * (2.0 / args.clean_level)

    labels = DBSCAN(
        eps=eps,
        min_samples=args.dbscan_min_points
    ).fit_predict(xyz_f)

    valid = labels >= 0

    if valid.sum() == 0:
        print("⚠️ No clusters → fallback")
        final_idx = idx_map
    else:
        largest = np.bincount(labels[valid]).argmax()
        final_idx = idx_map[labels == largest]

    print(f"📊 Final points: {len(final_idx):,}")

    # =========================
    # REBUILD
    # =========================
    new_vertex = vertex[final_idx]

    xyz_new = np.vstack([
        new_vertex["x"],
        new_vertex["y"],
        new_vertex["z"]
    ]).T

    # =========================
    # MODE
    # =========================
    if args.recenter:
        center = xyz_new.mean(axis=0)
        xyz_new -= center
        print(f"📍 Recenter applied")

    elif args.supersplat:
        center = xyz_new.mean(axis=0)
        xyz_new -= center

        scale = np.max(np.linalg.norm(xyz_new, axis=1))
        if scale > 0:
            xyz_new /= scale

        print(f"📏 Normalized scale: {scale:.6f}")

    # =========================
    # SAVE
    # =========================
    new_vertex["x"] = xyz_new[:, 0]
    new_vertex["y"] = xyz_new[:, 1]
    new_vertex["z"] = xyz_new[:, 2]

    el = PlyElement.describe(new_vertex, "vertex")
    PlyData([el], text=False).write(str(args.output))

    print(f"✅ Saved: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n💥 ERROR:")
        traceback.print_exc()
        sys.exit(1)
