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
    p.add_argument("--std-ratio", type=float, default=1.5)

    p.add_argument("--dbscan-eps", type=float, default=0.05)
    p.add_argument("--dbscan-min-points", type=int, default=50)

    p.add_argument(
        "--center-percentile",
        type=float,
        default=95.0,
        help="Keep points within this percentile distance from center"
    )

    # NEW FLAG
    p.add_argument(
        "--supersplat",
        action="store_true",
        help="Apply Supersplat-style centering + normalization"
    )

    return p.parse_args()


def main():
    args = parse_args()

    print(f"📥 Loading: {args.input}")

    ply = PlyData.read(str(args.input))
    vertex = ply["vertex"].data

    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T
    n = xyz.shape[0]

    print(f"📊 Points: {n:,}")

    # =========================
    # CENTER FILTER
    # =========================
    print("🎯 Center distance filtering...")

    center = np.median(xyz, axis=0)
    dist_center = np.linalg.norm(xyz - center, axis=1)

    threshold = np.percentile(dist_center, args.center_percentile)

    mask_center = dist_center < threshold
    xyz_f = xyz[mask_center]

    print(f"📊 After center filter: {len(xyz_f):,}")

    idx_map = np.where(mask_center)[0]

    # =========================
    # SOR
    # =========================
    print("🧹 SOR filtering...")

    nn = NearestNeighbors(n_neighbors=args.nb_neighbors).fit(xyz_f)
    dists, _ = nn.kneighbors(xyz_f)

    mean_dist = dists.mean(axis=1)
    thresh = mean_dist.mean() + args.std_ratio * mean_dist.std()

    mask_sor = mean_dist < thresh
    xyz_f = xyz_f[mask_sor]

    print(f"📊 After SOR: {len(xyz_f):,}")

    idx_map = idx_map[mask_sor]

    # =========================
    # DBSCAN
    # =========================
    print("🔗 DBSCAN clustering...")

    labels = DBSCAN(
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_points
    ).fit_predict(xyz_f)

    valid = labels >= 0

    if valid.sum() == 0:
        print("⚠️ No clusters found → fallback SOR+center only")
        final_idx = idx_map
    else:
        largest = np.bincount(labels[valid]).argmax()
        keep = labels == largest
        final_idx = idx_map[keep]

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
    # SUPERSPLAT MODE
    # =========================
    if args.supersplat:
        print("🚀 Supersplat mode ON")

        center = xyz_new.mean(axis=0)
        xyz_new = xyz_new - center

        scale = np.max(np.linalg.norm(xyz_new, axis=1))
        if scale > 0:
            xyz_new = xyz_new / scale

        print(f"📍 Center: {center}")
        print(f"📏 Scale normalization: {scale:.6f}")

        new_vertex["x"] = xyz_new[:, 0]
        new_vertex["y"] = xyz_new[:, 1]
        new_vertex["z"] = xyz_new[:, 2]

    # =========================
    # SAVE
    # =========================
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
