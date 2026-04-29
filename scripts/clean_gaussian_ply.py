#!/usr/bin/env python3

from pathlib import Path
import argparse
import sys
import traceback

import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)

    # core density control
    p.add_argument("--nb-neighbors", type=int, default=16)

    # outlier rejection (soft)
    p.add_argument("--outlier-ratio", type=float, default=3.0)

    # spatial trimming (very light)
    p.add_argument("--center-percentile", type=float, default=98.0)

    # optional aggressivity
    p.add_argument("--clean-level", type=float, default=1.0)

    # modes
    p.add_argument("--recenter", action="store_true")
    p.add_argument("--supersplat", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    print(f"📥 Loading: {args.input}")

    ply = PlyData.read(str(args.input))
    vertex = ply["vertex"].data

    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T
    n = len(xyz)

    print(f"📊 Input points: {n:,}")

    # =========================
    # 1. GLOBAL CENTER TRIM (VERY SAFE)
    # =========================
    center = np.median(xyz, axis=0)
    dist_center = np.linalg.norm(xyz - center, axis=1)

    thresh_center = np.percentile(dist_center, args.center_percentile)
    mask = dist_center < thresh_center

    xyz_f = xyz[mask]
    idx_map = np.where(mask)[0]

    print(f"📊 After center trim: {len(xyz_f):,}")

    # =========================
    # 2. LOCAL DENSITY FILTER (CORE SPLAT FILTER)
    # =========================
    if len(xyz_f) > args.nb_neighbors:

        nn = NearestNeighbors(n_neighbors=args.nb_neighbors)
        nn.fit(xyz_f)

        dists, _ = nn.kneighbors(xyz_f)
        mean_dist = dists.mean(axis=1)

        # robust threshold (NOT percentile stacking)
        thr = mean_dist.mean() + args.outlier_ratio * mean_dist.std()

        mask_density = mean_dist < thr

        xyz_f = xyz_f[mask_density]
        idx_map = idx_map[mask_density]

    print(f"📊 After density filter: {len(xyz_f):,}")

    # =========================
    # 3. NO DBSCAN (INTENTIONALLY REMOVED)
    # =========================
    # reason: destroys Gaussian continuity structure

    # =========================
    # REBUILD
    # =========================
    new_vertex = vertex[idx_map]

    xyz_new = np.vstack([
        new_vertex["x"],
        new_vertex["y"],
        new_vertex["z"]
    ]).T

    # =========================
    # MODE
    # =========================
    if args.recenter:
        print("📍 Recenter only")
        xyz_new -= xyz_new.mean(axis=0)

    elif args.supersplat:
        print("🚀 Supersplat mode")
        center = xyz_new.mean(axis=0)
        xyz_new -= center

        scale = np.max(np.linalg.norm(xyz_new, axis=1))
        if scale > 0:
            xyz_new /= scale

        print(f"📏 scale: {scale:.6f}")

    # =========================
    # SAVE
    # =========================
    new_vertex["x"] = xyz_new[:, 0]
    new_vertex["y"] = xyz_new[:, 1]
    new_vertex["z"] = xyz_new[:, 2]

    PlyData([PlyElement.describe(new_vertex, "vertex")], text=False).write(str(args.output))

    print(f"✅ Saved: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n💥 ERROR:")
        traceback.print_exc()
        sys.exit(1)
