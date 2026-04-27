#!/usr/bin/env python3
"""
Clean a Gaussian Splat PLY by removing outliers and keeping
the main connected component.

Usage:
    python clean_gaussian_ply.py input.ply output.ply \
        --nb-neighbors 32 \
        --std-ratio 1.5 \
        --dbscan-eps 0.05 \
        --dbscan-min-points 50
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean Gaussian Splat PLY files."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input PLY file",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Output cleaned PLY file",
    )

    parser.add_argument(
        "--nb-neighbors",
        type=int,
        default=32,
        help="Number of neighbors for statistical outlier removal (default: 32)",
    )

    parser.add_argument(
        "--std-ratio",
        type=float,
        default=1.5,
        help="Standard deviation ratio for outlier removal (default: 1.5)",
    )

    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.05,
        help="DBSCAN epsilon radius (default: 0.05)",
    )

    parser.add_argument(
        "--dbscan-min-points",
        type=int,
        default=50,
        help="Minimum points per DBSCAN cluster (default: 50)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)

    print(f"📥 Loading: {args.input}")
    print("⚙️ Parameters:")
    print(f"   nb_neighbors      = {args.nb_neighbors}")
    print(f"   std_ratio         = {args.std_ratio}")
    print(f"   dbscan_eps        = {args.dbscan_eps}")
    print(f"   dbscan_min_points = {args.dbscan_min_points}")

    pcd = o3d.io.read_point_cloud(str(args.input))

    if len(pcd.points) == 0:
        print("❌ Empty point cloud")
        sys.exit(1)

    print(f"📊 Original points: {len(pcd.points):,}")

    # ==========================================
    # Statistical Outlier Removal
    # ==========================================
    print("🧹 Removing statistical outliers...")

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
    )

    print(f"📊 After SOR: {len(pcd.points):,}")

    # ==========================================
    # Largest DBSCAN Cluster
    # ==========================================
    print("🔗 Finding largest cluster...")

    labels = np.array(
        pcd.cluster_dbscan(
            eps=args.dbscan_eps,
            min_points=args.dbscan_min_points,
            print_progress=False,
        )
    )

    valid_labels = labels[labels >= 0]

    if len(valid_labels) > 0:
        largest_label = np.bincount(valid_labels).argmax()
        indices = np.where(labels == largest_label)[0]
        pcd = pcd.select_by_index(indices)
        print(f"📊 Largest cluster: {len(pcd.points):,} points")
    else:
        print("⚠️ No clusters found; keeping filtered cloud")

    # ==========================================
    # Save
    # ==========================================
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not o3d.io.write_point_cloud(str(args.output), pcd):
        print("❌ Failed to write output")
        sys.exit(1)

    print(f"✅ Saved: {args.output}")
    print(f"📊 Final points: {len(pcd.points):,}")


if __name__ == "__main__":
    main()
