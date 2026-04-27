#!/usr/bin/env python3
"""
Clean a Gaussian Splat PLY by removing outliers and keeping
the main connected component.
"""

from pathlib import Path
import argparse
import sys
import traceback

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean Gaussian Splat PLY files."
    )

    parser.add_argument("input", type=Path, help="Input PLY file")
    parser.add_argument("output", type=Path, help="Output cleaned PLY file")

    parser.add_argument(
        "--nb-neighbors",
        type=int,
        default=32,
        help="SOR neighbor count",
    )

    parser.add_argument(
        "--std-ratio",
        type=float,
        default=1.5,
        help="SOR standard deviation ratio",
    )

    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.05,
        help="DBSCAN epsilon",
    )

    parser.add_argument(
        "--dbscan-min-points",
        type=int,
        default=50,
        help="DBSCAN minimum cluster size",
    )

    return parser.parse_args()


def inspect_ply_header(path: Path, max_lines: int = 30) -> None:
    """Display the beginning of the PLY file for debugging."""
    print("\n📄 PLY header preview:")
    try:
        with path.open("rb") as f:
            for i in range(max_lines):
                line = f.readline()
                if not line:
                    break
                try:
                    print(f"   {line.decode('utf-8', errors='replace').rstrip()}")
                except Exception:
                    print(f"   {line!r}")

                if line.strip() == b"end_header":
                    break
    except Exception as e:
        print(f"   ⚠️ Unable to read header: {e}")


def fail(msg: str, code: int = 1) -> None:
    print(f"\n❌ {msg}")
    sys.exit(code)


def main() -> None:
    args = parse_args()

    try:
        # ==========================================
        # Validate input
        # ==========================================
        if not args.input.exists():
            fail(f"Input file not found: {args.input}")

        if not args.input.is_file():
            fail(f"Input path is not a file: {args.input}")

        if args.input.suffix.lower() != ".ply":
            print(
                f"⚠️ Unexpected extension: '{args.input.suffix}' "
                "(expected .ply)"
            )

        file_size = args.input.stat().st_size

        print("────────────────────────────────────────────")
        print(f"📥 Input file : {args.input}")
        print(f"📤 Output file: {args.output}")
        print(f"📦 Size       : {file_size:,} bytes")
        print("────────────────────────────────────────────")

        if file_size == 0:
            fail("Input file is empty (0 bytes)")

        print("⚙️ Parameters:")
        print(f"   nb_neighbors      = {args.nb_neighbors}")
        print(f"   std_ratio         = {args.std_ratio}")
        print(f"   dbscan_eps        = {args.dbscan_eps}")
        print(f"   dbscan_min_points = {args.dbscan_min_points}")

        inspect_ply_header(args.input)

        # ==========================================
        # Load point cloud
        # ==========================================
        print("\n📥 Loading point cloud with Open3D...")

        pcd = o3d.io.read_point_cloud(
            str(args.input),
            format="ply",
            remove_nan_points=True,
            remove_infinite_points=True,
        )

        point_count = len(pcd.points)

        if point_count == 0:
            print("\n⚠️ Open3D returned an empty point cloud.")
            print("Possible causes:")
            print("   • Invalid or corrupted PLY file")
            print("   • Unsupported Gaussian Splat PLY format")
            print("   • Missing vertex data")
            print("   • Header/content mismatch")
            fail("Unable to load any 3D points")

        print(f"📊 Original points: {point_count:,}")

        # ==========================================
        # Statistical Outlier Removal
        # ==========================================
        print("\n🧹 Removing statistical outliers...")

        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=args.nb_neighbors,
            std_ratio=args.std_ratio,
        )

        print(f"📊 After SOR: {len(pcd.points):,}")

        if len(pcd.points) == 0:
            fail("All points were removed by statistical filtering")

        # ==========================================
        # Largest DBSCAN Cluster
        # ==========================================
        print("\n🔗 Finding largest cluster...")

        labels = np.array(
            pcd.cluster_dbscan(
                eps=args.dbscan_eps,
                min_points=args.dbscan_min_points,
                print_progress=False,
            )
        )

        valid_labels = labels[labels >= 0]

        if len(valid_labels) > 0:
            cluster_sizes = np.bincount(valid_labels)
            largest_label = cluster_sizes.argmax()
            largest_size = cluster_sizes[largest_label]

            print(f"📦 Clusters found: {len(cluster_sizes)}")
            print(f"🏆 Largest cluster: #{largest_label}")
            print(f"📊 Largest size   : {largest_size:,}")

            indices = np.where(labels == largest_label)[0]
            pcd = pcd.select_by_index(indices)
        else:
            print("⚠️ No DBSCAN clusters found; keeping SOR result")

        final_points = len(pcd.points)

        if final_points == 0:
            fail("No points remain after clustering")

        # ==========================================
        # Save
        # ==========================================
        print("\n💾 Saving cleaned point cloud...")

        args.output.parent.mkdir(parents=True, exist_ok=True)

        if args.output.exists():
            print(f"🗑️ Removing existing file: {args.output}")
            args.output.unlink()

        success = o3d.io.write_point_cloud(
            str(args.output),
            pcd,
            write_ascii=False,
            compressed=False,
        )

        if not success:
            fail("Open3D failed to write the output file")

        output_size = args.output.stat().st_size

        print("\n✅ Cleaning completed successfully")
        print(f"📄 Output file : {args.output}")
        print(f"📦 Output size : {output_size:,} bytes")
        print(f"📊 Final points: {final_points:,}")

    except Exception as e:
        print("\n💥 Unexpected error:")
        print(f"   {type(e).__name__}: {e}")
        print()
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
