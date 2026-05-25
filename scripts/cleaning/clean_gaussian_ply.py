#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


# ==========================================================
# Math helpers
# ==========================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_opacity(raw_opacity):
    """
    Nerfstudio stores opacity in logit space.
    Convert to [0, 1].
    """
    return sigmoid(raw_opacity)


def decode_scales(vertices):
    """
    Nerfstudio stores scale_* in log-space.
    Convert to real Gaussian scales.
    """
    scales_log = np.column_stack([
        vertices["scale_0"],
        vertices["scale_1"],
        vertices["scale_2"]
    ])

    return np.exp(scales_log)


def compute_scale_metric(scales, metric):
    """
    Convert 3D scale to scalar score.
    """

    if metric == "max":
        return np.max(scales, axis=1)

    elif metric == "mean":
        return np.mean(scales, axis=1)

    elif metric == "norm":
        return np.linalg.norm(scales, axis=1)

    raise ValueError(metric)


# ==========================================================
# Filters
# ==========================================================

def opacity_filter(vertices, min_opacity):
    opacity = decode_opacity(
        vertices["opacity"]
    )

    keep = opacity >= min_opacity

    return keep, opacity


def scale_filter(
    vertices,
    percentile,
    metric
):
    """
    Robust scale filtering.

    Remove only anomalously large splats
    based on percentile.
    """

    scales = decode_scales(vertices)

    scale_metric = compute_scale_metric(
        scales,
        metric
    )

    threshold = np.percentile(
        scale_metric,
        percentile
    )

    keep = scale_metric <= threshold

    return (
        keep,
        scale_metric,
        threshold
    )


def spatial_filter(
    vertices,
    spatial_k,
    spatial_factor,
    min_neighbors
):
    """
    Robust scene-scale adaptive filter.

    Estimate a scene-scale invariant radius:

        radius =
            median(distance_to_kth_neighbor)
            * spatial_factor

    Then remove isolated splats.
    """

    xyz = np.column_stack([
        vertices["x"],
        vertices["y"],
        vertices["z"]
    ])

    tree = cKDTree(xyz)

    # ------------------------------------
    # Estimate scene scale
    # ------------------------------------

    distances, _ = tree.query(
        xyz,
        k=spatial_k + 1
    )

    kth_distance = distances[:, -1]

    median_distance = np.median(
        kth_distance
    )

    radius = (
        median_distance
        * spatial_factor
    )

    # ------------------------------------
    # Count neighbors
    # ------------------------------------

    neighbors = tree.query_ball_point(
        xyz,
        radius
    )

    counts = np.array(
        [
            len(n) - 1
            for n in neighbors
        ],
        dtype=np.int32
    )

    keep = counts >= min_neighbors

    return (
        keep,
        counts,
        radius,
        median_distance
    )


# ==========================================================
# Main
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""
    Filter Gaussian Splats (.ply) exported by Nerfstudio / splatfacto.

    Filters (all optional):
      - opacity : remove weak / nearly transparent splats
      - scale   : remove unusually large blurry splats
      - spatial : remove isolated floating splats

    Typical interior preset:

    python filter_splats.py \
        --input scene.ply \
        --output clean.ply \
        --min-opacity 0.01 \
        --max-scale-percentile 99.5 \
        --spatial-k 8 \
        --spatial-factor 5 \
        --min-neighbors 3
    """
    )

    parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help=(
            "Input Gaussian splat PLY "
            "(e.g. scene.ply)"
        )
    )

    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help=(
            "Output filtered PLY with all attributes preserved "
            "(e.g. clean_scene.ply)"
        )
    )

    # ======================================================
    # OPACITY
    # ======================================================

    parser.add_argument(
        "--min-opacity",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Remove splats whose decoded opacity "
            "(after sigmoid) is below threshold "
            "(e.g. 0.005 conservative, "
            "0.01 recommended, "
            "0.02 stronger)"
        )
    )

    # ======================================================
    # SCALE
    # ======================================================

    parser.add_argument(
        "--max-scale-percentile",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Remove splats larger than this scale percentile "
            "to suppress oversized blurry Gaussians without "
            "assuming scene units "
            "(e.g. 99.9 very conservative, "
            "99.5 recommended, "
            "99 stronger, "
            "95 aggressive)"
        )
    )

    parser.add_argument(
        "--scale-metric",
        choices=["max", "mean", "norm"],
        default="max",
        help=(
            "Method used to convert 3D Gaussian scale into "
            "a scalar size for filtering "
            "(e.g. max=largest axis [recommended], "
            "mean=average size, "
            "norm=L2 norm)"
        )
    )

    # ======================================================
    # SPATIAL
    # ======================================================

    parser.add_argument(
        "--spatial-k",
        type=int,
        default=8,
        metavar="INT",
        help=(
            "k-th nearest neighbor used to estimate typical "
            "scene spacing before computing adaptive radius "
            "(e.g. 5–8 recommended, "
            "10+ smoother)"
        )
    )

    parser.add_argument(
        "--spatial-factor",
        type=float,
        default=5.0,
        metavar="FLOAT",
        help=(
            "Multiplier applied to estimated scene spacing "
            "to define neighbor search radius for floater removal "
            "(e.g. 3–4 aggressive, "
            "5 recommended, "
            "8 conservative)"
        )
    )

    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=0,
        metavar="INT",
        help=(
            "Remove splats with fewer than this number of "
            "neighbors inside adaptive radius to suppress "
            "isolated floaters "
            "(e.g. 2–3 conservative, "
            "4–5 recommended, "
            "8+ aggressive)"
        )
    )

    # ======================================================
    # MISC
    # ======================================================

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run filters and print statistics without writing "
            "output file (e.g. parameter tuning)"
        )
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(
        f"\nLoading PLY:\n"
        f"{input_path}"
    )

    ply = PlyData.read(
        input_path
    )

    if "vertex" not in ply:
        raise RuntimeError(
            "No vertex element found"
        )

    vertices = ply["vertex"].data

    total = len(vertices)

    keep_mask = np.ones(
        total,
        dtype=bool
    )

    removed_opacity = 0
    removed_scale = 0
    removed_spatial = 0

    # ======================================================
    # OPACITY FILTER
    # ======================================================

    if args.min_opacity is not None and args.min_opacity > 0:

        keep_opacity, opacity = (
            opacity_filter(
                vertices,
                args.min_opacity
            )
        )

        removed_opacity = np.sum(
            ~keep_opacity
        )

        keep_mask &= keep_opacity

        print(
            "\n[Opacity filter]"
        )
        print(
            f"min opacity : "
            f"{args.min_opacity}"
        )
        print(
            f"removed     : "
            f"{removed_opacity:,}"
        )

    # ======================================================
    # SCALE FILTER
    # ======================================================

    if (
      args.max_scale_percentile is not None
      and args.max_scale_percentile > 0
    ):

        (
            keep_scale,
            scale_metric,
            threshold
        ) = scale_filter(
            vertices,
            percentile=args.max_scale_percentile,
            metric=args.scale_metric
        )

        removed_scale = np.sum(
            ~keep_scale
        )

        keep_mask &= keep_scale

        print(
            "\n[Scale filter]"
        )
        print(
            f"metric      : "
            f"{args.scale_metric}"
        )
        print(
            f"percentile  : "
            f"{args.max_scale_percentile}"
        )
        print(
            f"threshold   : "
            f"{threshold:.6f}"
        )
        print(
            f"removed     : "
            f"{removed_scale:,}"
        )

    # ======================================================
    # SPATIAL FILTER
    # ======================================================

    if (
        args.min_neighbors > 0
        and args.spatial_k > 0
        and args.spatial_factor > 0
    ):

        (
            keep_spatial,
            counts,
            radius,
            median_distance
        ) = spatial_filter(
            vertices,
            spatial_k=args.spatial_k,
            spatial_factor=args.spatial_factor,
            min_neighbors=args.min_neighbors
        )

        removed_spatial = np.sum(
            ~keep_spatial
        )

        keep_mask &= keep_spatial

        print(
            "\n[Spatial filter]"
        )
        print(
            f"spatial_k        : "
            f"{args.spatial_k}"
        )
        print(
            f"median kNN dist  : "
            f"{median_distance:.6f}"
        )
        print(
            f"auto radius      : "
            f"{radius:.6f}"
        )
        print(
            f"spatial factor   : "
            f"{args.spatial_factor}"
        )
        print(
            f"min neighbors    : "
            f"{args.min_neighbors}"
        )
        print(
            f"removed          : "
            f"{removed_spatial:,}"
        )

    # ======================================================
    # SUMMARY
    # ======================================================

    kept = np.sum(
        keep_mask
    )

    removed = (
        total - kept
    )

    print(
        "\n========== SUMMARY =========="
    )

    print(
        f"Total splats     : "
        f"{total:,}"
    )

    print(
        f"Final kept       : "
        f"{kept:,}"
    )

    print(
        f"Final removed    : "
        f"{removed:,}"
    )

    print(
        f"Retention        : "
        f"{100 * kept / total:.2f}%"
    )

    if args.dry_run:
        print(
            "\nDry run enabled."
        )
        return

    # ======================================================
    # SAVE
    # ======================================================

    filtered_vertices = (
        vertices[
            keep_mask
        ]
    )

    vertex_element = (
        PlyElement.describe(
            filtered_vertices,
            "vertex"
        )
    )

    PlyData(
        [vertex_element],
        text=False
    ).write(output_path)

    print(
        f"\nSaved:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
