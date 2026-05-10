import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import urllib.request

import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

SAM_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


# ---------------------------------------------------
# DEVICE
# ---------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        print("⚙️ CUDA detected -> GPU mode ON")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        return torch.device("cuda")

    print("⚙️ CPU mode")
    return torch.device("cpu")


# ---------------------------------------------------
# CHECKPOINT
# ---------------------------------------------------
def download_checkpoint(model_type, path):
    if os.path.exists(path):
        return

    print(f"📥 downloading SAM {model_type}")

    url = SAM_URLS[model_type]

    os.makedirs(os.path.dirname(path), exist_ok=True)

    urllib.request.urlretrieve(url, path)

    print("✅ checkpoint ready")


# ---------------------------------------------------
# IO
# ---------------------------------------------------
def list_images(folder):
    if not os.path.isdir(folder):
        raise RuntimeError(f"Folder not found: {folder}")

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    files.sort()

    return files


def resize(img, max_size):
    h, w = img.shape[:2]

    s = min(1.0, max_size / max(h, w))

    if s < 1:
        img = cv2.resize(
            img,
            (int(w * s), int(h * s)),
            interpolation=cv2.INTER_AREA
        )

    return img


# ---------------------------------------------------
# MASK VALIDATION
# ---------------------------------------------------
def is_valid_mask(seg):
    return seg.sum() > 10


# ---------------------------------------------------
# GEOMETRY
# ---------------------------------------------------
def build_center_mask(h, w, margin):
    m = int(min(h, w) * margin)

    mask = np.zeros((h, w), dtype=bool)
    mask[m:h - m, m:w - m] = True

    return mask


def build_border_zones(h, w, margin_ratio):
    m = int(min(h, w) * margin_ratio)

    border = np.zeros((h, w), dtype=bool)

    border[:m, :] = True
    border[-m:, :] = True
    border[:, :m] = True
    border[:, -m:] = True

    return border


# ---------------------------------------------------
# INDEXED VIS
# ---------------------------------------------------
def build_indexed(masks, h, w):
    seg_img = np.zeros((h, w), dtype=np.int32)

    for i, m in enumerate(masks, start=1):
        seg_img[m["segmentation"].astype(bool)] = i

    return seg_img


def colorize_indexed(seg_img):
    vis = np.zeros((*seg_img.shape, 3), dtype=np.uint8)

    ids = np.unique(seg_img)

    rng = np.random.default_rng(0)

    for i in ids:
        if i == 0:
            continue

        vis[seg_img == i] = rng.integers(
            0,
            255,
            3,
            dtype=np.uint8
        )

    return vis


# ---------------------------------------------------
# DOWNSCALE HELPERS
# ---------------------------------------------------
def maybe_generate_downscaled_mask(
    mask_path,
    images2_dir,
    masks2_dir,
    name,
    override=False
):
    """
    Generate masks_2/<name>
    from masks/<name>
    using images_2 resolution.
    """

    if not os.path.isdir(images2_dir):
        return

    image2_path = os.path.join(images2_dir, name)

    if not os.path.exists(image2_path):
        return

    os.makedirs(masks2_dir, exist_ok=True)

    out2_path = os.path.join(masks2_dir, name)

    if os.path.exists(out2_path) and not override:
        return

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return

    img2 = cv2.imread(image2_path)

    if img2 is None:
        return

    h2, w2 = img2.shape[:2]

    mask2 = cv2.resize(
        mask,
        (w2, h2),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imwrite(out2_path, mask2)

    print(f"🧩 generated masks_2/{name}")


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument(
        "--sam_checkpoint",
        default="checkpoints/sam.pth"
    )

    parser.add_argument(
        "--model_type",
        default="vit_b"
    )

    parser.add_argument(
        "--max_size",
        type=int,
        default=1024
    )

    parser.add_argument(
        "--margin_ratio",
        type=float,
        default=0.30
    )

    parser.add_argument(
        "--points_per_side",
        type=int,
        default=8
    )

    parser.add_argument(
        "--verbose",
        action="store_true"
    )

    parser.add_argument(
        "--override",
        action="store_true"
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---------------------------------------------------
    # images_2 / masks_2 detection
    # ---------------------------------------------------
    parent_dir = os.path.dirname(args.input)

    images2_dir = os.path.join(parent_dir, "images_2")
    masks2_dir = os.path.join(parent_dir, "masks_2")

    has_images2 = os.path.isdir(images2_dir)

    if has_images2:
        print(f"🧩 detected images_2: {images2_dir}")

    # ---------------------------------------------------
    # DEVICE
    # ---------------------------------------------------
    device = get_device()

    # ---------------------------------------------------
    # CHECKPOINT
    # ---------------------------------------------------
    download_checkpoint(
        args.model_type,
        args.sam_checkpoint
    )

    # ---------------------------------------------------
    # LOAD SAM
    # ---------------------------------------------------
    print("🧠 loading SAM...")

    sam = sam_model_registry[args.model_type](
        checkpoint=args.sam_checkpoint
    )

    sam.to(device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.88,
        crop_n_layers=0,
        min_mask_region_area=200
    )

    images = list_images(args.input)

    print(f"📦 images: {len(images)}")

    # ---------------------------------------------------
    # LOOP
    # ---------------------------------------------------
    for name in tqdm(images):

        out_path = os.path.join(args.outdir, name)
        vis_path = os.path.join(args.outdir, "seg_" + name)

        # ---------------------------------------------------
        # EXISTING MASK
        # ---------------------------------------------------
        mask_exists = os.path.exists(out_path)

        # ---------------------------------------------------
        # SKIP REGEN IF EXISTS
        # ---------------------------------------------------
        if mask_exists and not args.override:

            if has_images2:
                maybe_generate_downscaled_mask(
                    mask_path=out_path,
                    images2_dir=images2_dir,
                    masks2_dir=masks2_dir,
                    name=name,
                    override=args.override
                )

            continue

        # ---------------------------------------------------
        # LOAD IMAGE
        # ---------------------------------------------------
        img_path = os.path.join(args.input, name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = resize(img, args.max_size)

        h, w = img.shape[:2]

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # ---------------------------------------------------
        # SAM INFERENCE
        # ---------------------------------------------------
        with torch.inference_mode():
            masks = mask_generator.generate(img)

        center_mask = build_center_mask(
            h,
            w,
            args.margin_ratio
        )

        border_mask = build_border_zones(
            h,
            w,
            0.05
        )

        # ---------------------------------------------------
        # GLOBAL SAM COVERAGE
        # ---------------------------------------------------
        sam_union = np.zeros((h, w), dtype=bool)

        for m in masks:
            sam_union |= m["segmentation"].astype(bool)

        zero_center = center_mask & (~sam_union)

        kept = []

        if args.verbose:
            print(f"\n🖼️ {name} | raw masks={len(masks)}")

        # ---------------------------------------------------
        # FILTERING
        # ---------------------------------------------------
        for m in masks:

            seg = m["segmentation"].astype(bool)

            if not is_valid_mask(seg):
                continue

            touches_border = np.any(seg & border_mask)

            center_pixels = np.sum(seg & center_mask)

            area = seg.sum()

            center_ratio = center_pixels / (area + 1e-6)

            border_density = np.sum(seg & border_mask) / (area + 1e-6)

            if not touches_border:
                keep = True
            else:
                keep = (
                    (center_ratio > 0.5)
                    or
                    (border_density < 0.2)
                )

            if keep:
                kept.append(seg)

        # ---------------------------------------------------
        # MERGE
        # ---------------------------------------------------
        final = np.zeros((h, w), dtype=bool)

        for k in kept:
            final |= k

        # add missing center pixels
        final |= zero_center

        final = final.astype(np.uint8) * 255

        kernel = np.ones((5, 5), np.uint8)

        final = cv2.morphologyEx(
            final,
            cv2.MORPH_CLOSE,
            kernel
        )

        cv2.imwrite(out_path, final)

        # ---------------------------------------------------
        # GENERATE masks_2
        # ---------------------------------------------------
        if has_images2:
            maybe_generate_downscaled_mask(
                mask_path=out_path,
                images2_dir=images2_dir,
                masks2_dir=masks2_dir,
                name=name,
                override=True
            )

        # ---------------------------------------------------
        # DEBUG VIS
        # ---------------------------------------------------
        if args.verbose:
            seg_img = build_indexed(masks, h, w)

            vis = colorize_indexed(seg_img)

            cv2.imwrite(vis_path, vis)

    print("🏁 DONE")


if __name__ == "__main__":
    main()
