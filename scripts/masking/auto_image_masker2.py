import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import urllib.request
import time

import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

SAM_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


# -----------------------------
# DEVICE HELPERS
# -----------------------------
def get_device():
    if torch.cuda.is_available():
        print("⚙️ CUDA detected -> GPU mode ON")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        return torch.device("cuda")
    print("⚙️ CPU mode")
    return torch.device("cpu")


# -----------------------------
# DOWNLOAD CHECKPOINT
# -----------------------------
def download_checkpoint(model_type, path):
    if os.path.exists(path):
        return

    print(f"📥 downloading SAM {model_type}")
    url = SAM_URLS[model_type]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(url, path)

    print("✅ checkpoint ready")


# -----------------------------
# IO
# -----------------------------
def list_images(folder):
    if not os.path.isdir(folder):
        raise RuntimeError(f"Folder not found: {folder}")

    files = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS)]
    files.sort()
    return files


def resize(img, max_size):
    h, w = img.shape[:2]
    s = min(1.0, max_size / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


# -----------------------------
# MASK VALIDATION
# -----------------------------
def is_valid_mask(seg):
    return seg.sum() > 10


# -----------------------------
# MASK GEOMETRY (CPU ONCE)
# -----------------------------
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


# -----------------------------
# INDEXED VIS
# -----------------------------
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
        vis[seg_img == i] = rng.integers(0, 255, 3, dtype=np.uint8)

    return vis


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--sam_checkpoint", default="checkpoints/sam.pth")
    parser.add_argument("--model_type", default="vit_b")

    parser.add_argument("--max_size", type=int, default=1024)
    parser.add_argument("--margin_ratio", type=float, default=0.30)
    parser.add_argument("--points_per_side", type=int, default=8)

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--override", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # -----------------------------
    # DEVICE
    # -----------------------------
    device = get_device()

    # -----------------------------
    # CHECKPOINT
    # -----------------------------
    download_checkpoint(args.model_type, args.sam_checkpoint)

    # -----------------------------
    # LOAD SAM
    # -----------------------------
    print("🧠 loading SAM...")

    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
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

    # -----------------------------
    # LOOP
    # -----------------------------
    for name in tqdm(images):

        out_path = os.path.join(args.outdir, name)
        vis_path = os.path.join(args.outdir, "seg_" + name)

        if not args.override and os.path.exists(out_path):
            continue

        img = cv2.imread(os.path.join(args.input, name))
        if img is None:
            continue

        img = resize(img, args.max_size)
        h, w = img.shape[:2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # -----------------------------
        # SAM INFERENCE (GPU if available)
        # -----------------------------
        with torch.inference_mode():
            masks = mask_generator.generate(img)

        center_mask = build_center_mask(h, w, args.margin_ratio)
        border_mask = build_border_zones(h, w, 0.05)

        # -----------------------------
        # GLOBAL SAM COVERAGE (NEW FIX)
        # -----------------------------
        sam_union = np.zeros((h, w), dtype=bool)

        for m in masks:
            sam_union |= m["segmentation"].astype(bool)

        zero_center = center_mask & (~sam_union)

        kept = []

        if args.verbose:
            print(f"\n🖼️ {name} | raw masks={len(masks)}")

        # -----------------------------
        # FILTERING
        # -----------------------------
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
                keep = (center_ratio > 0.5) or (border_density < 0.2)

            if keep:
                kept.append(seg)

        # -----------------------------
        # MERGE
        # -----------------------------
        final = np.zeros((h, w), dtype=bool)

        for k in kept:
            final |= k

        # 🔥 ADD MISSING CENTER PIXELS (FIX)
        final |= zero_center

        final = final.astype(np.uint8) * 255

        kernel = np.ones((5, 5), np.uint8)
        final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite(out_path, final)

        # -----------------------------
        # DEBUG VIS
        # -----------------------------
        if args.verbose:
            seg_img = build_indexed(masks, h, w)
            vis = colorize_indexed(seg_img)
            cv2.imwrite(vis_path, vis)

    print("🏁 DONE")


if __name__ == "__main__":
    main()
