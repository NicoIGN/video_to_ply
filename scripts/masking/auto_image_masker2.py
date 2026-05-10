import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import urllib.request

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

SAM_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


# -----------------------------
# UTILITIES
# -----------------------------

def download_checkpoint(model_type, checkpoint_path):
    if os.path.exists(checkpoint_path):
        return

    if model_type not in SAM_URLS:
        raise ValueError(f"Unknown model_type: {model_type}")

    url = SAM_URLS[model_type]

    print(f"📥 Downloading SAM checkpoint ({model_type})...")
    print(f"   → {url}")
    print(f"   → {checkpoint_path}")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    urllib.request.urlretrieve(url, checkpoint_path)

    print("✅ Download complete")


def list_images(folder):
    if not os.path.isdir(folder):
        raise RuntimeError(f"Input folder not found: {folder}")

    files = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS)]
    files.sort()
    return files


def touches_border(mask, border_ratio):
    h, w = mask.shape
    b = max(1, int(min(h, w) * border_ratio))

    return (
        mask[:b, :].any() or
        mask[-b:, :].any() or
        mask[:, :b].any() or
        mask[:, -b:].any()
    )


def mask_score(mask):
    return mask.sum()


# -----------------------------
# MAIN
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Production SAM pipeline for 3D reconstruction"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--masks", required=True)

    parser.add_argument(
        "--sam_checkpoint",
        default="checkpoints/sam_vit_h.pth",
        help="Path to SAM checkpoint (auto-downloaded if missing)"
    )

    parser.add_argument(
        "--model_type",
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model type"
    )

    parser.add_argument("--min_area_ratio", type=float, default=0.01)
    parser.add_argument("--border_ratio", type=float, default=0.05)
    parser.add_argument("--max_masks", type=int, default=50)

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.masks, exist_ok=True)

    print("🔥 SAM PIPELINE START")
    print(f"📂 input: {args.input}")
    print(f"📂 output: {args.masks}")
    print(f"🧠 model: {args.model_type}")

    # -----------------------------
    # AUTO DOWNLOAD CHECKPOINT
    # -----------------------------
    download_checkpoint(args.model_type, args.sam_checkpoint)

    # -----------------------------
    # LOAD SAM
    # -----------------------------
    print("🧠 loading SAM model...")

    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        min_mask_region_area=200
    )

    print("✅ SAM ready")

    images = list_images(args.input)

    print(f"📦 images found: {len(images)}")

    # -----------------------------
    # LOOP
    # -----------------------------
    for name in tqdm(images):

        path = os.path.join(args.input, name)
        img = cv2.imread(path)

        if img is None:
            print(f"❌ skip: {name}")
            continue

        h, w = img.shape[:2]
        img_area = h * w

        masks = mask_generator.generate(img)

        if args.verbose:
            print(f"\n🖼️ {name} → {len(masks)} masks")

        kept = []

        for m in masks:
            seg = m["segmentation"].astype(np.uint8)

            area_ratio = seg.sum() / img_area
            border = touches_border(seg, args.border_ratio)

            if border and area_ratio < args.min_area_ratio:
                continue

            kept.append(seg)

        if len(kept) == 0:
            final = np.zeros((h, w), dtype=np.uint8)
        else:
            kept = sorted(kept, key=mask_score, reverse=True)[:args.max_masks]
            merged = np.max(np.stack(kept), axis=0)
            final = (merged * 255).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite(os.path.join(args.masks, name), final)

    print("🏁 DONE")


if __name__ == "__main__":
    main()
