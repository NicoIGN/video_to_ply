# auto_image_masker.py

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def list_images(folder):
    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Automatic foreground mask generation for image sequences"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input images directory"
    )

    parser.add_argument(
        "--masks",
        required=True,
        help="Output masks directory"
    )

    parser.add_argument(
        "--model",
        default="yolov8x-seg.pt",
        help="YOLO segmentation model"
    )

    args = parser.parse_args()

    os.makedirs(args.masks, exist_ok=True)

    image_files = list_images(args.input)

    if len(image_files) == 0:
        raise RuntimeError(f"No images found in: {args.input}")

    print(f"Found {len(image_files)} images")

    # Load segmentation model
    model = YOLO(args.model)

    for image_name in tqdm(image_files):
        image_path = os.path.join(args.input, image_name)

        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Warning: failed to load {image_path}")
            continue

        h, w = frame.shape[:2]

        # Run segmentation
        results = model(frame, verbose=False)[0]

        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()

            # Select largest segmented object
            areas = [m.sum() for m in masks]
            best_idx = int(np.argmax(areas))

            mask = masks[best_idx]

            # Convert to uint8 mask
            mask_img = (mask * 255).astype(np.uint8)

            # Optional cleanup
            kernel = np.ones((5, 5), np.uint8)
            mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)

        else:
            mask_img = np.zeros((h, w), dtype=np.uint8)

        output_path = os.path.join(args.masks, image_name)

        cv2.imwrite(output_path, mask_img)

    print("Done.")


if __name__ == "__main__":
    main()
