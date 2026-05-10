import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser("Auto video mask generator")

    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--frames", required=True, help="Output frames folder")
    parser.add_argument("--masks", required=True, help="Output masks folder")

    args = parser.parse_args()

    os.makedirs(args.frames, exist_ok=True)
    os.makedirs(args.masks, exist_ok=True)

    # Load segmentation model
    model = YOLO("yolov8x-seg.pt")

    cap = cv2.VideoCapture(args.input)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0

    print(f"Processing {total} frames...")

    for _ in tqdm(range(total)):
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # save frame
        frame_path = os.path.join(args.frames, f"{frame_id:05d}.png")
        cv2.imwrite(frame_path, frame)

        # inference
        results = model(frame, verbose=False)[0]

        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()  # (N, H, W)

            # choose largest object (foreground heuristic)
            areas = [m.sum() for m in masks]
            best_idx = int(np.argmax(areas))

            mask = masks[best_idx]

            mask_img = (mask * 255).astype(np.uint8)

        else:
            mask_img = np.zeros((h, w), dtype=np.uint8)

        mask_path = os.path.join(args.masks, f"{frame_id:05d}.png")
        cv2.imwrite(mask_path, mask_img)

        frame_id += 1

    cap.release()
    print("Done.")


if __name__ == "__main__":
    main()
