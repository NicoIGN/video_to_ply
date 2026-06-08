import os
import cv2
import shutil
import numpy as np
import argparse
from glob import glob


def sharpness(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


def diff(a, b):
    return cv2.absdiff(a, b).mean()


def main(tmp_dir, out_dir, num_frames, start_index):

    files = sorted(glob(os.path.join(tmp_dir, "*.png")))

    imgs = []
    sharps = []

    # ======================
    # compute sharpness
    # ======================
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        imgs.append((f, img))
        sharps.append(sharpness(img))

    sharps = np.array(sharps)

    if len(sharps) == 0:
        print("❌ No valid frames")
        return

    # ======================
    # adaptive blur threshold
    # ======================
    blur_threshold = np.percentile(sharps, 30)

    filtered = []

    for (f, img), s in zip(imgs, sharps):
        if s < blur_threshold:
            continue

        filtered.append((f, img))

    print(f"📊 After blur filter: {len(filtered)}")

    # ======================
    # adaptive diff threshold
    # ======================
    diffs = []

    for i in range(1, len(filtered)):
        diffs.append(diff(filtered[i][1], filtered[i - 1][1]))

    if len(diffs) > 0:
        diff_threshold = np.median(diffs) * 0.5
    else:
        diff_threshold = 0

    print(f"📐 Adaptive diff threshold: {diff_threshold}")

    candidates = []
    last = None

    for f, img in filtered:

        if last is not None:
            d = diff(img, last)

            if d < diff_threshold:
                continue

        candidates.append(f)
        last = img

    print(f"📊 Candidates: {len(candidates)}")

    # ======================
    # final sampling
    # ======================
    if len(candidates) > num_frames:
        step = len(candidates) / num_frames
        selected = [
            candidates[int(i * step)]
            for i in range(num_frames)
        ]
    else:
        selected = candidates

    os.makedirs(out_dir, exist_ok=True)

    for i, f in enumerate(selected):

        dst = os.path.join(
            out_dir,
            f"frame_{start_index + i:05d}.png"
        )

        shutil.copy2(f, dst)

    print(f"✅ Selected frames: {len(selected)}")

    if len(selected) > 0:
        print(
            f"🔢 Frame range: "
            f"frame_{start_index:05d}.png -> "
            f"frame_{start_index + len(selected) - 1:05d}.png"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tmp_dir",
        required=True
    )

    parser.add_argument(
        "--out_dir",
        required=True
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        required=True
    )

    parser.add_argument(
        "--start_index",
        type=int,
        default=0
    )

    args = parser.parse_args()

    main(
        args.tmp_dir,
        args.out_dir,
        args.num_frames,
        args.start_index
    )
