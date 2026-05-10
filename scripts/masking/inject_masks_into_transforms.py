import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser("Inject masks into transforms.json")

    parser.add_argument(
        "--transforms",
        required=True,
        help="Path to transforms.json"
    )

    parser.add_argument(
        "--masks",
        required=True,
        help="Path to masks directory"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output transforms.json (default: overwrite)"
    )

    args = parser.parse_args()

    transforms_path = args.transforms
    masks_dir = args.masks

    output_path = args.output or transforms_path

    with open(transforms_path, "r") as f:
        data = json.load(f)

    frames = data.get("frames", [])

    if len(frames) == 0:
        raise RuntimeError("No frames found in transforms.json")

    print(f"Found {len(frames)} frames")

    # check mask existence helper
    def find_mask(frame_file):
        base = os.path.basename(frame_file)
        return os.path.join("masks", base)

    missing = 0

    for frame in frames:
        file_path = frame.get("file_path", None)

        if file_path is None:
            continue

        mask_path = find_mask(file_path)

        full_mask_path = os.path.join(masks_dir, os.path.basename(mask_path))

        if not os.path.exists(full_mask_path):
            missing += 1
            continue

        frame["mask_path"] = mask_path

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Done. Missing masks: {missing}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
