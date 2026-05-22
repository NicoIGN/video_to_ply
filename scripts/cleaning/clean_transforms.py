#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Remove missing frames from a Nerfstudio transforms.json"
    )

    parser.add_argument(
        "--transforms",
        required=True,
        help="Path to transforms.json",
    )

    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing images",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output transforms path (default: overwrite input)",
    )

    args = parser.parse_args()

    transforms_path = Path(args.transforms).resolve()
    image_dir = Path(args.image_dir).resolve()

    if not transforms_path.exists():
        raise FileNotFoundError(f"Transforms file not found: {transforms_path}")

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    with open(transforms_path, "r") as f:
        transforms = json.load(f)

    frames = transforms.get("frames", [])
    valid_frames = []
    missing_frames = []

    for frame in frames:
        file_path = frame["file_path"]

        # ex: "./images/frame_00021.png" -> "frame_00021.png"
        filename = Path(file_path).name

        image_path = image_dir / filename

        if image_path.exists():
            valid_frames.append(frame)
            print(f"OK       {filename}")
        else:
            missing_frames.append(filename)
            print(f"MISSING  {filename}")

    transforms["frames"] = valid_frames

    output_path = (
        Path(args.output).resolve()
        if args.output
        else transforms_path
    )

    with open(output_path, "w") as f:
        json.dump(transforms, f, indent=4)

    print()
    print("====================================")
    print(f"Input frames   : {len(frames)}")
    print(f"Valid frames   : {len(valid_frames)}")
    print(f"Missing frames : {len(missing_frames)}")
    print(f"Written to     : {output_path}")
    print("====================================")


if __name__ == "__main__":
    main()
