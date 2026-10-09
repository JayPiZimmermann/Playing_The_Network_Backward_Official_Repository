#!/usr/bin/env python3
"""
Download ImageNet validation images that have matching ImageNet-S masks.

Only extracts the ~752 images needed (not the full 50k validation set).

Usage:
    # Extract from ILSVRC2012 tar (recommended)
    python data/download_imagenet_val.py /path/to/ILSVRC2012_img_val.tar

    # Symlink from existing ImageNet val directory
    python data/download_imagenet_val.py --link /path/to/imagenet/val
"""

import argparse
import os
import sys
import tarfile
from pathlib import Path


def get_needed_images(seg_root: Path) -> set:
    """Collect needed image stems from ImageNet-S mask filenames."""
    needed = set()
    for class_dir in seg_root.iterdir():
        if not class_dir.is_dir():
            continue
        for mask_file in class_dir.glob("*.png"):
            needed.add(mask_file.stem)  # e.g. ILSVRC2012_val_00004677
    return needed


def extract_from_tar(tar_path: str, needed: set, out_dir: Path):
    """Extract only needed images from ILSVRC2012_img_val.tar."""
    existing = {p.stem for p in out_dir.glob("*.JPEG")}
    to_extract = needed - existing

    if not to_extract:
        print("All images already present!")
        return

    print(f"Extracting {len(to_extract)} images from {tar_path}...")

    extracted = 0
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar:
            stem = Path(member.name).stem
            if stem in to_extract:
                # Extract directly to out_dir with just the filename
                member.name = Path(member.name).name
                tar.extract(member, path=str(out_dir))
                extracted += 1
                if extracted % 100 == 0:
                    print(f"  Extracted {extracted}/{len(to_extract)}...")
                if extracted >= len(to_extract):
                    break

    print(f"Done! Extracted {extracted} images to {out_dir}")


def link_from_dir(src_dir: str, needed: set, out_dir: Path):
    """Create symlinks from an existing ImageNet val directory."""
    src = Path(src_dir)
    if not src.exists():
        print(f"Source directory not found: {src}")
        sys.exit(1)

    existing = {p.stem for p in out_dir.glob("*.JPEG")}
    to_link = needed - existing

    if not to_link:
        print("All images already present!")
        return

    linked = 0
    missing = 0
    for stem in sorted(to_link):
        fname = f"{stem}.JPEG"
        # Try flat layout
        src_path = src / fname
        if not src_path.exists():
            # Try class-organized layout (val/nXXXXXXXX/ILSVRC*.JPEG)
            matches = list(src.glob(f"*/{fname}"))
            if matches:
                src_path = matches[0]
            else:
                missing += 1
                continue

        dst_path = out_dir / fname
        dst_path.symlink_to(src_path.resolve())
        linked += 1

    print(f"Linked {linked} images to {out_dir}")
    if missing:
        print(f"Warning: {missing} images not found in source directory")


def main():
    parser = argparse.ArgumentParser(
        description="Get ImageNet val images matching ImageNet-S masks"
    )
    parser.add_argument(
        "tar_path", nargs="?", default=None,
        help="Path to ILSVRC2012_img_val.tar"
    )
    parser.add_argument(
        "--link", metavar="DIR",
        help="Symlink from an existing ImageNet val/ directory"
    )
    args = parser.parse_args()

    if args.tar_path is None and args.link is None:
        parser.print_help()
        print("\nExamples:")
        print("  python data/download_imagenet_val.py /path/to/ILSVRC2012_img_val.tar")
        print("  python data/download_imagenet_val.py --link /path/to/imagenet/val")
        sys.exit(1)

    data_dir = Path(__file__).parent
    seg_root = data_dir / "imagenet-s" / "ImageNetS50" / "validation-segmentation"
    out_dir = data_dir / "imagenet" / "val"

    if not seg_root.exists():
        print(f"Segmentation masks not found at {seg_root}")
        print("Run: bash data/download_imagenet_s.sh")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    needed = get_needed_images(seg_root)
    print(f"Need {len(needed)} validation images matching segmentation masks")

    if args.link:
        link_from_dir(args.link, needed, out_dir)
    else:
        tar_path = args.tar_path
        if not os.path.exists(tar_path):
            print(f"Tar file not found: {tar_path}")
            sys.exit(1)
        extract_from_tar(tar_path, needed, out_dir)


if __name__ == "__main__":
    main()
