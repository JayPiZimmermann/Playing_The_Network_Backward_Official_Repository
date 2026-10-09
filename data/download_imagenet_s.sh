#!/bin/bash
# Download ImageNet-S segmentation annotations.
#
# Usage:
#   bash data/download_imagenet_s.sh          # downloads ImageNetS50 (smallest)
#   bash data/download_imagenet_s.sh 300      # downloads ImageNetS300
#   bash data/download_imagenet_s.sh all      # downloads all splits
#
# After downloading, you also need ImageNet validation images.
# Place or symlink them at:  data/imagenet/val/
#
# Expected final layout:
#   data/
#     imagenet/val/ILSVRC2012_val_*.JPEG      (50k images)
#     imagenet-s/ImageNetS50/validation-segmentation/n*/*.png
#     imagenet-s/ImageNetS300/validation-segmentation/n*/*.png   (optional)
#     imagenet-s/ImageNetS919/validation-segmentation/n*/*.png   (optional)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="$SCRIPT_DIR/imagenet-s"
SPLIT="${1:-50}"

mkdir -p "$DEST"

download() {
    local name=$1
    local url=$2
    local zip="$DEST/${name}.zip"

    if [ -d "$DEST/$name" ]; then
        echo "  $name already exists, skipping."
        return
    fi

    echo "  Downloading $name ..."
    wget -q --show-progress -O "$zip" "$url"
    echo "  Extracting ..."
    unzip -q -o "$zip" -d "$DEST"
    rm "$zip"
    echo "  Done: $DEST/$name"
}

echo "ImageNet-S annotation download (split=$SPLIT)"
echo "Destination: $DEST"
echo ""

case "$SPLIT" in
    50)
        download "ImageNetS50" \
            "https://github.com/LUSSeg/ImageNet-S/releases/download/ImageNet-S/ImageNetS50-a0fe9d82231f9bc4787ee76e304dfa51.zip"
        ;;
    300)
        download "ImageNetS300" \
            "https://github.com/LUSSeg/ImageNet-S/releases/download/ImageNet-S/ImageNetS300-dd8e42e156bac415a35c3c239de297bf.zip"
        ;;
    919)
        download "ImageNetS919" \
            "https://github.com/LUSSeg/ImageNet-S/releases/download/ImageNet-S/ImageNetS919-5f7f58ae1003d21da9409a8576bf7680.zip"
        ;;
    all)
        download "ImageNetS50" \
            "https://github.com/LUSSeg/ImageNet-S/releases/download/ImageNet-S/ImageNetS50-a0fe9d82231f9bc4787ee76e304dfa51.zip"
        download "ImageNetS300" \
            "https://github.com/LUSSeg/ImageNet-S/releases/download/ImageNet-S/ImageNetS300-dd8e42e156bac415a35c3c239de297bf.zip"
        download "ImageNetS919" \
            "https://github.com/LUSSeg/ImageNet-S/releases/download/ImageNet-S/ImageNetS919-5f7f58ae1003d21da9409a8576bf7680.zip"
        ;;
    *)
        echo "Unknown split: $SPLIT  (use 50, 300, 919, or all)"
        exit 1
        ;;
esac

echo ""
echo "Next step:"
echo "  Place ImageNet validation images at: $SCRIPT_DIR/imagenet/val/"
echo "  (or symlink:  ln -s /path/to/ILSVRC2012/val $SCRIPT_DIR/imagenet/val)"
