#!/usr/bin/env bash
# Download Pascal VOC 2012 trainval (segmentation set) into data/VOCdevkit/.
#
# After this script finishes you will have:
#   data/VOCdevkit/VOC2012/JPEGImages/             (~17,125 .jpg)
#   data/VOCdevkit/VOC2012/SegmentationClass/      (~2,913 .png masks)
#   data/VOCdevkit/VOC2012/ImageSets/Segmentation/{train,val,trainval}.txt
#
# That is sufficient for the 2913-image segmentation set used by
# data/pascal_voc.py.
set -euo pipefail

# data/voc/VOCdevkit/VOC2012 matches torchvision.datasets.VOCSegmentation's
# expected layout.
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/voc"
TARGET="$DATA_DIR/VOCdevkit/VOC2012"
mkdir -p "$DATA_DIR"

if [ -d "$TARGET/SegmentationClass" ]; then
    echo "Pascal VOC 2012 already present at $TARGET — skipping download."
    exit 0
fi

cd "$DATA_DIR"

# The official host (host.robots.ox.ac.uk) is intermittently up; fall back
# to known mirrors in order.
URLS=(
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    "http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar"
    "https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar"
)

TARBALL="VOCtrainval_11-May-2012.tar"
if [ ! -f "$TARBALL" ]; then
    for url in "${URLS[@]}"; do
        echo "Trying $url ..."
        if curl -fL --connect-timeout 30 --retry 2 -o "$TARBALL" "$url"; then
            break
        fi
        rm -f "$TARBALL"
    done
fi

if [ ! -f "$TARBALL" ]; then
    echo "ERROR: Could not download VOC2012 from any mirror." >&2
    echo "Please download manually and place the .tar in $DATA_DIR." >&2
    exit 1
fi

echo "Extracting $TARBALL ..."
tar -xf "$TARBALL"

if [ ! -d "$TARGET/SegmentationClass" ]; then
    echo "ERROR: extraction did not produce the expected layout under $TARGET" >&2
    exit 1
fi

echo "Done. Pascal VOC 2012 available at $TARGET"
