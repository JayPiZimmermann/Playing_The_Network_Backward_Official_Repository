"""
Pascal VOC 2012 segmentation loader for class-specific attribution evaluation.

Default data root ``data/voc/VOCdevkit/VOC2012`` matches the layout of
``torchvision.datasets.VOCSegmentation`` and what
``data/download_pascal_voc.sh`` extracts to.

The dataset emits ONE sample per (image, foreground-VOC-class) pair, so an
image that contains both a person and a horse generates two samples — one
with the horse mask + an ImageNet target representative of "horse", one
with the person mask if such a target exists (else dropped).  This gives
the framework a class-specific heatmap per (image, class) without changing
any downstream code.

Each sample is a tuple compatible with ``data.imagenet_s.ImageNetS``:

    (image_t, mask, label, class_name, image_name)

* ``image_t``     normalised float tensor  (3, image_size, image_size)
* ``mask``        binary numpy array        (image_size, image_size)
* ``label``       ImageNet class index used as the explanation target
                  (the first/most-representative index in the VOC→ImageNet
                  candidate list for the VOC class)
* ``class_name``  VOC class string ('cat', 'horse', ...)
* ``image_name``  unique identifier  '2008_000067__cat'
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]

# VOC class index (1-20) -> list of matching ImageNet class indices.
# Mapping based on WordNet synset overlap between VOC and ImageNet-1k.
_VOC_TO_INET_LISTS: dict = {
    1: [404],
    2: [444, 671],
    3: list(range(80, 101)),
    4: [472, 554, 625, 814],
    5: [440, 720, 898, 907],
    6: [779, 874, 654],
    7: list(range(407, 417)) + [436, 468, 511, 609, 627, 656, 661, 705, 717, 734, 751, 817, 864],
    8: list(range(281, 294)),
    9: [423, 559, 765, 857],
    10: [345, 346, 347],
    11: [532],
    12: list(range(151, 269)),
    13: list(range(339, 345)),
    14: [670, 665],
    15: [],
    16: [738],
    17: [348, 349, 350],
    18: [831, 832],
    19: [466, 547, 820],
    20: [761, 851, 664],
}

logger = logging.getLogger(__name__)


# Curated single-target picks per VOC class.  The candidate lists above
# are unordered; for attribution we need ONE ImageNet target per (image,
# class) pair, and the obvious choice is the most semantically central
# index (e.g. tvmonitor → 664 'monitor', not 761 'remote control').
VOC_TO_IMAGENET: dict = {
    "aeroplane":   404,  # airliner
    "bicycle":     671,  # mountain bike
    "bird":         80,  # black grouse (first index in bird candidate range 80-100)
    "boat":        472,  # canoe
    "bottle":      898,  # water bottle
    "bus":         779,  # school bus
    "car":         817,  # sports car
    "cat":         281,  # tabby cat
    "chair":       559,  # folding chair
    "cow":         345,  # ox
    "diningtable": 532,  # dining table
    "dog":         207,  # golden retriever
    "horse":       339,  # sorrel
    "motorbike":   670,  # motor scooter
    "person":      None,  # ImageNet has no faithful 'person' class
    "pottedplant": 738,  # pot
    "sheep":       348,  # ram
    "sofa":        831,  # studio couch
    "train":       466,  # bullet train
    "tvmonitor":   664,  # monitor
}

# Sanity: every curated single-target must be inside the candidate list
# for that VOC class (or both list and target empty).
for _name, _target in VOC_TO_IMAGENET.items():
    _voc_idx = VOC_CLASSES.index(_name)
    _cands = _VOC_TO_INET_LISTS.get(_voc_idx, [])
    if _target is None:
        continue
    assert _target in _cands, (
        f"VOC_TO_IMAGENET[{_name!r}] = {_target} is not in the candidate list "
        f"{_cands} — fix the mapping above so the curated index is canonical."
    )


def _representative_target(voc_class_name: str) -> Optional[int]:
    """Single-int explanation target for a VOC class, or ``None`` if unmapped."""
    return VOC_TO_IMAGENET.get(voc_class_name)


# Default data root used by data/download_pascal_voc.sh and
# torchvision.datasets.VOCSegmentation.
DEFAULT_VOC_ROOT = Path(__file__).resolve().parent / "voc" / "VOCdevkit" / "VOC2012"


class PascalVOC2012(Dataset):
    """Pascal VOC 2012 segmentation, one (image, foreground-class) per sample.

    Parameters
    ----------
    voc_root :
        Path to the ``VOCdevkit/VOC2012`` directory (containing
        ``JPEGImages/``, ``SegmentationClass/``, ``ImageSets/Segmentation``).
        Defaults to ``data/voc/VOCdevkit/VOC2012``.
    image_set :
        ``"trainval"`` (default) reads ``trainval.txt`` (~2913 images);
        ``"train"`` and ``"val"`` are also accepted.  Our eval-side val/
        test split is decided downstream by
        ``bwgame.evaluation.dataset.build_val_split`` /
        ``build_test_split``.  The default is the larger ``trainval``
        pool because it covers more (image, class) pairs for tuning.
    image_size :
        Square crop edge — must match the model under evaluation.  We use
        the same ``Resize(max(256, image_size)) -> CenterCrop(image_size)``
        pipeline as ``ImageNetS`` so attributions are evaluated on the
        same resolution.
    min_area :
        Drop (image, class) pairs whose binary mask covers fewer than
        this many pixels on the *original* (pre-resize) annotation
        (default 64).  Filters out sliver objects whose Pointing Game /
        Attribution Localisation are dominated by noise.  The filter is
        applied at construction time using ``_mask_class_areas`` so the
        sample list reflects the post-filter pool used in the paper.
    skip_unmapped :
        If True (default), drop classes whose representative ImageNet
        target is ``None`` (currently only ``person``).  Set to False to
        emit those samples with ``label = -1``.
    download :
        If True and the data root is missing, fetch the official VOC2012
        tarball into the parent directory using
        ``torchvision.datasets.VOCSegmentation`` (~2 GB).
    """

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        voc_root: Optional[str] = None,
        image_set: str = "trainval",
        image_size: int = 224,
        min_area: int = 64,
        skip_unmapped: bool = True,
        download: bool = False,
    ):
        self.voc_root = Path(voc_root) if voc_root else DEFAULT_VOC_ROOT
        self.image_size = image_size
        self.min_area = min_area
        self.skip_unmapped = skip_unmapped

        self.images_dir = self.voc_root / "JPEGImages"
        self.masks_dir = self.voc_root / "SegmentationClass"
        split_file = self.voc_root / "ImageSets" / "Segmentation" / f"{image_set}.txt"

        if not split_file.exists() and download:
            logger.info("Pascal VOC 2012 not present at %s — downloading via torchvision.",
                        self.voc_root)
            self._download_via_torchvision()
        if not split_file.exists():
            raise FileNotFoundError(
                f"VOC split file not found: {split_file}\n"
                f"Either run: bash data/download_pascal_voc.sh\n"
                f"Or instantiate PascalVOC2012(..., download=True)."
            )

        resize_size = max(256, image_size)
        self.resize_size = resize_size
        self.image_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

        with open(split_file) as f:
            stems = [ln.strip() for ln in f if ln.strip()]

        # Enumerate (image, class) pairs by scanning each segmentation mask.
        # Done eagerly so __len__ is exact and DataLoader sharding is safe.
        self.samples: List[dict] = []
        skipped_no_class = 0
        skipped_unmapped = 0
        skipped_small = 0
        for stem in stems:
            mask_path = self.masks_dir / f"{stem}.png"
            img_path = self.images_dir / f"{stem}.jpg"
            if not mask_path.exists() or not img_path.exists():
                continue
            class_pixels = self._mask_class_areas(mask_path)
            if not class_pixels:
                skipped_no_class += 1
                continue
            for cls_idx, npix in class_pixels.items():
                cls_name = VOC_CLASSES[cls_idx]
                target = _representative_target(cls_name)
                if skip_unmapped and target is None:
                    skipped_unmapped += 1
                    continue
                if npix < self.min_area:
                    skipped_small += 1
                    continue
                self.samples.append({
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "voc_class_idx": cls_idx,
                    "voc_class_name": cls_name,
                    "imagenet_target": target,
                    "stem": stem,
                })
        if not self.samples:
            raise FileNotFoundError(
                f"No (image, class) samples found under {self.voc_root}.\n"
                f"Check that download_pascal_voc.sh ran successfully."
            )
        logger.info(
            "PascalVOC2012(%s): %d (image, class) samples "
            "(skipped: %d images with no foreground, %d 'person' / unmapped, %d below min_area)",
            image_set, len(self.samples), skipped_no_class, skipped_unmapped, skipped_small,
        )

    # ------------------------------------------------------------------
    # Download fallback
    # ------------------------------------------------------------------
    def _download_via_torchvision(self) -> None:
        """Populate ``data/voc/VOCdevkit/VOC2012`` via torchvision."""
        from torchvision.datasets import VOCSegmentation
        # torchvision wants the *parent* of VOCdevkit as `root`.
        root_parent = self.voc_root.parent.parent  # .../VOCdevkit/VOC2012 -> .../voc
        root_parent.mkdir(parents=True, exist_ok=True)
        VOCSegmentation(root=str(root_parent), year="2012",
                        image_set="trainval", download=True)

    # ------------------------------------------------------------------
    # mask helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _mask_class_areas(mask_path: Path) -> dict:
        """Return {class_idx: pixel_count} on the *original* mask, ignoring 0 (bg) and 255 (boundary)."""
        m = np.array(Image.open(mask_path))
        unique, counts = np.unique(m, return_counts=True)
        return {int(c): int(n) for c, n in zip(unique, counts) if 0 < int(c) < 255}

    def _crop_class_mask(self, mask_path: str, voc_class_idx: int) -> np.ndarray:
        """Resize+CenterCrop the per-class binary mask to match the image."""
        m = np.array(Image.open(mask_path))
        binary = (m == voc_class_idx).astype(np.uint8) * 255
        pil = Image.fromarray(binary)
        pil = transforms.Resize(self.resize_size, interpolation=transforms.InterpolationMode.NEAREST)(pil)
        pil = transforms.CenterCrop(self.image_size)(pil)
        return np.array(pil) > 0

    def mask_visible(self, idx: int) -> bool:
        """True if the class object is visible in the evaluated crop."""
        s = self.samples[idx]
        return bool(self._crop_class_mask(s["mask_path"], s["voc_class_idx"]).any())

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def get_label(self, idx: int) -> int:
        """Fast label lookup (used by stratified split builders)."""
        target = self.samples[idx]["imagenet_target"]
        return -1 if target is None else int(target)

    def __getitem__(self, idx: int) -> Tuple:
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        image_t = self.image_transform(image)

        mask = self._crop_class_mask(s["mask_path"], s["voc_class_idx"])

        cls_name = s["voc_class_name"]
        target = s["imagenet_target"]
        label = -1 if target is None else int(target)

        image_name = f"{s['stem']}__{cls_name}"
        return image_t, mask, label, cls_name, image_name

    def denormalize(self, image_t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.MEAN, device=image_t.device).view(3, 1, 1)
        std = torch.tensor(self.STD, device=image_t.device).view(3, 1, 1)
        return torch.clamp(image_t * std + mean, 0, 1)
