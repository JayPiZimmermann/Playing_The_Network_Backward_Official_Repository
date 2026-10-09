"""
ImageNet-S dataset loader.

Loads ImageNet validation images with segmentation masks from ImageNet-S.
Segmentation masks provide ground-truth object boundaries for evaluating
gradient-based explanations.

Setup:
    bash data/download_imagenet_s.sh      # download annotations
    # then place/symlink ImageNet val images at data/imagenet/val/
"""

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# ImageNet class index → human-readable name (top-level only, loaded lazily)
_IMAGENET_CLASSES = None


def _get_imagenet_classes():
    """Load ImageNet 1000-class names from torchvision."""
    global _IMAGENET_CLASSES
    if _IMAGENET_CLASSES is None:
        from torchvision.models import ResNet18_Weights
        _IMAGENET_CLASSES = ResNet18_Weights.DEFAULT.meta["categories"]
    return _IMAGENET_CLASSES


# Mapping from WordNet IDs to ImageNet class indices
_WNID_TO_IDX = None


def _get_wnid_to_idx():
    """Build wnid → class_index mapping from bundled imagenet_class_index.json."""
    global _WNID_TO_IDX
    if _WNID_TO_IDX is None:
        import json
        json_path = Path(__file__).parent / "imagenet_class_index.json"
        with open(json_path) as f:
            data = json.load(f)
        _WNID_TO_IDX = {v[0]: int(k) for k, v in data.items()}
    return _WNID_TO_IDX


class ImageNetS(Dataset):
    """
    ImageNet-S: ImageNet validation images with segmentation masks.

    Each sample returns (image, mask, label, class_name, image_name) where:
    - image: (3, 224, 224) normalized tensor
    - mask: (224, 224) binary numpy array (True = object)
    - label: ImageNet class index (int, 0-999)
    - class_name: human-readable class name
    - image_name: e.g. 'ILSVRC2012_val_00049441.JPEG'

    Sample pool for ImageNet-S50 (the split used in the paper):
    The release ships 752 validation masks in 50 class folders; all 752
    images are ILSVRC2012 validation images.  For 136 of them the
    ImageNet-S class differs from the official ILSVRC2012 label (near
    synonyms such as purse/wallet or ladybug/leaf beetle).  With
    ``label_agreement=True`` (default, the paper's pool) only the 616
    images whose ImageNet-S class equals the ILSVRC2012 label are kept, so
    the explained class is both the ground-truth label and the segmented
    object.  The labels come from ``imagenet_s50_val_ilsvrc_labels.csv``
    (derived from the ILSVRC2012 devkit), so the pool does not depend on
    the on-disk layout.  The pool is split into 50 val (1 per class via
    ``build_val_split``, seed=42) and 566 test images, of which 565 have the
    object visible in the evaluated centre crop (``reproduce.common.build_split``).
    """

    # Standard ImageNet normalization
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        imagenet_root: Optional[str] = None,
        imagenet_s_root: Optional[str] = None,
        split: str = "ImageNetS50",
        image_size: int = 224,
        label_agreement: bool = True,
    ):
        data_dir = Path(__file__).parent

        if imagenet_root is None:
            imagenet_root = str(data_dir / "imagenet")
        if imagenet_s_root is None:
            imagenet_s_root = str(data_dir / "imagenet-s")

        self.imagenet_root = Path(imagenet_root)
        self.imagenet_s_root = Path(imagenet_s_root)
        self.split = split
        self.image_size = image_size
        self.label_agreement = label_agreement
        self.ilsvrc_labels = self._load_ilsvrc_labels(data_dir) if split == "ImageNetS50" else {}

        self.seg_root = self.imagenet_s_root / split / "validation-segmentation"
        if not self.seg_root.exists():
            raise FileNotFoundError(
                f"Segmentation masks not found at {self.seg_root}\n"
                f"Run: bash data/download_imagenet_s.sh"
            )

        resize_size = max(256, image_size)
        self.resize_size = resize_size
        self.transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

        self.samples = self._build_samples()
        if len(self.samples) == 0:
            val_path = self.imagenet_root / "val"
            raise FileNotFoundError(
                f"No matching image+mask pairs found.\n"
                f"ImageNet val images expected at: {val_path}\n"
                f"Segmentation masks at: {self.seg_root}"
            )

    @staticmethod
    def _load_ilsvrc_labels(data_dir: Path) -> dict:
        """Image stem -> official ILSVRC2012 wnid for the ImageNet-S50 images."""
        import csv
        path = data_dir / "imagenet_s50_val_ilsvrc_labels.csv"
        with open(path, newline="") as f:
            return {r["image"]: r["ilsvrc2012_wnid"] for r in csv.DictReader(f)}

    def _build_samples(self) -> List[dict]:
        samples = []
        for class_dir in sorted(self.seg_root.iterdir()):
            if not class_dir.is_dir():
                continue
            wnid = class_dir.name  # e.g. 'n02747177'
            for mask_file in sorted(class_dir.glob("*.png")):
                ilsvrc_wnid = self.ilsvrc_labels.get(mask_file.stem, wnid)
                if self.label_agreement and ilsvrc_wnid != wnid:
                    continue
                img_name = mask_file.stem + ".JPEG"
                # Flat val/ layout first, then class-organized (val/<ILSVRC wnid>/)
                for img_path in (self.imagenet_root / "val" / img_name,
                                 self.imagenet_root / "val" / ilsvrc_wnid / img_name,
                                 self.imagenet_root / "val" / wnid / img_name):
                    if img_path.exists():
                        samples.append({
                            "image_path": str(img_path),
                            "mask_path": str(mask_file),
                            "wnid": wnid,
                            "image_name": img_name,
                        })
                        break
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def get_label(self, idx: int) -> int:
        """Fast label lookup without decoding the image (used by stratified split builders)."""
        wnid = self.samples[idx]["wnid"]
        return _get_wnid_to_idx()[wnid]

    def _crop_mask(self, mask_path: str) -> np.ndarray:
        """Binary object mask after the same Resize + CenterCrop as the image."""
        binary = (np.array(Image.open(mask_path)) > 0).astype(np.uint8)
        mask_pil = Image.fromarray(binary * 255)
        mask_pil = transforms.Resize(self.resize_size, interpolation=transforms.InterpolationMode.NEAREST)(mask_pil)
        mask_pil = transforms.CenterCrop(self.image_size)(mask_pil)
        return np.array(mask_pil) > 0

    def mask_visible(self, idx: int) -> bool:
        """True if the annotated object is visible in the evaluated crop."""
        return bool(self._crop_mask(self.samples[idx]["mask_path"]).any())

    def __getitem__(self, idx: int) -> Tuple:
        s = self.samples[idx]

        # Image
        image = Image.open(s["image_path"]).convert("RGB")
        image_t = self.transform(image)

        # Mask — resize+crop to match the image transform
        mask = self._crop_mask(s["mask_path"])

        # Class info
        wnid = s["wnid"]
        wnid_to_idx = _get_wnid_to_idx()
        label = wnid_to_idx[wnid]
        classes = _get_imagenet_classes()
        class_name = classes[label]

        return image_t, mask, label, class_name, s["image_name"]

    def denormalize(self, image_t: torch.Tensor) -> torch.Tensor:
        """Convert normalized tensor back to [0, 1] for display."""
        mean = torch.tensor(self.MEAN, device=image_t.device).view(3, 1, 1)
        std = torch.tensor(self.STD, device=image_t.device).view(3, 1, 1)
        return torch.clamp(image_t * std + mean, 0, 1)
