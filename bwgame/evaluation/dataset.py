"""Dataset utilities for evaluation.

Provides functions to create validation and test splits from ImageNet-S,
including the 1-image-per-class tuning split for hyperparameter selection.
"""

import logging
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

logger = logging.getLogger(__name__)


def _stratified_indices(
    dataset: Dataset,
    n_per_class: int = 1,
    seed: int = 42,
) -> Tuple[List[int], Dict[int, List[int]]]:
    """Select n_per_class samples per class from the dataset.

    The label used for stratification is sample[2] (the int class label of
    the explanation target).  Datasets that expose a ``get_label(idx)``
    method are queried via that fast path; otherwise we fall back to
    ``dataset[idx]`` which decodes the image.

    Returns:
        (selected_indices, class_to_indices_map)
    """
    rng = random.Random(seed)
    class_to_indices: Dict[int, List[int]] = defaultdict(list)

    has_fast = hasattr(dataset, "get_label")
    for idx in range(len(dataset)):
        if has_fast:
            label = int(dataset.get_label(idx))
        else:
            sample = dataset[idx]
            label = sample[2] if isinstance(sample[2], int) else sample[2].item()
        class_to_indices[label].append(idx)

    selected = []
    for label in sorted(class_to_indices.keys()):
        indices = class_to_indices[label]
        chosen = rng.sample(indices, min(n_per_class, len(indices)))
        selected.extend(chosen)

    return selected, dict(class_to_indices)


def build_val_split(
    dataset: Dataset,
    n_per_class: int = 1,
    seed: int = 42,
) -> Subset:
    """Build a validation/tuning split: n_per_class images per class.

    For ImageNet-S50: 50 classes x 1 = 50 images for hyperparameter tuning.

    Args:
        dataset: Full ImageNet-S dataset.
        n_per_class: Number of images per class for the val split.
        seed: Random seed for reproducibility.

    Returns:
        Subset of the dataset for validation/tuning.
    """
    val_indices, class_map = _stratified_indices(dataset, n_per_class, seed)
    logger.info(
        "Val split: %d images from %d classes (n_per_class=%d)",
        len(val_indices), len(class_map), n_per_class,
    )
    return Subset(dataset, val_indices)


def build_test_split(
    dataset: Dataset,
    val_indices: Optional[List[int]] = None,
    n_per_class_val: int = 1,
    seed: int = 42,
) -> Subset:
    """Build a test split: all images NOT in the val split.

    For the ImageNet-S50 pool used in the paper (616 images whose ImageNet-S
    class equals the ILSVRC2012 label, see ``data.imagenet_s.ImageNetS``),
    ``build_val_split(n_per_class=1, seed=42)`` holds out 50 images (one per
    class) and this helper returns the remaining 566.
    ``reproduce.common.build_split`` then keeps the 565 images whose object is
    visible in the evaluated crop.

    Args:
        dataset: Full ImageNet-S dataset.
        val_indices: Pre-computed val indices. If None, computes them.
        n_per_class_val: Used if val_indices is None.
        seed: Random seed.

    Returns:
        Subset of the dataset for testing.
    """
    if val_indices is None:
        val_indices, _ = _stratified_indices(dataset, n_per_class_val, seed)

    val_set = set(val_indices)
    test_indices = [i for i in range(len(dataset)) if i not in val_set]
    logger.info("Test split: %d images (excluded %d val)", len(test_indices), len(val_set))
    return Subset(dataset, test_indices)


def split_voc_trainval(
    dataset,
    val_fraction: float = 1.0 / 6.0,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    """Disjoint per-image random split of a PascalVOC2012 dataset.

    The dataset is expected to expose ``samples`` as a list of dicts with a
    ``stem`` key — i.e. the trainval pool produced by
    ``data.pascal_voc.PascalVOC2012(image_set='trainval')``. The split is
    done at *image* granularity (not per (image, class) row), so all
    foreground-class samples for a given image land in the same subset. The
    test subset receives ``1 - val_fraction`` of the images.

    Default ``val_fraction = 1/6`` gives val:test ≈ 1:5 on VOC2012 trainval
    (roughly 485 val / 2428 test images out of 2913).
    """
    if not hasattr(dataset, "samples"):
        raise TypeError(
            "split_voc_trainval expects a PascalVOC2012 dataset with a "
            "'samples' attribute"
        )
    rng = random.Random(seed)
    stems = sorted({s["stem"] for s in dataset.samples})
    rng.shuffle(stems)
    n_val = max(1, int(round(len(stems) * val_fraction)))
    val_stems = set(stems[:n_val])

    val_indices: List[int] = []
    test_indices: List[int] = []
    for i, s in enumerate(dataset.samples):
        (val_indices if s["stem"] in val_stems else test_indices).append(i)

    logger.info(
        "VOC trainval split (seed=%d, val_fraction=%.4f): "
        "val=%d images / %d (image, class) samples; "
        "test=%d images / %d (image, class) samples",
        seed, val_fraction,
        len(val_stems), len(val_indices),
        len(stems) - len(val_stems), len(test_indices),
    )
    return Subset(dataset, val_indices), Subset(dataset, test_indices)


class EvalCollator:
    """Collator for evaluation that returns batch dicts with all needed fields."""

    def __call__(self, batch):
        images = torch.stack([b[0] for b in batch])
        masks = np.stack([b[1] for b in batch])
        labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
        class_names = [b[3] for b in batch]
        image_names = [b[4] for b in batch]
        return {
            "images": images,
            "masks": masks,
            "labels": labels,
            "class_names": class_names,
            "image_names": image_names,
        }
