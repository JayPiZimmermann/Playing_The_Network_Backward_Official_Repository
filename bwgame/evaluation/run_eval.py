"""Pretrained ImageNet backbones of the paper (torchvision)."""

import torch


def load_model(model_name: str, device: torch.device):
    """Load a pretrained backbone (``vgg16``, ``resnet50`` or ``vit_b_16``).

    The weights are pinned to explicit torchvision weight enums (not
    ``.DEFAULT``) so that a torchvision upgrade cannot silently change the
    checkpoint.

    Returns:
        model on device in eval mode.
    """
    import torchvision.models as models

    model_factory = {
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
        "vgg16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
        "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1),
    }
    if model_name not in model_factory:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_factory)}")
    factory, weights = model_factory[model_name]
    return factory(weights=weights).to(device).eval()
