"""The frozen organizer BBox and E2 image preprocessing contract."""

from __future__ import annotations

import math

from PIL import Image, ImageOps
from torchvision.transforms import functional as TF


VERSION = "organizer-bbox-rgb-max448-lanczos-pad-bicubic-imagenet-v1"


def crop_bbox(image: Image.Image, bbox: tuple[int, int, int, int], max_side: int = 448) -> Image.Image:
    if len(bbox) != 4 or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value != int(value) for value in bbox
    ):
        raise ValueError("BBox must contain four finite integer xywh values")
    x, y, width, height = map(int, bbox)
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > image.width or y + height > image.height:
        raise ValueError("BBox must be nonempty and inside the image")
    crop = image.convert("RGB").crop((x, y, x + width, y + height))
    crop.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return crop


def tensor_from_crop(image: Image.Image, size: int = 336):
    image = ImageOps.pad(
        image.convert("RGB"), (size, size),
        method=Image.Resampling.BICUBIC, color=(123, 116, 103),
    )
    return TF.normalize(TF.to_tensor(image), [.485, .456, .406], [.229, .224, .225])
