from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from PIL import Image,ImageDraw
from torchvision.transforms.functional import to_pil_image

from adetailer import PredictOutput
from adetailer.common import create_mask_from_bbox
import numpy as np
if TYPE_CHECKING:
    import torch
    from ultralytics import YOLO, YOLOWorld


def bbox_inpaint(
    bboxes,
    image: Image.Image,

) -> PredictOutput:

    bboxes = bboxes

    masks = create_mask_from_bbox(bboxes, image.size)

    preview = image.copy()

    # Draw rectangles on the preview image
    draw = ImageDraw.Draw(preview)
    for bbox in bboxes:
        # Convert bounding box coordinates to integers (adjust if needed)
        int_bbox = [int(x) for x in bbox]
        draw.rectangle(int_bbox, outline=(0, 255, 0), width=2)  # Green rectangles


    return PredictOutput(bboxes=bboxes, masks=masks, preview=preview)

