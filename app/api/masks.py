import logging

import numpy as np
from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.masks import EraseRequest, GrowRequest, MaskData
from app.services.imaging import ImageError, decode_mask, encode_mask, mask_size
from app.services.masks import erase_from_mask, resize_mask

router = APIRouter(prefix="/api/masks", tags=["masks"])
logger = logging.getLogger(__name__)


def _reject(message: str) -> HTTPException:
    logger.warning("Mask request rejected: %s", message)
    return HTTPException(422, message)


def _decode(png: bytes) -> np.ndarray:
    try:
        width, height = mask_size(png)
    except ImageError as exc:
        raise _reject(str(exc)) from exc
    if max(width, height) > settings.max_image_edge:
        raise _reject("The mask is larger than a photo can be.")
    return decode_mask(png)


def _result(mask: np.ndarray) -> MaskData:
    if not mask.any():
        raise _reject("That would remove the whole part. Delete the part instead.")
    return MaskData(mask_png=encode_mask(mask))


@router.post("/grow", response_model=MaskData)
def grow_mask(body: GrowRequest) -> MaskData:
    return _result(resize_mask(_decode(body.mask_png), body.pixels))


@router.post("/erase", response_model=MaskData)
def erase_mask(body: EraseRequest) -> MaskData:
    mask, erased = _decode(body.mask_png), _decode(body.erase_png)
    if mask.shape != erased.shape:
        raise _reject("The outline and the part come from different photos.")
    return _result(erase_from_mask(mask, erased))
