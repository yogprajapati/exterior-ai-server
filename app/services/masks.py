import cv2
import numpy as np


def resize_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(pixels) + 1,) * 2)
    change = cv2.dilate if pixels > 0 else cv2.erode
    return change(mask.astype(np.uint8), kernel) > 0


def erase_from_mask(mask: np.ndarray, erased: np.ndarray) -> np.ndarray:
    return mask & ~erased
