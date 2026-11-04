import io

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 60_000_000
JPEG_QUALITY = 88


class ImageError(ValueError):
    pass


def decode_upload(data: bytes, max_edge: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Return the orientation-fixed BGR image (long edge capped) and the original (width, height)."""
    try:
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageError("The file is not a readable image (use JPEG, PNG or WebP).") from exc
    original = pil.size
    scale = max_edge / max(original)
    if scale < 1:
        pil = pil.resize((round(original[0] * scale), round(original[1] * scale)), Image.LANCZOS)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR), original


def decode_image(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def encode_jpeg(image: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ImageError("The image could not be encoded.")
    return buf.tobytes()


def mask_size(png: bytes) -> tuple[int, int]:
    """(width, height) of a PNG, fully decoded so corrupt data is rejected before it is saved."""
    try:
        with Image.open(io.BytesIO(png)) as mask:
            mask.load()
            return mask.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageError("A region mask is not a valid PNG.") from exc


def decode_mask(png: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE) > 127


def encode_mask(mask: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    if not ok:
        raise ImageError("A mask could not be encoded.")
    return buf.tobytes()
