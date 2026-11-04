import logging
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from gradio_client import Client, handle_file

from app.core.config import settings

logger = logging.getLogger(__name__)
EDIT_EDGE = 1024  # the editor works on a reduced copy; the original's pixels are kept outside the redesigned regions
MODEL_STEP = 16  # FLUX.2 klein needs sides that are multiples of this


class EditorUnavailable(RuntimeError):
    """The image editor cannot be used right now (no token, quota used up, service down)."""


@dataclass(frozen=True)
class SpaceEditor:
    """A free Hugging Face ZeroGPU Space that edits a photo from an instruction; `call` adapts to that Space's own API."""

    space: str
    call: Callable[[Client, str, str, int, tuple[int, int]], object]  # client, photo path, prompt, seed, (width, height)


# Choose with EDIT_MODEL. Free GPU quota is per account (about 3.5 min a day) and each call reserves the time its Space asks for (see the notes).
EDITORS: dict[str, SpaceEditor] = {
    # 28 steps, about 35 s of GPU per edit; reserves 90 s. Best structure preservation of those tested.
    "kontext": SpaceEditor("black-forest-labs/FLUX.1-Kontext-Dev", lambda c, src, prompt, seed, size: c.predict(handle_file(src), prompt, seed, False, 2.5, 28, api_name="/infer")),
    # 4 steps, a few seconds of GPU; reserves 85 s. Apache-2.0 model, so the most edits a day.
    "klein-4b": SpaceEditor("black-forest-labs/FLUX.2-klein-4B", lambda c, src, prompt, seed, size: c.predict(
        prompt, [{"image": handle_file(src)}], "Distilled (4 steps)", seed, False, size[0], size[1], 4, 1.0, False, api_name="/generate")),
    # Qwen-Image-Edit-2511 with a lightning LoRA, 4 steps; reserves 60 s.
    "qwen-lightning": SpaceEditor("akhaliq/Qwen-Image-Edit-2511-Lightning", lambda c, src, prompt, seed, size: c.predict(handle_file(src), prompt, seed, False, 1.0, 4, api_name="/infer")),
    # Qwen-Image-Edit with a lightning LoRA, 8 steps; reserves 60 s.
    "qwen-fast": SpaceEditor("multimodalart/Qwen-Image-Edit-Fast", lambda c, src, prompt, seed, size: c.predict(handle_file(src), prompt, seed, False, 1.0, 8, False, api_name="/infer")),
}


def _reduced(image: np.ndarray) -> np.ndarray:
    """At most EDIT_EDGE on the long side, with sides rounded to MODEL_STEP."""
    scale = min(1.0, EDIT_EDGE / max(image.shape[:2]))
    height, width = (max(MODEL_STEP, round(side * scale / MODEL_STEP) * MODEL_STEP) for side in image.shape[:2])
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def edit_image(original: np.ndarray, prompt: str, seed: int) -> np.ndarray:
    """Instruction edit by the chosen Space (free GPU time, a daily quota per Hugging Face account)."""
    editor = EDITORS[settings.edit_model]
    reduced = _reduced(original)
    with tempfile.TemporaryDirectory() as folder:
        source = os.path.join(folder, "photo.jpg")
        cv2.imwrite(source, reduced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        started = time.perf_counter()
        try:
            result = editor.call(Client(editor.space, token=settings.hf_token, verbose=False), source, prompt, seed, (reduced.shape[1], reduced.shape[0]))
        except Exception as exc:  # gradio_client raises many types for quota, queue and network failures
            raise EditorUnavailable(f"{type(exc).__name__}: {exc}"[:300]) from exc
        logger.info("Edit by %s returned in %.1f s", settings.edit_model, time.perf_counter() - started)
        return cv2.imread(_image_path(result))


def _image_path(result: object) -> str:
    """The edited image's file, whichever way the Space returns it: (image, seed), or a one-item gallery."""
    image = result[0] if isinstance(result, tuple) else result
    image = image[0] if isinstance(image, list) else image
    image = image.get("image", image) if isinstance(image, dict) else image
    return image if isinstance(image, str) else image["path"]
