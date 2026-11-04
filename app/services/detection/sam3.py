import logging
import time
from dataclasses import dataclass
from threading import Lock

import numpy as np
import torch
from cachetools import LRUCache
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)
SCORE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5


@dataclass(frozen=True)
class PhotoFeatures:
    """SAM 3's image encoding, kept so every concept prompt on the same photo reuses it."""

    embeds: object
    size: tuple[int, int]  # (width, height)


class Sam3:
    """Text-prompted segmentation (SAM 3, promptable concept segmentation): every instance of a phrase gets its own mask."""

    def __init__(self) -> None:
        self._lock = Lock()  # the model is not thread-safe, and one photo at a time keeps memory bounded
        self._model = None
        self._processor = None
        self._features: LRUCache = LRUCache(maxsize=2)

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import Sam3Model, Sam3Processor

        started = time.perf_counter()
        torch.set_num_threads(settings.torch_threads)
        self._processor = Sam3Processor.from_pretrained(settings.sam3_model)
        self._model = Sam3Model.from_pretrained(settings.sam3_model, torch_dtype=torch.float32).eval()
        logger.info("SAM 3 loaded in %.1f s", time.perf_counter() - started)

    def encode(self, key: str, rgb: np.ndarray) -> PhotoFeatures:
        with self._lock:
            if key in self._features:
                return self._features[key]
            self._load()
            started = time.perf_counter()
            with torch.no_grad():
                embeds = self._model.get_vision_features(pixel_values=self._processor(images=Image.fromarray(rgb), return_tensors="pt").pixel_values)
            features = PhotoFeatures(embeds, (rgb.shape[1], rgb.shape[0]))
            self._features[key] = features
            logger.info("Photo %s encoded in %.1f s", key, time.perf_counter() - started)
            return features

    def find(self, features: PhotoFeatures, phrase: str) -> list[np.ndarray]:
        """Boolean full-size masks, one per instance of `phrase`, best first."""
        with self._lock, torch.no_grad():
            started = time.perf_counter()
            out = self._model(vision_embeds=features.embeds, **self._processor(text=phrase, return_tensors="pt"))
            result = self._processor.post_process_instance_segmentation(
                out, threshold=SCORE_THRESHOLD, mask_threshold=MASK_THRESHOLD, target_sizes=[features.size[::-1]]
            )[0]
            order = result["scores"].argsort(descending=True)
            masks = [m.numpy() for m in result["masks"].bool()[order]]
            logger.info("'%s': %d found in %.1f s", phrase, len(masks), time.perf_counter() - started)
            return masks


sam3 = Sam3()
