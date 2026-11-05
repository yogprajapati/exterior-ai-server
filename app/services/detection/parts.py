import logging
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from app.schemas.project import RegionType
from app.services.detection.sam3 import Sam3, sam3

logger = logging.getLogger(__name__)

# Phrases SAM 3 is asked for, and the part each finds. Wording was tuned on photos of European and Indian houses.
PART_PHRASES: dict[str, RegionType] = {
    "window": RegionType.WINDOW,
    "door": RegionType.DOOR,
    "gate": RegionType.GATE,
    "garage": RegionType.GATE,
    "balcony": RegionType.BALCONY,
    "balcony railing": RegionType.RAILING,
    "railing": RegionType.RAILING,
    "pillar": RegionType.PILLAR,
    "parapet": RegionType.PARAPET,
    "cornice": RegionType.ROOF_EDGE,
}
WALL_PHRASES = ("building", "wall")
EXCLUDED_PHRASES = ("roof", "tree", "sky")  # never repainted: these were the false "wall" of the old detector

MIN_PART_SHARE = 0.0001  # of the photo; smaller masks are noise
MIN_WALL_PIECE_SHARE = 0.002
DUPLICATE_OVERLAP = 0.6  # a mask this covered by an earlier one of the same type is the same part
CLEAN_KERNEL = np.ones((3, 3), np.uint8)


@dataclass(frozen=True)
class DetectedPart:
    type: RegionType
    mask: np.ndarray


@dataclass(frozen=True)
class Detection:
    parts: list[DetectedPart]
    building_share: float


def _union(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    return np.logical_or.reduce(masks) if masks else np.zeros(shape, bool)


def _new_parts(masks: list[np.ndarray], region_type: RegionType, taken: dict[RegionType, np.ndarray], min_pixels: float) -> list[DetectedPart]:
    """Keeps masks that are big enough and not already found under another phrase for the same part type."""
    kept = []
    for mask in masks:
        area = mask.sum()
        if area < min_pixels or (taken[region_type] & mask).sum() > DUPLICATE_OVERLAP * area:
            continue
        taken[region_type] |= mask
        kept.append(DetectedPart(region_type, mask))
    return kept


def _wall(context: np.ndarray, others: np.ndarray, min_pixels: float) -> np.ndarray:
    """The building (and any walls) minus every other part, the roof, trees and sky; specks and slivers removed."""
    wall = cv2.morphologyEx((context & ~others).astype(np.uint8), cv2.MORPH_OPEN, CLEAN_KERNEL)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(wall, connectivity=8)
    keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_pixels]
    return np.isin(labels, keep)


def find_parts(key: str, rgb: np.ndarray, phrase: str, region_type: RegionType, existing: list[np.ndarray], model: Sam3 = sam3) -> list[DetectedPart]:
    masks = model.find(model.encode(key, rgb), phrase)
    taken = {region_type: _union(existing, rgb.shape[:2])}
    return _new_parts(masks, region_type, taken, MIN_PART_SHARE * rgb.shape[0] * rgb.shape[1])


def detect(key: str, rgb: np.ndarray, on_progress: Callable[[float, str], None] = lambda *_: None, model: Sam3 = sam3) -> Detection:
    """Finds the parts of the house on one photo: SAM 3 concept prompts, de-duplicated, with the wall as what is left of the building."""
    phrases = (*PART_PHRASES, *WALL_PHRASES, *EXCLUDED_PHRASES)
    on_progress(0.0, "Reading the photo")
    features = model.encode(key, rgb)
    shape, total = rgb.shape[:2], rgb.shape[0] * rgb.shape[1]
    found = {}
    for i, phrase in enumerate(phrases):
        on_progress((i + 1) / (len(phrases) + 1), f"Looking for “{phrase}”")
        found[phrase] = model.find(features, phrase)

    taken = {region_type: np.zeros(shape, bool) for region_type in RegionType}
    parts = [part for phrase, region_type in PART_PHRASES.items() for part in _new_parts(found[phrase], region_type, taken, MIN_PART_SHARE * total)]
    building = _union([m for phrase in WALL_PHRASES for m in found[phrase]], shape)
    others = _union([p.mask for p in parts] + [m for phrase in EXCLUDED_PHRASES for m in found[phrase]], shape)
    wall = _wall(building, others, MIN_WALL_PIECE_SHARE * total)
    if wall.any():
        parts.append(DetectedPart(RegionType.WALL, wall))
    logger.info("Photo %s: %d parts, building covers %.0f%%", key, len(parts), 100 * building.mean())
    return Detection(parts, float(building.mean()))
