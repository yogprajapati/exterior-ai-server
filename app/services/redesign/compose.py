import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app.schemas.catalog import Material
from app.schemas.project import RegionType
from app.services.layering import RegionMask

logger = logging.getLogger(__name__)
REGISTER_EDGE = 640  # px: registration runs on a reduced copy
EDGE_BLUR, CANNY_LOW, CANNY_HIGH = 1.5, 40, 110
EDGE_TOLERANCE_PX = 2
ADDED_TOLERANCE_PX = 3
BORDER_PX = 12  # the editor's frame rarely matches the photo's outermost pixels
FEATHER_SIGMA = 1.2
LIGHTNESS_SHARE = 0.8  # how far calibration moves lightness toward the chosen colour (chroma moves fully)
OPAQUE = 0.9  # a tint at least this strong is a solid colour (a glass railing is not)


@dataclass(frozen=True)
class Composite:
    image: np.ndarray  # BGR, the original except inside the redesigned regions
    geometry: float  # share of the original's edges outside those regions that the edit kept (low only when registration failed)
    added: float  # share of the edit's edges on repainted surfaces that the photo did not have (new windows, panels, objects)
    change: float  # mean absolute pixel change inside the redesigned regions (0..255)


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0), EDGE_BLUR)


def register(original: np.ndarray, generated: np.ndarray) -> np.ndarray:
    """Resize the editor's output to the photo and undo the small shift or zoom editors add (ECC affine)."""
    height, width = original.shape[:2]
    generated = cv2.resize(generated, (width, height), interpolation=cv2.INTER_CUBIC)
    scale = min(1.0, REGISTER_EDGE / max(height, width))
    small = lambda image: cv2.resize(_gray(image), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(small(original), small(generated), warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
    except cv2.error:
        logger.warning("Registration did not converge; using the resized edit as is")
        return generated
    warp[:, 2] /= scale
    return cv2.warpAffine(generated, warp, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def geometry_score(original: np.ndarray, aligned: np.ndarray, redesigned: np.ndarray) -> float:
    """Edges of the original that must survive (those outside the redesigned regions) and are still there in the edit."""
    edges = lambda image: cv2.Canny(_gray(image), CANNY_LOW, CANNY_HIGH) > 0
    kept = edges(original) & ~(cv2.dilate(redesigned.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    kept[:BORDER_PX], kept[-BORDER_PX:], kept[:, :BORDER_PX], kept[:, -BORDER_PX:] = False, False, False, False
    near = cv2.dilate(edges(aligned).astype(np.uint8), np.ones((2 * EDGE_TOLERANCE_PX + 1,) * 2, np.uint8)) > 0
    return float((kept & near).sum() / max(kept.sum(), 1))


def added_structure(original: np.ndarray, aligned: np.ndarray, surfaces: np.ndarray) -> float:
    """A repaint changes colour, not structure: edges the edit draws on those surfaces that the photo lacks are things the editor made up."""
    edges = lambda image: cv2.Canny(_gray(image), CANNY_LOW, CANNY_HIGH) > 0
    known = cv2.dilate(edges(original).astype(np.uint8), np.ones((2 * ADDED_TOLERANCE_PX + 1,) * 2, np.uint8)) > 0
    inside = cv2.erode(surfaces.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    drawn = edges(aligned) & inside
    return float((drawn & ~known).sum() / max(drawn.sum(), 1))


def _hex_lab(color: str) -> np.ndarray:
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]


def _is_solid_colour(region: RegionMask, material: Material) -> bool:
    """Flat-colour materials; windows are excluded because their frame and glass share one region."""
    spec = material.render
    return region.type != RegionType.WINDOW and (spec.mode == "paint" or (spec.mode == "tint" and spec.strength >= OPAQUE))


def calibrate(generated: np.ndarray, regions: list[RegionMask], materials: dict[str, Material]) -> np.ndarray:
    """Editors drift on colour ("sage" comes out lime), so each flat-colour material is moved to its exact catalog colour; texture and shading stay."""
    lab = cv2.cvtColor(generated, cv2.COLOR_BGR2LAB).astype(np.float32)
    groups: dict[str, tuple[Material, list[np.ndarray]]] = {}
    for region in regions:
        material = materials[region.id]
        if _is_solid_colour(region, material) and region.mask.any():
            groups.setdefault(material.id, (material, []))[1].append(region.mask)
    for material, masks in groups.values():
        mask = np.logical_or.reduce(masks)
        shift = _hex_lab(material.render.color) - np.median(lab[mask], axis=0)
        lab[mask] += shift * np.array([LIGHTNESS_SHARE, 1, 1], np.float32)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def compose(original: np.ndarray, generated: np.ndarray, regions: list[RegionMask], materials: dict[str, Material]) -> Composite:
    """The original photo, with the edited pixels (colour-calibrated) used only inside the regions the design changes."""
    aligned = register(original, generated)
    changed = [r for r in regions if r.id in materials]
    allowed = np.logical_or.reduce([r.mask for r in changed]) if changed else np.zeros(original.shape[:2], bool)
    graded = calibrate(aligned, changed, materials)
    alpha = cv2.GaussianBlur(allowed.astype(np.float32), (0, 0), FEATHER_SIGMA)[..., None]
    image = (graded.astype(np.float32) * alpha + original.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    flat = [r.mask for r in changed if materials[r.id].render.mode == "paint" and r.type != RegionType.WINDOW]
    added = added_structure(original, aligned, np.logical_or.reduce(flat)) if flat else 0.0
    change = float(np.abs(graded.astype(np.int16) - original)[allowed].mean()) if allowed.any() else 0.0
    return Composite(image, geometry_score(original, aligned, allowed), added, change)
