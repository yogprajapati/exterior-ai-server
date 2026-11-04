from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from app.schemas.catalog import Material, RenderSpec
from app.schemas.project import Design, RegionType
from app.services.catalog import TEXTURE_DIR, get_catalog
from app.services.geometry import PlaneModel
from app.services.layering import RegionMask

LIGHT_BLUR_SIGMA = 5.0
DETAIL_BLUR_SIGMA = 1.5
SNAP_PX = 3  # a region's edge may move this far to follow the photo's own edges
GUIDE_RADIUS = 4
GUIDE_EPS = 1e-3
EDGE_CONTRAST = 3.0
CROP_PAD = SNAP_PX + GUIDE_RADIUS + 1
SHADE_RANGE = (0.3, 1.25)
SHADE_GAMMA = 0.75  # softens how strongly the old colour's light and dark areas carry over
TINT_RANGE = (0.3, 1.5)
FRAME_SHARE = 0.08  # window frame width as a share of the window's smaller side
RECTANGLE_FILL = 0.85  # a part this close to a rectangle is drawn as one
SMOOTH_SHARE = 0.06  # outline smoothing radius as a share of a part's smaller side
SKY_BGR = np.array([0.95, 0.88, 0.8], np.float32)


@dataclass(frozen=True)
class Shading:
    """The photo's own lighting: smooth luminance plus fine surface detail."""

    gray: np.ndarray
    light: np.ndarray
    detail: np.ndarray


def _shading(image: np.ndarray) -> Shading:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    return Shading(gray, cv2.GaussianBlur(gray, (0, 0), LIGHT_BLUR_SIGMA), gray - cv2.GaussianBlur(gray, (0, 0), DETAIL_BLUR_SIGMA))


@lru_cache(maxsize=16)
def _texture(file: str) -> np.ndarray:
    return cv2.imread(str(TEXTURE_DIR / file)).astype(np.float32) / 255


def preload_textures() -> None:
    """Decode every catalog texture at start-up so the first render does not pay for it."""
    for material in get_catalog().values():
        if material.render.texture:
            _texture(material.render.texture)


def _hex_bgr(color: str) -> np.ndarray:
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return np.array([b, g, r], np.float32) / 255


def _tiled(texture: np.ndarray, model: PlaneModel, box: tuple[int, int, int, int], tile_ft: float) -> np.ndarray:
    """Sample the texture at real-world scale, following the plane's perspective."""
    x0, y0, x1, y1 = box
    xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    feet = model.to_feet(np.stack([xs, ys], -1).reshape(-1, 2)).reshape(y1 - y0, x1 - x0, 2)
    th, tw = texture.shape[:2]
    map_x = (feet[..., 0] / tile_ft % 1 * tw).astype(np.float32)
    map_y = (feet[..., 1] / tile_ft % 1 * th).astype(np.float32)
    return cv2.remap(texture, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def _guided_filter(guide: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Edge-aware smoothing of `source`: it follows the edges of `guide` (He et al., guided filter)."""
    box = lambda x: cv2.boxFilter(x, -1, (2 * GUIDE_RADIUS + 1,) * 2)
    mean_g, mean_s = box(guide), box(source)
    slope = (box(guide * source) - mean_g * mean_s) / (box(guide * guide) - mean_g**2 + GUIDE_EPS)
    return box(slope) * guide + box(mean_s - slope * mean_g)


def _smoothed(mask: np.ndarray) -> np.ndarray:
    """Cleans a part's outline for the preview (a near-rectangle becomes one, anything else is opened and closed); measurements keep the original mask."""
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    if w and np.count_nonzero(mask) >= RECTANGLE_FILL * w * h:
        clean = np.zeros_like(mask)
        clean[y : y + h, x : x + w] = True
        return clean
    radius = max(2, round(SMOOTH_SHARE * min(mask.shape) / 1.3))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    return cv2.morphologyEx(cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel) > 0


def _alpha(mask: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Opacity of a region: solid inside, and within a few pixels of the border it snaps to the photo's edges."""
    solid = mask.astype(np.uint8)
    ring = np.ones((2 * SNAP_PX + 1,) * 2, np.uint8)
    core, reach = cv2.erode(solid, ring) > 0, cv2.dilate(solid, ring) > 0
    matte = np.clip((_guided_filter(gray, mask.astype(np.float32)) - 0.5) * EDGE_CONTRAST + 0.5, 0, 1)
    return np.where(core, 1.0, np.where(reach, matte, 0.0)).astype(np.float32)


def _crop_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    if w == 0:
        return None
    height, width = mask.shape
    return max(x - CROP_PAD, 0), max(y - CROP_PAD, 0), min(x + w + CROP_PAD, width), min(y + h + CROP_PAD, height)


def _surface(spec: RenderSpec, model: PlaneModel, box: tuple[int, int, int, int], light: np.ndarray, mask: np.ndarray, detail: np.ndarray) -> np.ndarray:
    """Paint or texture lit by the photo's own shading."""
    reference = max(float(np.median(light[mask][::7])), 1e-3)
    shade = np.clip((light / reference) ** SHADE_GAMMA, *SHADE_RANGE)[..., None]
    base = _hex_bgr(spec.color) if spec.mode == "paint" else _tiled(_texture(spec.texture), model, box, spec.tile_ft)
    return np.clip(base * shade + spec.relief * detail[..., None], 0, 1) * 255


def _recolor(current: np.ndarray, color: str, gray: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    """Moves a region to `color` but keeps its own light and dark, so bars, frames and gaps stay visible."""
    if not mask.any():
        return current
    reference = max(float(np.median(gray[mask][::7])), 1e-3)
    target = _hex_bgr(color) * np.clip(gray / reference, *TINT_RANGE)[..., None] * 255
    return current * (1 - strength) + np.clip(target, 0, 255) * strength


def _window(current: np.ndarray, frame: Material | None, glass: Material | None, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Frame recoloured; glass tinted and given a soft sky reflection that fades toward the bottom. A part left out of the design stays as photographed."""
    height, width = mask.shape
    frame_px = max(2, round(FRAME_SHARE * min(width, height)))
    pane = cv2.erode(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * frame_px + 1,) * 2)) > 0
    painted = _recolor(current, frame.render.color, gray, mask & ~pane, 1.0) if frame else current
    if glass and pane.any():
        spec = glass.render
        tinted = _hex_bgr(spec.color) * np.clip(0.35 + 0.9 * gray, 0, 1)[..., None] * 255
        tint = current * (1 - spec.strength) + tinted * spec.strength
        fade = ((1 - np.linspace(0, 1, height, dtype=np.float32)) ** 1.5)[:, None, None] * spec.gloss * 0.5
        painted = np.where(pane[..., None], tint * (1 - fade) + SKY_BGR * 255 * fade, painted)
    return painted


def _paint_region(out: np.ndarray, shading: Shading, region: RegionMask, claimed: np.ndarray, model: PlaneModel, material: Material | None, glass: Material | None) -> None:
    box = _crop_box(region.mask)
    if box is None:
        return
    x0, y0, x1, y1 = box
    crop = (slice(y0, y1), slice(x0, x1))
    mask, gray = region.mask[crop], shading.gray[crop]
    if region.type != RegionType.WALL:
        mask = _smoothed(mask) & ~(claimed[crop] & ~region.mask[crop])
    if not mask.any():  # smoothing can remove a very small part
        return
    alpha = (_alpha(mask, gray) * ~(claimed[crop] & ~mask))[..., None]  # never spill onto another region
    current = out[crop]
    if region.type == RegionType.WINDOW:
        painted = _window(current, material, glass, gray, mask)
    elif material.render.mode == "tint":
        painted = _recolor(current, material.render.color, gray, mask, material.render.strength)
    else:
        painted = _surface(material.render, model, box, shading.light[crop], mask, shading.detail[crop])
    out[crop] = current * (1 - alpha) + painted * alpha


def render(image: np.ndarray, regions: list[RegionMask], models: dict[str, PlaneModel], design: Design) -> np.ndarray:
    """Apply the design's materials to the photo; only the marked regions (plus a few pixels snapped to their edges) change."""
    catalog = get_catalog()
    shading = _shading(image)
    out = image.astype(np.float32)
    claimed = np.logical_or.reduce([r.mask for r in regions]) if regions else np.zeros(image.shape[:2], bool)
    for region in regions:
        material, glass = catalog.get(design.assignments.get(region.id, "")), catalog.get(design.glass.get(region.id, ""))
        if material or glass:
            _paint_region(out, shading, region, claimed, models[region.plane_id], material, glass)
    return np.clip(out, 0, 255).astype(np.uint8)
