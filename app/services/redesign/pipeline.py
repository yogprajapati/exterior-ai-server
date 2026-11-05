import logging
from collections.abc import Callable

import numpy as np

from app.schemas.project import Design
from app.schemas.render import RenderResult
from app.services import render
from app.services.catalog import get_catalog
from app.services.geometry import PlaneModel
from app.services.imaging import encode_jpeg
from app.services.layering import RegionMask
from app.services.redesign.compose import compose
from app.services.redesign.editor import EditorUnavailable, edit_image
from app.services.redesign.prompt import build_prompt

logger = logging.getLogger(__name__)
SEEDS = (0, 1)  # a second try with another seed when the first edit fails the checks
MIN_GEOMETRY = 0.5  # share of the building's edges outside the changed parts that must still line up: below it, the edit could not be registered
MAX_ADDED = 0.08  # share of edges on repainted surfaces that may be new (made-up windows or panels)
MIN_CHANGE = 3.0  # mean pixel change inside the redesigned regions below which the edit did nothing


def _preview(image: np.ndarray, regions: list[RegionMask], planes: dict[str, PlaneModel], design: Design, note: str) -> RenderResult:
    return RenderResult(encode_jpeg(render.render(image, regions, planes, design)), "preview", note)


def redesign(image: np.ndarray, regions: list[RegionMask], planes: dict[str, PlaneModel], design: Design, report: Callable[[float, str], None]) -> RenderResult:
    """AI edit constrained to the chosen parts and calibrated to the catalog colours; a plain recolour preview when the editor is unavailable or fails the checks."""
    catalog = get_catalog()
    frames = {region_id: catalog[material_id] for region_id, material_id in design.assignments.items()}
    glass = {region_id: catalog[material_id] for region_id, material_id in design.glass.items()}
    changed = [r for r in regions if r.id in frames or r.id in glass]
    if not changed:
        return RenderResult(encode_jpeg(image), "preview", "No materials are chosen yet.")
    prompt = build_prompt(changed, frames, glass)
    for attempt, seed in enumerate(SEEDS):
        report(attempt / len(SEEDS), "Redesigning your house (this takes up to a minute)")
        try:
            generated = edit_image(image, prompt, seed)
        except EditorUnavailable as exc:
            logger.warning("Image editor unavailable: %s", exc)
            return _preview(image, regions, planes, design, "The AI redesign is unavailable right now, so this is a simple recolour preview.")
        result = compose(image, generated, changed, {**glass, **frames})
        logger.info("Edit %d checked: geometry %.2f, added %.2f, change %.1f", attempt + 1, result.geometry, result.added, result.change)
        if result.geometry >= MIN_GEOMETRY and result.added <= MAX_ADDED and result.change >= MIN_CHANGE:
            return RenderResult(encode_jpeg(result.image), "ai")
    return _preview(image, regions, planes, design, "The AI redesign invented parts that are not in your photo, so this is a simple recolour preview.")
