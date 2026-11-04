import logging
import time
from collections.abc import Callable, Hashable
from threading import Lock
from typing import TypeVar

from cachetools import LRUCache

from app.repositories.projects import ProjectRecord
from app.schemas.estimate import Estimate
from app.schemas.project import Design
from app.services import render
from app.services.estimator import build_estimate
from app.services.geometry import PlaneModel
from app.services.imaging import decode_image, encode_jpeg
from app.services.layering import RegionMask, effective_masks
from app.services.redesign.pipeline import RenderResult, redesign

T = TypeVar("T")
_results: LRUCache = LRUCache(maxsize=32)
_masks: LRUCache = LRUCache(maxsize=2)  # decoded masks are large; one entry per recently used project
_lock = Lock()
logger = logging.getLogger(__name__)


def _memo(cache: LRUCache, key: Hashable, label: str, compute: Callable[[], T]) -> T:
    """Memoise per saved project state, so repeat views and the report reuse earlier work."""
    with _lock:
        if key in cache:
            return cache[key]
    started = time.perf_counter()
    value = compute()
    logger.info("%s computed in %.0f ms", label, (time.perf_counter() - started) * 1000)
    with _lock:
        cache[key] = value
    return value


def project_masks(project: ProjectRecord) -> list[RegionMask]:
    return _memo(_masks, (project.id, project.updated_at), f"Masks of project {project.id}", lambda: effective_masks(project.state.regions))


def estimate_design(project: ProjectRecord, design: Design) -> Estimate:
    return _memo(_results, ("estimate", project.id, design.id, project.updated_at), f"Estimate of project {project.id}, design {design.id}",
                 lambda: build_estimate(project.state, design, project_masks(project)))


def _planes(project: ProjectRecord) -> dict[str, PlaneModel]:
    return {p.id: PlaneModel.from_plane(p) for p in project.state.planes}


def preview_design(project: ProjectRecord, design: Design) -> bytes:
    """The quick recolour preview: instant, so every choice shows at once; the AI redesign is a separate, slower step."""
    return _memo(_results, ("preview", project.id, design.id, project.updated_at), f"Preview of project {project.id}, design {design.id}",
                 lambda: encode_jpeg(render.render(decode_image(project.image), project_masks(project), _planes(project), design)))


def cached_render(project: ProjectRecord, design: Design) -> RenderResult | None:
    with _lock:
        return _results.get(render_key(project, design))


def render_design(project: ProjectRecord, design: Design, report: Callable[[float, str], None] = lambda *_: None) -> RenderResult:
    """Redesigns the photo (slow: an AI edit), reusing an earlier result for the same saved state."""
    return _memo(_results, render_key(project, design), f"Render of project {project.id}, design {design.id}",
                 lambda: redesign(decode_image(project.image), project_masks(project), _planes(project), design, report))


def render_key(project: ProjectRecord, design: Design) -> tuple:
    return ("render", project.id, design.id, project.updated_at)
