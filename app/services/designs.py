import logging
import time
from collections.abc import Callable, Hashable
from threading import Lock
from typing import TypeVar

from cachetools import LRUCache

from app.repositories.projects import ProjectRecord
from app.schemas.estimate import Estimate
from app.schemas.project import Design
from app.schemas.render import RenderResult
from app.services import render
from app.services.estimator import build_estimate
from app.services.geometry import PlaneModel
from app.services.imaging import decode_image, encode_jpeg
from app.services.layering import RegionMask, effective_masks
from app.services.redesign.pipeline import redesign

T = TypeVar("T")
_results: LRUCache = LRUCache(maxsize=32)
_masks: LRUCache = LRUCache(maxsize=2)
_lock = Lock()
logger = logging.getLogger(__name__)


def _memo(cache: LRUCache, key: Hashable, label: str, compute: Callable[[], T]) -> T:
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
    return _memo(_results, ("preview", project.id, design.id, project.updated_at), f"Preview of project {project.id}, design {design.id}",
                 lambda: encode_jpeg(render.render(decode_image(project.image), project_masks(project), _planes(project), design)))


def render_design(project: ProjectRecord, design: Design, report: Callable[[float, str], None]) -> RenderResult:
    return redesign(decode_image(project.image), project_masks(project), _planes(project), design, report)
