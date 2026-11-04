import logging
from uuid import uuid4

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.api.deps import ProjectDep
from app.repositories.projects import ProjectRecord
from app.schemas.jobs import FindRequest, JobStatus
from app.schemas.project import Region
from app.services.detection.parts import DetectedPart, detect, find_parts
from app.services.imaging import decode_image, encode_mask
from app.services.jobs import Job, jobs

router = APIRouter(prefix="/api/projects/{project_id}", tags=["detection"])
logger = logging.getLogger(__name__)
SMALL_HOUSE_SHARE = 0.15
MAX_PHRASE_LENGTH = 60
NO_PLANE = "unassigned"  # detection can start before the size step is done; the plane is filled in when the parts are fetched


def _rgb(project: ProjectRecord):
    return cv2.cvtColor(decode_image(project.image), cv2.COLOR_BGR2RGB)


def _regions(parts: list[DetectedPart], plane_id: str) -> list[Region]:
    return [Region(id=uuid4().hex[:12], type=p.type, plane_id=plane_id, mask_png=encode_mask(p.mask)) for p in parts]


def _status(job: Job | None) -> JobStatus:
    return JobStatus(status="idle") if job is None else JobStatus(status=job.status, progress=job.progress, label=job.label, error=job.error, note=job.note)


def _first_plane(project: ProjectRecord) -> str:
    if not project.state.planes:
        raise HTTPException(409, "Set the size of the house before looking for its parts.")
    return project.state.planes[0].id


@router.get("/detect", response_model=JobStatus)
async def read_detection(project: ProjectDep) -> JobStatus:
    return _status(jobs.get(("detect", project.id)))


@router.post("/detect", response_model=JobStatus, status_code=202)
async def start_detection(project: ProjectDep) -> JobStatus:
    """Finds the parts of the house in the background (it can run while the user sets the size); fetch them from /detect/parts."""

    def work(job: Job) -> list[Region]:
        detection = detect(project.id, _rgb(project), job.report)
        if not detection.parts:
            job.note = "We couldn't find a house in this photo. Please mark the parts yourself, or try a clearer photo of the front of the house."
        elif detection.building_share < SMALL_HOUSE_SHARE:
            job.note = "The house looks quite small in this photo, so the outlines may be rough. A closer photo gives better results."
        return _regions(detection.parts, NO_PLANE)

    return _status(jobs.start(("detect", project.id), work))


@router.get("/detect/parts", response_model=list[Region])
async def read_detected_parts(project: ProjectDep) -> list[Region]:
    """The parts found by a finished detection, placed on the project's first surface."""
    job = jobs.get(("detect", project.id))
    if job is None or job.status != "done":
        raise HTTPException(404, "Detection has not finished.")
    plane_id = _first_plane(project)
    return [region.model_copy(update={"plane_id": plane_id}) for region in job.result]


@router.post("/find", response_model=list[Region])
async def find_more(body: FindRequest, project: ProjectDep) -> list[Region]:
    """Looks for one more kind of part by name; the result is returned for the user to accept, not saved."""
    phrase = body.phrase.strip()
    if not phrase or len(phrase) > MAX_PHRASE_LENGTH:
        raise HTTPException(422, f"Describe the part in a few words (up to {MAX_PHRASE_LENGTH} characters).")
    plane_id = _first_plane(project)
    parts = await run_in_threadpool(find_parts, project.id, _rgb(project), phrase, body.type)
    logger.info("Project %s: '%s' found %d", project.id, phrase, len(parts))
    return _regions(parts, plane_id)
