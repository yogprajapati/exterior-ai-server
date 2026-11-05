import hashlib
import logging
from uuid import uuid4

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.api.deps import DetectionRepoDep, ProjectDep
from app.repositories.projects import ProjectRecord
from app.schemas.detection import StoredDetection
from app.schemas.jobs import FindRequest, JobStatus
from app.schemas.project import Region
from app.services.detection.parts import DetectedPart, Detection, detect, find_parts
from app.services.imaging import decode_image, decode_mask, encode_mask
from app.services.jobs import Job, jobs

router = APIRouter(prefix="/api/projects/{project_id}", tags=["detection"])
logger = logging.getLogger(__name__)
SMALL_HOUSE_SHARE = 0.15
MAX_PHRASE_LENGTH = 60
NO_PLANE = "unassigned"
NO_HOUSE_NOTE = "We couldn't find a house in this photo. Please mark the parts yourself, or try a clearer photo of the front of the house."
SMALL_HOUSE_NOTE = "The house looks quite small in this photo, so the outlines may be rough. A closer photo gives better results."


def _digest(project: ProjectRecord) -> str:
    return hashlib.sha256(project.image).hexdigest()


def _rgb(project: ProjectRecord):
    return cv2.cvtColor(decode_image(project.image), cv2.COLOR_BGR2RGB)


def _regions(parts: list[DetectedPart]) -> list[Region]:
    return [Region(id=uuid4().hex[:12], type=p.type, plane_id=NO_PLANE, mask_png=encode_mask(p.mask)) for p in parts]


def _note(detection: Detection) -> str | None:
    if not detection.parts:
        return NO_HOUSE_NOTE
    return SMALL_HOUSE_NOTE if detection.building_share < SMALL_HOUSE_SHARE else None


def _status(job: Job) -> JobStatus:
    return JobStatus.model_validate(job, from_attributes=True)


def _finished(stored: StoredDetection) -> JobStatus:
    return JobStatus(status="done", progress=1.0, note=stored.note)


@router.get("/detect", response_model=JobStatus)
async def read_detection(project: ProjectDep, detections: DetectionRepoDep) -> JobStatus:
    job = jobs.get(("detect", project.id))
    if job:
        return _status(job)
    stored = await detections.get(_digest(project))
    return JobStatus(status="idle") if stored is None else _finished(stored)


@router.post("/detect", response_model=JobStatus, status_code=202)
async def start_detection(project: ProjectDep, detections: DetectionRepoDep) -> JobStatus:
    digest = _digest(project)
    stored = await detections.get(digest)
    if stored:
        return _finished(stored)

    def work(job: Job) -> StoredDetection:
        detection = detect(project.id, _rgb(project), job.report)
        job.note = _note(detection)
        return StoredDetection(regions=_regions(detection.parts), note=job.note)

    async def save(found: StoredDetection) -> None:
        await detections.save(digest, found)

    return _status(jobs.start(("detect", project.id), work, save))


@router.get("/detect/parts", response_model=list[Region])
async def read_detected_parts(project: ProjectDep, detections: DetectionRepoDep) -> list[Region]:
    stored = await detections.get(_digest(project))
    if stored is None:
        raise HTTPException(404, "Detection has not finished.")
    return stored.regions


@router.post("/find", response_model=list[Region])
async def find_more(body: FindRequest, project: ProjectDep, detections: DetectionRepoDep) -> list[Region]:
    phrase = body.phrase.strip()
    if not phrase or len(phrase) > MAX_PHRASE_LENGTH:
        logger.warning("Project %s: phrase rejected (%d characters)", project.id, len(phrase))
        raise HTTPException(422, f"Describe the part in a few words (up to {MAX_PHRASE_LENGTH} characters).")
    stored = await detections.get(_digest(project))
    known = project.state.regions or (stored.regions if stored else [])
    existing = [decode_mask(region.mask_png) for region in known if region.type == body.type]
    parts = await run_in_threadpool(find_parts, project.id, _rgb(project), phrase, body.type, existing)
    logger.info("Project %s: '%s' found %d new", project.id, phrase, len(parts))
    return _regions(parts)
