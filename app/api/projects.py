import logging
from dataclasses import replace
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from app.api.deps import ProjectDep, RepoDep
from app.core.config import settings
from app.repositories.projects import ProjectRecord
from app.schemas.project import (
    Plane,
    ProjectCreated,
    ProjectOut,
    ProjectState,
    ProjectUpdate,
    ScaleCheck,
)
from app.services.imaging import ImageError, decode_upload, encode_jpeg
from app.services.quality import assess
from app.services.scale import check_scale
from app.services.validation import StateError, validate_regions, validate_state

router = APIRouter(prefix="/api/projects", tags=["projects"])
logger = logging.getLogger(__name__)


def _out(project: ProjectRecord) -> ProjectOut:
    return ProjectOut(id=project.id, width=project.width, height=project.height, created_at=project.created_at,
                      updated_at=project.updated_at, **dict(project.state))


@router.post("", response_model=ProjectCreated, status_code=201)
async def create_project(file: UploadFile, repo: RepoDep) -> ProjectCreated:
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        logger.warning("Upload rejected: larger than %d MB", settings.max_upload_mb)
        raise HTTPException(413, f"The photo is larger than {settings.max_upload_mb} MB.")
    try:
        image, original_size = await run_in_threadpool(decode_upload, data, settings.max_image_edge)
    except ImageError as exc:
        logger.warning("Upload rejected: %s", exc)
        raise HTTPException(400, str(exc)) from exc
    quality = await run_in_threadpool(assess, image, original_size)
    if not quality.accepted:
        logger.warning("Photo rejected by the quality check: %s", [issue.code for issue in quality.issues])
        raise HTTPException(422, quality.model_dump())
    project_id = uuid4().hex
    height, width = image.shape[:2]
    await repo.create(project_id, encode_jpeg(image), width, height)
    logger.info("Project %s created: %dx%d, quality warnings %s", project_id, width, height, [issue.code for issue in quality.issues])
    return ProjectCreated(id=project_id, width=width, height=height, quality=quality)


@router.get("/{project_id}", response_model=ProjectOut)
async def read_project(project: ProjectDep) -> ProjectOut:
    return _out(project)


@router.patch("/{project_id}", response_model=ProjectOut)
async def update_project(body: ProjectUpdate, project: ProjectDep,
                         repo: RepoDep) -> ProjectOut:
    changes = {name: value for name, value in body if value is not None}
    merged = ProjectState(**{**dict(project.state), **changes})
    try:
        if body.regions is not None:
            validate_regions(body.regions, (project.width, project.height))
        validate_state(merged, (project.width, project.height))
    except StateError as exc:
        logger.warning("Project %s update rejected: %s", project.id, exc)
        raise HTTPException(422, str(exc)) from exc
    updated_at = await repo.save_fields(project.id, merged.model_dump(include=set(changes)))
    logger.info("Project %s updated: %s", project.id, sorted(changes))
    return _out(replace(project, state=merged, updated_at=updated_at))


@router.post("/{project_id}/scale-check", response_model=ScaleCheck)
async def scale_check(plane: Plane, project: ProjectDep) -> ScaleCheck:
    return check_scale(plane, (project.width, project.height))


@router.get("/{project_id}/image")
async def read_image(project: ProjectDep) -> Response:
    return Response(project.image, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
