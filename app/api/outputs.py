import logging

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from app.api.deps import DesignDep
from app.repositories.projects import ProjectRecord
from app.schemas.estimate import Estimate
from app.schemas.jobs import RenderStatus
from app.schemas.project import Design
from app.services.designs import (
    cached_render,
    estimate_design,
    preview_design,
    render_design,
    render_key,
)
from app.services.jobs import Job, jobs
from app.services.report import build_report

router = APIRouter(prefix="/api/projects/{project_id}/designs/{design_id}", tags=["outputs"])
logger = logging.getLogger(__name__)


@router.get("/estimate", response_model=Estimate)
async def read_estimate(selected: DesignDep) -> Estimate:
    project, design = selected
    return await run_in_threadpool(estimate_design, project, design)


@router.get("/preview")
async def read_preview(selected: DesignDep) -> Response:
    project, design = selected
    jpeg = await run_in_threadpool(preview_design, project, design)
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@router.post("/render", response_model=RenderStatus, status_code=202)
async def start_render(selected: DesignDep) -> RenderStatus:
    """Redesigns the photo in the background (an AI edit takes up to a minute); poll /render/status, then fetch /render."""
    project, design = selected
    if cached_render(project, design):
        return RenderStatus(status="done", progress=1.0)

    def work(job: Job) -> None:
        render_design(project, design, job.report)

    return _render_status(project, design, jobs.start(render_key(project, design), work))


@router.get("/render/status", response_model=RenderStatus)
async def read_render_status(selected: DesignDep) -> RenderStatus:
    project, design = selected
    return _render_status(project, design, jobs.get(render_key(project, design)))


@router.get("/render")
async def read_render(selected: DesignDep) -> Response:
    project, design = selected
    result = cached_render(project, design)
    if result is None:
        raise HTTPException(404, "This design has not been rendered yet.")
    return Response(result.jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


def _render_status(project: ProjectRecord, design: Design, job: Job | None) -> RenderStatus:
    result = cached_render(project, design)
    if result:
        return RenderStatus(status="done", progress=1.0, method=result.method, note=result.note)
    return RenderStatus(status="idle") if job is None else RenderStatus(status=job.status, progress=job.progress, label=job.label, error=job.error)


@router.get("/report")
async def read_report(selected: DesignDep) -> Response:
    project, design = selected

    def build() -> bytes:
        redesigned = cached_render(project, design)  # the AI redesign when it has been made, else the quick preview
        after = redesigned.jpeg if redesigned and redesigned.method == "ai" else preview_design(project, design)
        return build_report(project.id, design, project.image, after, estimate_design(project, design))

    pdf = await run_in_threadpool(build)
    logger.info("Report for project %s, design %s built: %d KB", project.id, design.id, len(pdf) // 1024)
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="renovation-estimate-{project.id[:8]}.pdf"'})
