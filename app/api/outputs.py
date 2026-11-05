import logging

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from app.api.deps import DesignDep, RenderRepoDep
from app.schemas.estimate import Estimate
from app.schemas.jobs import RenderStatus
from app.schemas.render import RenderResult
from app.services.designs import estimate_design, preview_design, render_design
from app.services.jobs import Job, jobs
from app.services.report import build_report

router = APIRouter(prefix="/api/projects/{project_id}/designs/{design_id}", tags=["outputs"])
logger = logging.getLogger(__name__)


def _done(result: RenderResult) -> RenderStatus:
    return RenderStatus(status="done", progress=1.0, method=result.method, note=result.note)


def _running(job: Job | None) -> RenderStatus:
    return RenderStatus(status="idle") if job is None else RenderStatus.model_validate(job, from_attributes=True)


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
async def start_render(selected: DesignDep, renders: RenderRepoDep) -> RenderStatus:
    project, design = selected
    existing = await renders.get(project.id, design.id, project.updated_at)
    if existing:
        return _done(existing)

    def work(job: Job) -> RenderResult:
        return render_design(project, design, job.report)

    async def save(result: RenderResult) -> None:
        await renders.save(project.id, design.id, project.updated_at, result)

    key = ("render", project.id, design.id, project.updated_at)
    return _running(jobs.start(key, work, save))


@router.get("/render/status", response_model=RenderStatus)
async def read_render_status(selected: DesignDep, renders: RenderRepoDep) -> RenderStatus:
    project, design = selected
    existing = await renders.get(project.id, design.id, project.updated_at)
    if existing:
        return _done(existing)
    return _running(jobs.get(("render", project.id, design.id, project.updated_at)))


@router.get("/render")
async def read_render(selected: DesignDep, renders: RenderRepoDep) -> Response:
    project, design = selected
    existing = await renders.get(project.id, design.id, project.updated_at)
    if existing is None:
        raise HTTPException(404, "This design has not been rendered yet.")
    return Response(existing.jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@router.get("/report")
async def read_report(selected: DesignDep, renders: RenderRepoDep) -> Response:
    project, design = selected
    redesigned = await renders.get(project.id, design.id, project.updated_at)

    def build() -> bytes:
        after = redesigned.jpeg if redesigned and redesigned.method == "ai" else preview_design(project, design)
        return build_report(project.id, design, project.image, after, estimate_design(project, design))

    pdf = await run_in_threadpool(build)
    logger.info("Report for project %s, design %s built: %d KB", project.id, design.id, len(pdf) // 1024)
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="renovation-estimate-{project.id[:8]}.pdf"'})
