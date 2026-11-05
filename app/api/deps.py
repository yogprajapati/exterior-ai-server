import logging
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from app.repositories.detections import DetectionRepository
from app.repositories.projects import ProjectRecord, ProjectRepository
from app.repositories.renders import RenderRepository
from app.schemas.project import Design

logger = logging.getLogger(__name__)


def get_repo(request: Request) -> ProjectRepository:
    return ProjectRepository(request.app.state.db)


def get_detections(request: Request) -> DetectionRepository:
    return DetectionRepository(request.app.state.db)


def get_renders(request: Request) -> RenderRepository:
    return RenderRepository(request.app.state.db)


RepoDep = Annotated[ProjectRepository, Depends(get_repo)]
DetectionRepoDep = Annotated[DetectionRepository, Depends(get_detections)]
RenderRepoDep = Annotated[RenderRepository, Depends(get_renders)]


async def get_project(project_id: str, repo: RepoDep) -> ProjectRecord:
    project = await repo.get(project_id)
    if project is None:
        logger.info("Project %s not found", project_id)
        raise HTTPException(404, "Project not found.")
    return project


ProjectDep = Annotated[ProjectRecord, Depends(get_project)]


def get_design(design_id: str, project: ProjectDep) -> tuple[ProjectRecord, Design]:
    design = next((d for d in project.state.designs if d.id == design_id), None)
    if design is None:
        logger.info("Design %s not found in project %s", design_id, project.id)
        raise HTTPException(404, "Design not found.")
    return project, design


DesignDep = Annotated[tuple[ProjectRecord, Design], Depends(get_design)]
