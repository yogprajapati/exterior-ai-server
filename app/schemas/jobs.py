from typing import Literal

from pydantic import BaseModel

from app.schemas.project import RegionType


class JobStatus(BaseModel):
    status: Literal["idle", "running", "done", "failed"]
    progress: float = 0.0
    label: str = ""
    error: str | None = None
    note: str | None = None


class RenderStatus(JobStatus):
    method: Literal["ai", "preview"] | None = None  # how the finished image was made; `note` says why when it is a preview


class FindRequest(BaseModel):
    phrase: str
    type: RegionType
