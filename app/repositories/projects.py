from dataclasses import dataclass
from datetime import UTC, datetime

from pymongo.asynchronous.database import AsyncDatabase

from app.schemas.project import ProjectState


@dataclass(frozen=True)
class ProjectRecord:
    id: str
    width: int
    height: int
    created_at: datetime
    updated_at: datetime
    image: bytes
    state: ProjectState


def utc_now() -> datetime:
    now = datetime.now(UTC)
    return now.replace(microsecond=now.microsecond // 1000 * 1000)  # MongoDB stores milliseconds


class ProjectRepository:
    def __init__(self, db: AsyncDatabase):
        self._projects = db.projects

    async def create(self, project_id: str, image: bytes, width: int, height: int) -> None:
        now = utc_now()
        await self._projects.insert_one({
            "_id": project_id, "image": image, "width": width, "height": height, "created_at": now, "updated_at": now,
            **ProjectState().model_dump(),
        })

    async def get(self, project_id: str) -> ProjectRecord | None:
        doc = await self._projects.find_one({"_id": project_id})
        if doc is None:
            return None
        # MongoDB returns naive UTC datetimes
        stamp = lambda key: doc[key].replace(tzinfo=UTC)
        return ProjectRecord(project_id, doc["width"], doc["height"], stamp("created_at"), stamp("updated_at"), doc["image"], ProjectState.model_validate(doc))

    async def save_fields(self, project_id: str, fields: dict) -> datetime:
        updated_at = utc_now()
        await self._projects.update_one({"_id": project_id}, {"$set": {**fields, "updated_at": updated_at}})
        return updated_at
