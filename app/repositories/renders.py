from datetime import datetime

from pymongo.asynchronous.database import AsyncDatabase

from app.repositories.projects import utc_now
from app.schemas.render import RenderResult


class RenderRepository:
    def __init__(self, db: AsyncDatabase):
        self._renders = db.renders

    async def get(self, project_id: str, design_id: str, version: datetime) -> RenderResult | None:
        doc = await self._renders.find_one({"_id": f"{project_id}:{design_id}", "version": version})
        return None if doc is None else RenderResult(doc["jpeg"], doc["method"], doc["note"])

    async def save(self, project_id: str, design_id: str, version: datetime, result: RenderResult) -> None:
        key = f"{project_id}:{design_id}"
        if await self._renders.find_one({"_id": key, "version": {"$gt": version}}, {"_id": 1}):
            return
        await self._renders.replace_one(
            {"_id": key},
            {"_id": key, "version": version, "jpeg": result.jpeg, "method": result.method, "note": result.note, "created_at": utc_now()},
            upsert=True,
        )
