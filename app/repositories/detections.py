from pymongo.asynchronous.database import AsyncDatabase

from app.repositories.projects import utc_now
from app.schemas.detection import StoredDetection


class DetectionRepository:
    def __init__(self, db: AsyncDatabase):
        self._detections = db.detections

    async def get(self, digest: str) -> StoredDetection | None:
        doc = await self._detections.find_one({"_id": digest})
        return None if doc is None else StoredDetection.model_validate(doc)

    async def save(self, digest: str, detection: StoredDetection) -> None:
        await self._detections.replace_one({"_id": digest}, {"_id": digest, **detection.model_dump(), "created_at": utc_now()}, upsert=True)
