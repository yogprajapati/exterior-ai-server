from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import ServerSelectionTimeoutError

from app.core.config import settings

TTL_SECONDS = 30 * 86_400
TTL_FIELDS = {"projects": "updated_at", "renders": "created_at", "detections": "created_at"}


def create_client() -> AsyncMongoClient:
    return AsyncMongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=8000)


async def init_indexes(db: AsyncDatabase) -> None:
    try:
        for collection, field in TTL_FIELDS.items():
            await db[collection].create_index(field, expireAfterSeconds=TTL_SECONDS)
    except ServerSelectionTimeoutError:
        raise RuntimeError("Cannot reach MongoDB. Start it, or set MONGODB_URI in server/.env.") from None
