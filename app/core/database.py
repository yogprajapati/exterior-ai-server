from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import settings

PROJECT_TTL_SECONDS = 30 * 86_400


def create_client() -> AsyncMongoClient:
    return AsyncMongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=8000)


async def init_indexes(db: AsyncDatabase) -> None:
    # TTL keeps the free-tier database from filling with abandoned anonymous projects
    await db.projects.create_index("updated_at", expireAfterSeconds=PROJECT_TTL_SECONDS)
