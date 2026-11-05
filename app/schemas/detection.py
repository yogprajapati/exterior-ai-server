from pydantic import BaseModel

from app.schemas.project import Region


class StoredDetection(BaseModel):
    regions: list[Region]
    note: str | None = None
