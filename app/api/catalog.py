from fastapi import APIRouter

from app.schemas.catalog import Material
from app.services.catalog import get_catalog

router = APIRouter(prefix="/api/catalog", tags=["catalog"])


@router.get("", response_model=list[Material])
async def list_materials() -> list[Material]:
    return list(get_catalog().values())
