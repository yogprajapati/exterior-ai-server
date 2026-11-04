import logging

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health")
async def health(request: Request) -> dict:
    try:
        await request.app.state.db.command("ping")
        db_ok = True
    except Exception:
        logger.exception("Health check: database unreachable")
        db_ok = False
    return {"status": "ok", "db": db_ok}
