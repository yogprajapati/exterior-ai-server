import logging

from app.core.config import settings


def configure_logging() -> None:
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    logging.getLogger("uvicorn.access").disabled = True  # replaced by the request log in main.py, which adds timings
