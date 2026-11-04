import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import catalog, detection, health, outputs, projects
from app.core.config import settings
from app.core.database import create_client, init_indexes
from app.core.logging import configure_logging
from app.services.catalog import TEXTURE_DIR
from app.services.render import preload_textures

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = create_client()
    app.state.db = client[settings.mongodb_db]
    await init_indexes(app.state.db)
    preload_textures()
    logger.info("Started: database '%s' ready, textures preloaded", settings.mongodb_db)
    yield
    await client.close()
    logger.info("Stopped")


app = FastAPI(title="Exterior Renovation Assistant", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(RequestValidationError)
async def log_invalid_request(request: Request, exc: RequestValidationError):
    logger.warning("%s %s rejected: %s", request.method, request.url.path, [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()][:5])
    return await request_validation_exception_handler(request, exc)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("%s %s failed", request.method, request.url.path)
        raise
    level = logging.DEBUG if request.url.path == "/health" else logging.INFO  # keep-alive pings would drown the log
    logger.log(level, "%s %s -> %d in %.0f ms", request.method, request.url.path, response.status_code, (time.perf_counter() - started) * 1000)
    return response


for module in (health, catalog, projects, detection, outputs):
    app.include_router(module.router)
app.mount("/textures", StaticFiles(directory=TEXTURE_DIR), name="textures")
