import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)
MAX_JOBS = 16
Status = Literal["running", "done", "failed"]


@dataclass
class Job:
    status: Status = "running"
    progress: float = 0.0
    label: str = ""
    error: str | None = None
    note: str | None = None

    def report(self, progress: float, label: str) -> None:
        self.progress, self.label = progress, label


class JobBoard:
    def __init__(self) -> None:
        self._jobs: dict[tuple, Job] = {}
        self._tasks: set[asyncio.Task] = set()

    def get(self, key: tuple) -> Job | None:
        return self._jobs.get(key)

    def start(self, key: tuple, work: Callable[[Job], Any], after: Callable[[Any], Awaitable[None]] | None = None) -> Job:
        job = self._jobs.get(key)
        if job and job.status == "running":
            return job
        self._jobs.pop(key, None)
        job = self._jobs[key] = Job()
        self._evict()
        task = asyncio.create_task(self._run(key, job, work, after))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    def _evict(self) -> None:
        finished = [key for key, job in self._jobs.items() if job.status != "running"]
        for key in finished[: max(len(self._jobs) - MAX_JOBS, 0)]:
            del self._jobs[key]

    async def _run(self, key: tuple, job: Job, work: Callable[[Job], Any], after: Callable[[Any], Awaitable[None]] | None) -> None:
        try:
            result = await run_in_threadpool(work, job)
            if after:
                await after(result)
            job.status, job.progress = "done", 1.0
        except Exception as exc:
            logger.exception("Job %s failed", key)
            job.status, job.error = "failed", str(exc) or exc.__class__.__name__


jobs = JobBoard()
