import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)
Status = Literal["running", "done", "failed"]


@dataclass
class Job:
    status: Status = "running"
    progress: float = 0.0  # 0..1
    label: str = ""
    error: str | None = None
    note: str | None = None  # something to tell the user about a finished job
    result: object = None  # what the work returned

    def report(self, progress: float, label: str) -> None:
        self.progress, self.label = progress, label


class JobBoard:
    """Slow work (detection, redesign) runs in the background, one job per key; the client polls its status."""

    def __init__(self) -> None:
        self._jobs: dict[tuple, Job] = {}
        self._tasks: set[asyncio.Task] = set()

    def get(self, key: tuple) -> Job | None:
        return self._jobs.get(key)

    def start(self, key: tuple, work: Callable[[Job], object]) -> Job:
        """Start `work(job)` on a worker thread unless it is already running."""
        job = self._jobs.get(key)
        if job and job.status == "running":
            return job
        job = self._jobs[key] = Job()
        task = asyncio.create_task(self._run(key, job, work))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def _run(self, key: tuple, job: Job, work) -> None:
        try:
            job.result = await run_in_threadpool(work, job)
            job.status, job.progress = "done", 1.0
        except Exception as exc:  # shown to the user, so keep it readable
            logger.exception("Job %s failed", key)
            job.status, job.error = "failed", str(exc) or exc.__class__.__name__


jobs = JobBoard()
