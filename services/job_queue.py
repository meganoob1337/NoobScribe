"""Redis Queue helpers for transcription jobs."""
from __future__ import annotations

import logging

import redis
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job

from config import get_config

logger = logging.getLogger(__name__)

QUEUE_NAME = "transcription"

# RQ statuses where the worker is done or metadata is gone — safe to re-enqueue.
RQ_TERMINAL_STATUSES = frozenset({"finished", "failed", "canceled", "stopped"})


def get_redis_connection() -> redis.Redis:
    return redis.from_url(get_config().redis_url)


def check_redis_connection() -> bool:
    try:
        conn = get_redis_connection()
        conn.ping()
        return True
    except Exception as e:
        logger.debug("Redis ping failed: %s", e)
        return False


def enqueue_transcription_job(job_id: str) -> str:
    """Enqueue a transcription job; returns the RQ job id."""
    queue = get_transcription_queue()
    from workers.transcription_tasks import process_transcription_job

    rq_job = queue.enqueue(
        process_transcription_job,
        job_id,
        job_timeout="24h",
        result_ttl=86400,
        failure_ttl=86400,
    )
    return rq_job.id


def rq_job_exists(rq_job_id: str) -> bool:
    """True while RQ still retains job metadata (result/failure TTL)."""
    if not rq_job_id:
        return False
    try:
        conn = get_redis_connection()
        Job.fetch(rq_job_id, connection=conn)
        return True
    except NoSuchJobError:
        return False
    except Exception as e:
        logger.debug("RQ job fetch failed for %s: %s", rq_job_id, e)
        return False


def get_transcription_queue():
    """Return the transcription RQ queue."""
    conn = get_redis_connection()
    return Queue(QUEUE_NAME, connection=conn)


def get_queue_depth() -> int:
    """Number of jobs waiting in the transcription queue."""
    try:
        return get_transcription_queue().count
    except Exception as e:
        logger.debug("Queue depth check failed: %s", e)
        return 0


def count_workers_for_queue() -> int:
    """Count RQ workers registered for the transcription queue."""
    try:
        from rq.worker import Worker

        conn = get_redis_connection()
        return sum(
            1 for worker in Worker.all(connection=conn) if QUEUE_NAME in worker.queue_names()
        )
    except Exception as e:
        logger.debug("Worker count check failed: %s", e)
        return 0


def get_queue_stats() -> dict:
    """Redis queue health: worker count and pending job depth."""
    redis_ok = check_redis_connection()
    if not redis_ok:
        return {
            "redis_ok": False,
            "workers_registered": 0,
            "queue_depth": 0,
        }
    return {
        "redis_ok": True,
        "workers_registered": count_workers_for_queue(),
        "queue_depth": get_queue_depth(),
    }


def get_rq_job_status(rq_job_id: str) -> Optional[str]:
    """Return RQ job status string, or None if metadata is missing."""
    if not rq_job_id:
        return None
    try:
        conn = get_redis_connection()
        job = Job.fetch(rq_job_id, connection=conn)
        return job.get_status()
    except NoSuchJobError:
        return None
    except Exception as e:
        logger.debug("RQ job status fetch failed for %s: %s", rq_job_id, e)
        return None


def rq_job_needs_reenqueue(rq_status: Optional[str]) -> bool:
    """True when a DB-queued job should get a fresh RQ enqueue."""
    if rq_status is None:
        return True
    return rq_status in RQ_TERMINAL_STATUSES


def cancel_rq_job(rq_job_id: str) -> None:
    """Best-effort RQ cancel for queued jobs; running jobs rely on cooperative DB cancel."""
    if not rq_job_id:
        return
    try:
        conn = get_redis_connection()
        job = Job.fetch(rq_job_id, connection=conn)
        job.cancel()
        try:
            from rq.command import send_stop_job_command

            send_stop_job_command(conn, rq_job_id)
        except Exception:
            pass
    except NoSuchJobError:
        pass
    except Exception as e:
        logger.warning("Failed to cancel RQ job %s: %s", rq_job_id, e)
