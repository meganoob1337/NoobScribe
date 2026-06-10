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
    conn = get_redis_connection()
    queue = Queue(QUEUE_NAME, connection=conn)
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
