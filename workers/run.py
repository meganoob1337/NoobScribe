"""Start the RQ transcription worker."""
from __future__ import annotations

import logging
import sys

import redis
from rq import SimpleWorker, Queue

from config import get_config
from database.sync_db import init_sync_db
from services.job_queue import QUEUE_NAME
from services.worker_bootstrap import init_worker_resources

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = get_config()
    logger.info("Initializing transcription worker (queue=%s)", QUEUE_NAME)
    init_sync_db()
    init_worker_resources()

    conn = redis.from_url(cfg.redis_url)
    queue = Queue(QUEUE_NAME, connection=conn)
    worker = SimpleWorker([queue], connection=conn)
    logger.info("Worker listening on %s at %s", QUEUE_NAME, cfg.redis_url)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
