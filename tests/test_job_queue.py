"""Unit tests for transcription job queue helpers."""
from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.job_queue import rq_job_needs_reenqueue
from services.transcription_jobs import (
    JOB_STATUS_CANCELLED,
    JOB_STATUS_FAILED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_RUNNING,
    reconcile_stale_queued_jobs,
    update_job_progress_sync,
)


class TestRqJobNeedsReenqueue(unittest.TestCase):
    def test_missing_or_terminal_statuses(self):
        self.assertTrue(rq_job_needs_reenqueue(None))
        self.assertTrue(rq_job_needs_reenqueue("finished"))
        self.assertTrue(rq_job_needs_reenqueue("failed"))
        self.assertTrue(rq_job_needs_reenqueue("canceled"))
        self.assertTrue(rq_job_needs_reenqueue("stopped"))

    def test_active_statuses(self):
        self.assertFalse(rq_job_needs_reenqueue("queued"))
        self.assertFalse(rq_job_needs_reenqueue("started"))
        self.assertFalse(rq_job_needs_reenqueue("deferred"))
        self.assertFalse(rq_job_needs_reenqueue("scheduled"))


class TestCreateAndEnqueueJob(unittest.IsolatedAsyncioTestCase):
    @patch("services.transcription_jobs.enqueue_transcription_job")
    async def test_commits_before_enqueue(self, mock_enqueue):
        from services.transcription_jobs import create_and_enqueue_job

        order = []

        async def track_commit():
            order.append("commit")

        def track_enqueue(job_id):
            order.append("enqueue")
            return "rq-123"

        mock_enqueue.side_effect = track_enqueue
        session = MagicMock()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.commit = AsyncMock(side_effect=track_commit)

        job_id = await create_and_enqueue_job(
            session,
            job_type="recording_transcribe",
            recording_id="rec-1",
            params={"diarize": True},
        )

        self.assertTrue(job_id)
        self.assertEqual(order, ["commit", "enqueue", "commit"])
        mock_enqueue.assert_called_once()


class TestReconcileStaleQueuedJobs(unittest.IsolatedAsyncioTestCase):
    @patch("services.transcription_jobs.enqueue_transcription_job", return_value="rq-new")
    @patch("services.transcription_jobs.get_rq_job_status", return_value="finished")
    async def test_re_enqueues_stale_finished_rq_job(self, _mock_status, mock_enqueue):
        stale_job = MagicMock()
        stale_job.id = "job-1"
        stale_job.rq_job_id = "rq-old"
        stale_job.status = JOB_STATUS_QUEUED
        stale_job.created_at = datetime.now(timezone.utc) - timedelta(minutes=5)

        session = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [stale_job]
        session.execute = AsyncMock(return_value=result)
        session.flush = AsyncMock()

        stats = await reconcile_stale_queued_jobs(session, stale_seconds=60)

        self.assertEqual(stats["examined"], 1)
        self.assertEqual(stats["re_enqueued"], 1)
        self.assertEqual(stats["skipped"], 0)
        mock_enqueue.assert_called_once_with("job-1")
        self.assertEqual(stale_job.rq_job_id, "rq-new")

    @patch("services.transcription_jobs.get_rq_job_status", return_value="queued")
    async def test_skips_jobs_still_waiting_in_redis(self, _mock_status):
        waiting_job = MagicMock()
        waiting_job.id = "job-2"
        waiting_job.rq_job_id = "rq-wait"
        waiting_job.status = JOB_STATUS_QUEUED
        waiting_job.created_at = datetime.now(timezone.utc) - timedelta(minutes=5)

        session = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [waiting_job]
        session.execute = AsyncMock(return_value=result)

        stats = await reconcile_stale_queued_jobs(session, stale_seconds=60)

        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(stats["re_enqueued"], 0)

    @patch(
        "services.transcription_jobs.enqueue_transcription_job",
        side_effect=RuntimeError("redis down"),
    )
    @patch("services.transcription_jobs.get_rq_job_status", return_value=None)
    async def test_marks_failed_when_reenqueue_errors(self, _mock_status, _mock_enqueue):
        stale_job = MagicMock()
        stale_job.id = "job-3"
        stale_job.rq_job_id = None
        stale_job.status = JOB_STATUS_QUEUED
        stale_job.created_at = datetime.now(timezone.utc) - timedelta(minutes=5)

        session = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [stale_job]
        session.execute = AsyncMock(return_value=result)
        session.flush = AsyncMock()

        stats = await reconcile_stale_queued_jobs(session, stale_seconds=60)

        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stale_job.status, JOB_STATUS_FAILED)
        self.assertIn("Reconcile failed", stale_job.error_message)


class TestUpdateJobProgressSync(unittest.TestCase):
    def test_skips_write_when_cancelled(self):
        job = MagicMock()
        job.status = JOB_STATUS_CANCELLED
        session = MagicMock()
        session.get.return_value = job

        ok = update_job_progress_sync(session, "job-1", stage="transcribing", current=1, total=2)

        self.assertFalse(ok)
        session.flush.assert_not_called()

    def test_writes_progress_when_running(self):
        job = MagicMock()
        job.status = JOB_STATUS_RUNNING
        job.progress_stage = None
        session = MagicMock()
        session.get.return_value = job

        ok = update_job_progress_sync(session, "job-1", stage="transcribing", current=2, total=4)

        self.assertTrue(ok)
        self.assertEqual(job.progress_stage, "transcribing")
        self.assertEqual(job.progress_current, 2)
        self.assertEqual(job.progress_total, 4)
        session.flush.assert_called_once()


if __name__ == "__main__":
    unittest.main()
