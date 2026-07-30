#!/usr/bin/env python3
"""
Evaluate NoobScribe end-to-end transcription + diarization performance.

How it works:
1) Edit RECORDING_IDS below to point to existing recording UUIDs.
2) For each UUID, trigger /v1/recordings/{recording_id}/transcribe with diarize=true.
3) Poll /v1/transcription-jobs/{job_id} until terminal status.
4) Measure per-recording and total end-to-end duration.
5) Sample worker-container VRAM and report overall peak plus per-stage peaks
   (model load / diarization / transcription) based on job progress_stage windows.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests

RESULTS_DIR = Path(__file__).resolve().parent / "performance_results"

# ---------------------------------------------------------------------------
# Configure your benchmark input here (existing recording UUIDs).
# ---------------------------------------------------------------------------
RECORDING_IDS = [
    "925c463c-1916-4a52-9230-65b1d716085e",
]

DEFAULT_WORKER_CANDIDATES = [
    "noobscribe-worker",
    "noobscribe-worker-no-traefik",
    "noobscribe-worker-cpu",
    "noobscribe-worker-cpu-no-traefik",
]

TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}


@dataclass
class RecordingRunResult:
    recording_id: str
    ok: bool
    status: str
    job_id: Optional[str]
    trigger_seconds: Optional[float]
    processing_seconds: Optional[float]
    end_to_end_seconds: Optional[float]
    loading_diarization_model_seconds: Optional[float]
    diarization_seconds: Optional[float]
    loading_transcription_model_seconds: Optional[float]
    transcription_seconds: Optional[float]
    peak_vram_mib: Optional[int]
    peak_vram_diarization_mib: Optional[int]
    peak_vram_transcription_mib: Optional[int]
    error: Optional[str]


def _run_command(cmd: Sequence[str], timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _ensure_recording_ids(recording_ids: Sequence[str]) -> List[str]:
    if not recording_ids:
        raise ValueError(
            "RECORDING_IDS is empty. Add at least one existing recording UUID at the top of this script."
        )

    resolved: List[str] = []
    for raw in recording_ids:
        rid = str(raw).strip()
        if not rid:
            raise ValueError("Recording UUID list contains an empty entry.")
        try:
            uuid.UUID(rid)
        except ValueError as exc:
            raise ValueError(f"Invalid recording UUID: {rid}") from exc
        resolved.append(rid)
    return resolved


def _list_running_container_names() -> List[str]:
    cp = _run_command(["docker", "ps", "--format", "{{.Names}}"])
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr.strip() or "Failed to list Docker containers.")
    return [line.strip() for line in cp.stdout.splitlines() if line.strip()]


def default_base_url() -> str:
    """
    Build default API URL from NOOBSCRIBE_HOST when present.

    Examples:
    - NOOBSCRIBE_HOST=noobscribe.example.com -> https://noobscribe.example.com
    - NOOBSCRIBE_HOST=http://localhost:8000 -> http://localhost:8000
    - unset -> http://localhost:8000
    """
    raw = (os.getenv("NOOBSCRIBE_HOST") or "").strip()
    if not raw:
        return "http://localhost:8000"

    if raw.startswith(("http://", "https://")):
        return raw.rstrip("/")

    return f"https://{raw.rstrip('/')}"


def resolve_worker_container(explicit_name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if shutil.which("docker") is None:
        return None, "docker CLI is not available"

    try:
        running = set(_list_running_container_names())
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)

    if explicit_name:
        if explicit_name in running:
            return explicit_name, None
        return None, f"Requested worker container '{explicit_name}' is not running"

    for name in DEFAULT_WORKER_CANDIDATES:
        if name in running:
            return name, None

    return None, "No known NoobScribe worker container is currently running"


class WorkerVramMonitor:
    """
    Samples GPU VRAM usage attributed to processes inside a Docker container.

    Strategy:
    - get container pids via `docker top <container> -eo pid`
    - get GPU usage per pid via `nvidia-smi --query-compute-apps=pid,used_memory`
    - sum usage for pids present in the container
    """

    def __init__(self, container_name: Optional[str], sample_interval_s: float = 0.5) -> None:
        self.container_name = container_name
        self.sample_interval_s = sample_interval_s
        self.available = False
        self.unavailable_reason: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._samples: List[Tuple[float, int]] = []
        self._lock = threading.Lock()

    def _set_unavailable(self, reason: str) -> None:
        self.available = False
        self.unavailable_reason = reason

    def start(self) -> None:
        if self.container_name is None:
            self._set_unavailable("Worker container not resolved")
            return
        if shutil.which("docker") is None:
            self._set_unavailable("docker CLI is not available")
            return
        if shutil.which("nvidia-smi") is None:
            self._set_unavailable("nvidia-smi is not available")
            return

        cp = _run_command(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.container_name],
            timeout=5,
        )
        if cp.returncode != 0:
            self._set_unavailable(cp.stderr.strip() or f"Could not inspect container {self.container_name}")
            return
        if cp.stdout.strip().lower() != "true":
            self._set_unavailable(f"Container '{self.container_name}' is not running")
            return

        probe = _run_command(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            timeout=5,
        )
        if probe.returncode != 0:
            self._set_unavailable(probe.stderr.strip() or "nvidia-smi query failed")
            return

        self.available = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.available:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.sample_interval_s * 4, 1.0))

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            now = time.monotonic()
            usage = self._sample_container_vram_mib()
            with self._lock:
                self._samples.append((now, usage))
            self._stop_event.wait(self.sample_interval_s)

    def _sample_container_vram_mib(self) -> int:
        container_pids = self._get_container_pids()
        if not container_pids:
            return 0

        pid_to_mem = self._get_gpu_memory_by_pid()
        return sum(mem for pid, mem in pid_to_mem.items() if pid in container_pids)

    def _get_container_pids(self) -> set[int]:
        cp = _run_command(["docker", "top", self.container_name, "-eo", "pid"], timeout=5)
        if cp.returncode != 0:
            return set()

        pids: set[int] = set()
        for line in cp.stdout.splitlines():
            raw = line.strip()
            if not raw or raw.lower() == "pid":
                continue
            try:
                pids.add(int(raw))
            except ValueError:
                continue
        return pids

    @staticmethod
    def _get_gpu_memory_by_pid() -> Dict[int, int]:
        cp = _run_command(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            timeout=5,
        )
        if cp.returncode != 0:
            return {}

        usage: Dict[int, int] = {}
        for line in cp.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                mem_mib = int(float(parts[1]))
            except ValueError:
                continue
            usage[pid] = mem_mib
        return usage

    def peak_between(self, start_monotonic: float, end_monotonic: float) -> Optional[int]:
        if not self.available:
            return None
        with self._lock:
            values = [mem for ts, mem in self._samples if start_monotonic <= ts <= end_monotonic]
        return max(values) if values else 0

    def global_peak(self) -> Optional[int]:
        if not self.available:
            return None
        with self._lock:
            if not self._samples:
                return 0
            return max(mem for _, mem in self._samples)


def trigger_transcription_job(
    session: requests.Session,
    base_url: str,
    recording_id: str,
    response_format: str,
    language: Optional[str],
    request_timeout_s: float,
) -> Tuple[str, float]:
    endpoint = f"{base_url.rstrip('/')}/v1/recordings/{recording_id}/transcribe"
    form_data = {
        "response_format": response_format,
        "timestamps": "true",
        "word_timestamps": "false",
        "diarize": "true",
        "include_diarization_in_text": "true",
    }
    if language:
        form_data["language"] = language

    trigger_start = time.monotonic()
    response = session.post(
        endpoint,
        data=form_data,
        timeout=request_timeout_s,
    )
    trigger_end = time.monotonic()

    if response.status_code != 202:
        body = response.text.strip()
        raise RuntimeError(
            f"Trigger failed for recording '{recording_id}' with status {response.status_code}: {body[:500]}"
        )

    try:
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Trigger returned non-JSON 202 response for recording '{recording_id}': {exc}"
        ) from exc

    job_id = payload.get("job_id")
    if not job_id:
        raise RuntimeError(f"Trigger response for recording '{recording_id}' does not contain job_id")

    return str(job_id), trigger_end - trigger_start


def wait_for_job(
    session: requests.Session,
    base_url: str,
    job_id: str,
    poll_interval_s: float,
    timeout_s: float,
    request_timeout_s: float,
) -> Tuple[Dict[str, object], Dict[str, Tuple[float, float]]]:
    """
    Poll until terminal status.

    Returns:
      (payload, stage_windows) where stage_windows maps stage name ->
      (start_monotonic, end_monotonic) observed via progress_stage transitions.
    """
    endpoint = f"{base_url.rstrip('/')}/v1/transcription-jobs/{job_id}"
    started = time.monotonic()
    last_status: Optional[str] = None
    last_stage: Optional[str] = None
    stage_starts: Dict[str, float] = {}
    stage_windows: Dict[str, Tuple[float, float]] = {}

    while True:
        now = time.monotonic()
        response = session.get(endpoint, timeout=request_timeout_s)
        response.raise_for_status()
        payload = response.json()
        status = str(payload.get("status", "unknown"))
        stage = payload.get("progress_stage")
        stage_name = str(stage) if stage else None

        if status != last_status or stage_name != last_stage:
            if last_stage is not None and last_stage in stage_starts and last_stage not in stage_windows:
                stage_windows[last_stage] = (stage_starts[last_stage], now)
            if stage_name is not None and stage_name not in stage_starts:
                stage_starts[stage_name] = now

            progress_current = payload.get("progress_current")
            progress_total = payload.get("progress_total")
            if progress_current is not None and progress_total is not None:
                print(
                    f"    job={job_id} status={status} stage={stage_name} "
                    f"progress={progress_current}/{progress_total}"
                )
            else:
                print(f"    job={job_id} status={status} stage={stage_name}")
            last_status = status
            last_stage = stage_name

        if status in TERMINAL_JOB_STATUSES:
            end = time.monotonic()
            if last_stage is not None and last_stage in stage_starts and last_stage not in stage_windows:
                stage_windows[last_stage] = (stage_starts[last_stage], end)
            return payload, stage_windows

        elapsed = time.monotonic() - started
        if elapsed > timeout_s:
            raise TimeoutError(f"Timed out waiting for job {job_id} after {timeout_s:.1f}s")

        time.sleep(poll_interval_s)


def evaluate_recordings(
    recording_ids: Sequence[str],
    base_url: str,
    poll_interval_s: float,
    job_timeout_s: float,
    request_timeout_s: float,
    vram_monitor: WorkerVramMonitor,
    response_format: str,
    language: Optional[str],
) -> List[RecordingRunResult]:
    results: List[RecordingRunResult] = []
    session = requests.Session()

    for recording_id in recording_ids:
        print(f"\nProcessing recording: {recording_id}")
        run_start = time.monotonic()
        job_id: Optional[str] = None
        trigger_seconds: Optional[float] = None
        status = "failed"
        error: Optional[str] = None
        stage_windows: Dict[str, Tuple[float, float]] = {}

        try:
            job_id, trigger_seconds = trigger_transcription_job(
                session=session,
                base_url=base_url,
                recording_id=recording_id,
                response_format=response_format,
                language=language,
                request_timeout_s=request_timeout_s,
            )
            print(f"  queued job_id={job_id} trigger_seconds={trigger_seconds:.3f}")

            payload, stage_windows = wait_for_job(
                session=session,
                base_url=base_url,
                job_id=job_id,
                poll_interval_s=poll_interval_s,
                timeout_s=job_timeout_s,
                request_timeout_s=request_timeout_s,
            )
            status = str(payload.get("status", "unknown"))
            if status != "completed":
                error = payload.get("error_message") or f"Job terminal status: {status}"
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            status = "failed"

        run_end = time.monotonic()
        end_to_end_seconds = run_end - run_start
        peak_vram = vram_monitor.peak_between(run_start, run_end)

        loading_diarization_model_seconds: Optional[float] = None
        diarization_seconds: Optional[float] = None
        loading_transcription_model_seconds: Optional[float] = None
        transcription_seconds: Optional[float] = None
        peak_vram_diarization: Optional[int] = None
        peak_vram_transcription: Optional[int] = None

        load_diar_window = stage_windows.get("loading_diarization_model")
        if load_diar_window is not None:
            loading_diarization_model_seconds = max(
                load_diar_window[1] - load_diar_window[0], 0.0
            )

        diar_window = stage_windows.get("diarization")
        if diar_window is not None:
            diarization_seconds = max(diar_window[1] - diar_window[0], 0.0)
            if vram_monitor.available:
                peak_vram_diarization = vram_monitor.peak_between(diar_window[0], diar_window[1])

        load_asr_window = stage_windows.get("loading_transcription_model")
        if load_asr_window is not None:
            loading_transcription_model_seconds = max(
                load_asr_window[1] - load_asr_window[0], 0.0
            )

        asr_window = stage_windows.get("transcribing")
        if asr_window is not None:
            transcription_seconds = max(asr_window[1] - asr_window[0], 0.0)
            if vram_monitor.available:
                peak_vram_transcription = vram_monitor.peak_between(asr_window[0], asr_window[1])

        processing_seconds: Optional[float] = None
        if trigger_seconds is not None:
            processing_seconds = max(end_to_end_seconds - trigger_seconds, 0.0)

        ok = status == "completed" and error is None
        results.append(
            RecordingRunResult(
                recording_id=recording_id,
                ok=ok,
                status=status,
                job_id=job_id,
                trigger_seconds=trigger_seconds,
                processing_seconds=processing_seconds,
                end_to_end_seconds=end_to_end_seconds,
                loading_diarization_model_seconds=loading_diarization_model_seconds,
                diarization_seconds=diarization_seconds,
                loading_transcription_model_seconds=loading_transcription_model_seconds,
                transcription_seconds=transcription_seconds,
                peak_vram_mib=peak_vram,
                peak_vram_diarization_mib=peak_vram_diarization,
                peak_vram_transcription_mib=peak_vram_transcription,
                error=error,
            )
        )

    return results


def build_summary(
    results: Sequence[RecordingRunResult],
    total_runtime_s: float,
    worker_container: Optional[str],
    monitor: WorkerVramMonitor,
    url: str,
) -> Dict[str, object]:
    success_count = sum(1 for r in results if r.ok)
    e2e_values = [
        r.end_to_end_seconds
        for r in results
        if r.ok and r.end_to_end_seconds is not None
    ]
    avg_e2e = (sum(e2e_values) / len(e2e_values)) if e2e_values else None
    max_e2e = max(e2e_values) if e2e_values else None
    input_recording_ids = [r.recording_id for r in results]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "url": url,
        "input_recording_ids": input_recording_ids,
        "worker_container": worker_container,
        "vram_monitoring": {
            "available": monitor.available,
            "reason_unavailable": monitor.unavailable_reason,
            "global_peak_mib": monitor.global_peak(),
        },
        "aggregate": {
            "total_recordings": len(results),
            "succeeded": success_count,
            "failed": len(results) - success_count,
            "total_wall_seconds": total_runtime_s,
            "avg_e2e_seconds": avg_e2e,
            "max_e2e_seconds": max_e2e,
        },
        "recordings": [r.__dict__ for r in results],
    }


def save_summary(summary: Dict[str, object], results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = results_dir / f"{stamp}.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return out_path


def print_summary(
    results: Sequence[RecordingRunResult],
    total_runtime_s: float,
    worker_container: Optional[str],
    monitor: WorkerVramMonitor,
    url: str,
) -> Path:
    print("\n" + "=" * 90)
    print("NoobScribe performance summary")
    print("=" * 90)
    print(f"API URL: {url}")
    print(f"Worker container: {worker_container or 'unresolved'}")
    if monitor.available:
        print(f"Global peak VRAM: {monitor.global_peak()} MiB")
    else:
        print(f"Global peak VRAM: unavailable ({monitor.unavailable_reason})")
    print()

    for r in results:
        peak_vram_text = f"{r.peak_vram_mib} MiB" if r.peak_vram_mib is not None else "n/a"
        diar_vram_text = (
            f"{r.peak_vram_diarization_mib} MiB"
            if r.peak_vram_diarization_mib is not None
            else "n/a"
        )
        asr_vram_text = (
            f"{r.peak_vram_transcription_mib} MiB"
            if r.peak_vram_transcription_mib is not None
            else "n/a"
        )
        trigger_text = f"{r.trigger_seconds:.3f}s" if r.trigger_seconds is not None else "n/a"
        processing_text = (
            f"{r.processing_seconds:.3f}s"
            if r.processing_seconds is not None
            else "n/a"
        )
        load_diar_time_text = (
            f"{r.loading_diarization_model_seconds:.3f}s"
            if r.loading_diarization_model_seconds is not None
            else "n/a"
        )
        diar_time_text = (
            f"{r.diarization_seconds:.3f}s" if r.diarization_seconds is not None else "n/a"
        )
        load_asr_time_text = (
            f"{r.loading_transcription_model_seconds:.3f}s"
            if r.loading_transcription_model_seconds is not None
            else "n/a"
        )
        asr_time_text = (
            f"{r.transcription_seconds:.3f}s" if r.transcription_seconds is not None else "n/a"
        )
        e2e_text = f"{r.end_to_end_seconds:.3f}s" if r.end_to_end_seconds is not None else "n/a"
        print(
            f"- recording_id={r.recording_id} status={r.status} trigger={trigger_text} "
            f"processing={processing_text} e2e={e2e_text} "
            f"load_diarization_model={load_diar_time_text} diarization={diar_time_text} "
            f"load_transcription_model={load_asr_time_text} transcription={asr_time_text} "
            f"peak_vram={peak_vram_text} "
            f"peak_vram_diarization={diar_vram_text} peak_vram_transcription={asr_vram_text}"
        )
        if r.error:
            print(f"  error: {r.error}")

    summary = build_summary(results, total_runtime_s, worker_container, monitor, url)
    aggregate = summary["aggregate"]
    assert isinstance(aggregate, dict)

    print("\nAggregate:")
    print(f"  total_recordings={aggregate['total_recordings']}")
    print(f"  succeeded={aggregate['succeeded']}")
    print(f"  failed={aggregate['failed']}")
    print(f"  total_wall_seconds={total_runtime_s:.3f}")
    avg_e2e = aggregate["avg_e2e_seconds"]
    max_e2e = aggregate["max_e2e_seconds"]
    print(f"  avg_e2e_seconds={avg_e2e:.3f}" if avg_e2e is not None else "  avg_e2e_seconds=n/a")
    print(f"  max_e2e_seconds={max_e2e:.3f}" if max_e2e is not None else "  max_e2e_seconds=n/a")

    out_path = save_summary(summary)
    print("\nJSON summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved results to: {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    default_url = default_base_url()
    parser = argparse.ArgumentParser(
        description="Evaluate NoobScribe transcription+diarization E2E performance."
    )
    parser.add_argument(
        "--recording-ids",
        nargs="+",
        default=None,
        help=(
            "Optional recording UUID list to override RECORDING_IDS. "
            "Example: --recording-ids <uuid1> <uuid2>"
        ),
    )
    parser.add_argument(
        "--url",
        default=default_url,
        help=f"Base NoobScribe API URL (default: {default_url})",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds for job status (default: 2.0)",
    )
    parser.add_argument(
        "--job-timeout",
        type=float,
        default=7200.0,
        help="Maximum seconds to wait per transcription job (default: 7200)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="HTTP request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--worker-container",
        default=None,
        help=(
            "Optional Docker worker container name override "
            "(auto-detection tries noobscribe-worker variants)."
        ),
    )
    parser.add_argument(
        "--vram-sample-interval",
        type=float,
        default=0.5,
        help="VRAM sample interval in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--response-format",
        default="verbose_json",
        choices=["json", "verbose_json", "text", "srt", "vtt"],
        help="Requested transcription format (default: verbose_json)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code (e.g. de, en). Omit for auto language ID.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configured_recording_ids = args.recording_ids if args.recording_ids else RECORDING_IDS

    try:
        recording_ids = _ensure_recording_ids(configured_recording_ids)
    except Exception as exc:  # noqa: BLE001
        print(f"Input validation error: {exc}", file=sys.stderr)
        return 2

    worker_container, worker_resolution_error = resolve_worker_container(args.worker_container)
    if worker_container:
        print(f"Using worker container for VRAM tracking: {worker_container}")
    else:
        print(f"VRAM tracking disabled: {worker_resolution_error}")

    monitor = WorkerVramMonitor(
        container_name=worker_container,
        sample_interval_s=args.vram_sample_interval,
    )
    monitor.start()
    if monitor.available:
        print("VRAM monitor started.")
    else:
        print(f"VRAM monitor unavailable: {monitor.unavailable_reason}")

    total_start = time.monotonic()
    try:
        results = evaluate_recordings(
            recording_ids=recording_ids,
            base_url=args.url,
            poll_interval_s=args.poll_interval,
            job_timeout_s=args.job_timeout,
            request_timeout_s=args.request_timeout,
            vram_monitor=monitor,
            response_format=args.response_format,
            language=args.language,
        )
    finally:
        monitor.stop()
    total_end = time.monotonic()

    total_runtime_s = total_end - total_start
    print_summary(
        results=results,
        total_runtime_s=total_runtime_s,
        worker_container=worker_container,
        monitor=monitor,
        url=args.url,
    )

    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
