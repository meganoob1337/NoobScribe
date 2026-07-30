# Originally from https://github.com/jfgonsalves/parakeet-diarized (commit 6abadfd)
# Copyright (c) jfgonsalves - MIT License
# Modified by meganoob1337 for the NoobScribe project
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from audio import convert_audio_to_wav, cut_audio_segment, get_wav_duration_seconds
from config import get_config, use_cuda
from database.db import get_session, init_db, init_engine
from database.models import Recording as RecordingORM
from database.speakers import SpeakerDB
from models import (
    EmbeddingSnippet,
    EmbeddingSnippetListResponse,
    ModelInfo,
    ModelList,
    SpeakerCreate,
    SpeakerEmbeddingDetail,
    SpeakerEmbeddingListResponse,
    SpeakerList,
    SpeakerNearestMatch,
    SpeakerNearestRequest,
    SpeakerNearestResponse,
    SpeakerRematchResponse,
    SpeakerResponse,
    SpeakerUpdate,
    TranscriptionJobCreateResponse,
)
from routers.jobs import router as jobs_router
from routers.recordings import router as recordings_router
from services.job_queue import check_redis_connection, get_queue_stats
from services.transcription_enqueue import (
    build_transcribe_params,
    enqueue_openai_transcription,
    enqueue_recording_transcription,
)
from services.transcription_jobs import JOB_TYPE_SPEAKER_EXTRACT, reconcile_stale_queued_jobs
from services.rematch_stored_transcriptions import rematch_all_stored_transcriptions
from services.speaker_embedding_snippets import collect_snippets_for_enrolled_embedding
from transcription import load_model

# Initialize logging
logger = logging.getLogger(__name__)

# Global variable for model
asr_model = None

# Global variable for speaker database
speaker_db = None
_speaker_db_init_error: Optional[str] = None

# Get configuration
config = get_config()


def init_speaker_db() -> bool:
    """Initialize Chroma speaker store (idempotent). Returns True on success."""
    global speaker_db, _speaker_db_init_error
    if speaker_db is not None:
        return True
    try:
        speaker_db = SpeakerDB(
            db_path=config.chromadb_path,
            similarity_threshold=config.speaker_similarity_threshold,
        )
        _speaker_db_init_error = None
        logger.info("Speaker database initialized at %s", config.chromadb_path)
        return True
    except Exception as e:
        _speaker_db_init_error = str(e)
        speaker_db = None
        logger.error("Failed to initialize speaker database: %s", _speaker_db_init_error)
        return False


def require_speaker_db() -> SpeakerDB:
    """Return speaker DB or raise 503 with the last init error when unavailable."""
    if not init_speaker_db():
        detail = "Speaker database not initialized"
        if _speaker_db_init_error:
            detail += f": {_speaker_db_init_error}"
        raise HTTPException(status_code=503, detail=detail)
    return speaker_db


class WebUiStaticFiles(StaticFiles):
    """
    Static file handler for /ui that prevents stale mobile-cache assets.
    """

    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        if response.status_code >= 400:
            return response

        # HTML should always be fetched fresh to pick up new asset references.
        if path.endswith(".html") or path == "":
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response

        # Assets revalidate every request; browser may keep local copy but must check server.
        response.headers["Cache-Control"] = "no-cache, max-age=0, must-revalidate"
        return response


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application

    Returns:
        Configured FastAPI app
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global asr_model

        init_engine()
        await init_db()
        logger.info("SQLite database initialized")

        try:
            if use_cuda():
                logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
            else:
                logger.warning("Using CPU for inference (this will be slow)")

            if getattr(config, "use_api", False):
                asr_model = None
                logger.info(
                    "Using remote STT API at %s (model=%s)",
                    config.stt_base_url,
                    config.model_id,
                )
            elif config.queue_enabled or config.skip_model_load:
                asr_model = None
                logger.info(
                    "API skipping ASR model load (transcription handled by workers; queue_enabled=%s)",
                    config.queue_enabled,
                )
            elif config.load_models_on_demand:
                asr_model = None
                logger.info(
                    "API skipping ASR model preload (LOAD_MODELS_ON_DEMAND=1); "
                    "model loads during transcription"
                )
            else:
                asr_model = load_model(config.resolved_asr_model)
                logger.info(
                    "ASR model loaded (%s, backend=%s)",
                    config.model_path or config.asr_model,
                    config.resolved_asr_model.spec.backend,
                )
        except Exception as e:
            logger.error("Error loading ASR model during startup: %s", str(e))

        init_speaker_db()

        if config.diarization_model_path:
            logger.info(
                "Local diarization model path set; offline diarization available without HF token"
            )
        hf_token = config.get_hf_token()
        if hf_token:
            logger.info("HuggingFace access token found; remote diarization models available")
        else:
            logger.info("No HuggingFace access token (use DIARIZATION_MODEL_PATH for offline)")

        if config.queue_enabled and check_redis_connection():
            try:
                from database.db import _session_factory

                if _session_factory is not None:
                    async with _session_factory() as session:
                        result = await reconcile_stale_queued_jobs(session)
                        if result["re_enqueued"] or result["failed"]:
                            logger.info("Startup job reconcile: %s", result)
            except Exception as e:
                logger.warning("Startup job reconcile failed: %s", e)

        yield

    app = FastAPI(title="NoobScribe API", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(recordings_router, prefix="/v1")
    app.include_router(jobs_router, prefix="/v1")

    @app.get("/", include_in_schema=False)
    async def root_redirect_to_ui():
        """Redirect browsers to the Web UI."""
        return RedirectResponse(url="/ui", status_code=302)

    webui_root = Path(__file__).resolve().parent / "webui"
    if webui_root.is_dir():
        app.mount("/ui", WebUiStaticFiles(directory=str(webui_root), html=True), name="ui")

    @app.post(
        "/v1/audio/transcriptions",
        response_model=TranscriptionJobCreateResponse,
        status_code=202,
    )
    async def transcribe_audio(
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
        timestamps: bool = Form(False),
        timestamp_granularities: Optional[List[str]] = Form(None),
        vad_filter: bool = Form(False),
        word_timestamps: bool = Form(False),
        diarize: bool = Form(True),
        include_diarization_in_text: Optional[bool] = Form(None),
        session: AsyncSession = Depends(get_session),
    ):
        """
        Queue transcription of an uploaded audio file. Poll
        ``GET /v1/transcription-jobs/{job_id}`` for status and the result.
        """
        _ = (model, prompt, timestamp_granularities, vad_filter)

        logger.info("Transcription queued: %s, format: %s", file.filename, response_format)

        temp_dir = Path(config.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_file = temp_dir / f"upload_{os.urandom(8).hex()}{Path(file.filename).suffix}"

        try:
            with open(temp_file, "wb") as f:
                content = await file.read()
                f.write(content)

            params = build_transcribe_params(
                language=language,
                response_format=response_format,
                timestamps=timestamps,
                word_timestamps=word_timestamps,
                diarize=diarize,
                include_diarization_in_text=include_diarization_in_text,
                temperature=temperature,
                extra={"temp_file": str(temp_file.resolve())},
            )
            job_id = await enqueue_openai_transcription(session, params)
            return TranscriptionJobCreateResponse(job_id=job_id, status="queued")
        except HTTPException:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except OSError:
                pass
            raise
        except Exception as e:
            logger.error("Error queueing transcription: %s", str(e))
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except OSError:
                pass
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        """
        Check the health of the API and the loaded model
        """
        global asr_model

        queue_stats = get_queue_stats()

        return {
            "status": "ok",
            "version": "1.0.0",
            "model_loaded": (
                asr_model is not None
                or getattr(config, "use_api", False)
                or (
                    getattr(config, "load_models_on_demand", False)
                    and not config.queue_enabled
                    and not config.skip_model_load
                )
            ),
            "speaker_db_initialized": init_speaker_db(),
            "model_id": config.get_transcription_model_name(),
            "asr_model": getattr(config, "asr_model", None),
            "asr_backend": (
                config.resolved_asr_model.spec.backend
                if getattr(config, "resolved_asr_model", None) is not None
                else None
            ),
            "force_cpu": config.force_cpu,
            "cuda_available": use_cuda(),
            "gpu_info": torch.cuda.get_device_name(0) if use_cuda() else None,
            "queue_enabled": config.queue_enabled,
            "redis_ok": queue_stats["redis_ok"],
            "workers_registered": queue_stats["workers_registered"],
            "queue_depth": queue_stats["queue_depth"],
            "config": config.as_dict(),
        }

    @app.get("/v1/models")
    async def list_models():
        """
        List available models (compatibility with OpenAI API)
        """
        from services.asr_models import list_local_asr_models

        if getattr(config, "use_api", False):
            models = [
                ModelInfo(
                    id=config.model_id,
                    created=1677649963,
                    owned_by="noobscribe",
                    root=config.model_id,
                    permission=[
                        {
                            "id": "modelperm-remote",
                            "object": "model_permission",
                            "created": 1677649963,
                            "allow_create_engine": False,
                            "allow_sampling": True,
                            "allow_logprobs": True,
                            "allow_search_indices": False,
                            "allow_view": True,
                            "allow_fine_tuning": False,
                            "organization": "*",
                            "group": None,
                            "is_blocking": False,
                        }
                    ],
                )
            ]
        else:
            models = [
                ModelInfo(
                    id=spec.id,
                    created=1677649963,
                    owned_by="noobscribe",
                    root=spec.id,
                    permission=[
                        {
                            "id": f"modelperm-{spec.id}",
                            "object": "model_permission",
                            "created": 1677649963,
                            "allow_create_engine": False,
                            "allow_sampling": True,
                            "allow_logprobs": True,
                            "allow_search_indices": False,
                            "allow_view": True,
                            "allow_fine_tuning": False,
                            "organization": "*",
                            "group": None,
                            "is_blocking": False,
                        }
                    ],
                )
                for spec in list_local_asr_models()
            ]

        return ModelList(data=models)

    @app.get("/v1/speakers", response_model=SpeakerList)
    async def list_speakers():
        """
        Get all speakers in the database
        """
        db = require_speaker_db()

        try:
            speakers = db.get_all_speakers()
            speaker_responses = [
                SpeakerResponse(
                    id=speaker.id,
                    display_name=speaker.display_name,
                    created_at=speaker.created_at,
                    embedding_count=speaker.embedding_count,
                )
                for speaker in speakers
            ]
            return SpeakerList(data=speaker_responses)
        except Exception as e:
            logger.error("Error listing speakers: %s", str(e))
            raise HTTPException(status_code=500, detail=f"Failed to list speakers: {str(e)}")

    @app.post("/v1/speakers/rematch", response_model=SpeakerRematchResponse)
    async def rematch_stored_speakers(session: AsyncSession = Depends(get_session)):
        """
        Re-match every stored transcription's diarization speaker rows against the
        current Chroma speaker index (same logic as after speaker create/update/delete).
        """
        db = require_speaker_db()

        try:
            n = await rematch_all_stored_transcriptions(session, db)
            if n:
                logger.info("Rematched speakers_json on %s stored transcriptions (manual rematch)", n)
            return SpeakerRematchResponse(updated_transcriptions=n)
        except Exception as e:
            logger.error("Error rematching stored transcriptions: %s", str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Failed to rematch stored transcriptions: {str(e)}",
            )

    @app.post("/v1/speakers/nearest-embeddings", response_model=SpeakerNearestResponse)
    async def nearest_speaker_embeddings(body: SpeakerNearestRequest):
        """
        Return the nearest stored speaker embeddings to a query vector, regardless
        of whether they meet the configured similarity threshold.
        """
        db = require_speaker_db()

        if not body.embedding:
            raise HTTPException(status_code=400, detail="embedding must be a non-empty array")

        try:
            emb = np.asarray(body.embedding, dtype=np.float32)
            if emb.size == 0:
                raise HTTPException(status_code=400, detail="embedding must be a non-empty array")
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid embedding: {e}") from e

        threshold = config.speaker_similarity_threshold
        try:
            hits = db.query_nearest_embeddings(emb, limit=body.limit)
            return SpeakerNearestResponse(
                threshold=threshold,
                data=[
                    SpeakerNearestMatch(
                        speaker_id=h["speaker_id"],
                        display_name=h["display_name"],
                        embedding_index=h["embedding_index"],
                        similarity=h["similarity"],
                        meets_threshold=h["similarity"] >= threshold,
                    )
                    for h in hits
                ],
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error querying nearest embeddings: %s", str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Failed to query nearest embeddings: {str(e)}",
            )

    @app.post(
        "/v1/speakers/extract-from-audio",
        response_model=TranscriptionJobCreateResponse,
        status_code=202,
    )
    async def extract_speaker_embeddings_from_audio(
        file: UploadFile = File(...),
        name: Optional[str] = Form(None),
        language: Optional[str] = Form(None),
        response_format: str = Form("verbose_json"),
        temperature: float = Form(0.0),
        timestamps: bool = Form(True),
        word_timestamps: bool = Form(False),
        diarize: bool = Form(True),
        include_diarization_in_text: Optional[bool] = Form(None),
        session: AsyncSession = Depends(get_session),
    ):
        """
        Upload audio, store as a hidden recording, and queue transcription +
        diarization. Poll ``GET /v1/transcription-jobs/{job_id}`` for speaker
        embeddings and transcript in ``result``.
        """
        _ = temperature
        cfg = get_config()
        rid = str(uuid.uuid4())
        suffix = Path(file.filename or "audio").suffix or ".bin"
        dest = Path(cfg.recordings_path) / f"{rid}{suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)

        content = await file.read()
        dest.write_bytes(content)

        duration = None
        mime = file.content_type
        try:
            if suffix.lower() == ".wav":
                duration = get_wav_duration_seconds(str(dest))
            else:
                wav = convert_audio_to_wav(str(dest))
                duration = get_wav_duration_seconds(wav)
                if os.path.exists(wav):
                    os.unlink(wav)
        except Exception as e:
            logger.debug("Could not compute duration on enrollment upload: %s", e)

        display_name = (name or "").strip() or (file.filename or "Speaker enrollment")
        rec = RecordingORM(
            id=rid,
            name=display_name,
            original_filename=file.filename or "upload",
            file_path=str(dest.resolve()),
            duration_seconds=duration,
            file_size_bytes=len(content),
            mime_type=mime,
            hide_in_recordings=True,
        )
        session.add(rec)
        await session.flush()

        try:
            params = build_transcribe_params(
                language=language,
                response_format=response_format,
                timestamps=timestamps,
                word_timestamps=word_timestamps,
                diarize=diarize,
                include_diarization_in_text=include_diarization_in_text,
                temperature=temperature,
            )
            job_id = await enqueue_recording_transcription(
                session,
                rid,
                params,
                job_type=JOB_TYPE_SPEAKER_EXTRACT,
            )
            return TranscriptionJobCreateResponse(
                job_id=job_id,
                status="queued",
                recording_id=rid,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("extract-from-audio queue failed for recording %s: %s", rid, e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/speakers", response_model=SpeakerResponse)
    async def create_speaker(
        speaker_data: SpeakerCreate,
        session: AsyncSession = Depends(get_session),
    ):
        """
        Create a new speaker with a display name and initial embedding
        """
        db = require_speaker_db()

        try:
            embedding = np.array(speaker_data.embedding, dtype=np.float32)
            speaker_id = db.create_speaker(
                display_name=speaker_data.display_name,
                embedding=embedding,
            )
            speaker = db.get_speaker_by_id(speaker_id)
            if not speaker:
                raise HTTPException(status_code=500, detail="Failed to retrieve created speaker")
            n = await rematch_all_stored_transcriptions(session, db)
            if n:
                logger.info("Rematched speakers_json on %s stored transcriptions after create_speaker", n)
            return SpeakerResponse(
                id=speaker.id,
                display_name=speaker.display_name,
                created_at=speaker.created_at,
                embedding_count=speaker.embedding_count,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error creating speaker: %s", str(e))
            raise HTTPException(status_code=500, detail=f"Failed to create speaker: {str(e)}")

    @app.put("/v1/speakers/{speaker_id}", response_model=SpeakerResponse)
    async def update_speaker(
        speaker_id: str,
        speaker_data: SpeakerUpdate,
        session: AsyncSession = Depends(get_session),
    ):
        """
        Add an additional embedding to an existing speaker
        """
        db = require_speaker_db()

        try:
            embedding = np.array(speaker_data.embedding, dtype=np.float32)
            success = db.add_embedding(speaker_id, embedding)
            if not success:
                raise HTTPException(status_code=404, detail=f"Speaker {speaker_id} not found")
            speaker = db.get_speaker_by_id(speaker_id)
            if not speaker:
                raise HTTPException(status_code=500, detail="Failed to retrieve updated speaker")
            n = await rematch_all_stored_transcriptions(session, db)
            if n:
                logger.info("Rematched speakers_json on %s stored transcriptions after update_speaker", n)
            return SpeakerResponse(
                id=speaker.id,
                display_name=speaker.display_name,
                created_at=speaker.created_at,
                embedding_count=speaker.embedding_count,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error updating speaker: %s", str(e))
            raise HTTPException(status_code=500, detail=f"Failed to update speaker: {str(e)}")

    @app.get("/v1/speakers/{speaker_id}/embeddings", response_model=SpeakerEmbeddingListResponse)
    async def list_speaker_embeddings(speaker_id: str):
        """
        List metadata for each enrolled embedding (index and created_at).
        """
        db = require_speaker_db()

        sp = db.get_speaker_by_id(speaker_id)
        if not sp:
            raise HTTPException(status_code=404, detail=f"Speaker {speaker_id} not found")

        try:
            rows = db.get_speaker_embeddings(speaker_id)
            return SpeakerEmbeddingListResponse(
                speaker_id=speaker_id,
                display_name=sp.display_name,
                data=[
                    SpeakerEmbeddingDetail(
                        embedding_index=r["embedding_index"],
                        created_at=r["created_at"],
                    )
                    for r in rows
                ],
            )
        except Exception as e:
            logger.error("Error listing speaker embeddings: %s", str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Failed to list speaker embeddings: {str(e)}",
            )

    @app.get(
        "/v1/speakers/{speaker_id}/embeddings/{embedding_index}/snippets",
        response_model=EmbeddingSnippetListResponse,
    )
    async def list_speaker_embedding_snippets(
        speaker_id: str,
        embedding_index: int,
        session: AsyncSession = Depends(get_session),
    ):
        """
        For one enrolled embedding, find stored transcriptions whose diarization speaker
        embedding matches (cosine similarity >= threshold) and return preview URLs for
        one matching segment per transcription.
        """
        db = require_speaker_db()

        if db.get_speaker_by_id(speaker_id) is None:
            raise HTTPException(status_code=404, detail=f"Speaker {speaker_id} not found")

        try:
            raw = await collect_snippets_for_enrolled_embedding(
                session,
                db,
                speaker_id,
                embedding_index,
                config.speaker_similarity_threshold,
            )
            if raw is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Embedding {embedding_index} not found for speaker {speaker_id}",
                )
            return EmbeddingSnippetListResponse(
                data=[
                    EmbeddingSnippet(
                        transcription_id=s["transcription_id"],
                        recording_id=s["recording_id"],
                        recording_name=s.get("recording_name"),
                        preview_url=s["preview_url"],
                        segment_text=s["segment_text"],
                        start=s["start"],
                        end=s["end"],
                    )
                    for s in raw
                ],
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error listing embedding snippets: %s", str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Failed to list embedding snippets: {str(e)}",
            )

    @app.delete("/v1/speakers/{speaker_id}/embeddings/{embedding_index}")
    async def delete_speaker_embedding(
        speaker_id: str,
        embedding_index: int,
        session: AsyncSession = Depends(get_session),
    ):
        """
        Remove one enrolled embedding. The last embedding cannot be deleted; delete the
        speaker instead. Triggers re-match of stored transcription speaker metadata.
        """
        db = require_speaker_db()

        ok, reason = db.delete_embedding(speaker_id, embedding_index)
        if not ok:
            if reason == "last_embedding":
                raise HTTPException(
                    status_code=400,
                    detail="Cannot delete the only embedding; delete the speaker instead",
                )
            raise HTTPException(
                status_code=404,
                detail=f"Speaker {speaker_id} or embedding {embedding_index} not found",
            )

        try:
            n = await rematch_all_stored_transcriptions(session, db)
            if n:
                logger.info(
                    "Rematched speakers_json on %s stored transcriptions after delete_embedding",
                    n,
                )
        except Exception as e:
            logger.error("Re-match after delete_embedding failed: %s", e)

        return JSONResponse(
            status_code=200,
            content={"message": f"Embedding {embedding_index} removed for speaker {speaker_id}"},
        )

    @app.get("/v1/audio/snippet")
    async def get_audio_snippet(
        recording_id: str,
        start: float,
        end: float,
        background_tasks: BackgroundTasks,
        session: AsyncSession = Depends(get_session),
    ):
        """
        On-demand WAV excerpt from a library recording (ffmpeg). Used by embedding snippet previews.
        """
        rec = await session.get(RecordingORM, recording_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Recording not found")
        if not rec.file_path or not os.path.exists(rec.file_path):
            raise HTTPException(status_code=404, detail="Audio file not found on disk")

        try:
            wav_path = cut_audio_segment(rec.file_path, start, end)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Snippet cut failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

        def _cleanup_snippet(path: str) -> None:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass

        background_tasks.add_task(_cleanup_snippet, wav_path)

        return FileResponse(
            wav_path,
            media_type="audio/wav",
            filename="snippet.wav",
        )

    @app.delete("/v1/speakers/{speaker_id}")
    async def delete_speaker(speaker_id: str):
        """
        Delete a speaker and all its embeddings from the database
        """
        db = require_speaker_db()

        try:
            success = db.delete_speaker(speaker_id)
            if not success:
                raise HTTPException(status_code=404, detail=f"Speaker {speaker_id} not found")
            return JSONResponse(
                status_code=200,
                content={"message": f"Speaker {speaker_id} deleted successfully"},
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error deleting speaker: %s", str(e))
            raise HTTPException(status_code=500, detail=f"Failed to delete speaker: {str(e)}")

    return app
