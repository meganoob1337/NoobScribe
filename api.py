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
    SpeakerResponse,
    SpeakerUpdate,
)
from routers.recordings import router as recordings_router
from services.rematch_stored_transcriptions import rematch_all_stored_transcriptions
from services.speaker_embedding_snippets import collect_snippets_for_enrolled_embedding
from services.stored_recording_transcribe import (
    persist_pipeline_transcription,
    run_transcription_for_recording_path,
    transcription_response_json_body,
)
from services.transcription_pipeline import run_transcription_pipeline
from transcription import format_srt, format_vtt, load_model

# Initialize logging
logger = logging.getLogger(__name__)

# Global variable for model
asr_model = None

# Global variable for speaker database
speaker_db = None

# Get configuration
config = get_config()


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
        global asr_model, speaker_db

        init_engine()
        await init_db()
        logger.info("SQLite database initialized")

        try:
            if use_cuda():
                logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
            else:
                logger.warning("Using CPU for inference (this will be slow)")

            model_id = config.model_id
            asr_model = load_model(model_id, model_path=config.model_path)
            logger.info("ASR model loaded (%s)", config.model_path or model_id)

            try:
                speaker_db = SpeakerDB(
                    db_path=config.chromadb_path,
                    similarity_threshold=config.speaker_similarity_threshold,
                )
                logger.info("Speaker database initialized at %s", config.chromadb_path)
            except Exception as e:
                logger.error("Failed to initialize speaker database: %s", str(e))
                speaker_db = None

            if config.diarization_model_path:
                logger.info(
                    "Local diarization model path set; offline diarization available without HF token"
                )
            hf_token = config.get_hf_token()
            if hf_token:
                logger.info("HuggingFace access token found; remote diarization models available")
            else:
                logger.info("No HuggingFace access token (use DIARIZATION_MODEL_PATH for offline)")

        except Exception as e:
            logger.error("Error during startup: %s", str(e))

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

    @app.get("/", include_in_schema=False)
    async def root_redirect_to_ui():
        """Redirect browsers to the Web UI."""
        return RedirectResponse(url="/ui", status_code=302)

    webui_root = Path(__file__).resolve().parent / "webui"
    if webui_root.is_dir():
        app.mount("/ui", WebUiStaticFiles(directory=str(webui_root), html=True), name="ui")

    @app.post("/v1/audio/transcriptions")
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
    ):
        """
        Transcribe audio using NVIDIA NeMo ASR (default: Canary 1B v2).

        This endpoint is compatible with the OpenAI Whisper API.
        """
        _ = (model, prompt, timestamp_granularities, vad_filter, temperature)

        global asr_model

        if not asr_model:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded yet. Please try again in a few moments.",
            )

        logger.info("Transcription requested: %s, format: %s", file.filename, response_format)

        temp_dir = Path(config.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_file = temp_dir / f"upload_{os.urandom(8).hex()}{Path(file.filename).suffix}"
        wav_file: Optional[str] = None
        pr = None

        try:
            with open(temp_file, "wb") as f:
                content = await file.read()
                f.write(content)

            wav_file = convert_audio_to_wav(str(temp_file))

            pr = run_transcription_pipeline(
                wav_file,
                asr_model,
                config,
                speaker_db,
                language=language,
                word_timestamps=word_timestamps,
                diarize=diarize,
                include_diarization_in_text=include_diarization_in_text,
                response_format=response_format,
                timestamps=timestamps,
            )

            response = pr.response

            if response_format == "json":
                out = response.dict() if hasattr(response, "dict") else response.model_dump()
            elif response_format == "text":
                out = PlainTextResponse(response.text)
            elif response_format == "srt":
                out = PlainTextResponse(format_srt(pr.all_segments))
            elif response_format == "vtt":
                out = PlainTextResponse(format_vtt(pr.all_segments))
            elif response_format == "verbose_json":
                out = response.dict() if hasattr(response, "dict") else response.model_dump()
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported response format: {response_format}")

            return out

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error during transcription: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if pr is not None:
                for chunk_path in pr.paths_to_cleanup:
                    try:
                        if os.path.exists(chunk_path):
                            os.unlink(chunk_path)
                    except OSError:
                        pass
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except OSError:
                pass
            if wav_file and os.path.exists(wav_file) and wav_file != str(temp_file):
                try:
                    os.unlink(wav_file)
                except OSError:
                    pass

    @app.get("/health")
    async def health_check():
        """
        Check the health of the API and the loaded model
        """
        global asr_model

        return {
            "status": "ok",
            "version": "1.0.0",
            "model_loaded": asr_model is not None,
            "model_id": config.model_id,
            "force_cpu": config.force_cpu,
            "cuda_available": use_cuda(),
            "gpu_info": torch.cuda.get_device_name(0) if use_cuda() else None,
            "config": config.as_dict(),
        }

    @app.get("/v1/models")
    async def list_models():
        """
        List available models (compatibility with OpenAI API)
        """
        models = [
            ModelInfo(
                id="whisper-1",
                created=1677649963,
                owned_by="noobscribe",
                root="whisper-1",
                permission=[
                    {
                        "id": "modelperm-1",
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

        return ModelList(data=models)

    @app.get("/v1/speakers", response_model=SpeakerList)
    async def list_speakers():
        """
        Get all speakers in the database
        """
        global speaker_db

        if speaker_db is None:
            raise HTTPException(status_code=503, detail="Speaker database not initialized")

        try:
            speakers = speaker_db.get_all_speakers()
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

    @app.post("/v1/speakers/extract-from-audio")
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
        Upload audio, store as a hidden recording (not listed under recordings), run
        transcription + diarization, persist the result, and return speaker embeddings
        and transcript (same shape as stored recording transcribe verbose_json).
        """
        _ = temperature
        global asr_model, speaker_db

        if asr_model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded yet. Please try again in a few moments.",
            )

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
            pr = run_transcription_for_recording_path(
                rec.file_path,
                asr_model,
                cfg,
                speaker_db,
                language=language,
                word_timestamps=word_timestamps,
                diarize=diarize,
                include_diarization_in_text=include_diarization_in_text,
                response_format=response_format,
                timestamps=timestamps,
            )
        except Exception as e:
            logger.error("extract-from-audio failed for recording %s: %s", rid, e)
            raise HTTPException(status_code=500, detail=str(e))

        tid = await persist_pipeline_transcription(
            session,
            rec,
            pr,
            diarize=diarize,
            word_timestamps=word_timestamps,
        )

        def _json_body() -> dict:
            d = transcription_response_json_body(pr, tid)
            d["recording_id"] = rid
            return d

        if response_format == "json":
            return _json_body()
        if response_format == "text":
            return PlainTextResponse(pr.response.text)
        if response_format == "srt":
            return PlainTextResponse(format_srt(pr.all_segments))
        if response_format == "vtt":
            return PlainTextResponse(format_vtt(pr.all_segments))
        if response_format == "verbose_json":
            return _json_body()
        raise HTTPException(status_code=400, detail=f"Unsupported response format: {response_format}")

    @app.post("/v1/speakers", response_model=SpeakerResponse)
    async def create_speaker(
        speaker_data: SpeakerCreate,
        session: AsyncSession = Depends(get_session),
    ):
        """
        Create a new speaker with a display name and initial embedding
        """
        global speaker_db

        if speaker_db is None:
            raise HTTPException(status_code=503, detail="Speaker database not initialized")

        try:
            embedding = np.array(speaker_data.embedding, dtype=np.float32)
            speaker_id = speaker_db.create_speaker(
                display_name=speaker_data.display_name,
                embedding=embedding,
            )
            speaker = speaker_db.get_speaker_by_id(speaker_id)
            if not speaker:
                raise HTTPException(status_code=500, detail="Failed to retrieve created speaker")
            n = await rematch_all_stored_transcriptions(session, speaker_db)
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
        global speaker_db

        if speaker_db is None:
            raise HTTPException(status_code=503, detail="Speaker database not initialized")

        try:
            embedding = np.array(speaker_data.embedding, dtype=np.float32)
            success = speaker_db.add_embedding(speaker_id, embedding)
            if not success:
                raise HTTPException(status_code=404, detail=f"Speaker {speaker_id} not found")
            speaker = speaker_db.get_speaker_by_id(speaker_id)
            if not speaker:
                raise HTTPException(status_code=500, detail="Failed to retrieve updated speaker")
            n = await rematch_all_stored_transcriptions(session, speaker_db)
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
        global speaker_db

        if speaker_db is None:
            raise HTTPException(status_code=503, detail="Speaker database not initialized")

        sp = speaker_db.get_speaker_by_id(speaker_id)
        if not sp:
            raise HTTPException(status_code=404, detail=f"Speaker {speaker_id} not found")

        try:
            rows = speaker_db.get_speaker_embeddings(speaker_id)
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
        global speaker_db

        if speaker_db is None:
            raise HTTPException(status_code=503, detail="Speaker database not initialized")

        if speaker_db.get_speaker_by_id(speaker_id) is None:
            raise HTTPException(status_code=404, detail=f"Speaker {speaker_id} not found")

        try:
            raw = await collect_snippets_for_enrolled_embedding(
                session,
                speaker_db,
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
        global speaker_db

        if speaker_db is None:
            raise HTTPException(status_code=503, detail="Speaker database not initialized")

        ok, reason = speaker_db.delete_embedding(speaker_id, embedding_index)
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
            n = await rematch_all_stored_transcriptions(session, speaker_db)
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
        global speaker_db

        if speaker_db is None:
            raise HTTPException(status_code=503, detail="Speaker database not initialized")

        try:
            success = speaker_db.delete_speaker(speaker_id)
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
