"""HTTP interface for the notebook's retrieval pipeline."""

from contextlib import asynccontextmanager
import logging
from threading import Lock

import httpx
from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool

from contracts import AnalysisResponse, ChatRequest
from settings import Settings

logger = logging.getLogger(__name__)


def load_engine():
    settings = Settings()
    settings.validate_artifacts()
    from pipeline import RetrievalPipeline
    return RetrievalPipeline(settings)


def create_app(engine_factory=load_engine):
    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = None
        app.state.not_ready = "Retrieval service is initializing."
        try:
            app.state.engine = await run_in_threadpool(engine_factory)
        except FileNotFoundError as error:
            app.state.not_ready = str(error)
            logger.warning("%s", error)
        except Exception:
            app.state.not_ready = "Model initialization failed. Check the backend logs and restart the service."
            logger.exception("Retrieval initialization failed")
        yield
        app.state.engine = None

    app = FastAPI(title="Transcript-RAG — Conversational Intelligence System", lifespan=lifespan)
    inference_lock = Lock()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    def require_ready():
        if app.state.engine is None:
            raise HTTPException(503, app.state.not_ready)
        return app.state.engine

    @app.get("/ready")
    def ready():
        require_ready()
        return {"status": "ready"}

    @app.post("/chat", response_model=AnalysisResponse)
    def chat(request: ChatRequest):
        engine = require_ready()
        # RAPTOR builds a tree per question; serialize GPU work, not HTTP traffic.
        if not inference_lock.acquire(blocking=False):
            raise HTTPException(429, "The model is answering another question. Try again shortly.", headers={"Retry-After": "5"})
        try:
            return AnalysisResponse.model_validate(engine.answer(request.message, request.history))
        except httpx.TimeoutException:
            raise HTTPException(504, "The model timed out. Try a shorter question.") from None
        except Exception:
            logger.exception("Retrieval request failed")
            raise HTTPException(502, "Answer generation failed. Check the backend logs and model server.") from None
        finally:
            inference_lock.release()

    return app


app = create_app()
