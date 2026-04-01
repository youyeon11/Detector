from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.analyze import router as analyze_router
from app.api.search import router as search_router
from app.api.stats import router as stats_router
from app.services.detector import DetectorService
from app.services.retriever import RetrieverError


def create_app() -> FastAPI:
    openapi_tags = [
        {
            "name": "detect",
            "description": "Run offensive text detection and return the resolved label.",
        },
        {
            "name": "analyze",
            "description": "Inspect normalized text and the generated query preview.",
        },
        {
            "name": "health",
            "description": "Check application and retriever availability.",
        },
        {
            "name": "stats",
            "description": "Read V4 aggregate statistics from Elasticsearch.",
        },
    ]
    app = FastAPI(
        title="Detector API",
        version="0.1.0",
        description="Offensive text detection and analysis API.",
        summary="Profanity detection API with analysis and health endpoints.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        swagger_ui_parameters={
            "displayRequestDuration": True,
            "docExpansion": "list",
            "filter": True,
        },
        openapi_tags=openapi_tags,
    )

    @app.exception_handler(RetrieverError)
    def handle_retriever_error(_: Request, exc: RetrieverError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc)},
        )

    @app.exception_handler(ValueError)
    def handle_value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc)},
        )

    @app.get("/health", tags=["health"], summary="Health check")
    def health() -> dict[str, str]:
        detector = DetectorService()
        status = "ok" if detector.lexicon_retriever.ping() else "degraded"
        return {"status": status}

    app.include_router(search_router)
    app.include_router(analyze_router)
    app.include_router(stats_router)
    return app


app = create_app()
