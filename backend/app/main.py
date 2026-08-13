from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.retrieval import router as retrieval_router
from app.core.config import get_settings

STATIC_DIRECTORY = Path(__file__).parent / "static"

settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0")
app.include_router(documents_router, prefix=settings.api_prefix)
app.include_router(retrieval_router, prefix=settings.api_prefix)
app.include_router(chat_router, prefix=settings.api_prefix)


@app.get("/", include_in_schema=False)
def test_console() -> FileResponse:
    """Serve the pipeline test console same-origin, so no CORS setup is needed."""
    return FileResponse(STATIC_DIRECTORY / "index.html")


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Liveness endpoint; dependency checks will be added separately."""
    return {"status": "ok", "service": settings.app_name}
