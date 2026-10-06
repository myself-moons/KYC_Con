"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.cases import router as cases_router
from app.api.documents import router as document_router
from app.api.extractions import router as extractions_router

app = FastAPI(title="KYC Document Intelligence API")
app.include_router(cases_router)
app.include_router(document_router)
app.include_router(extractions_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return a basic service health response."""
    return {"status": "ok"}
