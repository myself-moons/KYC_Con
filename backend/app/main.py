"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.documents import router as document_router

app = FastAPI(title="KYC Document Intelligence API")
app.include_router(document_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return a basic service health response."""
    return {"status": "ok"}
