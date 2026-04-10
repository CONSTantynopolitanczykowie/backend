import logging

from fastapi import FastAPI

from app.api.routes import router
from app.core.settings import get_settings
from app.services.session_manager import SessionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Beach People Counter API",
        version="0.1.0",
        description="YOLOv8n-based line crossing people counter",
    )
    app.state.session_manager = SessionManager(settings=settings)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    app.include_router(router)
    return app


app = create_app()
