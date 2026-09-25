import logging
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
import uvicorn
from .api import documents, query
from .core.config import get_settings
from .core.logging_config import configure_logging
from .core.middleware import RequestContextMiddleware
from .db import DocumentStore
from .pipeline.vectorstore import VectorStore

settings = get_settings()
configure_logging(
    settings.log_level,
    settings.log_dir,
    settings.log_max_bytes,
    settings.log_backup_count,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.store = DocumentStore(settings.resolved_database_url())
    app.state.vectorstore = VectorStore()
    logger.info("Semantic retrieval service ready (qdrant=%s)", settings.qdrant_url)
    yield


app = FastAPI(
    title="Semantic Retrieval Pipeline",
    description="Document ingestion + hybrid retrieval (no generation).",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestContextMiddleware)

app.include_router(documents.router)
app.include_router(query.router)

# all this custom api as swagger is not showing proper file select option #
def _use_binary_format(schema: dict) -> None:
    for schema_obj in schema.get("components", {}).get("schemas", {}).values():
        for prop in schema_obj.get("properties", {}).values():
            if "contentMediaType" in prop:
                prop["format"] = "binary"
                prop.pop("contentMediaType", None)
                prop.pop("contentEncoding", None)
            items = prop.get("items")
            if isinstance(items, dict) and "contentMediaType" in items:
                items["format"] = "binary"
                items.pop("contentMediaType", None)
                items.pop("contentEncoding", None)


def _custom_openapi() -> dict:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    _use_binary_format(schema)
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _custom_openapi


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":

    SERVER_HOST = os.getenv("HOST", "0.0.0.0")
    SERVER_PORT = int(os.getenv("PORT", 10005)) 
    config = uvicorn.Config(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info"
    )
    server = uvicorn.Server(config)
    server.run()
    print(f"Server started on f{SERVER_HOST}:{SERVER_PORT}")