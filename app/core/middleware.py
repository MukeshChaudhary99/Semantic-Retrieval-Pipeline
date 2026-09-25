import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from .logging_config import api_url_ctx, request_id_ctx

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Tags each request with an id, exposes it to logging, and logs access."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER, str(uuid.uuid4()))
        api_url = f"{request.method} {request.url.path}"

        request_id_token = request_id_ctx.set(request_id)
        api_url_token = api_url_ctx.set(api_url)
        start = time.perf_counter()
        try:
            logger.info("request started")
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "request completed",
                extra={"status_code": response.status_code, "duration_ms": duration_ms},
            )
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        except Exception:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.exception("request failed", extra={"duration_ms": duration_ms})
            raise
        finally:
            request_id_ctx.reset(request_id_token)
            api_url_ctx.reset(api_url_token)
