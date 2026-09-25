import json
import logging
import logging.config
from contextvars import ContextVar
from pathlib import Path

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
api_url_ctx: ContextVar[str] = ContextVar("api_url", default="-")

_RESERVED_RECORD_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message",
}


class RequestContextFilter(logging.Filter):
    """Injects the current request's id/url into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        record.api_url = api_url_ctx.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "function": record.funcName,
            "line": record.lineno,
            "request_id": getattr(record, "request_id", "-"),
            "api_url": getattr(record, "api_url", "-"),
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_ATTRS and key not in payload:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    log_dir = log_dir or Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "rag.log"

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "request_context": {"()": RequestContextFilter},
            },
            "formatters": {
                "json": {"()": JsonFormatter},
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                    "filters": ["request_context"],
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": str(log_file),
                    "maxBytes": max_bytes,
                    "backupCount": backup_count,
                    "encoding": "utf-8",
                    "formatter": "json",
                    "filters": ["request_context"],
                },
            },
            "root": {"handlers": ["default", "file"], "level": level},
            "loggers": {
                "uvicorn": {"handlers": ["default", "file"], "level": level, "propagate": False},
                "uvicorn.error": {"handlers": ["default", "file"], "level": level, "propagate": False},
                "uvicorn.access": {"handlers": ["default", "file"], "level": level, "propagate": False},
            },
        }
    )
