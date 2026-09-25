import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException, Request

from .config import get_settings

logger = logging.getLogger(__name__)

_LUA_SCRIPT_PATH = Path(__file__).parent / "lua" / "token_bucket.lua"


@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    retry_after: float  # seconds until at least one token is available


class RateLimiter(ABC):
    @abstractmethod
    async def check(self, key: str, capacity: int, refill_rate: float) -> RateLimitResult:
        """Try to consume one token from `key`'s bucket.
        """


class LocalRateLimiter(RateLimiter):

    def __init__(self) -> None:
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, key: str, capacity: int, refill_rate: float) -> RateLimitResult:
        now = time.monotonic()
        async with self._lock:
            tokens, last_ts = self._buckets.get(key, (float(capacity), now))
            tokens = min(capacity, tokens + (now - last_ts) * refill_rate)

            if tokens >= 1:
                tokens -= 1
                self._buckets[key] = (tokens, now)
                return RateLimitResult(True, int(tokens), 0.0)

            self._buckets[key] = (tokens, now)
            return RateLimitResult(False, 0, (1 - tokens) / refill_rate)


class RedisRateLimiter(RateLimiter):

    def __init__(self, url: str) -> None:
        import redis.asyncio as aioredis  # local import: optional dependency

        self._redis = aioredis.from_url(url, decode_responses=True)
        self._script = self._redis.register_script(_LUA_SCRIPT_PATH.read_text())

    async def check(self, key: str, capacity: int, refill_rate: float) -> RateLimitResult:
        # long enough for a fully-drained bucket to refill, plus slack —
        # bounds how long an idle client's key lingers in Redis
        ttl_seconds = max(2, round((capacity / refill_rate) * 2))
        try:
            allowed, tokens, retry_after = await self._script(
                keys=[f"ratelimit:{key}"],
                args=[capacity, refill_rate, 1, ttl_seconds],
            )
        except Exception:
            # Redis being down shouldn't take the whole API down with it —
            # fail open and let the request through.
            logger.exception("Redis rate limiter unavailable; failing open for %r", key)
            return RateLimitResult(True, capacity, 0.0)
        return RateLimitResult(bool(int(allowed)), int(float(tokens)), float(retry_after))


@lru_cache
def get_rate_limiter() -> RateLimiter:
    settings = get_settings()
    if settings.redis_rate_limit:
        logger.info("Rate limiter backend: redis (%s)", settings.redis_url)
        return RedisRateLimiter(settings.redis_url)
    logger.info("Rate limiter backend: local (in-process)")
    return LocalRateLimiter()


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    client_ip = forwarded.split(",")[0].strip() if forwarded else None
    if not client_ip:
        client_ip = request.client.host if request.client else "unknown"
    return f"{client_ip}:{request.url.path}"


def rate_limit(requests: int | None = None, window_seconds: int | None = None):
    """FastAPI dependency factory — `Depends(rate_limit())` for the config
    defaults, or `Depends(rate_limit(requests=10, window_seconds=60))`
    """

    async def dependency(request: Request) -> None:
        settings = get_settings()
        capacity = requests if requests is not None else settings.rate_limit_requests
        window = window_seconds if window_seconds is not None else settings.rate_limit_window_seconds
        refill_rate = capacity / window

        limiter = get_rate_limiter()
        result = await limiter.check(_client_key(request), capacity, refill_rate)

        if not result.allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(max(1, round(result.retry_after)))},
            )

    return dependency
