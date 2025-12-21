"""
API Authentication and Rate Limiting Middleware.

Provides production-ready security through:
- API key authentication (X-API-Key header)
- Rate limiting per API key (Redis-backed token bucket)
- Request tracking and audit logging
"""


import redis.asyncio as redis
from fastapi import Header, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

# API key header scheme
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class RateLimiter:
    """
    Redis-backed token bucket rate limiter.

    Implements sliding window rate limiting with Redis for distributed systems.
    """

    def __init__(
        self,
        redis_url: str = None,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    ):
        """
        Initialize rate limiter.

        Args:
            redis_url: Redis connection URL
            requests_per_minute: Maximum requests per minute per API key
            requests_per_hour: Maximum requests per hour per API key
        """
        self.redis_url = redis_url or settings.redis_url
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self._redis_client: redis.Redis | None = None

    async def get_redis_client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._redis_client is None:
            try:
                self._redis_client = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                await self._redis_client.ping()
                logger.info("Redis connection established for rate limiting")
            except Exception as e:
                logger.error(f"Failed to connect to Redis for rate limiting: {e}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Rate limiting service unavailable",
                )
        return self._redis_client

    async def check_rate_limit(self, api_key: str) -> tuple[bool, dict]:
        """
        Check if request is within rate limits.

        Args:
            api_key: API key to check

        Returns:
            Tuple of (allowed: bool, limit_info: dict)
        """
        try:
            client = await self.get_redis_client()

            # Check minute limit
            minute_key = f"rate_limit:minute:{api_key}"
            minute_count = await client.get(minute_key)

            if minute_count is None:
                # First request in this minute
                await client.setex(minute_key, 60, 1)
                minute_count = 1
            else:
                minute_count = int(minute_count)
                if minute_count >= self.rpm:
                    return False, {
                        "limit": self.rpm,
                        "remaining": 0,
                        "reset_in_seconds": await client.ttl(minute_key),
                        "limit_type": "per_minute",
                    }
                await client.incr(minute_key)
                minute_count += 1

            # Check hour limit
            hour_key = f"rate_limit:hour:{api_key}"
            hour_count = await client.get(hour_key)

            if hour_count is None:
                # First request in this hour
                await client.setex(hour_key, 3600, 1)
                hour_count = 1
            else:
                hour_count = int(hour_count)
                if hour_count >= self.rph:
                    return False, {
                        "limit": self.rph,
                        "remaining": 0,
                        "reset_in_seconds": await client.ttl(hour_key),
                        "limit_type": "per_hour",
                    }
                await client.incr(hour_key)
                hour_count += 1

            # Calculate remaining
            minute_remaining = self.rpm - minute_count
            hour_remaining = self.rph - hour_count

            return True, {
                "minute_limit": self.rpm,
                "minute_remaining": minute_remaining,
                "hour_limit": self.rph,
                "hour_remaining": hour_remaining,
            }

        except redis.RedisError as e:
            logger.error(f"Redis error in rate limiting: {e}", exc_info=True)
            # Fail open - allow request if Redis is down (configurable)
            if settings.environment == "production":
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Rate limiting service unavailable",
                )
            else:
                logger.warning("Rate limiting disabled due to Redis error (development mode)")
                return True, {"error": "rate_limiting_disabled"}

    async def close(self):
        """Close Redis connection."""
        if self._redis_client:
            await self._redis_client.close()


# Global rate limiter instance
rate_limiter = RateLimiter(
    requests_per_minute=settings.api_rate_limit_per_minute if hasattr(settings, "api_rate_limit_per_minute") else 60,
    requests_per_hour=settings.api_rate_limit_per_hour if hasattr(settings, "api_rate_limit_per_hour") else 1000,
)


async def verify_api_key_and_rate_limit(
    request: Request,
    x_api_key: str | None = Header(None),
) -> str:
    """
    Verify API key and check rate limits.

    Dependency function for FastAPI routes requiring authentication.

    Args:
        request: FastAPI request object
        x_api_key: API key from X-API-Key header

    Returns:
        Validated API key

    Raises:
        HTTPException: If API key is invalid or rate limit exceeded
    """
    # Check if API key authentication is enabled
    if not hasattr(settings, "api_keys_enabled") or not settings.api_keys_enabled:
        # Authentication disabled (development mode)
        if settings.environment != "production":
            logger.debug("API key authentication disabled (development mode)")
            return "development"
        else:
            logger.error("API key authentication disabled in production mode!")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication configuration error",
            )

    # Check API key presence
    if not x_api_key:
        logger.warning(
            "Missing API key",
            extra={
                "path": request.url.path,
                "client_host": request.client.host if request.client else None,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Include 'X-API-Key' header with your request.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Verify API key
    valid_api_keys = getattr(settings, "valid_api_keys", [])
    if isinstance(valid_api_keys, str):
        valid_api_keys = [k.strip() for k in valid_api_keys.split(",")]

    if x_api_key not in valid_api_keys:
        logger.warning(
            "Invalid API key attempted",
            extra={
                "path": request.url.path,
                "api_key_prefix": x_api_key[:8] + "..." if len(x_api_key) > 8 else "short",
                "client_host": request.client.host if request.client else None,
            },
        )
        # Log audit event
        logger.log_audit(
            action="api_key_rejected",
            resource="api_authentication",
            result="failure",
            api_key_prefix=x_api_key[:8],
            path=request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    # Check rate limits
    try:
        allowed, limit_info = await rate_limiter.check_rate_limit(x_api_key)

        if not allowed:
            logger.warning(
                "Rate limit exceeded for API key",
                extra={
                    "api_key_prefix": x_api_key[:8],
                    "limit_info": limit_info,
                    "path": request.url.path,
                },
            )
            # Log audit event
            logger.log_audit(
                action="rate_limit_exceeded",
                resource="api_rate_limiting",
                result="blocked",
                limit_type=limit_info.get("limit_type"),
                api_key_prefix=x_api_key[:8],
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {limit_info.get('reset_in_seconds', 60)} seconds.",
                headers={
                    "X-RateLimit-Limit": str(limit_info.get("limit", 0)),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(limit_info.get("reset_in_seconds", 60)),
                    "Retry-After": str(limit_info.get("reset_in_seconds", 60)),
                },
            )

        # Add rate limit headers to response
        request.state.rate_limit_info = limit_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in rate limiting: {e}", exc_info=True)
        # Fail open in development, fail closed in production
        if settings.environment == "production":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service unavailable",
            )

    # Log successful authentication
    logger.debug(
        "API key authenticated",
        extra={
            "api_key_prefix": x_api_key[:8],
            "path": request.url.path,
        },
    )

    # Attach API key to request state
    request.state.api_key = x_api_key

    return x_api_key


async def add_rate_limit_headers(request: Request, call_next):
    """
    Middleware to add rate limit headers to all responses.

    Args:
        request: FastAPI request
        call_next: Next middleware/handler

    Returns:
        Response with rate limit headers
    """
    response = await call_next(request)

    # Add rate limit info to headers if available
    if hasattr(request.state, "rate_limit_info"):
        limit_info = request.state.rate_limit_info
        response.headers["X-RateLimit-Limit-Minute"] = str(limit_info.get("minute_limit", 0))
        response.headers["X-RateLimit-Remaining-Minute"] = str(limit_info.get("minute_remaining", 0))
        response.headers["X-RateLimit-Limit-Hour"] = str(limit_info.get("hour_limit", 0))
        response.headers["X-RateLimit-Remaining-Hour"] = str(limit_info.get("hour_remaining", 0))

    return response
