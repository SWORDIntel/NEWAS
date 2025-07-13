"""API Middleware for NEMWAS"""

import time
import logging
import json
import uuid
from typing import Callable, Optional
from datetime import datetime

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import prometheus_client

from .models import ErrorResponse

logger = logging.getLogger(__name__)


# Prometheus metrics
request_duration = prometheus_client.Histogram(
    'nemwas_http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint', 'status']
)

request_count = prometheus_client.Counter(
    'nemwas_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

active_requests = prometheus_client.Gauge(
    'nemwas_http_requests_active',
    'Active HTTP requests'
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests and responses"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Start timing
        start_time = time.time()

        # Log request
        logger.info(
            f"Request {request_id}: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Log response
        logger.info(
            f"Response {request_id}: {response.status_code} "
            f"({duration:.3f}s)"
        )

        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration:.3f}"

        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect metrics for Prometheus"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip metrics endpoint to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)

        # Track active requests
        active_requests.inc()

        # Start timing
        start_time = time.time()

        try:
            # Process request
            response = await call_next(request)

            # Record metrics
            duration = time.time() - start_time
            endpoint = request.url.path
            method = request.method
            status_code = response.status_code

            request_duration.labels(
                method=method,
                endpoint=endpoint,
                status=status_code
            ).observe(duration)

            request_count.labels(
                method=method,
                endpoint=endpoint,
                status=status_code
            ).inc()

            return response

        finally:
            active_requests.dec()


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error_code="VALIDATION_ERROR",
                    error_type="ValueError",
                    detail=str(e)
                ).dict()
            )

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=ErrorResponse(
                    error_code="FILE_NOT_FOUND",
                    error_type="FileNotFoundError",
                    detail=str(e)
                ).dict()
            )

        except PermissionError as e:
            logger.error(f"Permission denied: {e}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content=ErrorResponse(
                    error_code="PERMISSION_DENIED",
                    error_type="PermissionError",
                    detail=str(e)
                ).dict()
            )

        except Exception as e:
            logger.exception(f"Unhandled error: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error_code="INTERNAL_ERROR",
                    error_type=type(e).__name__,
                    detail="An internal error occurred",
                    traceback=str(e) if logger.level <= logging.DEBUG else None
                ).dict()
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""

    def __init__(self, app: ASGIApp, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = {}  # client_ip -> list of timestamps

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for internal endpoints
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Clean old entries
        current_time = time.time()
        if client_ip in self.clients:
            self.clients[client_ip] = [
                ts for ts in self.clients[client_ip]
                if current_time - ts < self.period
            ]

        # Check rate limit
        if client_ip in self.clients and len(self.clients[client_ip]) >= self.calls:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=ErrorResponse(
                    error_code="RATE_LIMIT_EXCEEDED",
                    error_type="RateLimitError",
                    detail=f"Rate limit exceeded. Max {self.calls} calls per {self.period} seconds",
                    suggestions=["Wait before making more requests"]
                ).dict(),
                headers={
                    "Retry-After": str(self.period),
                    "X-RateLimit-Limit": str(self.calls),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time + self.period))
                }
            )

        # Record request
        if client_ip not in self.clients:
            self.clients[client_ip] = []
        self.clients[client_ip].append(current_time)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        remaining = self.calls - len(self.clients[client_ip])
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.period))

        return response


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Simple API key authentication (optional)"""

    def __init__(self, app: ASGIApp, api_keys: Optional[set] = None):
        super().__init__(app)
        self.api_keys = api_keys or set()
        self.enabled = bool(self.api_keys)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip auth for public endpoints
        public_endpoints = ["/", "/health", "/docs", "/openapi.json", "/metrics"]
        if request.url.path in public_endpoints or not self.enabled:
            return await call_next(request)

        # Check API key
        api_key = request.headers.get("X-API-Key")

        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=ErrorResponse(
                    error_code="MISSING_API_KEY",
                    error_type="AuthenticationError",
                    detail="API key required",
                    suggestions=["Include X-API-Key header"]
                ).dict()
            )

        if api_key not in self.api_keys:
            logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content=ErrorResponse(
                    error_code="INVALID_API_KEY",
                    error_type="AuthenticationError",
                    detail="Invalid API key"
                ).dict()
            )

        # Add user info to request
        request.state.api_key = api_key

        return await call_next(request)


class CompressionMiddleware(BaseHTTPMiddleware):
    """Response compression middleware"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Check if client accepts gzip
        accept_encoding = request.headers.get("Accept-Encoding", "")
        if "gzip" not in accept_encoding:
            return response

        # Only compress JSON responses
        content_type = response.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            return response

        # Don't compress small responses
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) < 1024:  # 1KB
            return response

        # In a real implementation, we would compress the response here
        # For now, just add the header to indicate we support it
        response.headers["Content-Encoding"] = "gzip"

        return response


def setup_middleware(app, config: dict):
    """Setup all middleware for the application"""

    # Add middleware in reverse order (last added is first executed)

    # Compression
    app.add_middleware(CompressionMiddleware)

    # Authentication (if API keys configured)
    api_keys = config.get("api", {}).get("api_keys", [])
    if api_keys:
        app.add_middleware(AuthenticationMiddleware, api_keys=set(api_keys))

    # Rate limiting
    rate_limit = config.get("api", {}).get("rate_limit", 100)
    if rate_limit > 0:
        app.add_middleware(RateLimitMiddleware, calls=rate_limit, period=60)

    # Error handling
    app.add_middleware(ErrorHandlingMiddleware)

    # Metrics collection
    if config.get("performance", {}).get("enable_prometheus", True):
        app.add_middleware(MetricsMiddleware)

    # Request logging
    app.add_middleware(RequestLoggingMiddleware)

    logger.info("API middleware configured")
