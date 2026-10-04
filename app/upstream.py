"""Problems with the outside services (Cohere, Groq, Weaviate) and how to wait them out.

Free plans limit how fast requests may come. When a service answers "too many requests" or is briefly unavailable,
the right response is to wait for the time it names, or a growing pause when it names none, and try again, within
a bound so that a person is never left waiting for minutes.
"""

import logging
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

import cohere.errors as cohere_errors
import groq
from weaviate import exceptions as weaviate_exceptions

logger = logging.getLogger(__name__)

T = TypeVar("T")

# The service is receiving requests faster than its plan allows.
RATE_LIMIT_ERRORS: tuple[type[Exception], ...] = (groq.RateLimitError, cohere_errors.TooManyRequestsError)

# The service could not be reached or failed on its side.
UNAVAILABLE_ERRORS: tuple[type[Exception], ...] = (
    groq.APIConnectionError,
    groq.InternalServerError,
    cohere_errors.ServiceUnavailableError,
    cohere_errors.GatewayTimeoutError,
    cohere_errors.InternalServerError,
    weaviate_exceptions.WeaviateConnectionError,
    weaviate_exceptions.WeaviateTimeoutError,
)

# Problems that are worth trying again; the others (a wrong key, a bad request) would fail the same way again.
RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    groq.RateLimitError,
    groq.APIConnectionError,
    groq.InternalServerError,
    cohere_errors.TooManyRequestsError,
    cohere_errors.ServiceUnavailableError,
    cohere_errors.GatewayTimeoutError,
    cohere_errors.InternalServerError,
)


def retry_after_seconds(error: BaseException) -> float | None:
    """Read how long a service asked callers to wait, from the headers of its error response.

    Args:
        error: An error raised by the Groq or Cohere client.

    Returns:
        The wait in seconds, or ``None`` if the service named none.
    """
    headers: Mapping[str, str] | None = None
    response = getattr(error, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(error, "headers", None)
    if not headers:
        return None
    lowered = {str(key).lower(): value for key, value in headers.items()}
    try:
        if "retry-after-ms" in lowered:
            return max(float(lowered["retry-after-ms"]) / 1000, 0.0)
        if "retry-after" in lowered:
            return max(float(lowered["retry-after"]), 0.0)
    except TypeError, ValueError:
        return None
    return None


@dataclass(frozen=True)
class RetryPolicy:
    """How patiently to retry.

    Attributes:
        attempts: The most tries, including the first.
        base_delay: The first pause when the service names none; it doubles each time.
        max_delay: The longest single pause. A service that asks for more is not waited for.
        budget: The longest total time spent pausing.
    """

    attempts: int = 4
    base_delay: float = 1.0
    max_delay: float = 20.0
    budget: float = 45.0


def _spread(delay: float) -> float:
    """Add up to a quarter of a pause at random, so that many callers do not all return at the same moment."""
    return random.uniform(0, delay * 0.25)  # noqa: S311 - this spreads out retries; it is not a security use


@dataclass
class Retrier:
    """Runs an operation again after a transient failure, as its policy allows.

    The pause and the randomness are replaceable so that tests do not really wait.
    """

    policy: RetryPolicy = field(default_factory=RetryPolicy)
    sleep: Callable[[float], None] = time.sleep
    jitter: Callable[[float], float] = field(default=lambda delay: _spread(delay))

    def call(self, operation: Callable[[], T], *, name: str = "request") -> T:
        """Run ``operation``, trying again when a retryable error occurs.

        Args:
            operation: The call to make. It should do one request.
            name: What the call is, for the log.

        Returns:
            What ``operation`` returns.

        Raises:
            Exception: The last error, when attempts or the time budget run out, when the service asks for a
                longer wait than allowed, or at once for errors that retrying cannot fix.
        """
        waited = 0.0
        for attempt in range(1, self.policy.attempts + 1):
            try:
                return operation()
            except RETRYABLE_ERRORS as error:
                asked = retry_after_seconds(error)
                if attempt == self.policy.attempts or (asked is not None and asked > self.policy.max_delay):
                    raise
                pause = (
                    asked
                    if asked is not None
                    else min(self.policy.base_delay * 2 ** (attempt - 1), self.policy.max_delay)
                )
                pause += self.jitter(pause)
                if waited + pause > self.policy.budget:
                    raise
                logger.warning(
                    "%s failed with %s; trying again in %.1f s (attempt %d of %d)",
                    name,
                    type(error).__name__,
                    pause,
                    attempt + 1,
                    self.policy.attempts,
                )
                self.sleep(pause)
                waited += pause
        raise AssertionError("unreachable")  # pragma: no cover - the loop always returns or raises


def describe_wait(seconds: float | None) -> str:
    """Say how long to wait in words, for a message to a person.

    Args:
        seconds: The wait the service asked for, or ``None`` if it named none.

    Returns:
        For example ``"about 23 seconds"`` or ``"about a minute"``.
    """
    if seconds is None or seconds > 90:
        return "about a minute"
    rounded = max(1, round(seconds))
    return f"about {rounded} second{'s' if rounded != 1 else ''}"


def public_error_details(error: BaseException) -> dict[str, Any]:
    """Collect what the page may be told about a rate limit.

    Args:
        error: A rate limit error.

    Returns:
        The wait to show to the person and the value for the ``Retry-After`` header.
    """
    asked = retry_after_seconds(error)
    return {"wait_text": describe_wait(asked), "retry_after": str(max(1, round(asked))) if asked is not None else "60"}
