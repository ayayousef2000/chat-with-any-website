"""Errors whose messages are safe to show to the end user."""


class AppError(Exception):
    """Base error whose message is safe to show to the end user."""

    status_code = 400


class InvalidURLError(AppError):
    """The URL is malformed, unresolvable or points to a non-public address."""

    status_code = 400


class FetchError(AppError):
    """The page could not be downloaded."""

    status_code = 502


class ExtractionError(AppError):
    """The page was downloaded but has no readable text."""

    status_code = 422


class NotIngestedError(AppError):
    """A question was asked about a URL that has not been loaded."""

    status_code = 404


class TooManyRequestsError(AppError):
    """The caller used the API more often than the limits allow.

    Attributes:
        retry_after: How many seconds to wait before trying again.
    """

    status_code = 429

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after
