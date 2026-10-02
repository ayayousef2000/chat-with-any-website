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
