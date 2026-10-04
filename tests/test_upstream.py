"""Waiting out rate limits and brief outages of the outside services."""

import logging
from collections.abc import Callable
from typing import Any

import cohere.errors as cohere_errors
import groq
import httpx
import pytest

from app.upstream import Retrier, RetryPolicy, _spread, describe_wait, public_error_details, retry_after_seconds

_REQUEST = httpx.Request("POST", "https://api.example.invalid/v1")


def groq_error(cls: type[groq.APIStatusError], status: int, **headers: str) -> groq.APIStatusError:
    return cls("failed", response=httpx.Response(status, request=_REQUEST, headers=headers), body=None)


def rate_limit(**headers: str) -> groq.APIStatusError:
    return groq_error(groq.RateLimitError, 429, **headers)


class Flaky:
    """An operation that raises the given errors one after the other and then succeeds."""

    def __init__(self, *errors: BaseException) -> None:
        self.errors = list(errors)
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def retrier(
    policy: RetryPolicy | None = None, jitter: Callable[[float], float] = lambda _: 0.0
) -> tuple[Retrier, list[float]]:
    pauses: list[float] = []
    return Retrier(policy or RetryPolicy(), sleep=pauses.append, jitter=jitter), pauses


# --- the retry rules --------------------------------------------------------------------------------------------


def test_a_call_that_works_is_made_once_without_waiting() -> None:
    operation = Flaky()
    r, pauses = retrier()
    assert r.call(operation) == "ok"
    assert (operation.calls, pauses) == (1, [])


def test_without_a_hint_the_pause_doubles_each_time() -> None:
    operation = Flaky(rate_limit(), rate_limit())
    r, pauses = retrier()
    assert r.call(operation) == "ok"
    assert operation.calls == 3
    assert pauses == [1.0, 2.0]


def test_the_pause_the_service_asks_for_is_used() -> None:
    operation = Flaky(rate_limit(**{"retry-after": "3"}))
    r, pauses = retrier()
    assert r.call(operation) == "ok"
    assert pauses == [3.0]


def test_a_pause_in_milliseconds_is_understood() -> None:
    r, pauses = retrier()
    r.call(Flaky(rate_limit(**{"retry-after-ms": "1500"})))
    assert pauses == [1.5]


def test_the_doubling_stops_at_the_longest_allowed_pause() -> None:
    r, pauses = retrier(RetryPolicy(attempts=6, base_delay=4, max_delay=10, budget=100))
    r.call(Flaky(*[rate_limit() for _ in range(5)]))
    assert pauses == [4.0, 8.0, 10.0, 10.0, 10.0]


def test_a_little_randomness_is_added_to_each_pause() -> None:
    r, pauses = retrier(jitter=lambda delay: delay * 0.1)
    r.call(Flaky(rate_limit()))
    assert pauses == [pytest.approx(1.1)]
    assert all(0 <= _spread(4) <= 1 for _ in range(50))


def test_it_gives_up_after_the_last_attempt_and_raises_the_last_error() -> None:
    last = rate_limit(**{"retry-after": "1"})
    operation = Flaky(rate_limit(), rate_limit(), rate_limit(), last)
    r, pauses = retrier(RetryPolicy(attempts=4))
    with pytest.raises(groq.RateLimitError) as raised:
        r.call(operation)
    assert raised.value is last
    assert operation.calls == 4
    assert len(pauses) == 3


def test_a_service_that_asks_for_a_longer_wait_than_allowed_is_not_waited_for() -> None:
    operation = Flaky(rate_limit(**{"retry-after": "50"}))
    r, pauses = retrier(RetryPolicy(max_delay=20))
    with pytest.raises(groq.RateLimitError):
        r.call(operation)
    assert (operation.calls, pauses) == (1, [])


def test_it_stops_when_the_total_waiting_time_would_pass_the_budget() -> None:
    operation = Flaky(rate_limit(), rate_limit(), rate_limit())
    r, pauses = retrier(RetryPolicy(attempts=5, base_delay=2, max_delay=20, budget=3))
    with pytest.raises(groq.RateLimitError):
        r.call(operation)
    assert pauses == [2.0]  # the next pause of 4 seconds would pass the budget of 3


@pytest.mark.parametrize(
    "error",
    [
        ValueError("a bug in the program"),
        groq_error(groq.AuthenticationError, 401),
        groq_error(groq.BadRequestError, 400),
        cohere_errors.UnauthorizedError(body={"message": "invalid token"}),
    ],
)
def test_errors_that_trying_again_cannot_fix_are_raised_at_once(error: BaseException) -> None:
    operation = Flaky(error)
    r, pauses = retrier()
    with pytest.raises(type(error)):
        r.call(operation)
    assert (operation.calls, pauses) == (1, [])


@pytest.mark.parametrize(
    "error",
    [
        rate_limit(),
        groq_error(groq.InternalServerError, 503),
        groq.APIConnectionError(request=_REQUEST),
        groq.APITimeoutError(request=_REQUEST),
        cohere_errors.TooManyRequestsError(body={"message": "limit"}),
        cohere_errors.ServiceUnavailableError(body={"message": "down"}),
        cohere_errors.GatewayTimeoutError(body={"message": "slow"}),
        cohere_errors.InternalServerError(body={"message": "oops"}),
    ],
)
def test_limits_and_outages_of_either_service_are_tried_again(error: BaseException) -> None:
    operation = Flaky(error)
    r, pauses = retrier()
    assert r.call(operation) == "ok"
    assert (operation.calls, len(pauses)) == (2, 1)


def test_each_retry_is_logged_with_what_was_called(caplog: pytest.LogCaptureFixture) -> None:
    r, _ = retrier()
    with caplog.at_level(logging.WARNING, logger="app.upstream"):
        r.call(Flaky(rate_limit()), name="Groq answer")
    assert "Groq answer failed with RateLimitError" in caplog.text
    assert "attempt 2 of 4" in caplog.text


# --- reading what the service asked for -------------------------------------------------------------------------


def test_the_wait_is_read_from_groq_and_cohere_errors() -> None:
    assert retry_after_seconds(rate_limit(**{"retry-after": "23"})) == 23.0
    assert retry_after_seconds(cohere_errors.TooManyRequestsError(body={}, headers={"Retry-After": "7"})) == 7.0


def test_milliseconds_win_over_seconds_and_names_may_be_any_case() -> None:
    error = rate_limit(**{"Retry-After": "9", "RETRY-AFTER-MS": "2500"})
    assert retry_after_seconds(error) == 2.5


@pytest.mark.parametrize(
    "error",
    [
        rate_limit(),
        rate_limit(**{"retry-after": "soon"}),
        ValueError("no headers at all"),
        cohere_errors.TooManyRequestsError(body={}),
    ],
)
def test_no_wait_is_reported_when_none_was_named(error: BaseException) -> None:
    assert retry_after_seconds(error) is None


def test_a_negative_wait_becomes_zero() -> None:
    assert retry_after_seconds(rate_limit(**{"retry-after": "-5"})) == 0.0


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (None, "about a minute"),
        (0.2, "about 1 second"),
        (1, "about 1 second"),
        (23.4, "about 23 seconds"),
        (89, "about 89 seconds"),
        (200, "about a minute"),
    ],
)
def test_waits_are_described_in_words(seconds: float | None, text: str) -> None:
    assert describe_wait(seconds) == text


def test_the_page_is_told_the_wait_and_the_header_value() -> None:
    details: dict[str, Any] = public_error_details(rate_limit(**{"retry-after": "23"}))
    assert details == {"wait_text": "about 23 seconds", "retry_after": "23"}
    assert public_error_details(rate_limit()) == {"wait_text": "about a minute", "retry_after": "60"}
