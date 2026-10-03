"""Runs a cleanup task on a timer in a background thread."""

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class Janitor:
    """Calls a task every few seconds until stopped. A failing run is logged and does not stop the timer."""

    def __init__(self, task: Callable[[], object], interval_seconds: float) -> None:
        """Prepare the timer; nothing runs until :meth:`start`.

        Args:
            task: What to run on every tick.
            interval_seconds: Seconds between runs.
        """
        self._task = task
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="page-janitor", daemon=True)

    def start(self) -> None:
        """Start the background thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and wait briefly for it to finish."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._task()
            except Exception:
                logger.exception("Cleanup of stored pages failed")
