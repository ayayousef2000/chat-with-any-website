"""Keeps track of which stored pages exist, when they were loaded and when they were last used.

The vector database holds the page text; this small SQLite file holds the usage times that decide when a page is
deleted. It is meant for a single server process.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PageRecord:
    """One stored page and the times that decide when it is deleted (seconds since the epoch)."""

    url: str
    title: str
    chunk_count: int
    loaded_at: float
    last_used_at: float


def is_expired(record: PageRecord, now: float, idle_seconds: float, max_age_seconds: float) -> bool:
    """Tell whether a page should be deleted.

    Args:
        record: The page.
        now: The current time.
        idle_seconds: How long a page may go without being used.
        max_age_seconds: How long after loading a page may be kept even while it is in use.

    Returns:
        ``True`` if the page was unused for too long or is older than the maximum age.
    """
    return now - record.last_used_at >= idle_seconds or now - record.loaded_at >= max_age_seconds


class PageRegistry:
    """A SQLite table of stored pages, safe to use from several threads."""

    def __init__(self, path: str) -> None:
        """Open the database, creating the file and the table if needed.

        Args:
            path: Location of the SQLite file. Missing folders are created.
        """
        self._path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS pages ("
                "url TEXT PRIMARY KEY, title TEXT NOT NULL, chunk_count INTEGER NOT NULL, "
                "loaded_at REAL NOT NULL, last_used_at REAL NOT NULL)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # One short-lived connection per operation keeps the registry safe across the server's worker threads.
        with closing(sqlite3.connect(self._path, timeout=10)) as connection, connection:
            yield connection

    @staticmethod
    def _record(row: tuple[str, str, int, float, float]) -> PageRecord:
        return PageRecord(url=row[0], title=row[1], chunk_count=row[2], loaded_at=row[3], last_used_at=row[4])

    def get(self, url: str) -> PageRecord | None:
        """Look up a page.

        Args:
            url: The normalized page address.

        Returns:
            The page, or ``None`` if it is not registered.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT url, title, chunk_count, loaded_at, last_used_at FROM pages WHERE url = ?", (url,)
            ).fetchone()
        return self._record(row) if row else None

    def all(self) -> list[PageRecord]:
        """List every registered page, least recently used first."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT url, title, chunk_count, loaded_at, last_used_at FROM pages ORDER BY last_used_at"
            ).fetchall()
        return [self._record(row) for row in rows]

    def record_load(self, url: str, title: str, chunk_count: int, now: float) -> None:
        """Register a page that was just loaded, replacing any earlier entry for the same address."""
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO pages (url, title, chunk_count, loaded_at, last_used_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(url) DO UPDATE SET title = excluded.title, chunk_count = excluded.chunk_count, "
                "loaded_at = excluded.loaded_at, last_used_at = excluded.last_used_at",
                (url, title, chunk_count, now, now),
            )

    def touch(self, url: str, now: float) -> None:
        """Mark a page as used just now."""
        with self._connect() as connection:
            connection.execute("UPDATE pages SET last_used_at = ? WHERE url = ?", (now, url))

    def remove(self, url: str) -> bool:
        """Forget a page.

        Returns:
            ``True`` if the page was registered.
        """
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM pages WHERE url = ?", (url,))
        return cursor.rowcount > 0

    def expired(self, now: float, idle_seconds: float, max_age_seconds: float) -> list[PageRecord]:
        """List the pages that are unused for too long or older than the maximum age."""
        return [record for record in self.all() if is_expired(record, now, idle_seconds, max_age_seconds)]

    def over_capacity(self, max_chunks: int) -> list[PageRecord]:
        """List the least recently used pages to remove so that no more than ``max_chunks`` chunks remain.

        The most recently used pages are kept; pages are chosen from the oldest use onwards.
        """
        records = self.all()
        total = sum(record.chunk_count for record in records)
        doomed: list[PageRecord] = []
        for record in records:
            if total <= max_chunks:
                break
            doomed.append(record)
            total -= record.chunk_count
        return doomed
