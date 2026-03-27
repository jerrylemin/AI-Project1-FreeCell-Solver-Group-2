"""Bounded queue helpers for keeping UI progress fresh."""

from __future__ import annotations

from queue import Empty, Full, Queue
from typing import Generic, List, TypeVar


T = TypeVar("T")


class FreshQueue(Generic[T]):
    """A bounded queue that prefers the newest items over stale ones."""

    def __init__(self, maxsize: int = 32):
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self._queue: Queue[T] = Queue(maxsize=maxsize)

    def put(self, item: T) -> None:
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except Full:
                try:
                    self._queue.get_nowait()
                except Empty:
                    return

    def drain(self) -> List[T]:
        items: List[T] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except Empty:
                return items

    def empty(self) -> bool:
        return self._queue.empty()
