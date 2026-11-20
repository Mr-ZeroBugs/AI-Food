"""Per-client request limit for the photo endpoint, so one client (or a leaked app key) cannot run up
the server bill. In-memory and per instance: with several Cloud Run instances the effective limit is
roughly limit x instances, which is fine for abuse control.

    RATE_LIMIT_PER_MIN   requests per client per rolling minute (default 20; 0 disables)
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Deque, Dict

WINDOW_S = 60.0


class RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    def allow(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] > WINDOW_S:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            if now - self._last_sweep > WINDOW_S:  # drop idle clients so memory stays bounded
                self._hits = {k: v for k, v in self._hits.items() if v and now - v[-1] <= WINDOW_S}
                self._last_sweep = now
            return True


limiter = RateLimiter(int(os.environ.get("RATE_LIMIT_PER_MIN", "20")))


def client_key(headers, client_host: str | None) -> str:
    """Cloud Run puts the caller's address first in X-Forwarded-For."""
    fwd = headers.get("x-forwarded-for", "")
    return (fwd.split(",")[0].strip() if fwd else "") or client_host or "unknown"
