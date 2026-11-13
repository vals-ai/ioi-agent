"""Live admission limits without cancelling already-running tasks."""

import asyncio
import json
from contextlib import asynccontextmanager


class AdmissionGate:
    def __init__(self, maximum, control=None):
        if maximum < 1:
            raise ValueError("Concurrency must be positive")
        self.maximum = maximum
        self.control = control
        self.active = 0
        self.semaphore = asyncio.Semaphore(maximum)

    def limit(self):
        if self.control and self.control.exists():
            try:
                return min(
                    self.maximum,
                    max(0, int(json.loads(self.control.read_text())["concurrency"])),
                )
            except (OSError, ValueError, KeyError, TypeError):
                return 0
        return self.maximum

    @asynccontextmanager
    async def slot(self):
        async with self.semaphore:
            while self.active >= self.limit():
                await asyncio.sleep(0.1)
            self.active += 1
            try:
                yield
            finally:
                self.active -= 1
