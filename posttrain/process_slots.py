"""Cross-process slots shared by agent tools and hidden-test grading."""

import asyncio
import fcntl
import json
import os
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path


def try_acquire():
    root = Path(
        os.environ.get(
            "IOI2_PROCESS_SLOTS_DIR", Path(__file__).parent / ".process_slots"
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    count = int(os.environ.get("IOI2_PROCESS_SLOTS", "6"))
    config = os.environ.get("IOI2_PROCESS_SLOTS_CONFIG")
    if config:
        try:
            count = int(json.loads(Path(config).read_text())["process_slots"])
        except (OSError, ValueError, KeyError):
            pass
    if count < 1:
        raise ValueError("process_slots must be positive")
    for index in range(count):
        handle = (root / str(index)).open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            handle.close()
    return None


@contextmanager
def process_slot():
    handle = try_acquire()
    while handle is None:
        time.sleep(0.05)
        handle = try_acquire()
    try:
        yield
    finally:
        handle.close()


@asynccontextmanager
async def async_process_slot():
    handle = try_acquire()
    while handle is None:
        await asyncio.sleep(0.05)
        handle = try_acquire()
    try:
        yield
    finally:
        handle.close()
