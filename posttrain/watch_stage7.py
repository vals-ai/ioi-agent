"""Monitor the six repeated dev samples and apply approved admission backoff."""

import json
import os
import time
from pathlib import Path

from progress import snapshot

ROOT = Path(os.environ.get("IOI2_ROOT", str(Path(__file__).resolve().parent)))


def next_limit(current, load, memory_error=False, rate_error=False):
    if rate_error:
        return max(1, current // 2)
    return min(current, 4) if load > 14 or memory_error else current


def main():
    jobs = json.loads((ROOT / "logs/stage7_launches.json").read_text())
    previous_429 = {name: 0 for name in jobs}
    seen_memory = set()
    while True:
        rows = []
        load = os.getloadavg()
        memory = set()
        for name in jobs:
            rows.append(snapshot(ROOT / "runs" / name))
            for path in (ROOT / "runs" / name).glob("*/stderr.log"):
                if b"MemoryError" in path.read_bytes():
                    memory.add(str(path))
        new_memory = memory - seen_memory
        rate_trigger = any(
            row["local_429_log_matches"] > previous_429[row["run"]] for row in rows
        )
        if load[0] > 14 or new_memory or rate_trigger:
            for name in jobs:
                path = ROOT / "logs" / f"{name}.control.json"
                control = json.loads(path.read_text())
                if not control.get("reason", "").startswith("stage7_"):
                    continue
                old = control["concurrency"]
                new = next_limit(old, load[0], bool(new_memory), rate_trigger)
                if new != old:
                    control.update(
                        concurrency=new,
                        reason="stage7_rate_backoff"
                        if rate_trigger
                        else "stage7_load_or_memory_backoff",
                        at=time.time(),
                    )
                    path.write_text(json.dumps(control))
            print(
                json.dumps(
                    {
                        "alert": True,
                        "load": load,
                        "new_memory_errors": sorted(new_memory),
                        "new_local_429": rate_trigger,
                    }
                ),
                flush=True,
            )
        seen_memory = memory
        previous_429 = {row["run"]: row["local_429_log_matches"] for row in rows}
        data = {
            "at": time.time(),
            "load": load,
            "runs": rows,
            "memory_error_files": sorted(memory),
            "rate_signal": "local logs only; no fleet signal for direct Tinker",
        }
        (ROOT / "logs/stage7_progress_latest.json").write_text(
            json.dumps(data, indent=2)
        )
        print(json.dumps(data), flush=True)
        if all(row["done"] == 12 for row in rows):
            break
        time.sleep(20)


if __name__ == "__main__":
    main()
