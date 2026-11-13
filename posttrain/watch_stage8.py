"""Stage-8 admission cutoff and load/error controls; never cancel active tasks."""

import datetime
import json
import os
import time
from pathlib import Path

from progress import snapshot
from synthetic_ioi import write_json

ROOT = Path(os.environ.get("IOI2_ROOT", str(Path(__file__).resolve().parent)))


def next_limit(current, initial, reason, now, cutoff, load, memory=False, rate=False):
    if now >= cutoff:
        return 0, "stage8_15_45_utc_cutoff"
    owned = (current == initial and reason is None) or str(reason).startswith("stage8_")
    if owned and (load > 14 or memory or rate):
        return min(current, max(1, current // 2)), "stage8_rate_load_memory_backoff"
    return current, reason


def main():
    cutoff = (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(hour=15, minute=45, second=0, microsecond=0)
        .timestamp()
    )
    latest = ROOT / "logs/stage8_progress_latest.json"
    previous = json.loads(latest.read_text()) if latest.exists() else {}
    seen_memory = set(previous.get("memory_error_files", []))
    rates = {
        r["run"]: r.get("local_429_log_matches", 0) for r in previous.get("runs", [])
    }
    while True:
        ledger = ROOT / "logs/stage8_launches.json"
        jobs = json.loads(ledger.read_text()) if ledger.exists() else {}
        rows = []
        load = os.getloadavg()
        for name in jobs:
            directory = ROOT / "runs" / name
            if jobs[name].get("kind") == "conversion":
                progress_path = ROOT / "synthetic_work/progress.json"
                progress = (
                    json.loads(progress_path.read_text())
                    if progress_path.exists()
                    else {}
                )
                row = {
                    "run": name,
                    "done": progress.get("completed", 0),
                    "accepted": progress.get("accepted", 0),
                    "total": 220,
                    "local_429_log_matches": (ROOT / "logs/syn-ioi-conversion.log")
                    .read_text(errors="replace")
                    .count("HTTP status 429"),
                }
            elif (directory / "run_config.json").exists():
                row = snapshot(directory)
            else:
                row = {"run": name, "initializing": True, "local_429_log_matches": 0}
            control_path = ROOT / "logs" / f"{name}.control.json"
            control = json.loads(control_path.read_text())
            memory = {
                str(p)
                for p in directory.glob("*/stderr.log")
                if "MemoryError" in p.read_text(errors="replace")
            }
            new_memory = memory - seen_memory
            seen_memory |= memory
            count = row.get("local_429_log_matches", 0)
            rate = count > rates.get(name, 0)
            rates[name] = count
            old = control["concurrency"]
            new, reason = next_limit(
                old,
                jobs[name]["concurrency"],
                control.get("reason"),
                time.time(),
                cutoff,
                load[0],
                bool(new_memory),
                rate,
            )
            if new < old:
                control.update(concurrency=new, reason=reason, at=time.time())
                write_json(control_path, control)
                print(
                    json.dumps(
                        {
                            "alert": True,
                            "run": name,
                            "before": old,
                            "after": new,
                            "reason": reason,
                            "memory": sorted(new_memory),
                            "local_429": rate,
                            "at": time.time(),
                        }
                    ),
                    flush=True,
                )
            row["active_control_limit"] = new
            rows.append(row)
        state = {
            "at": time.time(),
            "cutoff_utc": cutoff,
            "load": load,
            "runs": rows,
            "memory_error_files": sorted(seen_memory),
            "rate_signal": "local direct Tinker logs only; no fleet signal",
        }
        target = ROOT / "logs/stage8_progress_latest.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(target)
        print(json.dumps(state), flush=True)
        time.sleep(
            min(20, max(0.1, cutoff - time.time())) if time.time() < cutoff else 20
        )


if __name__ == "__main__":
    main()
