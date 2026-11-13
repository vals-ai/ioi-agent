"""Sample process RSS, cap live tool workers, and enforce the collection cutoff."""

import datetime
import json
import os
import resource
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(os.environ.get("IOI2_ROOT", str(Path(__file__).resolve().parent)))


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now.replace(hour=10, minute=45, second=0, microsecond=0).timestamp()
    stopped = False
    while True:
        top = []
        for line in subprocess.check_output(
            ["ps", "-eo", "pid=,rss=,comm=,args="], text=True
        ).splitlines():
            parts = line.strip().split(None, 3)
            if len(parts) < 4:
                continue
            pid, rss, name, command = parts
            if "agent_worker.py" in command and "python -c" not in command:
                try:
                    resource.prlimit(int(pid), resource.RLIMIT_AS, (4 * 1024**3,) * 2)
                except (OSError, ProcessLookupError):
                    pass
            if int(rss) > 25000:
                top.append(
                    {
                        "pid": int(pid),
                        "rss_kib": int(rss),
                        "name": name,
                        "agent_worker": "agent_worker.py" in command,
                    }
                )
        top.sort(key=lambda row: -row["rss_kib"])
        memory = (
            (Path(os.environ.get("IOI2_CGROUP", "/sys/fs/cgroup")) / "memory.current")
            .read_text()
            .strip()
        )
        print(
            json.dumps(
                {
                    "at": time.time(),
                    "load_average": os.getloadavg(),
                    "memory_bytes": int(memory),
                    "processes": top[:30],
                    "cutoff_utc": cutoff,
                }
            ),
            flush=True,
        )
        if time.time() >= cutoff and not stopped:
            for line in subprocess.check_output(
                ["ps", "-eo", "pid=,args="], text=True
            ).splitlines():
                fields = line.strip().split(None, 1)
                if len(fields) != 2:
                    continue
                pid, command = fields
                if command.split()[0].split("/")[-1] not in ("python", "python3"):
                    continue
                if (
                    "/progress.py" in command and "--watch" in command
                ) or "/watch_stage4.py" in command:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for name in ["coll-teacher-1", "coll-base-1", "ioi-teacher-1"]:
                path = ROOT / "logs" / f"{name}.control.json"
                path.write_text(
                    json.dumps(
                        {
                            "concurrency": 0,
                            "reason": "stage5_10_45_utc_cutoff",
                            "at": time.time(),
                        }
                    )
                )
            stopped = True
            print(
                json.dumps({"admission_stopped": True, "at": time.time()}), flush=True
            )
        time.sleep(15 if stopped else min(15, max(0.1, cutoff - time.time())))


if __name__ == "__main__":
    main()
