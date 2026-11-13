"""Incremental local collection status and optional rate-limit back-off."""

import argparse
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(os.environ.get("IOI2_ROOT", str(Path(__file__).resolve().parent)))
RUNS = ["coll-teacher-1", "coll-base-1", "heldout-base", "heldout-step30"]
FLOORS = {"coll-teacher-1": 6, "coll-base-1": 5, "heldout-base": 2, "heldout-step30": 2}
RATE = re.compile(
    r"HTTP(?:Error| status)?[^\n]{0,20}\b429\b|(?:status(?:_code)?|error code)[=: ]+429\b|429[^\n]{0,60}(?:too many requests|rate.?limit)|RateLimitError",
    re.I,
)


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def snapshot(run):
    config = read(run / "run_config.json", {})
    scores = [
        value
        for path in run.glob("*/score.json")
        if isinstance(value := read(path), dict)
        and isinstance(value.get("score"), (int, float))
    ]
    errors = 0
    for path in run.glob("*/score.json"):
        score = read(path, {})
        if not isinstance(score, dict):
            continue
        result = read(path.parent / "result.json", {}) or {}
        final_error = result.get("final_error") or {}
        errors += bool(
            (
                final_error
                and final_error.get("type")
                not in ("MaxTurnsExceeded", "MaxTimeExceeded")
            )
            or score.get("returncode") not in (None, 0, -9)
        )
    rate_events = 0
    for pattern in ["*/stderr.log", "*/stdout.log"]:
        for path in run.glob(pattern):
            try:
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 2_000_000))
                    rate_events += len(
                        RATE.findall(stream.read().decode(errors="replace"))
                    )
            except OSError:
                pass
    elapsed = max(1, time.time() - config.get("started_at", time.time()))
    control = (
        read(Path(config["control_file"]), {}) if config.get("control_file") else {}
    )
    return {
        "run": run.name,
        "done": len(scores),
        "total": config.get(
            "total",
            len(summary["tasks"])
            if isinstance(summary := read(run / "summary.json"), dict)
            and isinstance(summary.get("tasks"), list)
            else None,
        ),
        "mean_hidden_score": sum(x["score"] for x in scores) / len(scores)
        if scores
        else None,
        "full_solves": sum(x["score"] == config.get("full_score", 1) for x in scores),
        "score_units": "ioi_points" if config.get("full_score") == 100 else "fraction",
        "errors": errors,
        "timeouts": sum(x.get("timed_out", False) for x in scores),
        "tasks_per_hour": len(scores) * 3600 / elapsed,
        "elapsed_minutes": elapsed / 60,
        "concurrency": control.get("concurrency", config.get("concurrency")),
        "local_429_log_matches": rate_events,
        "rate_limit_signal": "local direct-Tinker client logs only; no gateway coverage",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--watch", type=float, default=0)
    parser.add_argument("--auto-backoff", action="store_true")
    args = parser.parse_args()
    paths = args.runs or [ROOT / "runs" / name for name in RUNS]
    last, adjusted = {}, {}
    load_checked = False
    while True:
        rows = [snapshot(path) for path in paths]
        for path, row in zip(paths, rows):
            name = path.name
            new_events = row["local_429_log_matches"] > last.get(name, 0)
            bad_errors = row["done"] >= 8 and row["errors"] / row["done"] > 0.3
            if (
                args.auto_backoff
                and (new_events or bad_errors)
                and time.time() - adjusted.get(name, 0) >= 180
            ):
                config = read(path / "run_config.json", {})
                if config.get("control_file"):
                    cap = row["concurrency"]
                    floor = FLOORS.get(name, 2)
                    new = max(floor, cap // 2) if cap > floor else 0
                    Path(config["control_file"]).write_text(
                        json.dumps(
                            {
                                "concurrency": new,
                                "reason": "local_429_or_many_terminal_errors",
                                "at": time.time(),
                            }
                        )
                    )
                    print(
                        json.dumps(
                            {
                                "alert": "backoff",
                                "run": name,
                                "before": cap,
                                "after": new,
                                "local_429_log_matches": row["local_429_log_matches"],
                                "errors": row["errors"],
                            }
                        ),
                        flush=True,
                    )
                    adjusted[name] = time.time()
            last[name] = row["local_429_log_matches"]
        load = os.getloadavg()
        if (
            args.auto_backoff
            and not load_checked
            and rows
            and min(row["elapsed_minutes"] for row in rows) >= 10
        ):
            slots = 8 if load[0] < 7 else 6
            (ROOT / "logs/process_slots.json").write_text(
                json.dumps(
                    {
                        "process_slots": slots,
                        "load_at_10min": load,
                        "checked_at": time.time(),
                    }
                )
            )
            load_checked = True
            print(
                json.dumps(
                    {
                        "event": "ten_minute_load_check",
                        "load_average": load,
                        "process_slots": slots,
                    }
                ),
                flush=True,
            )
        if args.auto_backoff and load[0] >= 14:
            (ROOT / "logs/process_slots.json").write_text(
                json.dumps({"process_slots": 4, "reason": "load>=14", "load": load})
            )
        print(
            json.dumps({"at": time.time(), "load_average": load, "runs": rows}),
            flush=True,
        )
        if not args.watch or all(
            row["total"] is not None and row["done"] >= row["total"] for row in rows
        ):
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
