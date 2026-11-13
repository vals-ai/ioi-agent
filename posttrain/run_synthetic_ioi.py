"""Run exact IOI loops as validated synthetic tasks arrive in an atomic manifest."""

import argparse
import asyncio
import datetime
import json
import os
import time
from pathlib import Path

from admission import AdmissionGate
from harness import dump, run_task
from synthetic_ioi import ROOT, write_json


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--samples-per-problem", type=int, default=2)
    parser.add_argument("--wall-seconds", type=float, default=600)
    parser.add_argument("--control-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "synthetic_ioi/manifest.json"
    )
    args = parser.parse_args()
    if (
        args.concurrency < 1
        or args.samples_per_problem < 1
        or not 0 < args.wall_seconds <= 600
    ):
        parser.error("Positive samples/concurrency and wall cap <=600 required")
    args.output.mkdir(exist_ok=False)
    config = {
        "checkpoint": args.checkpoint,
        "track": "ioi",
        "full_score": 100,
        "kind": "synthetic_ioi",
        "total": 0,
        "target_total": 300,
        "concurrency": args.concurrency,
        "samples_per_problem": args.samples_per_problem,
        "control_file": str(args.control_file),
        "started_at": time.time(),
        "pid": os.getpid(),
        "tasks": [],
    }
    write_json(args.output / "run_config.json", config)
    cutoff = (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(hour=15, minute=45, second=0, microsecond=0)
        .timestamp()
    )
    gate = AdmissionGate(args.concurrency, args.control_file)
    scheduled = set()
    futures = []

    async def one(index, row):
        context = gate.slot()
        try:
            remaining = cutoff - time.time()
            if remaining <= 0:
                return
            await asyncio.wait_for(context.__aenter__(), remaining)
        except asyncio.TimeoutError:
            return
        try:
            directory = Path(row["directory"])
            metadata = json.loads((directory / "problem.json").read_text())
            if (
                metadata.get("kind") != "synthetic_ioi"
                or metadata.get("source_split") != "train"
                or metadata.get("validation", {}).get("score") != 100
                or not row.get("duplicate_main_rejected")
            ):
                raise ValueError("Unvalidated synthetic task refused")
            output = args.output / f"{index:03d}-{directory.name}"
            try:
                await run_task(
                    "ioi",
                    row["id"],
                    directory / "statement.txt",
                    directory,
                    args.checkpoint,
                    output,
                    args.wall_seconds,
                )
            except Exception as error:
                output.mkdir(exist_ok=True)
                dump(
                    output / "score.json",
                    {
                        "score": 0,
                        "task_id": row["id"],
                        "track": "ioi",
                        "maximum_score": 100,
                        "returncode": 1,
                        "infrastructure_error": True,
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
                print(
                    json.dumps({"error": str(error), "output": str(output)}), flush=True
                )
        finally:
            await context.__aexit__(None, None, None)

    while time.time() < cutoff:
        if args.manifest.exists():
            rows = json.loads(args.manifest.read_text())
            for row in rows:
                if row["id"] in scheduled:
                    continue
                scheduled.add(row["id"])
                for _ in range(args.samples_per_problem):
                    index = len(futures)
                    futures.append(asyncio.create_task(one(index, row)))
                    config["tasks"].append(row["id"])
                config["total"] = len(futures)
                write_json(args.output / "run_config.json", config)
        complete = args.manifest.parent / "conversion_complete.json"
        if (
            complete.exists()
            and len(scheduled) >= json.loads(complete.read_text())["accepted"]
        ):
            break
        await asyncio.sleep(5)
    await asyncio.gather(*futures)
    scores = [json.loads(p.read_text()) for p in args.output.glob("*/score.json")]
    write_json(
        args.output / "summary.json",
        {
            "tasks": scores,
            "total_score": sum(s["score"] for s in scores),
            "maximum_score": 100 * len(scores),
            "admission_cutoff_utc": cutoff,
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
