"""Repeated IOI agent runs over validated historical task directories."""

import argparse
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from harness import dump, run_task


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, nargs="+", action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-397B-A17B")
    parser.add_argument("--samples-per-problem", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--wall-seconds", type=float, default=600)
    parser.add_argument("--control-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.samples_per_problem < 1
        or args.concurrency < 1
        or not 0 < args.wall_seconds <= 600
    ):
        parser.error(
            "Samples/concurrency must be positive; task deadline must be <=600s"
        )
    paths = [path.resolve() for group in args.task_dir for path in group]
    if args.manifest:
        paths.extend(
            Path(row["directory"]).resolve()
            for row in json.loads(args.manifest.read_text())
        )
    paths = list(dict.fromkeys(paths))
    if not paths:
        parser.error("Provide task directories or a validated manifest")
    tasks = []
    for path in paths:
        metadata = json.loads((path / "problem.json").read_text())
        year = metadata.get(
            "year", int(path.parent.name) if path.parent.name.isdigit() else None
        )
        if year is None or not 1989 <= int(year) <= 2023:
            parser.error("Training tasks require an explicit IOI year <=2023")
        if metadata.get("task_type") != "Batch" or not all(
            (path / name).exists()
            for name in [
                "statement.txt",
                "graders",
                "tests",
                "subtasks",
                "run_tests.sh",
            ]
        ):
            parser.error(f"Incomplete or unsupported IOI task: {path}")
        tasks.extend([(f"{year}/{path.name}", path)] * args.samples_per_problem)
    args.output.mkdir(parents=True, exist_ok=False)
    dump(
        args.output / "run_config.json",
        {
            "checkpoint": args.checkpoint,
            "track": "ioi",
            "full_score": 100,
            "total": len(tasks),
            "tasks": [task for task, _ in tasks],
            "samples_per_problem": args.samples_per_problem,
            "started_at": time.time(),
            "pid": os.getpid(),
            "concurrency": args.concurrency,
            "control_file": str(args.control_file.resolve())
            if args.control_file
            else None,
        },
    )
    active = 0

    @asynccontextmanager
    async def admission():
        nonlocal active
        while True:
            cap = args.concurrency
            if args.control_file:
                try:
                    cap = min(
                        cap,
                        max(
                            0,
                            int(
                                json.loads(args.control_file.read_text())["concurrency"]
                            ),
                        ),
                    )
                except (OSError, ValueError, KeyError):
                    pass
            if active < cap:
                active += 1
                break
            await asyncio.sleep(0.5)
        try:
            yield
        finally:
            active -= 1

    async def one(index, task, path):
        output = args.output / f"{index:03d}-{task.replace('/', '-')}"
        async with admission():
            try:
                return await run_task(
                    "ioi",
                    task,
                    path / "statement.txt",
                    path,
                    args.checkpoint,
                    output,
                    args.wall_seconds,
                )
            except Exception as error:
                output.mkdir(parents=True, exist_ok=True)
                score = {
                    "task_id": task,
                    "track": "ioi",
                    "maximum_score": 100,
                    "score": 0,
                    "returncode": 1,
                    "infrastructure_error": True,
                    "error": f"{type(error).__name__}: {error}",
                }
                dump(output / "score.json", score)
                print(json.dumps(score), flush=True)
                return score

    scores = await asyncio.gather(
        *(one(i, task, path) for i, (task, path) in enumerate(tasks))
    )
    summary = {
        "tasks": scores,
        "total_score": sum(row["score"] for row in scores),
        "maximum_score": 100 * len(scores),
    }
    dump(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
