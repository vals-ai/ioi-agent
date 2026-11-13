import argparse
import asyncio
import json
import os
import random
import time
from contextlib import asynccontextmanager
from pathlib import Path

from harness import dump, run_task


async def main():
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--problem-dir", type=Path, action="append")
    inputs.add_argument("--pool-dir", type=Path)
    parser.add_argument("--samples-per-problem", type=int, default=1)
    parser.add_argument("--eval-heldout", action="store_true")
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--control-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--memory-mb", type=int, default=1024)
    parser.add_argument("--wall-seconds", type=float, default=600)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.samples_per_problem < 1:
        parser.error("--samples-per-problem must be positive")
    problems = args.problem_dir
    if args.pool_dir:
        manifest = json.loads((args.pool_dir / "manifest.json").read_text())
        if (
            any(item.get("split") != "train" for item in manifest)
            and not args.eval_heldout
        ):
            parser.error(
                "--pool-dir collection requires a training manifest, not held-out"
            )
        problems = [args.pool_dir / item["id"] for item in manifest]
    if not problems:
        parser.error("No problems selected")
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(problems)
    config_path = args.output / "run_config.json"
    prior_config = (
        json.loads(config_path.read_text())
        if args.resume and config_path.is_file()
        else None
    )
    if args.resume and prior_config is None:
        parser.error("--resume requires an existing run_config.json")
    if prior_config and (
        prior_config["checkpoint"] != args.checkpoint
        or prior_config["total"] != len(problems) * args.samples_per_problem
        or prior_config["problem_order"] != [p.name for p in problems]
    ):
        parser.error(
            "--resume checkpoint, task count, and seeded order must match the original run"
        )
    args.output.mkdir(parents=True, exist_ok=args.resume)
    dump(
        args.output / "run_config.json",
        {
            "checkpoint": args.checkpoint,
            "total": len(problems) * args.samples_per_problem,
            "concurrency": args.concurrency,
            "shuffle_seed": args.shuffle_seed,
            "started_at": prior_config["started_at"] if prior_config else time.time(),
            "pid": os.getpid(),
            "control_file": str(args.control_file) if args.control_file else None,
            "problem_order": [p.name for p in problems],
        },
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    active = 0

    @asynccontextmanager
    async def admission():
        nonlocal active
        async with semaphore:
            while True:
                limit = args.concurrency
                if args.control_file and args.control_file.is_file():
                    try:
                        limit = min(
                            limit,
                            max(
                                0,
                                int(
                                    json.loads(args.control_file.read_text())[
                                        "concurrency"
                                    ]
                                ),
                            ),
                        )
                    except (OSError, ValueError, KeyError):
                        pass
                if active < limit:
                    active += 1
                    break
                await asyncio.sleep(0.5)
            try:
                yield
            finally:
                active -= 1

    async def one(index, problem):
        task_output = args.output / f"{index:03d}-{problem.name}"
        if args.resume and (task_output / "score.json").is_file():
            return json.loads((task_output / "score.json").read_text())
        async with admission():
            metadata_path = problem / "problem.json"
            metadata = (
                json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
            )
            hidden = problem / "tests"
            if not hidden.is_dir():
                hidden = problem / "hidden"
            try:
                return await run_task(
                    "olympiad",
                    problem.name,
                    problem / "statement.txt",
                    problem,
                    args.checkpoint,
                    task_output,
                    args.wall_seconds,
                    args.time_limit
                    if args.time_limit is not None
                    else metadata.get("time_limit", 2.0),
                    args.memory_mb,
                    hidden,
                )
            except Exception as error:
                task_output.mkdir(parents=True, exist_ok=True)
                score = {
                    "score": 0.0,
                    "task_id": problem.name,
                    "infrastructure_error": True,
                    "returncode": 1,
                    "error": f"{type(error).__name__}: {error}",
                }
                dump(task_output / "score.json", score)
                print(json.dumps(score), flush=True)
                return score

    scores = await asyncio.gather(
        *(
            one(i, p.resolve())
            for i, p in enumerate(
                p for p in problems for _ in range(args.samples_per_problem)
            )
        )
    )
    dump(
        args.output / "summary.json",
        {"tasks": scores, "mean_score": sum(s["score"] for s in scores) / len(scores)},
    )


if __name__ == "__main__":
    asyncio.run(main())
