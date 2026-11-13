import argparse
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

from admission import AdmissionGate
from harness import REPOS, dump, run_task


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--tasks", nargs="+", help="Default: all twelve 2024/2025 tasks"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--wall-seconds", type=float, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--control-file", type=Path)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    repo = REPOS / "ioi-agent"
    tasks = args.tasks or [
        str(p.relative_to(repo / "exams"))
        for y in ["2024", "2025"]
        for p in sorted((repo / "exams" / y).iterdir())
        if list(p.glob("*.pdf"))
    ]
    previous = (
        json.loads((args.output / "run_config.json").read_text()) if args.resume else {}
    )
    if previous and previous["checkpoint"] != args.checkpoint:
        parser.error("Resume checkpoint differs from existing run")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    dump(
        args.output / "run_config.json",
        {
            "checkpoint": args.checkpoint,
            "total": len(tasks),
            "tasks": tasks,
            "concurrency": args.concurrency,
            "started_at": previous.get("started_at", time.time()),
            "pid": os.getpid(),
            "track": "ioi",
            "full_score": 100,
            "control_file": str(args.control_file) if args.control_file else None,
        },
    )
    admission = AdmissionGate(args.concurrency, args.control_file)

    async def one(task):
        scored = args.output / task.replace("/", "_") / "score.json"
        if args.resume and scored.is_file():
            return json.loads(scored.read_text())
        async with admission.slot():
            assets = repo / "submission_scripts" / task
            for source in assets.rglob("*"):
                if source.is_file():
                    with source.open("rb") as f:
                        if f.read(42).startswith(
                            b"version https://git-lfs.github.com/spec/v1"
                        ):
                            raise RuntimeError(
                                f"LFS pointer: {source}. Run git lfs pull first."
                            )
            statement = args.output / (task.replace("/", "_") + ".txt")
            subprocess.run(
                [
                    "pdftotext",
                    "-layout",
                    str(next((repo / "exams" / task).glob("*.pdf"))),
                    str(statement),
                ],
                check=True,
            )
            return await run_task(
                "ioi",
                task,
                statement,
                assets,
                args.checkpoint,
                args.output / task.replace("/", "_"),
                args.wall_seconds,
            )

    scores = await asyncio.gather(*(one(t) for t in tasks))
    summary = {
        "total_score": sum(s["score"] for s in scores),
        "maximum_score": 100 * len(tasks),
        "tasks": scores,
    }
    dump(args.output / "summary.json", summary)
    print(f"IOI total: {summary['total_score']}/{summary['maximum_score']}")


if __name__ == "__main__":
    asyncio.run(main())
