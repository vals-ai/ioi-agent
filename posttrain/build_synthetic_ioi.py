"""Stream teacher-drafted, reference-validated synthetic function-grader tasks."""

import argparse
import asyncio
import datetime
import json
import os
import signal
import time
from pathlib import Path

import httpx

from admission import AdmissionGate
from harness import REPOS, compiler_limits
from process_slots import async_process_slot
from synthetic_ioi import (
    ROOT,
    TEACHER,
    materialize,
    parse_draft,
    prompt_for,
    select_candidates,
    write_json,
)


async def validate(directory, reference, work):
    command = [
        str(
            Path(
                os.environ.get(
                    "IOI2_IOI_PYTHON",
                    str(REPOS / "ioi-hackathon-agent/.venv/bin/python"),
                )
            )
        ),
        str(Path(__file__).resolve().parent / "validate_ioi_reference.py"),
        str(directory),
        str(reference),
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
        preexec_fn=compiler_limits,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 600)
    except asyncio.TimeoutError:
        os.killpg(process.pid, signal.SIGKILL)
        await process.communicate()
        raise ValueError("Reference validation timed out")
    (work / "validation.stderr").write_bytes(stderr)
    if process.returncode:
        raise ValueError(f"Reference validator exit {process.returncode}")
    result = json.loads(stdout)
    write_json(work / "reference_validation.json", result)
    expected = len(list((directory / "tests").glob("*.in")))
    if (
        result.get("total_score") != 100
        or not result.get("all_tests_passed")
        or result.get("passed_tests") != expected
    ):
        raise ValueError(
            "Adapted reference did not pass every retained test with score 100"
        )
    async with async_process_slot():
        binary = work / "with-main"
        with_main = work / "with-main.cpp"
        with_main.write_text(reference.read_text() + "\nint main() { return 0; }\n")
        command = [
            "g++",
            "-std=c++20",
            "-O2",
            "-I",
            str(directory / "graders"),
            "-include",
            "bits/stdc++.h",
            str(directory / "graders/grader.cpp"),
            str(with_main),
            "-o",
            str(binary),
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            preexec_fn=compiler_limits,
        )
        try:
            _, error = await asyncio.wait_for(process.communicate(), 60)
        except asyncio.TimeoutError:
            os.killpg(process.pid, signal.SIGKILL)
            await process.communicate()
            raise ValueError("Duplicate-main negative validation timed out")
        (work / "duplicate_main.stderr").write_bytes(error)
        if (
            not process.returncode
            or "multiple definition" not in error.decode(errors="replace")
            or "main" not in error.decode(errors="replace")
        ):
            raise ValueError(
                "main negative control did not fail specifically at duplicate-main linking"
            )
        empty = work / "empty.cpp"
        empty.write_text('#include "api.h"\n')
        command[-3] = str(empty)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            preexec_fn=compiler_limits,
        )
        try:
            _, error = await asyncio.wait_for(process.communicate(), 60)
        except asyncio.TimeoutError:
            os.killpg(process.pid, signal.SIGKILL)
            await process.communicate()
            raise ValueError("Missing-function negative validation timed out")
        (work / "missing_functions.stderr").write_bytes(error)
        if not process.returncode or "undefined reference" not in error.decode(
            errors="replace"
        ):
            raise ValueError("Grader does not require contestant functions")
    return {
        "score": 100,
        "all_tests_passed": True,
        "n_tests": expected,
        "duplicate_main_rejected": True,
        "missing_function_rejected": True,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=150)
    parser.add_argument("--candidates", type=int, default=220)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--control-file", type=Path, required=True)
    args = parser.parse_args()
    from build_ioi_train import Exclusions

    exclusions = Exclusions()
    output = ROOT / "synthetic_ioi"
    work_root = ROOT / "synthetic_work"
    references = ROOT / "synthetic_references"
    for path in [output, work_root, references]:
        path.mkdir(exist_ok=False)
    candidates = select_candidates(args.candidates)
    write_json(work_root / "candidates.json", candidates)
    accepted = []
    attempted = []
    write_json(output / "manifest.json", accepted)
    gate = AdmissionGate(args.concurrency, args.control_file)
    cutoff = (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(hour=15, minute=45, second=0, microsecond=0)
        .timestamp()
    )

    async def one(row):
        try:
            remaining = cutoff - time.time()
            if remaining <= 0:
                return
            context = gate.slot()
            await asyncio.wait_for(context.__aenter__(), remaining)
        except asyncio.TimeoutError:
            return
        try:
            if len(accepted) >= args.target:
                return
            source = Path(row["directory"])
            work = work_root / row["id"]
            work.mkdir()
            start = time.time()
            record = {"source_pool_id": row["id"], "started_at": start}
            try:
                prompt = prompt_for(source, Path(row["reference"]))
                if len(prompt) > 180000:
                    raise ValueError("Source/reference too long for safe drafting")
                async with httpx.AsyncClient(timeout=600) as client:
                    response = await client.post(
                        os.environ.get(
                            "TINKER_BASE_URL",
                            "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
                        )
                        + "/chat/completions",
                        headers={
                            "Authorization": "Bearer "
                            + (
                                os.environ.get("TINKER_API_KEY")
                                or os.environ["TINKER_DEVIN_KEY"]
                            )
                        },
                        json={
                            "model": TEACHER,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 16384,
                            "temperature": 0.5,
                        },
                    )
                if response.status_code == 429:
                    print("HTTP status 429 from synthetic drafting", flush=True)
                    raise ValueError("HTTP status 429")
                response.raise_for_status()
                payload = response.json()
                write_json(work / "response.json", payload)
                choice = payload["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError("Draft ended before completing its JSON")
                draft = parse_draft(choice["message"]["content"])
                reason = exclusions.check(source.name, draft["statement"])
                if reason:
                    raise ValueError("Protected statement overlap: " + reason)
                directory = output / row["id"]
                reference = references / (row["id"] + ".cpp")
                metadata = materialize(draft, source, directory, reference)
                reason = exclusions.check(
                    source.name, (directory / "statement.txt").read_text()
                )
                if reason:
                    raise ValueError("Materialized protected overlap: " + reason)
                result = await validate(directory, reference, work)
                metadata["validation"] = result
                write_json(directory / "problem.json", metadata)
                write_json(directory / "validation.json", result)
                accepted.append(
                    {
                        "id": "synthetic/" + row["id"],
                        "directory": str(directory),
                        "reference": str(reference),
                        "source_pool_id": row["id"],
                        "score": 100,
                        "n_tests": result["n_tests"],
                        "duplicate_main_rejected": True,
                    }
                )
                write_json(output / "manifest.json", accepted)
                record.update(status="accepted", **result)
            except Exception as error:
                record.update(
                    status="rejected", error=f"{type(error).__name__}: {error}"
                )
            record["seconds"] = time.time() - start
            attempted.append(record)
            write_json(work / "conversion.json", record)
            write_json(
                work_root / "progress.json",
                {
                    "at": time.time(),
                    "target": args.target,
                    "candidates": len(candidates),
                    "completed": len(attempted),
                    "accepted": len(accepted),
                    "rejected": len(attempted) - len(accepted),
                    "records": attempted,
                },
            )
            print(json.dumps(record), flush=True)
        finally:
            await context.__aexit__(None, None, None)

    await asyncio.gather(*(one(row) for row in candidates))
    write_json(
        output / "conversion_complete.json",
        {"at": time.time(), "accepted": len(accepted), "completed": len(attempted)},
    )


if __name__ == "__main__":
    asyncio.run(main())
