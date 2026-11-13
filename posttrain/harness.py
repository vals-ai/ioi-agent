import asyncio
import json
import os
import resource
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from process_slots import process_slot

ROOT = Path(os.environ.get("IOI2_ROOT", str(Path(__file__).resolve().parent)))
REPOS = Path(os.environ.get("IOI2_REPOS", str(Path(__file__).resolve().parents[2])))


def compiler_limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3,) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def grade(solution, tests, time_limit=2.0, memory_mb=1024):
    with process_slot():
        return _grade(solution, tests, time_limit, memory_mb)


def _grade(solution, tests, time_limit=2.0, memory_mb=1024):
    tests = Path(tests)
    inputs = sorted(tests.glob("*.in"))
    if not inputs:
        raise ValueError(f"No hidden tests in {tests}")
    for given in inputs:
        if not given.with_suffix(".out").is_file():
            raise ValueError(f"Missing expected output for {given}")
    if not Path(solution).is_file():
        return {
            "score": 0.0,
            "passed": 0,
            "total": len(inputs),
            "error": "No compiling submission",
        }
    with tempfile.TemporaryDirectory(dir=ROOT, prefix="grade-") as temp:
        binary = Path(temp) / "solution"
        compiled = subprocess.Popen(
            ["g++", "-std=c++20", "-O2", "-pipe", str(solution), "-o", str(binary)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=compiler_limits,
        )
        try:
            _, compile_error = compiled.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(compiled.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            compiled.communicate()
            return {
                "score": 0.0,
                "passed": 0,
                "total": len(inputs),
                "error": "Compilation timed out",
            }
        if compiled.returncode:
            return {
                "score": 0.0,
                "passed": 0,
                "total": len(inputs),
                "error": compile_error.decode(errors="replace"),
            }
        verdicts = []

        def limits():
            resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024**2,) * 2)
            resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2,) * 2)
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        for given in inputs:
            start = time.monotonic()
            with (
                given.open("rb") as stdin,
                tempfile.TemporaryFile() as stdout,
                tempfile.TemporaryFile() as stderr,
            ):
                process = subprocess.Popen(
                    [str(binary)],
                    stdin=stdin,
                    stdout=stdout,
                    stderr=stderr,
                    preexec_fn=limits,
                    start_new_session=True,
                )
                try:
                    process.wait(timeout=time_limit)
                    stdout.seek(0)
                    passed = (
                        process.returncode == 0
                        and stdout.read().split()
                        == given.with_suffix(".out").read_bytes().split()
                    )
                    verdict = (
                        "passed"
                        if passed
                        else ("runtime_error" if process.returncode else "wrong_answer")
                    )
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    verdict = "time_limit"
            verdicts.append(
                {
                    "test": given.name,
                    "verdict": verdict,
                    "seconds": time.monotonic() - start,
                }
            )
        passed = sum(v["verdict"] == "passed" for v in verdicts)
        return {
            "score": passed / len(inputs),
            "passed": passed,
            "total": len(inputs),
            "tests": verdicts,
        }


async def run_task(
    track,
    task_id,
    statement,
    assets,
    checkpoint,
    output,
    wall_seconds=600,
    time_limit=2.0,
    memory_mb=1024,
    hidden=None,
):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    workspace.mkdir()
    assets = Path(assets)
    if track == "olympiad":
        if (assets / "samples").is_dir():
            shutil.copytree(assets / "samples", workspace / "samples")
    else:
        for name in [
            "graders",
            "checker",
            "tests",
            "subtasks",
            "run_tests.sh",
            "problem.json",
        ]:
            source = assets / name
            if name == "tests" and source.is_dir():
                (workspace / name).symlink_to(
                    source.resolve(), target_is_directory=True
                )
            elif source.is_dir():
                shutil.copytree(source, workspace / name)
            elif source.exists():
                shutil.copy2(source, workspace / name)
    problem_file = workspace / "statement.txt"
    problem_file.write_text(Path(statement).read_text())
    repo = REPOS / (
        "ioi-hackathon-agent" if track == "ioi" else "hackathon-olympiad-agent"
    )
    command = [
        str(repo / ".venv/bin/python"),
        str(Path(__file__).resolve().parent / "agent_worker.py"),
        "--track",
        track,
        "--checkpoint",
        checkpoint,
        "--problem-dir",
        str(workspace),
        "--problem-file",
        str(problem_file),
        "--task-id",
        task_id,
        "--output-dir",
        str(output),
    ]
    dump(
        output / "config.json",
        {
            "command": command,
            "checkpoint": checkpoint,
            "track": track,
            "wall_seconds": wall_seconds,
            "time_limit": time_limit,
            "memory_mb": memory_mb,
        },
    )
    start = time.monotonic()
    with (
        (output / "stdout.log").open("wb") as stdout,
        (output / "stderr.log").open("wb") as stderr,
    ):
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=stdout,
            stderr=stderr,
            cwd=workspace,
            start_new_session=True,
        )
        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), wall_seconds)
        except asyncio.TimeoutError:
            timed_out = True
        finally:
            # Also reap tool subprocesses left by cancellation or a normal agent exit.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
    duration = time.monotonic() - start
    finalized = await asyncio.create_subprocess_exec(
        str(repo / ".venv/bin/python"),
        str(Path(__file__).resolve().parent / "finalize_run.py"),
        str(output),
    )
    if await finalized.wait():
        raise RuntimeError(f"Trajectory export failed in {output}")
    partial = output / "trajectory_atif.json"
    if partial.is_file():
        trajectory = json.loads(partial.read_text())
        trajectory.setdefault("extra", {})["harness"] = {
            "timed_out": timed_out,
            "returncode": process.returncode,
        }
        dump(partial, trajectory)
    if track == "olympiad":
        score = await asyncio.to_thread(
            grade, workspace / "solution.cpp", hidden, time_limit, memory_mb
        )
    else:
        states = sorted(output.glob("logs/**/turns/*/state.json"))
        state = json.loads(states[-1].read_text()) if states else {}
        scores = state.get("best_subtask_scores", {})
        score = {"score": sum(scores.values()), "best_subtask_scores": scores}
    score.update(
        task_id=task_id,
        track=track,
        maximum_score=100 if track == "ioi" else 1,
        duration_seconds=duration,
        timed_out=timed_out,
        returncode=process.returncode,
    )
    dump(output / "score.json", score)
    print(json.dumps(score), flush=True)
    return score
