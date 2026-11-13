"""Run the upstream agent unchanged except for local logging/deadline configuration."""

import argparse
import asyncio
import importlib
import json
import os
import resource
from pathlib import Path

import httpx2 as openai_httpx
from model_library.agent import TimeLimit
from model_library.atif import ATIFTrajectory

from finalize_run import readable
from process_slots import async_process_slot


def install_http_observer():
    send = openai_httpx.AsyncClient.send

    async def observed_send(client, *args, **kwargs):
        response = await send(client, *args, **kwargs)
        if response.status_code == 429:
            print(
                "HTTP status 429 received from Tinker (including SDK-retried requests)",
                flush=True,
            )
        return response

    openai_httpx.AsyncClient.send = observed_send


def install_subprocess_limits():
    spawn = asyncio.create_subprocess_exec

    async def bounded_spawn(*command, **kwargs):
        previous = kwargs.get("preexec_fn")

        def limits():
            if previous:
                previous()
            hard = resource.getrlimit(resource.RLIMIT_AS)[1]
            maximum = (
                min(2 * 1024**3, hard)
                if hard != resource.RLIM_INFINITY
                else 2 * 1024**3
            )
            resource.setrlimit(resource.RLIMIT_AS, (maximum, maximum))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        kwargs["preexec_fn"] = limits
        return await spawn(*command, **kwargs)

    asyncio.create_subprocess_exec = bounded_spawn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["ioi", "olympiad"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--problem-dir", required=True)
    parser.add_argument("--problem-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    os.environ["TINKER_API_KEY"] = (
        os.environ.get("TINKER_API_KEY") or os.environ["TINKER_DEVIN_KEY"]
    )
    install_http_observer()
    install_subprocess_limits()
    module = importlib.import_module(f"{args.track}_agent.run")
    if args.track == "olympiad":
        tool_module = importlib.import_module("olympiad_agent.tools")
        communicate = tool_module._communicate

        async def bounded_communicate(*args, **kwargs):
            async with async_process_slot():
                return await communicate(*args, **kwargs)

        tool_module._communicate = bounded_communicate
    else:
        tool_module = importlib.import_module("ioi_agent.tools")
        for cls, name in [
            (tool_module.CppExecutor, "_compile_and_run_cpp"),
            (tool_module.Submission, "_run_tests_with_solution"),
        ]:
            original_method = getattr(cls, name)

            async def bounded_method(self, *args, _method=original_method, **kwargs):
                async with async_process_slot():
                    return await _method(self, *args, **kwargs)

            setattr(cls, name, bounded_method)
    original = module.Agent

    class LocalAgent(original):
        def __init__(self, **kwargs):
            kwargs["log_dir"] = Path(args.output_dir) / "logs"
            kwargs["config"].time_limit = TimeLimit(
                max_seconds=600, include_retries=True
            )
            super().__init__(**kwargs)
            self.saved_turns = {}

        def _write_turn_dir(
            self, output_dir, turn_number, turn, state, history, logger
        ):
            super()._write_turn_dir(
                output_dir, turn_number, turn, state, history, logger
            )
            (Path(args.output_dir) / "raw_messages.json").write_text(
                json.dumps(readable(history), indent=2)
            )
            captured = turn.model_copy(
                update={
                    "query_result": turn.query_result.model_copy(
                        update={"history": list(history)}
                    )
                }
            )
            self.saved_turns[turn_number] = captured
            trajectory = ATIFTrajectory.from_agent_result(
                turns=[self.saved_turns[key] for key in sorted(self.saved_turns)],
                agent_name=args.track,
                model_name=args.checkpoint,
                tool_definitions=[tool.model_dump() for tool in self._tool_defs],
                trajectory_extra={"partial": True},
            )
            (Path(args.output_dir) / "trajectory_atif.json").write_text(
                json.dumps(trajectory.to_json_dict(), indent=2, default=str)
            )

    module.Agent = LocalAgent
    upstream_args = argparse.Namespace(
        model=None,
        checkpoint=args.checkpoint,
        reasoning_effort="high",
        temperature=None,
        top_p=None,
        max_output_tokens=16384,
        problem_file=args.problem_file,
        problem_dir=args.problem_dir,
        task_id=args.task_id,
        output=str(Path(args.output_dir) / "result.json"),
    )
    asyncio.run(module.run(upstream_args))


if __name__ == "__main__":
    main()
