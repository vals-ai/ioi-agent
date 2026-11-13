"""Validate a local reference through the upstream IOI Submission scorer."""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from ioi_agent.tools import Submission

from process_slots import async_process_slot


async def validate(assets, reference):
    tool = Submission(str(assets))
    logger = logging.getLogger("reference")
    async with async_process_slot():
        results = await tool._run_tests_with_solution(reference.read_text(), logger)
    scored = tool._calculate_subtask_scores(results, logger)
    return {
        **scored,
        "all_tests_passed": results["all_tests_passed"],
        "passed_tests": len(results["passed_tests"]),
        "stdout": results["stdout"][-16000:],
        "stderr": results["stderr"][-16000:],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("assets", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(validate(args.assets, args.reference))))
