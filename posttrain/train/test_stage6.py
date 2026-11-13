import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import run_ioi_dev
from admission import AdmissionGate
from build_sft import candidates, retain_candidate
from stage6_mix import allocate
from trajectory_metrics import distribution, summarize, trajectory_metrics


class Stage6Tests(unittest.TestCase):
    def test_target_fraction_keeps_all_and_invalid_sources(self):
        groups = [[{"target_tokens": 10}], [{"target_tokens": 20}]]
        selected = allocate(groups, 90)
        self.assertTrue({0, 1}.issubset(selected))
        total = sum(groups[index][0]["target_tokens"] for index in selected)
        self.assertEqual(total / (total + 90), 0.4)
        with self.assertRaises(ValueError):
            allocate([], 90)
        with self.assertRaises(ValueError):
            allocate(groups, 1)

    def test_rank_distinct_and_empty(self):
        group = []

        def item(score, tokens, identity):
            return ((-score, tokens), None, None, [], identity)

        retain_candidate(group, item(50, 10, "a"), 2)
        retain_candidate(group, item(100, 20, "b"), 2)
        retain_candidate(group, item(100, 15, "c"), 2)
        self.assertEqual([row[4] for row in group], ["c", "b"])
        retain_candidate(group, item(100, 12, "b"), 2)
        self.assertEqual([row[4] for row in group], ["b", "c"])
        self.assertEqual(
            retain_candidate(group, item(100, 25, "b"), 2), "duplicate_trajectory"
        )
        self.assertEqual(group[0][0], (-100, 12))
        with self.assertRaises(ValueError):
            retain_candidate([], item(1, 1, "a"), 0)

    def test_positive_ioi_and_dev_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, (problem, score) in enumerate(
                [("2021/a", 0), ("2021/a", 0.1), ("2024/a", 100)]
            ):
                directory = root / str(index)
                directory.mkdir()
                (directory / "score.json").write_text(
                    json.dumps({"task_id": problem, "score": score})
                )
            rows, reasons = candidates(
                [root], 1, 4, set(), minimum_ioi=0, maximum_ioi=4, positive_ioi=True
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(reasons["score"], 1)
            self.assertEqual(reasons["heldout_ioi"], 1)

    def test_usage_first_submission_and_empty_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            empty = trajectory_metrics(root)
            self.assertEqual(empty["turns"], 0)
            self.assertIsNone(empty["first_submission_tokens"])
            for index, (tokens, duration, completed, calls) in enumerate(
                [
                    (100, 2, "2026-01-01T00:00:02+00:00", []),
                    (300, 1, "2026-01-01T00:00:05+00:00", [{"name": "submission"}]),
                ]
            ):
                directory = root / "logs/a/turns" / f"turn_{index:03}"
                directory.mkdir(parents=True)
                (directory / "result.json").write_text(
                    json.dumps(
                        {
                            "query_result": {
                                "metadata": {
                                    "out_tokens": tokens,
                                    "duration_seconds": duration,
                                    "completed_at": completed,
                                },
                                "tool_calls": calls,
                            }
                        }
                    )
                )
            row = trajectory_metrics(root)
            self.assertEqual(row["first_submission_tokens"], 400)
            self.assertEqual(row["first_submission_seconds"], 5)
            self.assertEqual(row["max_turn_output_tokens"], 300)
            self.assertAlmostEqual(row["output_tokens_per_query_second"], 400 / 3)
            self.assertEqual(summarize([row, empty])["zero_submission_runs"], 1)
            self.assertIsNone(summarize([])["mean_first_submission_tokens"])
            self.assertEqual(distribution([]), {"count": 0})
            (directory / "result.json").write_text("{")
            with self.assertRaises(ValueError):
                trajectory_metrics(root)


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_pause_resume_and_release_after_error(self):
        with tempfile.TemporaryDirectory() as temp:
            control = Path(temp) / "control.json"
            control.write_text("{")
            gate = AdmissionGate(2, control)
            started = asyncio.Event()

            async def worker():
                async with gate.slot():
                    started.set()
                    raise ValueError("body")

            task = asyncio.create_task(worker())
            await asyncio.sleep(0.01)
            self.assertFalse(started.is_set())
            control.write_text("null")
            self.assertEqual(gate.limit(), 0)
            control.write_text('{"concurrency":1}')
            with self.assertRaisesRegex(ValueError, "body"):
                await asyncio.wait_for(task, 1)
            self.assertEqual(gate.active, 0)
            with self.assertRaises(ValueError):
                AdmissionGate(0)

    async def test_dev_cli_optional_control_preserves_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exam = root / "ioi-agent/exams/2024/a"
            exam.mkdir(parents=True)
            (exam / "problem.pdf").write_text("dummy")
            assets = root / "ioi-agent/submission_scripts/2024/a"
            assets.mkdir(parents=True)
            control = root / "control.json"
            control.write_text('{"concurrency":1}')
            output = root / "output"
            task = AsyncMock(return_value={"score": 10})
            with (
                patch.object(run_ioi_dev, "REPOS", root),
                patch.object(run_ioi_dev, "run_task", task),
                patch.object(run_ioi_dev.subprocess, "run"),
                patch(
                    "sys.argv",
                    [
                        "run_ioi_dev",
                        "--tasks",
                        "2024/a",
                        "--output",
                        str(output),
                        "--concurrency",
                        "6",
                        "--control-file",
                        str(control),
                    ],
                ),
            ):
                await run_ioi_dev.main()
            self.assertEqual(task.call_args.args[-1], 600)
            config = json.loads((output / "run_config.json").read_text())
            self.assertEqual(config["concurrency"], 6)
            self.assertEqual(config["control_file"], str(control))


if __name__ == "__main__":
    unittest.main()
