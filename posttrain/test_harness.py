import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from harness import grade, run_task
from process_slots import try_acquire


class SlotTests(unittest.TestCase):
    def test_slot_limit_and_release(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(
                os.environ, IOI2_PROCESS_SLOTS_DIR=temp, IOI2_PROCESS_SLOTS="2"
            ):
                first, second = try_acquire(), try_acquire()
                try:
                    self.assertIsNotNone(first)
                    self.assertIsNotNone(second)
                    self.assertIsNone(try_acquire())
                    first.close()
                    replacement = try_acquire()
                    self.assertIsNotNone(replacement)
                    replacement.close()
                finally:
                    first.close()
                    second.close()

    def test_collection_refuses_heldout_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.json").write_text(
                json.dumps([{"id": "test", "split": "heldout"}])
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parent / "run_olympiad.py"),
                    "--pool-dir",
                    str(root),
                    "--output",
                    str(root / "output"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("not held-out", result.stderr)
            self.assertFalse((root / "output").exists())


class DeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def test_wall_deadline_still_writes_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "statement.txt").write_text("Print 5.")
            (root / "tests").mkdir()
            (root / "tests/01.in").write_text("")
            (root / "tests/01.out").write_text("5")
            score = await run_task(
                "olympiad",
                "deadline",
                root / "statement.txt",
                root,
                "Qwen/Qwen3.5-9B",
                root / "output",
                wall_seconds=0.001,
                hidden=root / "tests",
            )
            self.assertTrue(score["timed_out"])
            self.assertEqual(score["returncode"], -9)
            self.assertEqual(score["score"], 0)
            self.assertTrue((root / "output/score.json").is_file())


class LocalAgentLoopTests(unittest.TestCase):
    def test_sixteen_concurrent_upstream_loops_and_artifacts(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                self.server.requests.append(request)
                submitted = any(m["role"] == "assistant" for m in request["messages"])
                if submitted:
                    delta = {"role": "assistant", "content": "EXIT"}
                    finish = "stop"
                else:
                    code = "#include <iostream>\nint main(){int a,b;std::cin>>a>>b;std::cout<<a+b;}"
                    delta = {
                        "role": "assistant",
                        "reasoning_content": "Add the inputs.",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "submission",
                                    "arguments": json.dumps({"cpp_code": code}),
                                },
                            }
                        ],
                    }
                    finish = "tool_calls"
                events = [
                    {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 10,
                            "total_tokens": 20,
                        },
                    },
                ]
                body = (
                    "".join(
                        "data: "
                        + json.dumps(
                            {
                                "id": "test",
                                "object": "chat.completion.chunk",
                                "created": 0,
                                "model": "test",
                                **event,
                            }
                        )
                        + "\n\n"
                        for event in events
                    )
                    + "data: [DONE]\n\n"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body.encode())))
                self.end_headers()
                self.wfile.write(body.encode())

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.requests = []
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                problem = root / "sum"
                problem.mkdir()
                (problem / "statement.txt").write_text(
                    "Read two integers and print their sum."
                )
                for name in ("samples", "tests"):
                    (problem / name).mkdir()
                    (problem / name / "01.in").write_text("2 3\n")
                    (problem / name / "01.out").write_text("5\n")
                (root / "manifest.json").write_text(
                    json.dumps([{"id": "sum", "split": "train"}])
                )
                command = [
                    sys.executable,
                    str(Path(__file__).parent / "run_olympiad.py"),
                    "--pool-dir",
                    str(root),
                    "--samples-per-problem",
                    "16",
                    "--concurrency",
                    "16",
                    "--output",
                    str(root / "output"),
                ]
                env = dict(
                    os.environ,
                    TINKER_API_KEY="local-test",
                    TINKER_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
                )
                started = time.monotonic()
                result = subprocess.run(
                    command, env=env, capture_output=True, text=True, timeout=180
                )
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                summary = json.loads((root / "output/summary.json").read_text())
                self.assertEqual(summary["mean_score"], 1)
                self.assertEqual(len(summary["tasks"]), 16)
                for output in (root / "output").glob("*-sum"):
                    trajectory = json.loads(
                        (output / "trajectory_atif.json").read_text()
                    )
                    self.assertEqual(trajectory["steps"][0]["source"], "user")
                    raw = json.loads((output / "raw_messages.json").read_text())
                    self.assertTrue(
                        any(
                            item.get("response", {}).get("reasoning_content")
                            == "Add the inputs."
                            for item in raw
                        )
                    )
                self.assertEqual(len(server.requests), 32)
                self.assertTrue(
                    all(
                        request.get("max_completion_tokens", request.get("max_tokens"))
                        == 16384
                        and request["reasoning_effort"] is True
                        for request in server.requests
                    )
                )
                report = {
                    "concurrency": 16,
                    "samples_per_problem": 16,
                    "completed": len(summary["tasks"]),
                    "mean_score": summary["mean_score"],
                    "wall_seconds": time.monotonic() - started,
                    "process_slots": 6,
                    "endpoint": "local_mock",
                    "real_cpp_compile_and_tests": True,
                }
                (root / "concurrency16_check.json").write_text(
                    json.dumps(report, indent=2)
                )
                print(json.dumps(report), flush=True)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class GradingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tests = self.root / "tests"
        self.tests.mkdir()
        (self.tests / "01.in").write_text("2 3\n")
        (self.tests / "01.out").write_text("5\n")
        self.solution = self.root / "solution.cpp"

    def tearDown(self):
        self.temp.cleanup()

    def test_pass_and_wrong_answer(self):
        self.solution.write_text(
            "#include <iostream>\nint main(){int a,b;std::cin>>a>>b;std::cout<<a+b;}"
        )
        result = grade(self.solution, self.tests)
        self.assertEqual(result["score"], 1)
        (self.tests / "02.in").write_text("1 2\n")
        (self.tests / "02.out").write_text("7\n")
        self.assertEqual(grade(self.solution, self.tests)["score"], 0.5)

    def test_empty_and_missing_output_rejected(self):
        (self.tests / "01.in").unlink()
        with self.assertRaises(ValueError):
            grade(self.solution, self.tests)
        (self.tests / "01.in").write_text("")
        (self.tests / "01.out").unlink()
        with self.assertRaises(ValueError):
            grade(self.solution, self.tests)

    def test_no_solution_compile_error_and_timeout(self):
        self.assertEqual(grade(self.solution, self.tests)["score"], 0)
        self.solution.write_text("invalid c++")
        self.assertIn("error", grade(self.solution, self.tests))
        self.solution.write_text('int main(){for(;;){asm volatile("");}}')
        self.assertEqual(
            grade(self.solution, self.tests, 0.05)["tests"][0]["verdict"], "time_limit"
        )

    def test_memory_limit(self):
        self.solution.write_text(
            "#include <cstdlib>\nint main(){return malloc(1024ull*1024*1024)?0:1;}"
        )
        self.assertEqual(
            grade(self.solution, self.tests, memory_mb=32)["tests"][0]["verdict"],
            "runtime_error",
        )


if __name__ == "__main__":
    unittest.main()
