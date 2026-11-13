import asyncio
import json
import os
import random
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import progress
import run_olympiad
from harness import grade
from process_slots import try_acquire


class ProgressTests(unittest.TestCase):
    def test_sdk_retried_429_is_observed(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.server.calls += 1
                status = 429 if self.server.calls == 1 else 200
                body = (
                    {
                        "error": {
                            "message": "local rate limit",
                            "type": "rate_limit_error",
                        }
                    }
                    if status == 429
                    else {
                        "id": "test",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "test",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": "ok"},
                            }
                        ],
                    }
                )
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.calls = 0
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            program = f"""import asyncio
from agent_worker import install_http_observer
from openai import AsyncOpenAI
install_http_observer()
async def check():
    async with AsyncOpenAI(api_key='local-test', base_url='http://127.0.0.1:{server.server_port}/v1') as client:
        response = await client.chat.completions.create(model='test', messages=[{{'role':'user','content':'test'}}])
        print(response.choices[0].message.content)
asyncio.run(check())"""
            result = subprocess.run(
                [
                    str(
                        Path(
                            os.environ.get(
                                "IOI2_REPOS", str(Path(__file__).resolve().parents[2])
                            )
                        )
                        / "hackathon-olympiad-agent/.venv/bin/python"
                    ),
                    "-c",
                    program,
                ],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(server.calls, 2)
            self.assertEqual(len(progress.RATE.findall(result.stdout)), 1)
            self.assertIn("ok", result.stdout)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_incremental_partial_and_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "run_config.json").write_text(
                json.dumps({"total": 4, "started_at": 1})
            )
            for name, value in [
                ("one", '{"score":1,"returncode":0}'),
                ("two", '{"score":0,"returncode":1}'),
                ("partial", "{"),
                ("null", "null"),
            ]:
                (root / name).mkdir()
                (root / name / "score.json").write_text(value)
            (root / "two/stderr.log").write_text("HTTP status 429 received from Tinker")
            row = progress.snapshot(root)
            self.assertEqual(
                (
                    row["done"],
                    row["total"],
                    row["mean_hidden_score"],
                    row["full_solves"],
                    row["errors"],
                ),
                (2, 4, 0.5, 1, 1),
            )
            self.assertEqual(row["local_429_log_matches"], 1)
            self.assertGreater(row["tasks_per_hour"], 0)
            empty = progress.snapshot(root / "empty")
            self.assertEqual(empty["done"], 0)
            self.assertIsNone(empty["mean_hidden_score"])
            self.assertIsNone(empty["total"])

    def test_legacy_total_and_live_slots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "summary.json").write_text('{"tasks":[{},{}]}')
            self.assertEqual(progress.snapshot(root)["total"], 2)
            config = root / "slots.json"
            config.write_text('{"process_slots":1}')
            with patch.dict(
                os.environ,
                IOI2_PROCESS_SLOTS_DIR=str(root),
                IOI2_PROCESS_SLOTS_CONFIG=str(config),
            ):
                first = try_acquire()
                self.assertIsNotNone(first)
                self.assertIsNone(try_acquire())
                config.write_text('{"process_slots":2}')
                second = try_acquire()
                self.assertIsNotNone(second)
                first.close()
                second.close()


class CollectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_task_exception_does_not_abort_collection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.json").write_text(
                json.dumps(
                    [{"id": "bad", "split": "train"}, {"id": "good", "split": "train"}]
                )
            )

            async def fake(*args):
                if args[1] == "bad":
                    raise RuntimeError("export failed")
                return {"score": 1}

            with (
                patch(
                    "sys.argv",
                    ["run", "--pool-dir", str(root), "--output", str(root / "out")],
                ),
                patch.object(run_olympiad, "run_task", fake),
            ):
                await run_olympiad.main()
            result = json.loads((root / "out/summary.json").read_text())
            self.assertEqual(result["mean_score"], 0.5)
            self.assertTrue(result["tasks"][0]["infrastructure_error"])

    async def test_fixed_shuffle_heldout_opt_in_and_live_backoff(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pool = root / "pool"
            pool.mkdir()
            names = ["a", "b", "c", "d", "e"]
            (pool / "manifest.json").write_text(
                json.dumps([{"id": name, "split": "heldout"} for name in names])
            )
            control = root / "control.json"
            control.write_text('{"concurrency":2}')
            active, started, peaks = 0, [], []

            async def fake(*args):
                nonlocal active
                active += 1
                started.append(args[1])
                peaks.append(active)
                if len(started) == 2:
                    control.write_text('{"concurrency":1}')
                await asyncio.sleep(0.02)
                active -= 1
                args[5].mkdir(exist_ok=True)
                (args[5] / "score.json").write_text('{"score":1}')
                return {"score": 1}

            argv = [
                "run",
                "--pool-dir",
                str(pool),
                "--eval-heldout",
                "--concurrency",
                "2",
                "--control-file",
                str(control),
                "--shuffle-seed",
                "1729",
                "--output",
                str(root / "out"),
            ]
            with patch("sys.argv", argv), patch.object(run_olympiad, "run_task", fake):
                await run_olympiad.main()
            self.assertEqual(peaks[:2], [1, 2])
            self.assertEqual(peaks[2:], [1, 1, 1])
            config = json.loads((root / "out/run_config.json").read_text())
            expected = list(names)
            random.Random(1729).shuffle(expected)
            self.assertEqual(config["problem_order"], expected)
            self.assertEqual(started[:2], expected[:2])
            self.assertEqual(sorted(started), sorted(names))
            self.assertEqual(config["total"], 5)
            self.assertNotEqual(started, names)
            with (
                patch("sys.argv", argv + ["--resume"]),
                patch.object(run_olympiad, "run_task", fake),
            ):
                await run_olympiad.main()
            self.assertEqual(len(started), 5)


class CompileTimeoutTests(unittest.TestCase):
    def test_compile_timeout_is_a_score_and_kills_process_group(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.in").write_text("")
            (root / "a.out").write_text("")
            (root / "solution.cpp").write_text("int main(){}")
            with (
                patch("harness.subprocess.Popen") as popen,
                patch("harness.os.killpg") as killpg,
            ):
                process = popen.return_value
                process.pid = 999999
                process.communicate.side_effect = [
                    subprocess.TimeoutExpired("g++", 60),
                    (b"", b""),
                ]
                score = grade(root / "solution.cpp", root)
                self.assertEqual(score["error"], "Compilation timed out")
                self.assertEqual(score["score"], 0)
                killpg.assert_called_once()


if __name__ == "__main__":
    unittest.main()
