import json
import os
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import build_ioi_train
import progress
import run_ioi_dev
import run_ioi_train


class IOIRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_dev_discovers_both_pdf_names_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for year, name in [(2024, "old"), (2025, "new")]:
                directory = root / f"ioi-agent/exams/{year}/{name}"
                directory.mkdir(parents=True)
                (
                    directory / ("problem.pdf" if year == 2024 else "new-ISC.pdf")
                ).write_text("mock")
                (root / f"ioi-agent/submission_scripts/{year}/{name}").mkdir(
                    parents=True
                )
            calls = []

            async def fake(*args):
                calls.append(args[1])
                args[5].mkdir()
                (args[5] / "score.json").write_text(
                    json.dumps({"task_id": args[1], "score": 100})
                )
                return {"task_id": args[1], "score": 100}

            with (
                patch.object(run_ioi_dev, "REPOS", root),
                patch.object(run_ioi_dev.subprocess, "run"),
                patch.object(run_ioi_dev, "run_task", fake),
                patch("sys.argv", ["run", "--output", str(root / "out")]),
            ):
                await run_ioi_dev.main()
            self.assertEqual(calls, ["2024/old", "2025/new"])
            with (
                patch.object(run_ioi_dev, "REPOS", root),
                patch.object(run_ioi_dev, "run_task", fake),
                patch("sys.argv", ["run", "--output", str(root / "out"), "--resume"]),
            ):
                await run_ioi_dev.main()
            self.assertEqual(len(calls), 2)
            self.assertEqual(
                json.loads((root / "out/summary.json").read_text())["maximum_score"],
                200,
            )

    async def test_training_samples_isolation_and_incremental_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = root / "2020/trees"
            task.mkdir(parents=True)
            for name in ["graders", "tests", "subtasks"]:
                (task / name).mkdir()
            (task / "statement.txt").write_text("statement")
            (task / "run_tests.sh").write_text("")
            (task / "problem.json").write_text('{"task_type":"Batch"}')
            count = 0

            async def fake(*args):
                nonlocal count
                count += 1
                if count == 1:
                    raise RuntimeError("export failure")
                args[5].mkdir()
                (args[5] / "score.json").write_text('{"score":100}')
                return {"score": 100}

            with (
                patch.object(run_ioi_train, "run_task", fake),
                patch(
                    "sys.argv",
                    [
                        "run",
                        "--task-dir",
                        str(task),
                        "--samples-per-problem",
                        "2",
                        "--output",
                        str(root / "out"),
                    ],
                ),
            ):
                await run_ioi_train.main()
            config = json.loads((root / "out/run_config.json").read_text())
            self.assertEqual(config["total"], 2)
            self.assertEqual(len(list((root / "out").glob("*/score.json"))), 2)
            self.assertEqual(
                json.loads((root / "out/summary.json").read_text())["total_score"], 100
            )
            row = progress.snapshot(root / "out")
            self.assertEqual(row["full_solves"], 1)
            self.assertEqual(row["errors"], 1)
            with (
                patch("sys.argv", ["run", "--output", str(root / "empty")]),
                self.assertRaises(SystemExit),
            ):
                await run_ioi_train.main()
            metadata = json.loads((task / "problem.json").read_text())
            metadata["year"] = 2024
            (task / "problem.json").write_text(json.dumps(metadata))
            with (
                patch(
                    "sys.argv",
                    ["run", "--task-dir", str(task), "--output", str(root / "dev")],
                ),
                self.assertRaises(SystemExit),
            ):
                await run_ioi_train.main()


class IOIBuildTests(unittest.TestCase):
    def test_reference_choices_require_complete_cpp(self):
        rows = [
            dict(
                label="correct.cpp",
                code="ok",
                subtask="a",
                subtask_points=100,
                subtask_max_points=100,
                subtask_score=1,
            ),
            dict(
                label="partial.cpp",
                code="bad",
                subtask="a",
                subtask_points=50,
                subtask_max_points=100,
                subtask_score=0.5,
            ),
        ]
        self.assertEqual(
            build_ioi_train.reference_choices(rows), [("correct.cpp", "ok")]
        )
        self.assertEqual(build_ioi_train.reference_choices([]), [])


class IOILocalLoopTests(unittest.TestCase):
    def test_function_grader_loop_and_saved_native_history(self):
        implementation = '#include "tiny.h"\nint solve(int x) { return x+1; }'

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                submitted = any(
                    message["role"] == "tool" for message in request["messages"]
                )
                message = {
                    "role": "assistant",
                    "content": "EXIT" if submitted else None,
                    "reasoning_content": "mock reasoning",
                }
                if not submitted:
                    message["tool_calls"] = [
                        {
                            "id": "call",
                            "type": "function",
                            "function": {
                                "name": "submission",
                                "arguments": json.dumps({"cpp_code": implementation}),
                            },
                        }
                    ]
                for call in message.get("tool_calls", []):
                    call["index"] = 0
                events = [
                    {
                        "choices": [
                            {"index": 0, "delta": message, "finish_reason": None}
                        ]
                    },
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop" if submitted else "tool_calls",
                            }
                        ]
                    },
                ]
                body = (
                    "".join(
                        "data: "
                        + json.dumps(
                            {
                                "id": "mock",
                                "object": "chat.completion.chunk",
                                "created": 0,
                                "model": "mock",
                                **event,
                            }
                        )
                        + "\n\n"
                        for event in events
                    )
                    + "data: [DONE]\n\n"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                task = root / "2020/tiny"
                for name in ["graders", "checker", "tests", "subtasks"]:
                    (task / name).mkdir(parents=True)
                (task / "statement.txt").write_text(
                    "Implement int solve(int x), returning x+1."
                )
                (task / "graders/tiny.h").write_text("int solve(int);")
                (task / "graders/grader.cpp").write_text(
                    '#include <iostream>\n#include "tiny.h"\nint main(){int x;std::cin>>x;std::cout<<solve(x);}'
                )
                (task / "checker/checker.cpp").write_text(
                    "#include <fstream>\n#include <iostream>\nint main(int argc,char**argv){std::ifstream a(argv[2]),b(argv[3]);int x=0,y=0;a>>x;b>>y;std::cout<<(x==y)<<std::endl;}"
                )
                (task / "tests/a.in").write_text("1\n")
                (task / "tests/a.out").write_text("2\n")
                (task / "subtasks/all.json").write_text(
                    '{"score":100,"testcases":["a"]}'
                )
                (task / "problem.json").write_text(
                    '{"task_type":"Batch","time_limit":2,"memory_limit":1073741824}'
                )
                source = Path(Path(__file__).resolve().parent / "ioi_run_tests.sh")
                (task / "run_tests.sh").write_text(source.read_text())
                result = subprocess.run(
                    [
                        "python",
                        str(Path(__file__).parent / "run_ioi_train.py"),
                        "--task-dir",
                        str(task),
                        "--output",
                        str(root / "out"),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=90,
                    env=dict(
                        os.environ,
                        TINKER_API_KEY="local-test",
                        TINKER_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
                    ),
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                output = next((root / "out").glob("000-*"))
                self.assertEqual(
                    json.loads((output / "score.json").read_text())["score"],
                    100,
                    {
                        "stdout": result.stdout,
                        "stderr": (output / "stderr.log").read_text(),
                        "turns": [
                            (
                                str(path),
                                json.loads(path.read_text()).get("tool_call_records"),
                            )
                            for path in output.glob("logs/**/turns/turn_*/result.json")
                        ],
                    },
                )
                self.assertTrue((output / "workspace/tests").is_symlink())
                self.assertTrue((output / "trajectory_atif.json").is_file())
                self.assertTrue(list(output.glob("logs/**/turns/turn_*/history.json")))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
