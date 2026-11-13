import ast
import os
import types
import unittest
from pathlib import Path

from audit_dev_verdicts import blocks, classify
from watch_stage7 import next_limit


class VerdictTests(unittest.TestCase):
    def test_approved_backoff_does_not_raise_a_lower_limit(self):
        self.assertEqual(next_limit(6, 10), 6)
        self.assertEqual(next_limit(6, 15), 4)
        self.assertEqual(next_limit(6, 1, memory_error=True), 4)
        self.assertEqual(next_limit(3, 15), 3)
        self.assertEqual(next_limit(6, 1, rate_error=True), 3)
        self.assertEqual(next_limit(1, 1, rate_error=True), 1)

    def test_actual_upstream_scoring_retains_per_subtask_maxima(self):
        for source in [
            str(
                Path(
                    os.environ.get(
                        "IOI2_REPOS", str(Path(__file__).resolve().parents[2])
                    )
                )
                / "ioi-hackathon-agent/src/ioi_agent/tools.py"
            ),
            str(
                Path(
                    os.environ.get(
                        "IOI2_REPOS", str(Path(__file__).resolve().parents[2])
                    )
                )
                / "ioi-agent/tool.py"
            ),
        ]:
            tree = ast.parse(Path(source).read_text())
            cls = next(
                n
                for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "Submission"
            )
            method = next(
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "_update_best_scores"
            )
            namespace = {}
            exec(
                compile(ast.Module(body=[method], type_ignores=[]), source, "exec"),
                namespace,
            )
            state = types.SimpleNamespace(best_subtask_scores={"a": 0, "b": 0})
            namespace["_update_best_scores"](state, {"a": 10, "b": 0})
            namespace["_update_best_scores"](state, {"a": 0, "b": 20})
            self.assertEqual(sum(state.best_subtask_scores.values()), 30)

    def test_link_and_compile_failures(self):
        result = classify(
            "Error: Failed to compile solution",
            "multiple definition of `main'",
            0,
            True,
            "int main() {}",
            {"solve"},
        )
        self.assertEqual(result["compile_style"], "main")
        result = classify(
            "Error: Failed to compile solution",
            "undefined reference to `solve(int)'",
            0,
            True,
            "int main() {}",
            {"solve"},
        )
        self.assertEqual(result["signature_status"], "missing_or_wrong")
        self.assertEqual(result["compile_style"], "signature")
        result = classify(
            "Error: Failed to compile solution",
            "solution.cpp:2: error: syntax",
            0,
            True,
            "int main() {}",
            {"solve"},
        )
        self.assertEqual(result["compile_style"], "other")

    def test_scores_wa_and_unobservable_execution_status(self):
        self.assertEqual(
            classify("Passed tests: a", "", 22, True, "", set())["verdict"], "partial"
        )
        self.assertEqual(
            classify("Passed tests:", "", 0, True, "", set())["verdict"], "wa"
        )
        self.assertEqual(
            classify(
                "Passed tests:\nError: Solution execution failed for test case a",
                "",
                0,
                True,
                "",
                set(),
            )["verdict"],
            "execution_failure_unspecified",
        )
        self.assertEqual(
            classify("", "Tests timed out", 0, True, "", set())["verdict"], "tle"
        )
        self.assertEqual(
            classify("", "Segmentation fault", 0, True, "", set())["verdict"],
            "runtime_error",
        )

    def test_empty_pending_and_last_code_block(self):
        self.assertEqual(blocks(""), {})
        self.assertEqual(
            classify("", "", 0, False, "", set())["verdict"], "pending_or_interrupted"
        )
        result = classify(
            "Passed tests:",
            "",
            0,
            True,
            "```cpp\nint main() {}\n```\n```cpp\nvoid solve() {}\n```",
            set(),
        )
        self.assertFalse(result["defines_main"])

    def test_log_blocks_associate_submission_numbers(self):
        log = "2026-11-13 08:30:35,679 DEBUG Test script stdout:\nPassed tests: a\n2026-11-13 08:30:35,680 INFO Test script stderr:\nwarning\n2026-11-13 08:30:35,744 INFO Submission #2 - Score: 22 - Best: 22 - Remaining: 48\n"
        result = blocks(log)
        self.assertEqual(result[2]["score"], 22)
        self.assertIn("warning", result[2]["stderr"])


if __name__ == "__main__":
    unittest.main()
