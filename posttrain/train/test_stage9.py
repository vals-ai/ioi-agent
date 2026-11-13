import json
import tempfile
import unittest
from pathlib import Path

from build_v2 import audit_record, balance, freeze
from submission_filter import defines_main, final_scoring_submission


class Stage9Tests(unittest.TestCase):
    def row(self, name, count):
        return {
            "trajectory": name,
            "input_ids": list(range(count + 1)),
            "target_ids": list(range(1, count + 2)),
            "weights": [0] + [1] * count,
            "target_tokens": count,
            "assistant_spans": [{"start": 2, "end": count + 2}],
            "sequence_tokens": count + 2,
        }

    def test_masks_and_balance_both_sides(self):
        for left, right in [(10, 90), (90, 10), (55, 45)]:
            rows = balance([self.row("ioi", left)], [self.row("other", right)])
            self.assertEqual({row["trajectory"] for row in rows}, {"ioi", "other"})
            fraction = sum(
                row["target_tokens"] for row in rows if row["trajectory"] == "ioi"
            ) / sum(row["target_tokens"] for row in rows)
            self.assertLess(abs(fraction - 0.55), 0.04)
        with self.assertRaises(ValueError):
            balance([], [self.row("a", 1)])
        row = self.row("a", 3)
        row["weights"][0] = 1
        with self.assertRaisesRegex(ValueError, "assistant spans"):
            audit_record(row)
        row = self.row("a", 3)
        row["target_ids"][0] = 99
        with self.assertRaisesRegex(ValueError, "Shifted"):
            audit_record(row)

    def test_main_lexing(self):
        for source in [
            "int main(){}",
            "int/*a*/main(){}",
            "#define entry main\nint entry(){}",
        ]:
            self.assertTrue(defines_main(source))
        for source in [
            "void solve(){}",
            "// main(){}\nvoid solve(){}",
            'const char*x="main()";',
        ]:
            self.assertFalse(defines_main(source))

    def put_turn(self, root, number, code, completed=True, score=100):
        directory = root / f"logs/run/turns/turn_{number:03}"
        directory.mkdir(parents=True)
        call = {
            "name": "submission",
            "call_id": str(number),
            "parsed_args": {"cpp_code": code},
        }
        records = (
            [
                {
                    "tool_call": call,
                    "tool_output": {"output": json.dumps({"total_score": score})},
                }
            ]
            if completed
            else []
        )
        (directory / "result.json").write_text(
            json.dumps(
                {"query_result": {"tool_calls": [call]}, "tool_call_records": records}
            )
        )

    def test_last_completed_scoring_call_not_pending(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with self.assertRaisesRegex(ValueError, "No completed"):
                final_scoring_submission(root)
            self.put_turn(root, 1, "void solve(){}")
            self.put_turn(root, 2, "int main(){}", completed=False)
            self.assertFalse(final_scoring_submission(root)["defines_main"])
            self.put_turn(root, 3, "int main(){}", score=0)
            self.assertTrue(final_scoring_submission(root)["defines_main"])
            self.put_turn(
                root, 4, "```cpp\nint main(){}\n```\n```cpp\nvoid solve(){}\n```"
            )
            self.assertFalse(final_scoring_submission(root)["defines_main"])

    def test_freeze_guards_cutoff_main_and_heldout(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for directory in [
                "pool",
                "pool_heldout",
                "ioi_train",
                "synthetic_ioi",
                "runs",
            ]:
                (root / directory).mkdir()
            for directory, ids in [
                ("pool", ["train"]),
                ("pool_heldout", ["heldout"]),
                ("ioi_train", ["2021/toy"]),
                ("synthetic_ioi", ["synthetic/toy"]),
            ]:
                (root / directory / "manifest.json").write_text(
                    json.dumps([{"id": value} for value in ids])
                )
            task = root / "runs/ioi-v1mix-rft/0"
            task.mkdir(parents=True)
            (task / "score.json").write_text(
                json.dumps({"task_id": "2021/toy", "score": 10})
            )
            self.put_turn(task, 1, "int main(){}", score=0)
            result = freeze(root, 10**12)
            self.assertEqual(
                result["historical-onpolicy"]["rejected"][
                    "final_scoring_submission_defines_main"
                ],
                1,
            )
            result = freeze(root, 0)
            self.assertEqual(
                result["historical-onpolicy"]["rejected"]["completed_after_freeze"], 1
            )
            (root / "pool_heldout/manifest.json").write_text('[{"id":"train"}]')
            with self.assertRaisesRegex(ValueError, "overlap"):
                freeze(root, 10**12)


if __name__ == "__main__":
    unittest.main()
