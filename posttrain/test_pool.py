import tempfile
import unittest
from pathlib import Path

from build_pool import (
    Exclusions,
    category,
    filter_metadata,
    plain_statement,
    select_tests,
    shingles,
    split_heldout,
)


class PoolTests(unittest.TestCase):
    def test_metadata_rejects_nonstandard_tasks(self):
        checker = Exclusions.__new__(Exclusions)
        checker.titles, checker.sets, checker.boilerplate = set(), [], set()
        row = {
            "source": 2,
            "cf_rating": 1800,
            "time_limit": {"seconds": 2, "nanos": 0},
            "input_file": "",
            "output_file": "",
            "cf_tags": ["dp"],
            "name": "Counting",
            "description": "Count things.\nInput\nAn integer.\nOutput\nOne integer.",
        }
        self.assertIsNone(filter_metadata(row, checker))
        for change in [
            {"cf_tags": ["constructive algorithms"]},
            {"description": "interactive\nInput\nOutput\n"},
            {"description": "Input\nOutput\nPrint any valid answer."},
            {"time_limit": {"seconds": 6, "nanos": 0}},
            {"cf_rating": 1400},
            {"input_file": "input.txt"},
        ]:
            self.assertIsNotNone(filter_metadata(row | change, checker))

    def test_title_and_eightgram_exclusions(self):
        checker = Exclusions.__new__(Exclusions)
        checker.titles = {"protected example title"}
        checker.boilerplate = set()
        checker.sets = [
            shingles(
                "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
            )
        ]
        self.assertEqual(
            checker.check("Protected example title", "unrelated"), "protected_title"
        )
        self.assertEqual(
            checker.check(
                "Different",
                "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen",
            ),
            "protected_8gram_overlap",
        )
        self.assertIsNone(
            checker.check("Different", "A completely unrelated problem statement here.")
        )
        self.assertEqual(
            checker.check("Different", "Adapted from IOI 2010"), "ioi_mention"
        )

    def test_test_cap_retains_largest_and_requires_ten(self):
        row = {
            "public_tests": {"input": ["sample"], "output": ["0"]},
            "private_tests": {
                "input": ["x" * i for i in range(1, 101)],
                "output": ["0"] * 100,
            },
            "generated_tests": {"input": [], "output": []},
        }
        public, hidden = select_tests(row)
        self.assertEqual(len(hidden), 40)
        self.assertTrue(all(("x" * i, "0") in hidden for i in range(93, 101)))
        row["private_tests"] = {"input": ["1"], "output": ["0"]}
        self.assertIsNone(select_tests(row))

    def test_format_and_categories(self):
        row = {
            "name": "123_A. Counting",
            "description": "Read numbers.\nInput\nAn integer n (1 <= n <= 100).\nOutput\nPrint n.\nExamples\nold example",
        }
        text = plain_statement(row, [("3\n", "3\n")])
        for value in [
            "Counting",
            "Input",
            "Output",
            "Constraints",
            "Sample Input\n3",
            "Sample Output\n3",
            "1 <= n <= 100",
        ]:
            self.assertIn(value, text)
        self.assertNotIn("old example", text)
        self.assertIn("data_structures", category(["data structures"]))

    def test_stratified_split_is_disjoint(self):
        import json

        with tempfile.TemporaryDirectory() as temp:
            pool = Path(temp) / "pool"
            pool.mkdir()
            for i in range(80):
                directory = pool / str(i)
                directory.mkdir()
                item = {
                    "id": str(i),
                    "split": "train",
                    "rating": 1500 + 100 * (i % 12),
                    "categories": ["data_structures" if i % 2 else "dp"],
                    "tags": ["dp"],
                }
                (directory / "problem.json").write_text(json.dumps(item))
            train, heldout = split_heldout(pool, Path(temp) / "heldout", 40)
            self.assertEqual(len(heldout), 40)
            self.assertFalse({x["id"] for x in train} & {x["id"] for x in heldout})
            self.assertTrue(all(x["split"] == "heldout" for x in heldout))


if __name__ == "__main__":
    unittest.main()
