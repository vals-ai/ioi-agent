import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import synthetic_ioi
from build_synthetic_ioi import validate
from synthetic_ioi import materialize, parse_draft

DRAFT = {
    "statement": "Add two integers. Implementation Details: implement add(a,b); return their sum. No main or I/O.",
    "header": "#pragma once\nlong long add(long long a, long long b);\n",
    "grader": '#include "api.h"\n#include <iostream>\nint main(){long long a,b; std::cin>>a>>b; std::cout<<add(a,b)<<"\\n";}\n',
    "reference": '#include "api.h"\nlong long add(long long a,long long b){return a+b;}\n',
}


class SyntheticTests(unittest.TestCase):
    def test_draft_normal_empty_and_invalid(self):
        self.assertEqual(parse_draft(json.dumps(DRAFT)), DRAFT)
        self.assertEqual(parse_draft("```json\n" + json.dumps(DRAFT) + "\n```"), DRAFT)
        for value in [
            "",
            "{}",
            json.dumps(DRAFT | {"reference": "int main(){return 0;}"}),
            json.dumps(
                DRAFT | {"reference": DRAFT["reference"] + " void x(){std::cout<<1;}"}
            ),
            json.dumps(DRAFT | {"grader": '#include "api.h"'}),
        ]:
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                parse_draft(value)

    def test_real_native_grader_and_negative_controls(self):
        with tempfile.TemporaryDirectory(
            dir=synthetic_ioi.ROOT, prefix="synthetic-test-"
        ) as name:
            root = Path(name)
            source = root / "pool/toy"
            (source / "tests").mkdir(parents=True)
            real_script = Path(synthetic_ioi.__file__).parent / "ioi_run_tests.sh"
            script = root / "ioi_train/2021/dna/run_tests.sh"
            script.parent.mkdir(parents=True)
            script.write_text(real_script.read_text())
            (source / "problem.json").write_text(
                json.dumps(
                    {
                        "split": "train",
                        "exclusion_checked_final": True,
                        "source_name": "Toy",
                        "time_limit": 2,
                        "statement_sha256": "test",
                        "reference_sha256": "test",
                    }
                )
            )
            for i, (given, expected) in enumerate(
                [("1 2", "3"), ("20 -5", "15"), ("-10 17", "7")]
            ):
                (source / f"tests/{i}.in").write_text(given + "\n")
                (source / f"tests/{i}.out").write_text(expected + "\n")
            destination = root / "toy"
            reference = root / "reference.cpp"
            with patch.object(synthetic_ioi, "ROOT", root):
                metadata = materialize(DRAFT, source, destination, reference)
                self.assertEqual(metadata["n_tests"], 3)
                with self.assertRaises(ValueError):
                    materialize(DRAFT, source, destination, reference)
            result = asyncio.run(validate(destination, reference, root))
            self.assertEqual(result["score"], 100)
            self.assertTrue(result["duplicate_main_rejected"])
            self.assertTrue(result["missing_function_rejected"])
            reference.write_text(DRAFT["reference"].replace("a+b", "a-b"))
            with self.assertRaisesRegex(ValueError, "did not pass"):
                asyncio.run(validate(destination, reference, root))


if __name__ == "__main__":
    unittest.main()
