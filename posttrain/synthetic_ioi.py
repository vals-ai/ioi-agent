"""Strict materialization of teacher-drafted typed-function IOI tasks."""

import hashlib
import json
import math
import os
import re
from pathlib import Path

ROOT = Path(os.environ.get("IOI2_ROOT", str(Path(__file__).resolve().parent)))
TEACHER = "Qwen/Qwen3.5-397B-A17B"
MAIN = re.compile(r"\bmain\s*\([^)]*\)\s*\{")
IO = re.compile(
    r"\b(?:cin|cout|cerr|scanf|printf|fprintf|fscanf|getchar|putchar|puts|gets|stdin|stdout|istream|ostream|freopen|fopen|ifstream|ofstream|fstream)\b"
)
UNSAFE = re.compile(
    r"\b(?:system|popen|fork|execve|execl|socket|filesystem|fstream|freopen|fopen)\s*(?:\(|::|>)"
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def code_only(text):
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)


def parse_draft(content):
    content = content.strip()
    if content.startswith("```"):
        content = "\n".join(content.splitlines()[1:-1])
    draft = json.loads(content)
    for key in ["statement", "header", "grader", "reference"]:
        if not isinstance(draft.get(key), str) or not draft[key].strip():
            raise ValueError(f"Missing draft field: {key}")
    reference, header, grader = (
        code_only(draft[k]) for k in ["reference", "header", "grader"]
    )
    if (
        MAIN.search(reference)
        or MAIN.search(header)
        or IO.search(reference)
        or IO.search(header)
    ):
        raise ValueError("Contestant API/reference must have no main or I/O")
    if not MAIN.search(grader):
        raise ValueError("Grader must own main")
    if any(
        UNSAFE.search(code_only(draft[k])) for k in ["reference", "header", "grader"]
    ):
        raise ValueError("Disallowed process/file operation")
    if re.search(r"\b(?:istream|ostream)\b", header):
        raise ValueError("A stream wrapper is not a typed contestant API")
    declarations = re.findall(r"\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*;", header)
    if not declarations or any(name in {"main", "system"} for name in declarations):
        raise ValueError("No typed function declaration")
    if not any(
        re.search(rf"\b{re.escape(name)}\s*\([^;{{}}]*\)\s*\{{", reference)
        for name in declarations
    ):
        raise ValueError("Reference does not define its declared API")
    if "#include" not in draft["grader"] or "api.h" not in draft["grader"]:
        raise ValueError("Grader must include api.h")
    if "Implementation Details" not in draft["statement"]:
        raise ValueError("Missing Implementation Details")
    return draft


def prompt_for(source, reference):
    return (
        """Convert this train-only competitive-programming problem into a genuine IOI
typed-function task. Return ONLY a JSON object with four string fields: statement,
header, grader, reference. No markdown wrapping around JSON.

Requirements:
- Preserve the mathematical problem, all constraints, indexing conventions, and
  exact original input/output semantics. Do not simplify the problem.
- Replace console I/O with a natural typed C++20 function API: scalars, strings,
  vectors/matrices for inputs and results, or typed init/query callbacks. Never
  pass the raw serialized input, stdin, streams, or output text to the function.
- statement must be a complete IOI-style statement with an "Implementation Details"
  section spelling out exact function signatures, parameter/return meanings,
  constraints and a worked example in terms of function arguments/results. Remove
  instructions to read input/print output; graders alone handle those operations.
  Explicitly say: do not define main, do not perform I/O, include api.h.
- header is self-contained guarded C++ header api.h, including required STL types
  and exact public function declarations (no implementation or main).
- grader is C++ with main, includes api.h, reads EXACT original test input, calls
  contestant functions and prints EXACT original required output. For multi-case
  input, call/reset the function separately per case. Handle all original formats.
  Use cin/cout only for I/O. No files, subprocesses, environmental inspection, or
  hardcoded answers. Call the contestant API for all solution logic.
- reference adapts the supplied known-correct solution into that API: include
  api.h, preserve algorithm correctness and integer widths, no main and absolutely
  no I/O. Return answers as typed values. Reset globals between cases/calls. Do not
  use string/stream wrappers around the old stdin/stdout algorithm.
- Prefer one batch function if practical. A void function is only appropriate for
  initialization alongside typed query-return functions. No template-only APIs.

ORIGINAL TRAIN PROBLEM:
"""
        + (source / "statement.txt").read_text()
        + "\n\nKNOWN-CORRECT ORIGINAL REFERENCE:\n"
        + reference.read_text()
    )


CHECKER = r"""#include <bits/stdc++.h>
using namespace std;
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    ifstream expected(argv[2]), actual(argv[3]);
    if (!expected || !actual) return 2;
    string a, b;
    while (expected >> a) {
        if (!(actual >> b) || a != b) { cout << 0 << '\n'; return 0; }
    }
    cout << (!(actual >> b) ? 1 : 0) << '\n';
}
"""


def materialize(draft, source, destination, reference_path):
    metadata = json.loads((source / "problem.json").read_text())
    if (
        source.parent.resolve() != (ROOT / "pool").resolve()
        or metadata.get("split") != "train"
        or not metadata.get("exclusion_checked_final")
    ):
        raise ValueError("Only exclusion-checked pool/train sources are allowed")
    if destination.exists():
        raise ValueError("Synthetic destination already exists")
    tests = sorted(
        (source / "tests").glob("*.in"), key=lambda p: (p.stat().st_size, p.name)
    )
    if len(tests) < 2 or any(not p.with_suffix(".out").is_file() for p in tests):
        raise ValueError("At least two complete test pairs required")
    destination.mkdir()
    for name in ["graders", "checker", "subtasks"]:
        (destination / name).mkdir()
    (destination / "tests").symlink_to(
        (source / "tests").resolve(), target_is_directory=True
    )
    (destination / "graders/api.h").write_text(draft["header"])
    (destination / "graders/grader.cpp").write_text(draft["grader"])
    (destination / "checker/checker.cpp").write_text(CHECKER)
    script = (Path(__file__).resolve().parent / "ioi_run_tests.sh").read_text()
    script = script.replace(
        "-include bits/stdc++.h", "-include bits/stdc++.h -include api.h"
    )
    (destination / "run_tests.sh").write_text(script)
    (destination / "run_tests.sh").chmod(0o755)
    names = [p.stem for p in tests]
    groups = (
        [
            ("01-small-input", 20, names[: max(1, math.ceil(len(names) * 0.15))]),
            ("02-medium-input", 30, names[: max(1, math.ceil(len(names) * 0.4))]),
            ("03-full", 50, names),
        ]
        if len(names) >= 3
        else [("01-small-input", 30, names[:1]), ("02-full", 70, names)]
    )
    for name, points, cases in groups:
        write_json(
            destination / "subtasks" / f"{name}.json",
            {"score": points, "testcases": cases},
        )
    statement = (
        draft["statement"]
        + "\n\n## Required C++ interface (api.h)\n```cpp\n"
        + draft["header"]
        + "\n```\n\nOnly implement the declared functions. Do not define main or perform I/O.\n\n## Scoring\n"
    )
    statement += "\n".join(
        f"- {name}: {points} points, all {len(cases)} tests in this input-size group must pass."
        for name, points, cases in groups
    )
    (destination / "statement.txt").write_text(statement)
    config = {
        "code": "syn-" + source.name,
        "name": metadata["source_name"] + " (function version)",
        "time_limit": max(2.0, metadata["time_limit"]),
        "memory_limit": 2 * 1024**3,
        "score_precision": 2,
        "task_type": "Batch",
        "task_type_params": {},
        "kind": "synthetic_ioi",
        "source_pool_id": source.name,
        "source_split": "train",
        "source_directory": str(source.resolve()),
        "source_statement_sha256": metadata["statement_sha256"],
        "source_reference_sha256": metadata["reference_sha256"],
        "draft_model": TEACHER,
        "n_tests": len(tests),
        "subtask_policy": "nested smallest-input-byte groups then full; all tests retained",
    }
    write_json(destination / "problem.json", config)
    write_json(destination / "graders/grader_config.json", config)
    reference_path.write_text(draft["reference"])
    return config


def select_candidates(maximum=220, seed=1729):
    import random

    rows = []
    heldout = {
        row["id"]
        for row in json.loads((ROOT / "pool_heldout/manifest.json").read_text())
    }
    for source in sorted((ROOT / "pool").iterdir()):
        if not source.is_dir() or not (source / "problem.json").is_file():
            continue
        meta = json.loads((source / "problem.json").read_text())
        reference = ROOT / "pool_references" / f"{source.name}.cpp"
        if (
            source.name in heldout
            or meta.get("split") != "train"
            or not reference.is_file()
        ):
            continue
        if (
            hashlib.sha256(reference.read_bytes()).hexdigest()
            != meta["reference_sha256"]
        ):
            raise ValueError("Original reference checksum mismatch")
        tags = set(meta.get("tags", []))
        priority = (
            4 * bool(tags & {"data structures", "dp"})
            + 3 * bool(tags & {"combinatorics"})
            + bool("range" in (source / "statement.txt").read_text().lower())
        )
        rows.append(
            {
                "id": source.name,
                "directory": str(source),
                "reference": str(reference),
                "priority": priority,
                "rating": meta.get("rating") or 3000,
            }
        )
    random.Random(seed).shuffle(rows)
    rows.sort(key=lambda r: (-r["priority"], not 1000 <= r["rating"] <= 2400))
    return rows[:maximum]
