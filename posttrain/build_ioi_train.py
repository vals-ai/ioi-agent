"""Materialize and validate pre-2024 IOI batch function-grader tasks."""

import collections
import concurrent.futures
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from build_pool import REPOS, ROOT, shingles, title_key

OUT = ROOT / "ioi_train"
STAGING = ROOT / "ioi_work"
REFERENCES = ROOT / "ioi_references"
PYTHON = REPOS / "ioi-hackathon-agent/.venv/bin/python"


def dump(path, data):
    path.write_text(json.dumps(data, indent=2))


class Exclusions:
    def __init__(self):
        dev = sorted((REPOS / "ioi-agent/exams").glob("202[45]/*/*.pdf"))
        future = sorted(
            (REPOS / "ioi-benchmark-service/src/ioi_benchmark_service/exams/2026").glob(
                "*/problem_statement.txt"
            )
        )
        if len(dev) != 12 or len(future) != 6:
            raise ValueError("Incomplete protected dev/2026 corpus")
        texts = []
        self.titles = set()
        for path in dev + future:
            text = (
                subprocess.check_output(
                    ["pdftotext", "-layout", str(path), "-"]
                ).decode()
                if path.suffix == ".pdf"
                else path.read_text()
            )
            if text.startswith("version https://git-lfs"):
                raise ValueError("Protected statement is an LFS pointer")
            texts.append(text)
            self.titles.add(title_key(path.parent.name))
            first = next((s.strip() for s in text.splitlines() if s.strip()), "")
            self.titles.add(title_key(first))
        groups = [shingles(text) for text in texts]
        counts = collections.Counter(s for group in groups for s in group)
        self.boilerplate = {
            s for s, n in counts.items() if n >= max(4, math.ceil(len(groups) * 0.2))
        }
        self.groups = [s - self.boilerplate for s in groups]
        self.inventory = {
            "dev_pdfs": len(dev),
            "ioi_2026_statements": len(future),
            "sha256": [hashlib.sha256(s.encode()).hexdigest() for s in texts],
        }

    def check(self, title, text):
        key = title_key(title)
        if key in self.titles:
            return "protected_title"
        words = set(key.split())
        if len(words) >= 3 and any(
            len(words & set(t.split())) / max(1, len(words | set(t.split()))) >= 0.8
            for t in self.titles
        ):
            return "similar_protected_title"
        candidate = shingles(text) - self.boilerplate
        for group in self.groups:
            shared = len(candidate & group)
            fraction = shared / max(1, min(len(candidate), len(group)))
            if (shared >= 4 and fraction >= 0.02) or (shared >= 2 and fraction >= 0.08):
                return "protected_8gram_overlap"
        return None


def reference_choices(rows):
    labels = {}
    for row in rows:
        if not row["label"].endswith((".cpp", ".cc", ".cxx")):
            continue
        entry = labels.setdefault(
            row["label"], {"code": row["code"], "points": {}, "all_ac": True}
        )
        entry["points"][row["subtask"]] = row["subtask_points"]
        if row["subtask_max_points"] > 0 and row["subtask_score"] < 1:
            entry["all_ac"] = False
    return [
        (name, value["code"])
        for name, value in sorted(labels.items())
        if value["all_ac"] and sum(value["points"].values()) >= 99.999
    ][:3]


def validate_one(item):
    year, task, choices = item
    directory = STAGING / str(year) / task
    attempts = []
    for index, (label, code) in enumerate(choices):
        reference = REFERENCES / f"{year}-{task}-{index}.cpp"
        reference.write_text(code)
        start = time.monotonic()
        process = subprocess.Popen(
            [
                str(PYTHON),
                str(Path(__file__).resolve().parent / "validate_ioi_reference.py"),
                str(directory),
                str(reference),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=dict(
                os.environ,
                IOI2_PROCESS_SLOTS_CONFIG=str(ROOT / "logs/process_slots.json"),
            ),
        )
        try:
            stdout, stderr = process.communicate(timeout=600)
            result = (
                json.loads(stdout)
                if not process.returncode
                else {"error": stderr.decode(errors="replace")[-4000:]}
            )
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            result = {"error": "Reference validation exceeded 600s"}
        result.update(label=label, duration_seconds=time.monotonic() - start)
        attempts.append(result)
        if result.get("total_score") == 100 and result.get("all_tests_passed"):
            destination = OUT / str(year) / task
            destination.parent.mkdir(parents=True, exist_ok=True)
            directory.rename(destination)
            return {
                "id": f"{year}/{task}",
                "year": year,
                "task": task,
                "directory": str(destination),
                "reference": str(reference),
                "reference_label": label,
                "score": 100,
                "n_tests": len(list((destination / "tests").glob("*.in"))),
                "attempts": attempts,
            }
    return {
        "id": f"{year}/{task}",
        "rejected": "reference_validation",
        "attempts": attempts,
    }


def main():
    for directory in [OUT, STAGING, REFERENCES]:
        directory.mkdir(exist_ok=False)
    api = HfApi()
    revisions = {
        name: api.dataset_info(name).sha
        for name in [
            "open-r1/ioi",
            "open-r1/ioi-sample-solutions",
            "open-r1/ioi-test-cases",
        ]
    }

    def fetch(repo, file):
        return hf_hub_download(
            repo, file, repo_type="dataset", revision=revisions[repo]
        )

    rows = pq.read_table(
        fetch("open-r1/ioi", "data/train-00000-of-00001.parquet")
    ).to_pylist()
    refs = pq.read_table(
        fetch("open-r1/ioi-sample-solutions", "data/train-00000-of-00001.parquet")
    ).to_pylist()
    grouped = collections.defaultdict(list)
    reference_rows = collections.defaultdict(list)
    for row in rows:
        if row["year"] <= 2023:
            grouped[row["year"], row["id"]].append(row)
    for row in refs:
        if int(row["year"]) <= 2023:
            reference_rows[int(row["year"]), row["problem_id"]].append(row)
    exclusions = Exclusions()
    rejected, selected = [], []
    script = (Path(__file__).resolve().parent / "ioi_run_tests.sh").read_text()
    for (year, task), subtasks in sorted(grouped.items()):
        row = subtasks[0]
        reason = (
            "unsupported_task_type"
            if row["task_type"] != "Batch"
            else exclusions.check(row["name"], row["statement"])
        )
        choices = reference_choices(reference_rows[year, task])
        if not reason and not choices:
            reason = "no_full_score_cpp_reference"
        if not reason and sum(s["score"] for s in subtasks) != 100:
            reason = "subtask_scores_not_100"
        if reason:
            rejected.append({"id": f"{year}/{task}", "reason": reason})
            continue
        directory = STAGING / str(year) / task
        (directory / "subtasks").mkdir(parents=True)
        (directory / "tests").mkdir()
        for name, content in row["grader_files"]:
            parts = PurePosixPath(name)
            if (
                parts.is_absolute()
                or ".." in parts.parts
                or parts.parts[0] not in ("graders", "checker")
            ):
                raise ValueError("Unsafe grader asset path")
            destination = directory / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content)
        config = json.loads((directory / "graders/grader_config.json").read_text())
        dump(directory / "problem.json", config)
        (directory / "run_tests.sh").write_text(script)
        (directory / "run_tests.sh").chmod(0o755)
        (directory / "statement.txt").write_text(
            row["statement"]
            + "\n\n## C++ interface\n\n```cpp\n"
            + row["starting_code"]
            + "\n```\n"
        )
        for subtask in subtasks:
            if not subtask["test_names"]:
                raise ValueError(f"Empty test group: {year}/{task}")
            dump(
                directory / "subtasks" / f"{subtask['subtask']}.json",
                {"score": subtask["score"], "testcases": subtask["test_names"]},
            )
        selected.append((year, task, choices))
    report = {
        "sources": revisions,
        "protected": exclusions.inventory,
        "thresholds": {
            "ngram": 8,
            "min_shared": 4,
            "min_fraction": 0.02,
            "small_min_shared": 2,
            "small_min_fraction": 0.08,
        },
        "candidates": len(grouped),
        "rejected": rejected,
        "selected_before_validation": len(selected),
    }
    dump(OUT / "exclusion_report.json", report)
    print(
        json.dumps({"selected_before_validation": len(selected), "rejected": rejected}),
        flush=True,
    )
    selected_keys = {(year, task) for year, task, _ in selected}
    files = [
        name
        for name in api.list_repo_files(
            "open-r1/ioi-test-cases",
            repo_type="dataset",
            revision=revisions["open-r1/ioi-test-cases"],
        )
        if name.endswith(".parquet") and re.match(r"202[0-3]/", name)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        downloaded = list(
            executor.map(lambda name: fetch("open-r1/ioi-test-cases", name), files)
        )
    for file in downloaded:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=1, use_threads=False):
            case = batch.to_pylist()[0]
            year, task = int(case["year"]), case["problem_id"]
            if (year, task) not in selected_keys:
                continue
            name = case["test_name"]
            if Path(name).name != name:
                raise ValueError("Unsafe test filename")
            directory = STAGING / str(year) / task / "tests"
            (directory / (name + ".in")).write_text(case["test_input"])
            (directory / (name + ".out")).write_text(case["test_output"])
    for year, task, _ in selected:
        directory = STAGING / str(year) / task
        required = {
            name
            for path in (directory / "subtasks").glob("*.json")
            for name in json.loads(path.read_text())["testcases"]
        }
        if not required or any(
            not (directory / "tests" / (name + suffix)).is_file()
            for name in required
            for suffix in (".in", ".out")
        ):
            raise ValueError(f"Missing required tests: {year}/{task}")
    print("All required test inputs and outputs materialized", flush=True)
    manifest, validations = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for result in executor.map(validate_one, selected):
            validations.append(result)
            if result.get("score") == 100:
                manifest.append(
                    {key: value for key, value in result.items() if key != "attempts"}
                )
            dump(OUT / "manifest.json", manifest)
            dump(OUT / "validation_report.json", validations)
            print(
                json.dumps(
                    {key: value for key, value in result.items() if key != "attempts"}
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                "validated": len(manifest),
                "rejected_reference": len(validations) - len(manifest),
            }
        ),
        flush=True,
    )
    dump(
        OUT / "build_complete.json",
        {"validated": len(manifest), "finished_at": time.time()},
    )


if __name__ == "__main__":
    main()
