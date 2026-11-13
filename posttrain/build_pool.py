"""Build a locally validated, contamination-filtered CodeContests pool."""

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from harness import grade

ROOT = Path(os.environ.get("IOI2_ROOT", str(Path(__file__).resolve().parent)))
REPOS = Path(os.environ.get("IOI2_REPOS", str(Path(__file__).resolve().parents[2])))
META_COLUMNS = [
    "name",
    "description",
    "source",
    "cf_rating",
    "cf_tags",
    "cf_contest_id",
    "cf_index",
    "time_limit",
    "memory_limit_bytes",
    "input_file",
    "output_file",
]
DATA_COLUMNS = ["public_tests", "private_tests", "generated_tests", "solutions"]
CATEGORIES = {
    "data_structures": {
        "data structures",
        "trees",
        "dsu",
        "divide and conquer",
        "schedules",
    },
    "combinatorics_math": {
        "combinatorics",
        "math",
        "number theory",
        "probabilities",
        "fft",
    },
    "dp": {"dp"},
    "graphs": {
        "graphs",
        "dfs and similar",
        "shortest paths",
        "flows",
        "graph matchings",
        "trees",
    },
    "greedy": {"greedy"},
    "strings": {"strings", "string suffix structures", "hashing"},
    "binary_search": {"binary search"},
}
AMBIGUOUS = re.compile(
    r"interactive|interactor|special checker|output[- ]only|absolute (?:or relative )?error|relative error|(?:print|output|find)\s+any\b|(?:several|multiple|many)\s+(?:possible |valid |correct )?(?:answers|solutions)|in any order|arbitrary order|construct\s+(?:a|an|any)\b|any (?:of them|correct|valid|suitable|optimal)\b",
    re.I,
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def title_key(title):
    title = re.sub(r"^\s*\d+[_ .-]*[A-Z]\d?\s*[-_.:]\s*", "", title)
    return " ".join(words(title))


def shingles(text):
    tokens = words(text)
    return {
        hashlib.blake2b(" ".join(tokens[i : i + 8]).encode(), digest_size=8).hexdigest()
        for i in range(len(tokens) - 7)
    }


class Exclusions:
    def __init__(self):
        texts = []
        self.inventory = collections.Counter()
        olympiad = (
            REPOS
            / "hackathon-olympiad-benchmark-service/src/olympiad_benchmark_service/data"
        )
        ioi = REPOS / "ioi-benchmark-service/src/ioi_benchmark_service/exams"
        for label, paths in [
            ("hackathon_olympiad", sorted(olympiad.glob("*/statement.txt"))),
            ("ioi_2024_2025_2026", sorted(ioi.glob("*/*/problem_statement.txt"))),
        ]:
            if not paths:
                raise RuntimeError(f"Missing exclusion corpus: {label}")
            for path in paths:
                text = path.read_text()
                if text.startswith("version https://git-lfs"):
                    raise RuntimeError("Exclusion corpus contains an LFS pointer")
                texts.append((path.parent.name, text))
                self.inventory[label] += 1
        if len(list((ioi / "2026").glob("*/problem_statement.txt"))) != 6:
            raise RuntimeError("Expected all six IOI 2026 scoring statements")
        pdfs = sorted((REPOS / "ioi-agent/exams").glob("202[45]/*/*.pdf"))
        if len(pdfs) != 12:
            raise RuntimeError("Expected all twelve IOI dev PDFs")
        for path in pdfs:
            result = subprocess.run(
                ["pdftotext", "-layout", str(path), "-"],
                capture_output=True,
                text=True,
                check=True,
            )
            texts.append((path.parent.name, result.stdout))
            self.inventory["ioi_dev_pdfs"] += 1
        self.titles = set()
        for name, text in texts:
            self.titles.add(title_key(name.replace("_", " ")))
            first = next(
                (line.strip() for line in text.splitlines() if line.strip()), ""
            )
            if 1 <= len(words(first)) <= 10:
                self.titles.add(title_key(first))
        sets = [shingles(text) for _, text in texts]
        frequency = collections.Counter(s for group in sets for s in group)
        boilerplate = {
            s for s, n in frequency.items() if n >= max(5, math.ceil(len(sets) * 0.2))
        }
        self.sets = [group - boilerplate for group in sets]
        self.boilerplate = boilerplate
        self.fingerprints = [
            hashlib.sha256(text.encode()).hexdigest() for _, text in texts
        ]
        archive_path = ROOT / "ioi_archive_fingerprints.json"
        if archive_path.is_file():
            archive = json.loads(archive_path.read_text())
            for entry in archive["entries"]:
                if entry["title"]:
                    self.titles.add(entry["title"])
                if entry["shingles"]:
                    self.sets.append(set(entry["shingles"]) - boilerplate)
                    self.fingerprints.append(entry["sha256"])
            self.inventory["historical_ioi_titles"] = len(archive["entries"])
            self.inventory["historical_ioi_statement_fingerprints"] = sum(
                bool(entry["shingles"]) for entry in archive["entries"]
            )

    def check(self, title, statement):
        key = title_key(title)
        if key in self.titles:
            return "protected_title"
        tokens = set(key.split())
        if len(tokens) >= 3 and any(
            len(tokens & set(t.split())) / max(1, len(tokens | set(t.split()))) >= 0.8
            for t in self.titles
        ):
            return "similar_protected_title"
        candidate = shingles(statement) - self.boilerplate
        for protected in self.sets:
            shared = len(candidate & protected)
            fraction = shared / max(1, min(len(candidate), len(protected)))
            if (shared >= 4 and fraction >= 0.02) or (shared >= 2 and fraction >= 0.08):
                return "protected_8gram_overlap"
        if re.search(r"\bioi\b|international olympiad in informatics", statement, re.I):
            return "ioi_mention"
        return None


def category(tags):
    return [name for name, choices in CATEGORIES.items() if set(tags) & choices]


def filter_metadata(row, exclusions):
    if row["source"] != 2:
        return "not_codeforces"
    if not 1500 <= row["cf_rating"] <= 2800:
        return "rating"
    limit = row.get("time_limit") or {}
    seconds = limit.get("seconds", 0) + limit.get("nanos", 0) / 1e9
    if not 0 < seconds <= 5:
        return "time_limit"
    if row["input_file"] not in ("", "stdin", "standard input") or row[
        "output_file"
    ] not in ("", "stdout", "standard output"):
        return "file_io"
    if set(row["cf_tags"]) & {
        "*special",
        "constructive algorithms",
    } or AMBIGUOUS.search(row["description"]):
        return "non_unique_or_nonstandard_output"
    if not re.search(r"(?im)^\s*input\s*$", row["description"]) or not re.search(
        r"(?im)^\s*output\s*$", row["description"]
    ):
        return "missing_io_sections"
    return exclusions.check(row["name"], row["description"])


def paired_tests(value):
    if isinstance(value, list):
        return [(x["input"], x["output"]) for x in value]
    return list(zip(value["input"], value["output"]))


def select_tests(row, cap=40):
    public = list(dict.fromkeys(paired_tests(row["public_tests"])))
    public_set = set(public)
    hidden = list(
        dict.fromkeys(
            paired_tests(row["private_tests"]) + paired_tests(row["generated_tests"])
        )
    )
    hidden = [pair for pair in hidden if pair not in public_set]
    if len(public) + len(hidden) < 10 or not hidden:
        return None
    if len(hidden) > cap:
        ordered = sorted(hidden, key=lambda pair: len(pair[0].encode()))
        largest = ordered[-8:]
        remaining = ordered[:-8]
        rng = random.Random(1729)
        hidden = rng.sample(remaining, cap - len(largest)) + largest
    return public, hidden


def plain_statement(row, public):
    description = row["description"].replace("\r\n", "\n").replace("\xa0", " ")
    description = re.sub(r"\$+", "", description)
    description = re.sub(r"^Problem description\.\s*", "", description, flags=re.I)
    description = re.sub(
        r"(?im)^\s*(?:Examples?|Sample Input)\s*:?\s*$[\s\S]*", "", description
    ).strip()
    title = re.sub(r"^\d+[_ .-]*[A-Z]\d?\s*[-_.:]\s*", "", row["name"])
    text = title + "\n\n" + description
    if not re.search(r"(?im)^\s*constraints\s*$", text):
        text += "\n\nConstraints\nThe bounds are specified in the Input section above."
    for given, expected in public:
        text += (
            "\n\nSample Input\n"
            + given.strip()
            + "\n\nSample Output\n"
            + expected.strip()
        )
    return text + "\n"


def validate(row, output, references):
    pid = f"cf-{row['cf_contest_id']}-{row['cf_index'].lower()}"
    selected = select_tests(row)
    if selected is None:
        return pid, None, "too_few_tests"
    public, hidden = selected
    solutions = row["solutions"]
    codes = (
        [item["solution"] for item in solutions if item["language"] == 2]
        if isinstance(solutions, list)
        else [
            code
            for lang, code in zip(solutions["language"], solutions["solution"])
            if lang == 2
        ]
    )
    if not codes:
        return pid, None, "no_cpp"
    codes = [
        code
        for code in codes
        if not re.search(r"freopen\s*\(|ifstream|ofstream|system\s*\(", code)
    ]
    if not codes:
        return pid, None, "reference_file_io"
    code = min(codes, key=len)
    limit = row["time_limit"]["seconds"] + row["time_limit"]["nanos"] / 1e9
    with tempfile.TemporaryDirectory(dir=ROOT / "pool_work") as temp:
        temp = Path(temp)
        solution = temp / "reference.cpp"
        solution.write_text(code)
        tests = temp / "validation"
        tests.mkdir()
        for i, (given, expected) in enumerate(public + hidden):
            (tests / f"{i:03d}.in").write_text(given)
            (tests / f"{i:03d}.out").write_text(expected)
        try:
            result = grade(solution, tests, limit, 1024)
        except subprocess.TimeoutExpired:
            return pid, None, "compile_timeout"
        if result["score"] != 1:
            return pid, None, "reference_failed"
    directory = output / pid
    directory.mkdir(exist_ok=True)
    for name, pairs in [("samples", public), ("tests", hidden)]:
        (directory / name).mkdir(exist_ok=True)
        for i, (given, expected) in enumerate(pairs):
            (directory / name / f"{i:03d}.in").write_text(given)
            (directory / name / f"{i:03d}.out").write_text(expected)
    (directory / "statement.txt").write_text(plain_statement(row, public))
    (references / f"{pid}.cpp").write_text(code)
    metadata = {
        "id": pid,
        "source": "deepmind/code_contests",
        "source_split": "train",
        "platform": "Codeforces",
        "source_name": row["name"],
        "source_url": f"https://codeforces.com/contest/{row['cf_contest_id']}/problem/{row['cf_index']}",
        "rating": row["cf_rating"],
        "tags": row["cf_tags"],
        "categories": category(row["cf_tags"]),
        "n_tests": len(public) + len(hidden),
        "n_hidden_tests": len(hidden),
        "n_public_tests": len(public),
        "TL": limit,
        "time_limit": limit,
        "memory_mb": 1024,
        "split": "train",
        "reference_passed": result["total"],
        "reference_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "statement_sha256": hashlib.sha256(
            (directory / "statement.txt").read_bytes()
        ).hexdigest(),
    }
    write_json(directory / "problem.json", metadata)
    write_json(directory / "validation.json", result)
    return pid, metadata, None


def stats(manifest):
    return {
        "count": len(manifest),
        "tags": dict(
            collections.Counter(tag for item in manifest for tag in item["tags"])
        ),
        "categories": dict(
            collections.Counter(tag for item in manifest for tag in item["categories"])
        ),
        "rating_buckets": dict(
            collections.Counter(
                f"{(item['rating'] // 300) * 300}-{min((item['rating'] // 300) * 300 + 299, 2800)}"
                for item in manifest
            )
        ),
    }


def split_heldout(pool, heldout, count, exclude_ids=()):
    manifest = [json.loads(p.read_text()) for p in sorted(pool.glob("*/problem.json"))]
    if count > len(manifest):
        raise RuntimeError("Not enough validated problems for heldout")
    rng = random.Random(1729)
    groups = collections.defaultdict(list)
    for item in manifest:
        if item["id"] in exclude_ids:
            continue
        primary = next(
            (name for name in CATEGORIES if name in item["categories"]), "other"
        )
        groups[(item["rating"] // 300, primary)].append(item)
    chosen = []
    eligible_count = sum(len(items) for items in groups.values())
    if count > eligible_count:
        raise RuntimeError("Not enough eligible problems for heldout")
    allocations = {
        key: count * len(items) / eligible_count for key, items in groups.items()
    }
    amounts = {key: int(value) for key, value in allocations.items()}
    for key in sorted(
        groups, key=lambda key: (allocations[key] - amounts[key], key), reverse=True
    )[: count - sum(amounts.values())]:
        amounts[key] += 1
    for key, items in sorted(groups.items()):
        rng.shuffle(items)
        chosen.extend(items[: amounts[key]])
    heldout.mkdir(exist_ok=True)
    held_ids = {item["id"] for item in chosen}
    for item in chosen:
        item["split"] = "heldout"
        shutil.move(str(pool / item["id"]), heldout / item["id"])
        write_json(heldout / item["id"] / "problem.json", item)
    train = [item for item in manifest if item["id"] not in held_ids]
    write_json(pool / "manifest.json", train)
    write_json(heldout / "manifest.json", chosen)
    write_json(
        pool / "statistics.json", {"train": stats(train), "heldout": stats(chosen)}
    )
    return train, chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=640)
    parser.add_argument("--heldout", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-shards", type=int, default=39)
    parser.add_argument("--minutes", type=float, default=55)
    args = parser.parse_args()
    start = time.monotonic()
    pool = ROOT / "pool"
    references = ROOT / "pool_references"
    cache = ROOT / "pool_cache"
    for path in [pool, references, cache, ROOT / "pool_work"]:
        path.mkdir(exist_ok=True)
    if (ROOT / "pool_heldout/manifest.json").exists():
        raise RuntimeError("Pool already split; do not rebuild over heldout")
    exclusions = Exclusions()
    report_path = pool / "exclusion_report.json"
    report = (
        json.loads(report_path.read_text())
        if report_path.exists()
        else {"counts": {}, "dropped": [], "checked_candidates": 0}
    )
    counts = collections.Counter(report["counts"])
    report["protected_corpus_counts"] = dict(exclusions.inventory)
    report["protected_sha256"] = exclusions.fingerprints
    report["method"] = (
        "Normalized title equality or >=0.8 title word Jaccard (>=3 words), plus hashed 8-word shingles; drop >=4 shared and >=2% overlap coefficient or >=2 shared and >=8%; remove corpus-wide boilerplate (>=20%, minimum 5 documents). Also reject IOI mentions. Protected text is never written into pool."
    )
    completed = {
        p.parent.name: json.loads(p.read_text()) for p in pool.glob("*/problem.json")
    }
    for pid, item in list(completed.items()):
        if "constructive algorithms" in item["tags"]:
            shutil.rmtree(pool / pid)
            del completed[pid]
            counts["non_unique_or_nonstandard_output"] += 1
    attempted_path = cache / "attempted.json"
    attempted = (
        json.loads(attempted_path.read_text()) if attempted_path.exists() else {}
    )
    files = sorted(
        f
        for f in HfApi().list_repo_files("deepmind/code_contests", repo_type="dataset")
        if "data/train-" in f and f.endswith(".parquet")
    )
    fs = HfFileSystem()
    for index, filename in enumerate(files[: args.max_shards]):
        if (
            len(completed) >= args.target
            or time.monotonic() - start > args.minutes * 60
        ):
            break
        print(
            json.dumps(
                {"event": "shard_start", "shard": index, "validated": len(completed)}
            ),
            flush=True,
        )
        metadata_path = cache / f"meta-{index:02d}.json"
        if metadata_path.exists():
            rows = json.loads(metadata_path.read_text())
        else:
            with fs.open(f"datasets/deepmind/code_contests/{filename}", "rb") as remote:
                rows = pq.ParquetFile(remote).read(columns=META_COLUMNS).to_pylist()
            write_json(metadata_path, rows)
        accepted = []
        for row_index, row in enumerate(rows):
            pid = f"cf-{row['cf_contest_id']}-{row['cf_index'].lower()}"
            if pid in attempted or pid in completed:
                continue
            reason = filter_metadata(row, exclusions)
            report["checked_candidates"] += 1
            if reason:
                counts[reason] += 1
                if (
                    reason.startswith("protected")
                    or reason.startswith("similar_protected")
                    or reason == "ioi_mention"
                ):
                    report["dropped"].append({"id": pid, "reason": reason})
                continue
            accepted.append((row_index, row))
        report["counts"] = dict(counts)
        write_json(report_path, report)
        if not accepted:
            continue
        print(
            json.dumps(
                {
                    "event": "fetch_test_solution_columns",
                    "shard": index,
                    "eligible": len(accepted),
                }
            ),
            flush=True,
        )
        with fs.open(f"datasets/deepmind/code_contests/{filename}", "rb") as remote:
            table = pq.ParquetFile(remote).read(columns=DATA_COLUMNS)
        category_counts = collections.Counter(
            cat for item in completed.values() for cat in item["categories"]
        )

        def priority(pair):
            _, row = pair
            cats = category(row["cf_tags"])
            score = sum(
                (
                    2.5
                    if cat == "data_structures"
                    else 2.0
                    if cat == "combinatorics_math"
                    else 1.0
                )
                / (10 + category_counts[cat])
                for cat in cats
            )
            return -score, hashlib.sha256(row["name"].encode()).hexdigest()

        accepted.sort(key=priority)
        futures = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            for row_index, row in accepted[: max(0, args.target - len(completed)) + 30]:
                full = row | table.slice(row_index, 1).to_pylist()[0]
                futures.append(executor.submit(validate, full, pool, references))
            for future in concurrent.futures.as_completed(futures):
                pid, item, reason = future.result()
                attempted[pid] = reason or "passed"
                if item:
                    completed[pid] = item
                else:
                    counts[reason] += 1
                write_json(attempted_path, attempted)
                write_json(pool / "manifest.json", list(completed.values()))
                report["counts"] = dict(counts)
                write_json(report_path, report)
                print(
                    json.dumps(
                        {
                            "event": "validation",
                            "id": pid,
                            "result": reason or "passed",
                            "validated": len(completed),
                        }
                    ),
                    flush=True,
                )
        del table
    train, heldout = split_heldout(pool, ROOT / "pool_heldout", args.heldout)
    report["final_train_count"] = len(train)
    report["final_heldout_count"] = len(heldout)
    report["elapsed_seconds"] = time.monotonic() - start
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "event": "complete",
                "train": stats(train),
                "heldout": stats(heldout),
                "elapsed_seconds": report["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
