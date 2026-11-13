"""Freeze train-only cohorts, render six components, and audit a 55% IOI mix."""

import argparse
import collections
import datetime
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from stage6_mix import allocate
from submission_filter import final_scoring_submission

CODE = Path(__file__).resolve().parents[1]
BASE = "Qwen/Qwen3.5-9B"
MIX = (
    "tinker://609385aa-c3bd-5b70-99a0-56fd62c64493:train:0/sampler_weights/ioi2-v1-mix"
)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2))


def freeze(root, cutoff):
    train = {row["id"] for row in read(root / "pool/manifest.json")}
    heldout = {row["id"] for row in read(root / "pool_heldout/manifest.json")}
    if train & heldout:
        raise ValueError("Training and held-out pool overlap")
    historical = {row["id"] for row in read(root / "ioi_train/manifest.json")}
    synthetic = {row["id"] for row in read(root / "synthetic_ioi/manifest.json")}
    plans = {
        "syn-teacher": {
            "runs": ["syn-ioi-teacher-1"],
            "ids": synthetic,
            "minimum": 100,
            "ioi": True,
            "flags": [
                "--selection",
                "shortest",
                "--max-per-problem",
                "2",
                "--max-ioi-per-problem",
                "2",
            ],
        },
        "syn-onpolicy": {
            "runs": ["syn-ioi-v1mix-1"],
            "ids": synthetic,
            "minimum": 100,
            "ioi": True,
            "flags": [
                "--max-per-problem",
                "1000000",
                "--max-ioi-per-problem",
                "1000000",
            ],
        },
        "historical-teacher": {
            "runs": ["ioi-teacher-1", "ioi-teacher-2"],
            "ids": historical,
            "minimum": 0,
            "ioi": True,
            "flags": [
                "--positive-ioi-score",
                "--selection",
                "score-shortest",
                "--max-ioi-per-problem",
                "4",
                "--max-per-problem",
                "4",
            ],
        },
        "historical-onpolicy": {
            "runs": ["ioi-v1mix-rft"],
            "ids": historical,
            "minimum": 0,
            "ioi": True,
            "flags": [
                "--positive-ioi-score",
                "--selection",
                "score-shortest",
                "--max-ioi-per-problem",
                "2",
                "--max-per-problem",
                "2",
            ],
        },
        "efficient-teacher": {
            "runs": ["coll-teacher-1"],
            "ids": train,
            "minimum": 1,
            "ioi": False,
            "flags": [
                "--selection",
                "shortest",
                "--max-per-problem",
                "1",
                "--max-first-submission-tokens",
                "8000",
                "--max-turn-output-tokens",
                "8000",
            ],
        },
        "olympiad-onpolicy": {
            "runs": [],
            "ids": train,
            "minimum": 1,
            "ioi": False,
            "flags": ["--max-per-problem", "1000000"],
        },
    }
    for run in sorted((root / "runs").iterdir()):
        config = run / "run_config.json"
        if config.is_file() and read(config).get("checkpoint") in {BASE, MIX}:
            plans["olympiad-onpolicy"]["runs"].append(run.name)
    for plan in plans.values():
        cohorts = []
        rejected = collections.Counter()
        for name in plan["runs"]:
            for path in sorted((root / "runs" / name).glob("*/score.json")):
                if path.stat().st_mtime > cutoff:
                    rejected["completed_after_freeze"] += 1
                    continue
                score = read(path)
                pid = score["task_id"]
                if pid not in plan["ids"] or pid in heldout:
                    rejected["not_eligible_train_task"] += 1
                    continue
                if score["score"] < plan["minimum"] or (
                    plan["ioi"] and score["score"] <= 0
                ):
                    rejected["score"] += 1
                    continue
                final = None
                if plan["ioi"]:
                    try:
                        final = final_scoring_submission(path.parent)
                    except (ValueError, KeyError, OSError):
                        rejected["unverifiable_final_scoring_submission"] += 1
                        continue
                    if final["defines_main"]:
                        rejected["final_scoring_submission_defines_main"] += 1
                        continue
                cohorts.append(
                    {
                        "trajectory": str(path.parent.resolve()),
                        "score_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "task_id": pid,
                        "score": score["score"],
                        "final_scoring_submission": final,
                    }
                )
        plan["runs"] = sorted({Path(row["trajectory"]).parent.name for row in cohorts})
        plan.update(cohort=cohorts, rejected=dict(rejected))
        del plan["ids"]
    return plans


def audit_record(row):
    size = len(row["input_ids"])
    if (
        not 0 < size < 65536
        or len(row["target_ids"]) != size
        or len(row["weights"]) != size
        or row["sequence_tokens"] != size + 1
    ):
        raise ValueError("Invalid record shape")
    expected = [0] * size
    for span in row["assistant_spans"]:
        start, end = span["start"] - 1, span["end"] - 1
        if not 0 <= start < end <= size:
            raise ValueError("Invalid assistant span")
        expected[start:end] = [1] * (end - start)
    if (
        expected != row["weights"]
        or sum(expected) != row["target_tokens"]
        or not any(expected)
    ):
        raise ValueError("Loss mask is not exactly the assistant spans")
    if row["input_ids"][1:] != row["target_ids"][:-1]:
        raise ValueError("Shifted token targets disagree")


def balance(ioi_rows, other_rows, fraction=0.55):
    def grouped(rows):
        groups = collections.defaultdict(list)
        for row in rows:
            audit_record(row)
            groups[row["trajectory"]].append(row)
        return list(groups.values())

    left, right = grouped(ioi_rows), grouped(other_rows)
    a = sum(row["target_tokens"] for row in ioi_rows)
    b = sum(row["target_tokens"] for row in other_rows)
    if not a or not b:
        raise ValueError("Both IOI and olympiad sides need valid data")
    if a / (a + b) <= fraction:
        chosen = allocate(left, b, fraction)
        return other_rows + [row for index in chosen for row in left[index]]
    chosen = allocate(right, a, 1 - fraction)
    return ioi_rows + [row for index in chosen for row in right[index]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(os.environ.get("IOI2_ROOT", CODE))
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--freeze-at", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    cutoff = datetime.datetime.fromisoformat(args.freeze_at).timestamp()
    directory = root / "train" / (args.name + ".data")
    directory.mkdir(parents=True, exist_ok=args.resume)
    frozen = directory / "freeze.json"
    if frozen.exists():
        saved = read(frozen)
        if saved["freeze_at"] != args.freeze_at:
            raise ValueError("Resume freeze timestamp does not match the saved cohort")
        plans = saved["components"]
    else:
        plans = freeze(root, cutoff)
        write(frozen, {"freeze_at": args.freeze_at, "components": plans})
    all_rows, ioi_rows, other_rows = [], [], []
    components = {}
    python = os.environ.get("IOI2_TRAIN_PYTHON", str(CODE / ".venv/bin/python"))
    commands = []
    for name, plan in plans.items():
        cohort = directory / (name + ".cohort.jsonl")
        if not cohort.exists():
            cohort.write_text("".join(json.dumps(row) + "\n" for row in plan["cohort"]))
        for row in plan["cohort"]:
            path = Path(row["trajectory"]) / "score.json"
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["score_sha256"]:
                raise ValueError("Frozen score changed")
        output = directory / (name + ".jsonl")
        if not plan["cohort"]:
            components[name] = {
                "datums": 0,
                "cohort_candidates": 0,
                "rejected_at_freeze": plan["rejected"],
            }
            continue
        command = [
            python,
            str(CODE / "train/build_sft.py"),
            "--output",
            str(output),
            "--cohort-jsonl",
            str(cohort),
            "--mode",
            "trajectory",
            "--min-ioi-score",
            str(plan["minimum"] if plan["ioi"] else 100),
            *plan["flags"],
        ]
        for run in plan["runs"]:
            command += ["--run-dir", str(root / "runs" / run)]
        commands.append(command)
        stats_path = output.with_suffix(".stats.json")
        if not (args.resume and output.exists() and stats_path.exists()):
            with (directory / (name + ".build.log")).open("w") as log:
                subprocess.run(
                    command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True
                )
        stats = read(stats_path)
        stats.update(
            cohort_candidates=len(plan["cohort"]), rejected_at_freeze=plan["rejected"]
        )
        components[name] = stats
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        (ioi_rows if plan["ioi"] else other_rows).extend(rows)
    write(directory / "commands.json", commands)
    all_rows = balance(ioi_rows, other_rows)
    ioi_trajectories = {row["trajectory"] for row in ioi_rows}
    stats = {
        "components": components,
        "freeze_at": args.freeze_at,
        "datums": len(all_rows),
        "unique_trajectories": len({row["trajectory"] for row in all_rows}),
        "sequence_tokens": sum(row["sequence_tokens"] for row in all_rows),
        "target_tokens": sum(row["target_tokens"] for row in all_rows),
        "all_masks_verified": True,
        "sources": {},
    }
    for row in all_rows:
        audit_record(row)
        source = stats["sources"].setdefault(
            row["source"], {"datums": 0, "sequence_tokens": 0, "target_tokens": 0}
        )
        source["datums"] += 1
        source["sequence_tokens"] += row["sequence_tokens"]
        source["target_tokens"] += row["target_tokens"]
    stats["ioi_target_fraction"] = (
        sum(
            row["target_tokens"]
            for row in all_rows
            if row["trajectory"] in ioi_trajectories
        )
        / stats["target_tokens"]
    )
    stats["warmup_steps_2ep"] = max(1, round(math.ceil(len(all_rows) / 16) * 2 * 0.05))
    output = root / "train" / (args.name + ".jsonl")
    with output.open("x") as stream:
        for row in all_rows:
            stream.write(json.dumps(row) + "\n")
    write(output.with_suffix(".stats.json"), stats)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
