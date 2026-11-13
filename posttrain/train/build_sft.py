"""Build target-turn-only SFT records from successful local agent trajectories."""

import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path

from tinker_cookbook.tokenizer_utils import get_tokenizer

from rendering import EndpointQwen35Renderer, merge_turns, render_turn, tool_specs
from trajectory_metrics import trajectory_metrics

ROOT = Path(os.environ.get("IOI2_ROOT", str(Path(__file__).resolve().parents[1])))


def retain_candidate(group, item, cap):
    if cap < 1:
        raise ValueError("Trajectory cap must be positive")
    duplicate = next(
        (row for row in group if item[4] is not None and row[4] == item[4]), None
    )
    if duplicate:
        if duplicate[0] <= item[0]:
            return "duplicate_trajectory"
        group.remove(duplicate)
    group.append(item)
    group.sort(key=lambda row: row[0])
    if len(group) > cap:
        group.pop()
        return "per_problem_cap"
    return "duplicate_trajectory" if duplicate else None


def read(path):
    return json.loads(path.read_text())


def candidates(
    runs, minimum, maximum, heldout, minimum_ioi=100, maximum_ioi=1, positive_ioi=False
):
    by_problem = collections.Counter()
    rejected = collections.Counter()
    kept = []
    eligible = []
    for run in runs:
        config = (
            read(run / "run_config.json") if (run / "run_config.json").is_file() else {}
        )
        paths = (
            [run / "score.json"]
            if (run / "score.json").is_file()
            else sorted(run.glob("*/score.json"))
        )
        for path in paths:
            try:
                score = read(path)
                pid = score["task_id"]
                is_ioi = score.get("track", config.get("track")) == "ioi" or bool(
                    re.match(r"^20\d\d[/_]", pid)
                )
                year = re.match(r"^(20\d\d)[/_]", pid)
                result = (
                    read(path.parent / "result.json")
                    if (path.parent / "result.json").exists()
                    else {}
                )
                final_error = result.get("final_error") or {}
                if is_ioi and year and int(year[1]) >= 2024:
                    rejected["heldout_ioi"] += 1
                elif pid in heldout:
                    rejected["heldout"] += 1
                elif (is_ioi and positive_ioi and score["score"] <= 0) or score[
                    "score"
                ] < (minimum_ioi if is_ioi else minimum):
                    rejected["score"] += 1
                elif (
                    final_error
                    and final_error.get("type")
                    not in ("MaxTurnsExceeded", "MaxTimeExceeded")
                ) or score.get("returncode") not in (
                    None,
                    0,
                    -9,
                ):
                    rejected["terminal_error"] += 1
                else:
                    eligible.append(
                        (path.parent, run.name, pid, is_ioi, score["score"])
                    )
            except (OSError, ValueError, KeyError):
                rejected["invalid_score"] += 1
    ordered = [row for row in eligible if not row[3]] + sorted(
        (row for row in eligible if row[3]), key=lambda row: -row[4]
    )
    for directory, source, pid, is_ioi, _ in ordered:
        if by_problem[pid] >= (maximum_ioi if is_ioi else maximum):
            rejected["per_problem_cap"] += 1
        else:
            kept.append((directory, source, pid))
            by_problem[pid] += 1
    return kept, rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=1.0)
    parser.add_argument("--min-ioi-score", type=float, default=100)
    parser.add_argument("--max-ioi-per-problem", type=int, default=1)
    parser.add_argument("--max-per-problem", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--mode", choices=["trajectory", "turn"], default="trajectory")
    parser.add_argument(
        "--selection", choices=["first", "shortest", "score-shortest"], default="first"
    )
    parser.add_argument("--positive-ioi-score", action="store_true")
    parser.add_argument("--distinct", action="store_true")
    parser.add_argument("--max-first-submission-tokens", type=int)
    parser.add_argument("--max-turn-output-tokens", type=int)
    parser.add_argument("--max-trajectory-tokens", type=int)
    parser.add_argument("--cohort-jsonl", type=Path)
    args = parser.parse_args()
    if args.distinct and args.selection == "first":
        parser.error("Distinct selection requires shortest or score-shortest")
    if (
        not 0 <= args.min_score <= 1
        or not 0 <= args.min_ioi_score <= 100
        or args.max_ioi_per_problem < 1
        or args.max_per_problem < 1
        or args.max_tokens < 2
    ):
        parser.error("Invalid score, trajectory cap, or token limit")
    if args.output.exists():
        parser.error("Output already exists; use a new filename for each build")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    heldout_path = ROOT / "pool_heldout/manifest.json"
    heldout = (
        {item["id"] for item in read(heldout_path)} if heldout_path.is_file() else set()
    )
    kept, rejected = candidates(
        args.run_dir,
        args.min_score,
        2**31 if args.selection != "first" else args.max_per_problem,
        heldout,
        args.min_ioi_score,
        2**31 if args.selection == "score-shortest" else args.max_ioi_per_problem,
        args.positive_ioi_score,
    )
    if args.cohort_jsonl:
        with args.cohort_jsonl.open() as cohort:
            paths = {
                str(Path(json.loads(line)["trajectory"]).resolve()) for line in cohort
            }
        kept = [row for row in kept if str(row[0].resolve()) in paths]
    renderer = EndpointQwen35Renderer(get_tokenizer(args.model))
    stats = {
        "renderer": "qwen3_5",
        "mode": args.mode,
        "selection": args.selection,
        "distinct": args.distinct,
        "positive_ioi_score": args.positive_ioi_score,
        "max_first_submission_tokens": args.max_first_submission_tokens,
        "max_turn_output_tokens": args.max_turn_output_tokens,
        "single_segment_trajectories": 0,
        "split_trajectories": 0,
        "prefix_verified_turns": 0,
        "verified_candidate_sequence_tokens": 0,
        "history_thinking": "HF endpoint gate: strip at/before last real user, preserve current tool chain",
        "tokenizer_model": args.model,
        "min_score": args.min_score,
        "min_ioi_score": args.min_ioi_score,
        "max_ioi_per_problem": args.max_ioi_per_problem,
        "max_per_problem": args.max_per_problem,
        "max_tokens": args.max_tokens,
        "trajectories_selected": len(kept),
        "trajectories": 0,
        "datums": 0,
        "tokens": 0,
        "target_tokens": 0,
        "per_source": {},
        "rejected": dict(rejected),
        "skipped_turns": {},
        "prompt_token_check": {
            "checked": 0,
            "matched": 0,
            "missing_usage": 0,
            "mismatches": [],
        },
    }
    skipped = collections.Counter()
    sources = collections.defaultdict(
        lambda: {"trajectories": 0, "datums": 0, "tokens": 0, "target_tokens": 0}
    )
    selected = collections.defaultdict(list)
    with args.output.open("w") as stream:

        def emit(directory, source, pid, records):
            for record in records:
                record.update(source=source, problem_id=pid, trajectory=str(directory))
                stream.write(json.dumps(record) + "\n")
                stats["datums"] += 1
                for key, amount in [
                    ("datums", 1),
                    ("tokens", record["sequence_tokens"]),
                    ("target_tokens", record["target_tokens"]),
                ]:
                    sources[source][key] += amount
                stats["tokens"] += record["sequence_tokens"]
                stats["target_tokens"] += record["target_tokens"]
            if records:
                stats["trajectories"] += 1
                sources[source]["trajectories"] += 1

        for directory, source, pid in kept:
            if args.max_first_submission_tokens or args.max_turn_output_tokens:
                metrics = trajectory_metrics(directory)
                if metrics["missing_usage_turns"]:
                    skipped["missing_efficiency_usage"] += 1
                    continue
                first = metrics["first_submission_tokens"]
                if args.max_first_submission_tokens and (
                    first is None or first > args.max_first_submission_tokens
                ):
                    skipped["late_first_submission"] += 1
                    continue
                if (
                    args.max_turn_output_tokens
                    and metrics["max_turn_output_tokens"] > args.max_turn_output_tokens
                ):
                    skipped["long_assistant_turn"] += 1
                    continue
            configs = sorted(directory.glob("logs/**/turns/init/config.json"))
            histories = sorted(directory.glob("logs/**/turns/turn_*/history.json"))
            if not configs or not histories:
                skipped["missing_native_history"] += 1
                continue
            specs = tool_specs(read(configs[-1])["tools"])
            records = []
            invalid = False
            for path in histories:
                try:
                    record = render_turn(renderer, read(path), specs)
                    if (
                        args.mode == "turn"
                        and record["sequence_tokens"] > args.max_tokens
                    ):
                        skipped["over_length"] += 1
                        continue
                    native = read(path.parent / "result.json")
                    metadata = native.get("query_result", {}).get("metadata") or {}
                    reported = metadata.get("total_input_tokens")
                    if reported is None and metadata.get("in_tokens") is not None:
                        reported = metadata["in_tokens"] + (
                            metadata.get("cache_read_tokens") or 0
                        )
                    if reported is not None:
                        check = stats["prompt_token_check"]
                        check["checked"] += 1
                        match = reported == record["prompt_tokens"]
                        check["matched"] += match
                        if not match:
                            check["mismatches"].append(
                                {
                                    "problem": pid,
                                    "source": source,
                                    "turn": path.parent.name,
                                    "rendered": record["prompt_tokens"],
                                    "reported": reported,
                                }
                            )
                            skipped["prompt_token_mismatch"] += 1
                            invalid = True
                            continue
                    else:
                        stats["prompt_token_check"]["missing_usage"] += 1
                    record.update(
                        turn=path.parent.name, reported_prompt_tokens=reported
                    )
                    records.append(record)
                except (ValueError, KeyError, OSError) as error:
                    skipped[type(error).__name__ + ":" + str(error)[:120]] += 1
                    invalid = True
            if args.mode == "trajectory":
                if invalid:
                    skipped["invalid_trajectory"] += 1
                    continue
                records = merge_turns(records)
                stats["prefix_verified_turns"] += sum(
                    row["prefix_verified_turns"] for row in records
                )
                stats["single_segment_trajectories"] += len(records) == 1
                stats["split_trajectories"] += len(records) > 1
            cost = sum(row["sequence_tokens"] for row in records)
            stats["verified_candidate_sequence_tokens"] += cost
            if args.max_trajectory_tokens and cost > args.max_trajectory_tokens:
                skipped["trajectory_over_length"] += 1
                continue
            if args.mode == "trajectory":
                skipped["over_length"] += sum(
                    row["sequence_tokens"] > args.max_tokens for row in records
                )
                records = [
                    row for row in records if row["sequence_tokens"] <= args.max_tokens
                ]
            if not records:
                continue
            score = read(directory / "score.json")["score"]
            rank = (-score, cost) if args.selection == "score-shortest" else (cost,)
            identity = None
            if args.distinct:
                digest = hashlib.sha256()
                for row in records:
                    generated = [
                        token
                        for token, weight in zip(row["target_ids"], row["weights"])
                        if weight
                    ]
                    digest.update(json.dumps(generated).encode())
                identity = digest.hexdigest()
            if args.selection in ("shortest", "score-shortest"):
                group = selected[pid]
                cap = (
                    args.max_ioi_per_problem
                    if re.match(r"^20\d\d[/_]", pid)
                    else args.max_per_problem
                )
                reason = retain_candidate(
                    group, (rank, directory, source, records, identity), cap
                )
                if reason:
                    rejected[reason] += 1
            else:
                emit(directory, source, pid, records)
        for pid, group in selected.items():
            for _, directory, source, records, _ in group:
                emit(directory, source, pid, records)
    stats["rejected"] = dict(rejected)
    stats["per_source"] = dict(sources)
    stats["mean_tokens_per_datum"] = (
        stats["tokens"] / stats["datums"] if stats["datums"] else None
    )
    stats["skipped_turns"] = dict(skipped)
    check = stats["prompt_token_check"]
    check["match_rate"] = (
        check["matched"] / check["checked"] if check["checked"] else None
    )
    args.output.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)
    if not stats["datums"]:
        raise SystemExit("No datums emitted")


if __name__ == "__main__":
    main()
