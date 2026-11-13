"""Budget diagnostics from native completed-response usage and tool calls."""

import argparse
import collections
import datetime
import json
import statistics
from pathlib import Path


def trajectory_metrics(directory):
    turns = []
    for path in sorted(directory.glob("logs/**/turns/turn_*/result.json")):
        data = json.loads(path.read_text())
        query = data.get("query_result")
        if query:
            turns.append(query)
    total = 0
    maximum = 0
    seconds = 0.0
    submissions = 0
    first_tokens = None
    first_seconds = None
    start = None
    missing = 0
    for query in turns:
        metadata = query.get("metadata") or {}
        count = metadata.get("total_output_tokens", metadata.get("out_tokens"))
        if count is None:
            missing += 1
            count = 0
        duration = metadata.get("duration_seconds") or 0
        completed = metadata.get("completed_at")
        completed = (
            datetime.datetime.fromisoformat(completed).timestamp()
            if completed
            else None
        )
        if start is None and completed is not None:
            start = completed - duration
        total += count
        maximum = max(maximum, count)
        seconds += duration
        calls = sum(
            call.get("name") == "submission" for call in query.get("tool_calls") or []
        )
        if calls and first_tokens is None:
            first_tokens = total
            first_seconds = (
                completed - start
                if completed is not None and start is not None
                else None
            )
        submissions += calls
    return {
        "turns": len(turns),
        "output_tokens": total,
        "max_turn_output_tokens": maximum,
        "first_submission_tokens": first_tokens,
        "first_submission_seconds": first_seconds,
        "submissions": submissions,
        "query_seconds": seconds,
        "output_tokens_per_query_second": total / seconds if seconds else None,
        "missing_usage_turns": missing,
    }


def summarize(rows):
    result = {
        "runs": len(rows),
        "zero_submission_runs": sum(not row["submissions"] for row in rows),
    }
    for key in [
        "turns",
        "output_tokens",
        "max_turn_output_tokens",
        "first_submission_tokens",
        "first_submission_seconds",
        "submissions",
    ]:
        values = [row[key] for row in rows if row[key] is not None]
        result["mean_" + key] = statistics.mean(values) if values else None
    duration = sum(row["query_seconds"] for row in rows)
    result["output_tokens_per_query_second"] = (
        sum(row["output_tokens"] for row in rows) / duration if duration else None
    )
    result["max_turn_output_tokens"] = max(
        (row["max_turn_output_tokens"] for row in rows), default=None
    )
    return result


def distribution(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": values[0],
        "p25": values[int(0.25 * (len(values) - 1))],
        "median": statistics.median(values),
        "p75": values[int(0.75 * (len(values) - 1))],
        "p90": values[int(0.9 * (len(values) - 1))],
        "p95": values[int(0.95 * (len(values) - 1))],
        "max": values[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("runs"))
    args = parser.parse_args()
    result = {
        "definition": "Completed-response tokens; first submission includes its assistant turn. Seconds from first LLM request start to submission response completion; excludes initial worker setup. Missing unfinished-response tokens are unknown. Rate is output tokens / summed LLM API seconds, including TTFT, excluding tool time.",
        "runs": {},
        "trajectories": {},
    }
    for name in ["dev-base", "dev-step30", "dev-v0", "ioi-teacher-1", "coll-teacher-1"]:
        tasks = collections.defaultdict(list)
        selected = []
        for path in sorted((args.root / name).glob("*/score.json")):
            score = json.loads(path.read_text())
            if name == "coll-teacher-1" and score["score"] != 1:
                continue
            metrics = trajectory_metrics(path.parent)
            result["trajectories"][str(path.parent.resolve())] = metrics
            tasks[score["task_id"]].append(metrics)
            selected.append(metrics)
        result["runs"][name] = {
            "overall": summarize(selected),
            "per_task": {task: summarize(rows) for task, rows in tasks.items()},
        }
        if name == "coll-teacher-1":
            result["teacher_olympiad_full"] = {
                "first_submission_tokens": distribution(
                    row["first_submission_tokens"] for row in selected
                ),
                "max_turn_output_tokens": distribution(
                    row["max_turn_output_tokens"] for row in selected
                ),
                "no_submission": sum(not row["submissions"] for row in selected),
            }
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {name: value["overall"] for name, value in result["runs"].items()}, indent=2
        )
    )
    print(json.dumps(result["teacher_olympiad_full"], indent=2))


if __name__ == "__main__":
    main()
