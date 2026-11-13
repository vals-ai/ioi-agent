"""Classify retained IOI submission diagnostics without exporting their code."""

import argparse
import collections
import json
import os
import re
from pathlib import Path

ROOT = Path(os.environ.get("IOI2_ROOT", str(Path(__file__).resolve().parent)))
RUNS = [
    "dev-base",
    "dev-base-2",
    "dev-step30",
    "dev-step30-2",
    "dev-v0",
    "dev-v1-ioi",
    "dev-v1-mix",
    "dev-v1-mix-lowlr",
]
EVENT = re.compile(
    r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+ (?:DEBUG|INFO|WARNING|ERROR|CRITICAL) (.*?)(?=^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+ (?:DEBUG|INFO|WARNING|ERROR|CRITICAL) |\Z)",
    re.M | re.S,
)


def blocks(text):
    stdout = stderr = ""
    result = {}
    for match in EVENT.finditer(text):
        message = match.group(1)
        if message.startswith("Test script stdout:"):
            stdout = message.partition(":")[2]
            stderr = ""
        elif message.startswith("Test script stderr:"):
            stderr = message.partition(":")[2]
        elif found := re.match(
            r"Submission #(\d+) - Score: (\d+) - Best: (\d+)", message
        ):
            number, score, best = map(int, found.groups())
            result[number] = {
                "stdout": stdout,
                "stderr": stderr,
                "score": score,
                "best": best,
            }
            stdout = stderr = ""
    return result


def classify(stdout, stderr, points, complete, code, required):
    text = stdout + "\n" + stderr
    lines = code.split("\n")
    fences = [i for i, line in enumerate(lines) if "```" in line]
    if len(fences) >= 2:
        extracted = "\n".join(lines[fences[-2] + 1 : fences[-1]])
        if extracted.strip():
            code = extracted
    compile_error = bool(
        re.search(
            r"failed to compile solution|compilation failed|undefined reference|multiple definition|collect2:.*error|(?:\.cpp|\.h):[^\n]*\berror:",
            text,
            re.I,
        )
    )
    main = bool(
        re.search(
            r"\bmain\s*\([^)]*\)\s*(?:->[^{}]+)?\{",
            re.sub(r"//[^\n]*|/\*.*?\*/", "", code, flags=re.S),
        )
    )
    missing_include = bool(
        re.search(
            r"fatal error:[^\n]*(?:No such file|file not found)|did you forget to.*#include|is defined in header",
            text,
            re.I,
        )
    )
    signature = bool(
        re.search(r"conflicting declaration|ambiguating new declaration", text, re.I)
    ) or any(
        re.search(r"undefined reference[^\n]*\b" + re.escape(name) + r"\s*\(", text)
        for name in required
    )
    main_error = bool(
        re.search(
            r"multiple definition[^\n]*\bmain\b|redefinition[^\n]*\bmain\b|conflicting declaration[^\n]*\bmain\b",
            text,
            re.I,
        )
    )
    style = (
        "main"
        if compile_error and main_error
        else "include"
        if compile_error and missing_include
        else "signature"
        if compile_error and signature
        else "other"
    )
    compiled = "Passed tests:" in stdout or points > 0
    if compile_error:
        verdict = "compile_error"
        compiled = False
    elif points > 0:
        verdict = "full" if points == 100 else "partial"
    elif re.search(
        r"Segmentation fault|SIGSEGV|SIGABRT|Assertion[^\n]*failed|Floating point exception|terminate called|std::bad_alloc",
        text,
        re.I,
    ):
        verdict = "runtime_error"
    elif re.search(r"timed out|time limit exceeded|\bTLE\b", text, re.I):
        verdict = "tle"
    elif "Solution execution failed" in stdout:
        verdict = "execution_failure_unspecified"
    elif "Checker execution failed" in stdout:
        verdict = "checker_failure_unspecified"
    elif compiled:
        verdict = "wa"
    elif not complete:
        verdict = "pending_or_interrupted"
    else:
        verdict = "unknown"
    status = (
        "link_compatible"
        if compiled
        else "missing_or_wrong"
        if signature
        else "not_established"
    )
    return {
        "verdict": verdict,
        "compiled": compiled,
        "compile_style": style if compile_error else None,
        "defines_main": main,
        "main_link_error": main_error,
        "signature_link_error": signature,
        "signature_status": status,
        "missing_include": missing_include,
    }


def object_output(record):
    value = (record.get("tool_output") or {}).get("output")
    try:
        value = json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def audit_run(name):
    rows, score_audit = [], []
    for directory in sorted((ROOT / "runs" / name).glob("*/score.json")):
        directory = directory.parent
        logs = list(directory.glob("logs/**/agent.log"))
        graded = {}
        for log in logs:
            graded.update(blocks(log.read_text()))
        task = directory.name.replace("_", "/", 1)
        required = set()
        for header in (
            ROOT / "runs" / name / directory.name / "workspace/graders"
        ).glob("*.h"):
            required.update(
                re.findall(
                    r"\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*;",
                    header.read_text(errors="replace"),
                )
            )
        best, single_scores = {}, []
        used = set()
        for path in sorted(directory.glob("logs/**/turns/turn_*/result.json")):
            data = json.loads(path.read_text())
            records = {
                str(r["tool_call"].get("call_id") or r["tool_call"].get("id")): r
                for r in data.get("tool_call_records", [])
            }
            for call in (data.get("query_result") or {}).get("tool_calls") or []:
                if call.get("name") != "submission":
                    continue
                key = str(call.get("call_id") or call.get("id"))
                record = records.get(key)
                output = object_output(record) if record else {}
                for subtask, points in output.get("best_subtask_scores", {}).items():
                    best[subtask] = max(best.get(subtask, 0), points)
                if "total_score" in output:
                    single_scores.append(output["total_score"])
                number = output.get("submission_count")
                block = graded.get(number, {})
                if block:
                    used.add(number)
                args = call.get("parsed_args") or call.get("args") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                code = args.get("cpp_code", "") if isinstance(args, dict) else ""
                row = {
                    "run": name,
                    "task": task,
                    "turn": path.parent.name,
                    "call_id": key,
                    "submission_number": number,
                    "points": output.get("total_score", block.get("score", 0)),
                    "completed": record is not None,
                    "diagnostics_found": bool(block),
                }
                row.update(
                    classify(
                        block.get("stdout", ""),
                        block.get("stderr", ""),
                        row["points"],
                        row["completed"],
                        code,
                        required,
                    )
                )
                if "Maximum submissions" in output.get("error", ""):
                    row["verdict"] = "submission_limit"
                rows.append(row)
        recorded = json.loads((directory / "score.json").read_text())["score"]
        score_audit.append(
            {
                "task": task,
                "recorded": recorded,
                "best_completed_subtasks": sum(best.values()),
                "max_single_submission": max(single_scores, default=0),
                "last_single_submission": single_scores[-1] if single_scores else None,
                "max_logged_best": max((b["best"] for b in graded.values()), default=0),
                "unmatched_log_blocks": sorted(set(graded) - used),
            }
        )
    return rows, score_audit


def main(output):
    result = {"runs": {}, "models": {}}
    for name in RUNS:
        rows, scoring = audit_run(name)
        result["runs"][name] = {"submissions": rows, "scoring": scoring}
        model = name.removeprefix("dev-").removesuffix("-2")
        summary = result["models"].setdefault(
            model,
            {
                "submissions": 0,
                "completed": 0,
                "compiled": 0,
                "verdicts": collections.Counter(),
                "compile_styles": collections.Counter(),
                "signature_status": collections.Counter(),
                "main_definitions": 0,
            },
        )
        for row in rows:
            summary["submissions"] += 1
            summary["completed"] += row["completed"]
            summary["compiled"] += row["compiled"]
            summary["main_definitions"] += row["defines_main"]
            summary["signature_status"][row["signature_status"]] += 1
            summary["verdicts"][row["verdict"]] += 1
            if row["compile_style"]:
                summary["compile_styles"][row["compile_style"]] += 1
    for summary in result["models"].values():
        known = summary["compiled"] + summary["verdicts"]["compile_error"]
        summary["compiled_fraction_known"] = (
            summary["compiled"] / known if known else None
        )
        summary["compiled_fraction_all_issued_lower_bound"] = (
            summary["compiled"] / summary["submissions"]
            if summary["submissions"]
            else None
        )
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["models"], indent=2))
    print(
        "SCORING_MISMATCHES",
        [
            (name, x)
            for name, run in result["runs"].items()
            for x in run["scoring"]
            if x["recorded"] != x["best_completed_subtasks"]
            or x["recorded"] != x["max_logged_best"]
        ],
    )
    print(
        "UNMATCHED_LOG_BLOCKS",
        sum(
            len(x["unmatched_log_blocks"])
            for run in result["runs"].values()
            for x in run["scoring"]
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
