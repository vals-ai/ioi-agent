"""Identify the last completed IOI scoring call without using dev assets."""

import json
import re


def defines_main(code):
    text = re.sub(
        r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
        " ",
        code,
        flags=re.S,
    )
    return bool(re.search(r"\bmain\s*\(|#\s*define\s+\w+\s+main\b", text))


def final_scoring_submission(directory):
    final = None
    for path in sorted(directory.glob("logs/**/turns/turn_*/result.json")):
        data = json.loads(path.read_text())
        records = {
            str(
                record["tool_call"].get("call_id") or record["tool_call"].get("id")
            ): record
            for record in data.get("tool_call_records", [])
        }
        for call in (data.get("query_result") or {}).get("tool_calls") or []:
            if call.get("name") != "submission":
                continue
            key = str(call.get("call_id") or call.get("id"))
            record = records.get(key)
            if record is None:
                continue
            output = (record.get("tool_output") or {}).get("output")
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except ValueError:
                    continue
            if not isinstance(output, dict) or "total_score" not in output:
                continue
            args = call.get("parsed_args") or call.get("args") or {}
            if isinstance(args, str):
                args = json.loads(args)
            code = args.get("cpp_code")
            if not isinstance(code, str) or not code.strip():
                raise ValueError("Final scoring call has no contestant code")
            blocks = re.findall(r"```[^\n]*\n(.*?)```", code, re.S)
            code = blocks[-1] if blocks else code
            final = {
                "turn": path.parent.name,
                "call_id": key,
                "defines_main": defines_main(code),
                "submission_score": output["total_score"],
            }
    if final is None:
        raise ValueError("No completed scoring submission")
    return final
