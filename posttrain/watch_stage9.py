"""Schedule frozen V2 training/evaluations and retain deadline/safety evidence."""

import argparse
import datetime
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

from progress import snapshot
from synthetic_ioi import write_json

CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("IOI2_ROOT", CODE))
PYTHON = os.environ.get("IOI2_TRAIN_PYTHON", str(CODE / ".venv/bin/python"))


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def time_at(hour, minute=0):
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(hour=hour, minute=minute, second=0, microsecond=0)
        .timestamp()
    )


def optional_fits(now, deadline, tokens, early=None):
    estimate = max(600, tokens * 3 / 20000 * 1.5)
    if early:
        estimate = max(estimate, early["seconds"] * tokens / early["tokens"] * 1.5 * 2)
    return now + estimate + 2200 + 300 < deadline, estimate


def launch(state, name, command, env):
    if name in state["jobs"]:
        raise RuntimeError("Duplicate job admission refused: " + name)
    with (ROOT / "logs" / (name + ".log")).open("ab") as log:
        process = subprocess.Popen(
            ["nice", "-n", "10", *command],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state["jobs"][name] = {
        "pid": process.pid,
        "command": command,
        "started_at": time.time(),
    }
    write_json(ROOT / "logs/stage9_scheduler.json", state)
    print(json.dumps({"launched": name, **state["jobs"][name]}), flush=True)


def report(state):
    result = {
        "at": time.time(),
        "models": {},
        "baselines": {},
        "jobs": state["jobs"],
        "optional_3ep": state.get("optional_3ep"),
        "alerts": state.get("alerts", []),
    }
    for model in state["models"]:
        sam = read(ROOT / "train" / model / "sampler.json")
        rows = []
        for sample in [1, 2, 3]:
            directory = ROOT / "runs" / f"dev-{model}-{sample}"
            scores = [read(p) for p in sorted(directory.glob("*/score.json"))]
            rows.append(
                {
                    "run": directory.name,
                    "done": len(scores),
                    "total": sum(row["score"] for row in scores),
                    "per_task": {row["task_id"]: row["score"] for row in scores},
                    "complete": len(scores) == 12,
                }
            )
        heldout = ROOT / "runs" / f"heldout-{model}"
        result["models"][model] = {
            "checkpoint": sam["sampler_path"] if sam else None,
            "dev": rows,
            "mean_dev_total": sum(row["total"] for row in rows) / 3
            if all(row["complete"] for row in rows)
            else None,
            "heldout": snapshot(heldout)
            if (heldout / "run_config.json").exists()
            else None,
        }
    for model, names in {
        "v1-mix": ["dev-v1-mix", "dev-v1mix-2", "dev-v1mix-3"],
        "step30": ["dev-step30", "dev-step30-2", "dev-step30-3"],
        "base": ["dev-base", "dev-base-2", "dev-base-3"],
    }.items():
        totals = [
            sum(read(p)["score"] for p in (ROOT / "runs" / name).glob("*/score.json"))
            for name in names
        ]
        result["baselines"][model] = {
            "runs": names,
            "totals": totals,
            "mean_dev_total": sum(totals) / 3,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    path = ROOT / "logs/stage9_scheduler.json"
    if path.exists() and not args.resume:
        raise RuntimeError("Scheduler state exists; explicit resume required")
    state = read(path) or {
        "jobs": {},
        "models": {},
        "evaluations": {},
        "alerts": [],
        "seen_memory": [],
        "previous_429": {},
    }
    env = dict(
        os.environ,
        IOI2_PROCESS_SLOTS_CONFIG=str(ROOT / "logs/process_slots.json"),
        IOI2_PROCESS_SLOTS_DIR=str(ROOT / ".process_slots"),
    )
    early_at, cutoff, deadline = time_at(14), time_at(15, 45), time_at(19)
    while True:
        now = time.time()
        failure = None
        training_rate = False
        for name in state["models"]:
            log = ROOT / "logs" / (name + ".log")
            count = (
                len(
                    re.findall(
                        r"(?:HTTP|status|code)[^\n]{0,40}\b429\b",
                        log.read_text(errors="replace"),
                        re.I,
                    )
                )
                if log.exists()
                else 0
            )
            key = "training:" + name
            if count > state["previous_429"].get(key, 0):
                training_rate = True
                state["rate_hold_until"] = now + 60
                print(
                    json.dumps(
                        {
                            "alert": True,
                            "training_local_429": name,
                            "hold_until": now + 60,
                        }
                    ),
                    flush=True,
                )
            state["previous_429"][key] = count
        hold = now < state.get("rate_hold_until", 0)
        if now >= early_at and "build-v2-early" not in state["jobs"]:
            launch(
                state,
                "build-v2-early",
                [
                    PYTHON,
                    "-u",
                    str(CODE / "train/build_v2.py"),
                    "--root",
                    str(ROOT),
                    "--name",
                    "v2-early",
                    "--freeze-at",
                    datetime.datetime.fromtimestamp(
                        early_at, datetime.timezone.utc
                    ).isoformat(),
                ],
                env,
            )
        collection_jobs = read(ROOT / "logs/stage8_launches.json") or {}
        drained = bool(collection_jobs) and not any(
            alive(row["pid"]) for row in collection_jobs.values()
        )
        if now >= cutoff and drained and "build-v2-final" not in state["jobs"]:
            state["final_freeze_at"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
            launch(
                state,
                "build-v2-final",
                [
                    PYTHON,
                    "-u",
                    str(CODE / "train/build_v2.py"),
                    "--root",
                    str(ROOT),
                    "--name",
                    "v2-final",
                    "--freeze-at",
                    state["final_freeze_at"],
                ],
                env,
            )
        for cohort, models in {
            "v2-early": [("ioi2-v2-early", 2)],
            "v2-final": [("ioi2-v2", 2), ("ioi2-v2-3ep", 3)],
        }.items():
            dataset = ROOT / "train" / (cohort + ".jsonl")
            stats = read(dataset.with_suffix(".stats.json"))
            builder = state["jobs"].get("build-" + cohort)
            if builder and not alive(builder["pid"]) and not stats:
                failure = "Dataset build exited without stats: " + cohort
            if not stats:
                continue
            for model, epochs in models:
                if (
                    hold
                    or model in state["jobs"]
                    or model == "ioi2-v2-3ep"
                    and state.get("optional_3ep") == "dropped_for_deadline"
                ):
                    continue
                if epochs == 3:
                    fits, estimate = optional_fits(
                        now, deadline, stats["sequence_tokens"]
                    )
                    if not fits:
                        state["optional_3ep"] = "dropped_for_deadline"
                        state["optional_3ep_train_estimate_seconds"] = estimate
                        continue
                warmup = max(1, round(math.ceil(stats["datums"] / 16) * epochs * 0.05))
                state["models"][model] = {"cohort": cohort, "epochs": epochs}
                launch(
                    state,
                    model,
                    [
                        PYTHON,
                        "-u",
                        str(CODE / "train/train_sft.py"),
                        "--data",
                        str(dataset),
                        "--name",
                        model,
                        "--output",
                        str(ROOT / "train" / model),
                        "--model",
                        "Qwen/Qwen3.5-9B",
                        "--lr",
                        "1e-4",
                        "--epochs",
                        str(epochs),
                        "--batch-size",
                        "16",
                        "--warmup-steps",
                        str(warmup),
                        "--sampling",
                        "epochs",
                    ],
                    env,
                )
        for model in state["models"]:
            sampler = read(ROOT / "train" / model / "sampler.json")
            if not sampler:
                if not alive(state["jobs"][model]["pid"]):
                    failure = "Training exited without sampler: " + model
                continue
            dev_names = [f"dev-{model}-{index}" for index in [1, 2, 3]]
            dev_complete = all(
                (ROOT / "runs" / name / "summary.json").exists() for name in dev_names
            )
            names = dev_names + (
                [f"heldout-{model}"] if dev_complete and now + 4800 < deadline else []
            )
            for name in names:
                if hold or name in state["evaluations"]:
                    continue
                heldout = name.startswith("heldout-")
                control = ROOT / "logs" / (name + ".control.json")
                write_json(control, {"concurrency": 6, "reason": "stage9_initial"})
                command = [
                    PYTHON,
                    "-u",
                    str(CODE / ("run_olympiad.py" if heldout else "run_ioi_dev.py")),
                    "--checkpoint",
                    sampler["sampler_path"],
                    "--concurrency",
                    "6",
                    "--wall-seconds",
                    "600",
                    "--control-file",
                    str(control),
                    "--output",
                    str(ROOT / "runs" / name),
                ]
                if heldout:
                    command += [
                        "--pool-dir",
                        str(ROOT / "pool_heldout"),
                        "--eval-heldout",
                        "--shuffle-seed",
                        "31415",
                    ]
                state["evaluations"][name] = {"model": model, "heldout": heldout}
                launch(state, name, command, env)
            if (
                dev_complete
                and now + 4800 >= deadline
                and f"heldout-{model}" not in state["evaluations"]
            ):
                state["models"][model]["heldout_skipped_for_deadline"] = True
        rows = []
        load = os.getloadavg()
        memory = set(state["seen_memory"])
        for name, evaluation in state["evaluations"].items():
            directory = ROOT / "runs" / name
            if not (directory / "run_config.json").exists():
                continue
            row = snapshot(directory)
            rows.append(row)
            errors = {
                str(p)
                for p in directory.glob("*/stderr.log")
                if "MemoryError" in p.read_text(errors="replace")
            }
            new_memory = errors - memory
            memory |= errors
            rate = training_rate or row["local_429_log_matches"] > state[
                "previous_429"
            ].get(name, 0)
            if rate:
                state["rate_hold_until"] = now + 60
            state["previous_429"][name] = row["local_429_log_matches"]
            control = ROOT / "logs" / (name + ".control.json")
            value = read(control)
            old = value["concurrency"]
            new = old
            if value.get("reason", "").startswith("stage9_"):
                if rate:
                    new = min(new, max(1, new // 2))
                elif load[0] > 14 or new_memory:
                    new = min(new, 4)
                if evaluation["heldout"] and now >= deadline:
                    new = 0
                if new < old:
                    value.update(
                        concurrency=new,
                        reason="stage9_rate_load_memory_backoff",
                        at=now,
                    )
                    write_json(control, value)
                    alert = {
                        "at": now,
                        "run": name,
                        "before": old,
                        "after": new,
                        "memory_errors": sorted(new_memory),
                        "local_429": rate,
                    }
                    state["alerts"].append(alert)
                    print(json.dumps({"alert": alert}), flush=True)
            if (
                not alive(state["jobs"][name]["pid"])
                and not (directory / "summary.json").exists()
            ):
                failure = "Evaluation parent exited without summary: " + name
        state.update(
            at=now,
            load=load,
            collection_drained=drained,
            seen_memory=sorted(memory),
            progress=rows,
            deadline_utc=deadline,
            early_freeze_utc=early_at,
            cutoff_utc=cutoff,
            rate_signal="local direct Tinker only; no provider-wide coverage",
        )
        if failure:
            state["failure"] = failure
            print(json.dumps({"alert": True, "failure": failure}), flush=True)
        write_json(path, state)
        write_json(ROOT / "logs/stage9_results_latest.json", report(state))
        print(
            json.dumps(
                {
                    "at": now,
                    "load": load,
                    "phase": "evaluating"
                    if state["evaluations"]
                    else "training"
                    if state["models"]
                    else "waiting_or_building",
                    "models": list(state["models"]),
                    "runs": rows,
                    "failure": failure,
                }
            ),
            flush=True,
        )
        expected = {"ioi2-v2-early", "ioi2-v2"}
        if state.get("optional_3ep") != "dropped_for_deadline":
            expected.add("ioi2-v2-3ep")
        complete = expected.issubset(state["models"]) and all(
            all(
                (ROOT / "runs" / f"dev-{model}-{i}" / "summary.json").exists()
                for i in [1, 2, 3]
            )
            and (
                (ROOT / "runs" / f"heldout-{model}" / "summary.json").exists()
                or state["models"][model].get("heldout_skipped_for_deadline")
            )
            for model in expected
        )
        if failure or complete or now >= deadline:
            write_json(ROOT / "logs/stage9_final_results.json", report(state))
            break
        time.sleep(
            min(20, max(0.1, early_at - time.time())) if time.time() < early_at else 20
        )


if __name__ == "__main__":
    main()
