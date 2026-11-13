"""Launch evaluations on sampler readiness and monitor resource/rate signals."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from progress import snapshot

ROOT = Path(os.environ.get("IOI2_ROOT", str(Path(__file__).resolve().parent)))


def read(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def launch(name, command, ledger, env):
    with (ROOT / "logs" / f"{name}.log").open("ab") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
    ledger[name] = {"pid": process.pid, "command": command, "started_at": time.time()}
    (ROOT / "logs/stage6_evaluations.json").write_text(json.dumps(ledger, indent=2))
    print(json.dumps({"launched": name, **ledger[name]}), flush=True)


def main(resume=False):
    training = read(ROOT / "logs/stage6_training.json")
    if not training:
        raise RuntimeError("Training ledger missing")
    if (ROOT / "logs/stage6_evaluations.json").exists() and not resume:
        raise RuntimeError("Evaluation ledger already exists; do not duplicate runs")
    ledger = (read(ROOT / "logs/stage6_evaluations.json") or {}) if resume else {}
    limits = {
        name: 6 for name, job in ledger.items() if "--control-file" in job["command"]
    }
    previous_429 = {}
    dev_reported = False
    rate_hold_until = 0
    env = dict(
        os.environ, IOI2_PROCESS_SLOTS_CONFIG=str(ROOT / "logs/process_slots.json")
    )
    env["TINKER_API_KEY"] = env["TINKER_DEVIN_KEY"]
    while True:
        for name, job in training.items():
            if time.time() < rate_hold_until:
                continue
            tag = name.removeprefix("ioi2-")
            if all(f"{kind}-{tag}" in ledger for kind in ["dev", "heldout"]):
                continue
            sampler = read(ROOT / "train" / name / "sampler.json")
            if sampler:
                path = sampler["sampler_path"]
                for kind in ["dev", "heldout"]:
                    run = f"{kind}-{tag}"
                    if run in ledger:
                        continue
                    control = ROOT / "logs" / f"{run}.control.json"
                    control.write_text(
                        json.dumps({"concurrency": 6, "reason": "stage6_initial"})
                    )
                    command = [
                        "python",
                        "-u",
                        str(
                            Path(__file__).resolve().parent
                            / ("run_ioi_dev.py" if kind == "dev" else "run_olympiad.py")
                        ),
                        "--checkpoint",
                        path,
                        "--concurrency",
                        "6",
                        "--wall-seconds",
                        "600",
                        "--output",
                        str(ROOT / "runs" / run),
                    ]
                    command.extend(["--control-file", str(control)])
                    limits[run] = 6
                    if kind == "heldout":
                        command.extend(
                            [
                                "--pool-dir",
                                str(ROOT / "pool_heldout"),
                                "--eval-heldout",
                                "--shuffle-seed",
                                "31415",
                            ]
                        )
                    launch(run, command, ledger, env)
            elif not alive(job["pid"]):
                raise RuntimeError(
                    f"Training {name} exited without sampler; inspect its log"
                )
        names = ["dev-base-2", "dev-step30-2", *ledger]
        rows = []
        load = os.getloadavg()
        for name in names:
            directory = ROOT / "runs" / name
            if not (directory / "run_config.json").exists():
                continue
            row = snapshot(directory)
            rows.append(row)
            old = previous_429.get(name, 0)
            count = row["local_429_log_matches"]
            if count > old:
                rate_hold_until = time.time() + 120
                print(
                    json.dumps(
                        {
                            "alert": "new_local_429",
                            "run": name,
                            "count": count,
                            "previous": old,
                            "at": time.time(),
                        }
                    ),
                    flush=True,
                )
                for heldout in limits:
                    limits[heldout] = max(1, limits[heldout] // 2)
            previous_429[name] = count
            if (
                name in ledger
                and row["done"] < row["total"]
                and not alive(ledger[name]["pid"])
            ):
                raise RuntimeError(f"Evaluation {name} disappeared before finishing")
        for name, limit in limits.items():
            control = ROOT / "logs" / f"{name}.control.json"
            current = read(control) or {}
            if not current.get("reason", "").startswith("stage6_"):
                continue
            desired = (
                0
                if load[0] > 12 and name.startswith("heldout-")
                else limit
                if load[0] < 8
                else current.get("concurrency", limit)
            )
            if desired != current.get("concurrency"):
                control.write_text(
                    json.dumps(
                        {
                            "concurrency": desired,
                            "reason": "stage6_cpu_priority"
                            if desired == 0
                            else "stage6_rate_or_load",
                            "at": time.time(),
                        }
                    )
                )
        data = {
            "at": time.time(),
            "load": load,
            "runs": rows,
            "sampling_signal": "local direct-Tinker logs only; fleet signal unavailable",
        }
        (ROOT / "logs/stage6_progress_latest.json").write_text(
            json.dumps(data, indent=2)
        )
        print(json.dumps(data), flush=True)
        dev = [row for row in rows if row["run"].startswith("dev-")]
        if len(dev) == 5 and all(row["done"] == 12 for row in dev) and not dev_reported:
            print(
                json.dumps({"all_dev_evaluations_complete": True, "at": time.time()}),
                flush=True,
            )
            (ROOT / "logs/stage6_dev_complete.json").write_text(
                json.dumps(data, indent=2)
            )
            dev_reported = True
        if (
            len(ledger) == 6
            and len(rows) == 8
            and all(row["done"] == row["total"] for row in rows)
        ):
            break
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args().resume)
