"""Export readable native messages and normalize completed/partial ATIF artifacts."""

import argparse
import json
import shutil
from pathlib import Path

from model_library.base import LLM
from pydantic import BaseModel


def readable(value):
    if isinstance(value, BaseModel):
        return readable(
            dict(vars(value)) | (getattr(value, "__pydantic_extra__", None) or {})
        )
    if isinstance(value, dict):
        return {key: readable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [readable(item) for item in value]
    return value


def finalize(output):
    output = Path(output)
    trajectories = sorted(output.glob("logs/**/trajectory_atif.json"))
    if trajectories:
        shutil.copy2(trajectories[-1], output / "trajectory_atif.json")
    histories = sorted(output.glob("logs/**/turns/turn_*/history.json"))
    if histories:
        # These are trusted files written locally by the upstream agent.
        history = LLM.deserialize_input(histories[-1])
        (output / "raw_messages.json").write_text(
            json.dumps(readable(history), indent=2)
        )
    path = output / "trajectory_atif.json"
    if path.is_file():
        trajectory = json.loads(path.read_text())
        if not any(s["source"] == "user" for s in trajectory["steps"]):
            initial = next(output.glob("logs/**/turns/init/history.json"), None)
            if initial:
                users = [
                    {"step_id": 0, "source": "user", "message": item.text}
                    for item in LLM.deserialize_input(initial)
                    if item.kind == "text"
                ]
                trajectory["steps"] = users + trajectory["steps"]
                for index, step in enumerate(trajectory["steps"], 1):
                    step["step_id"] = index
        (output / "trajectory_atif.json").write_text(json.dumps(trajectory, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    finalize(parser.parse_args().output)
