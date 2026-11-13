"""Materialize whole-trajectory IOI oversampling without dropping other sources."""

import argparse
import collections
import json
import random
from pathlib import Path


def allocate(ioi_groups, other_targets, fraction=0.4, seed=1729):
    if not ioi_groups or other_targets <= 0 or not 0 < fraction < 1:
        raise ValueError("Both training sides and a proper fraction are required")
    weights = [sum(row["target_tokens"] for row in group) for group in ioi_groups]
    if any(weight <= 0 for weight in weights):
        raise ValueError("Every IOI trajectory must have supervised tokens")
    desired = other_targets * fraction / (1 - fraction)
    chosen = list(range(len(ioi_groups)))
    current = sum(weights)
    if current > desired:
        raise ValueError(
            "IOI already exceeds requested fraction; cannot upweight it while retaining all data"
        )
    rng = random.Random(seed)
    while current + sum(weights) <= desired:
        chosen.extend(range(len(weights)))
        current += sum(weights)
    remaining = list(range(len(weights)))
    rng.shuffle(remaining)
    for index in remaining:
        if current + weights[index] <= desired:
            chosen.append(index)
            current += weights[index]
    best = min(
        range(len(weights)), key=lambda index: abs(desired - current - weights[index])
    )
    if abs(desired - current - weights[best]) < abs(desired - current):
        chosen.append(best)
    return chosen


def build(ioi_path, other_paths, output):
    if output.exists():
        raise ValueError("Mixture output already exists")
    groups = collections.defaultdict(list)
    with ioi_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            groups[row["trajectory"]].append(row)
    other = []
    for path in other_paths:
        with path.open() as stream:
            other.extend(json.loads(line) for line in stream)
    grouped = list(groups.values())
    chosen = allocate(grouped, sum(row["target_tokens"] for row in other))
    rows = other + [row for index in chosen for row in grouped[index]]
    stats = {
        "datums": len(rows),
        "unique_trajectories": len({row["trajectory"] for row in rows}),
        "sequence_tokens": sum(row["sequence_tokens"] for row in rows),
        "target_tokens": sum(row["target_tokens"] for row in rows),
        "ioi_unique_trajectories": len(groups),
        "ioi_trajectory_exposures": len(chosen),
        "sources": {},
    }
    for row in rows:
        if (
            len(row["input_ids"]) != len(row["weights"])
            or len(row["input_ids"]) != len(row["target_ids"])
            or sum(row["weights"]) != row["target_tokens"]
        ):
            raise ValueError("Invalid record shape or mask")
        source = stats["sources"].setdefault(
            row["source"], {"datums": 0, "target_tokens": 0, "sequence_tokens": 0}
        )
        source["datums"] += 1
        source["target_tokens"] += row["target_tokens"]
        source["sequence_tokens"] += row["sequence_tokens"]
    ioi_targets = sum(
        row["target_tokens"] for index in chosen for row in grouped[index]
    )
    stats["ioi_target_fraction"] = ioi_targets / stats["target_tokens"]
    with output.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    output.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ioi", type=Path, required=True)
    parser.add_argument("--other", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.ioi, args.other, args.output), indent=2))
