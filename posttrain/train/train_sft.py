"""Rank-32 Tinker SFT with weighted source sampling and target-token mean loss."""

import argparse
import collections
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tinker


@dataclass(frozen=True)
class RecordRef:
    path: Path
    offset: int


def index_datasets(paths):
    sources = collections.defaultdict(list)
    for path in paths:
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                row = json.loads(line)
                size = len(row["input_ids"])
                if (
                    not 0 < size < 65536
                    or len(row["target_ids"]) != size
                    or len(row["weights"]) != size
                ):
                    raise ValueError(f"Invalid record shape in {path}:{offset}")
                if not any(row["weights"]) or any(
                    value not in (0, 1) for value in row["weights"]
                ):
                    raise ValueError(f"Invalid loss mask in {path}:{offset}")
                sources[row["source"]].append(RecordRef(path, offset))
    if not sources:
        raise ValueError("No SFT records")
    return dict(sources)


def source_weights(sources, overrides):
    weights = {key: float(len(rows)) for key, rows in sources.items()}
    if overrides:
        weights = {key: 1.0 for key in sources}
        for entry in overrides:
            name, value = entry.rsplit("=", 1)
            if name not in sources:
                raise ValueError(f"Unknown source: {name}")
            weights[name] = float(value)
    if (
        any(not math.isfinite(value) or value < 0 for value in weights.values())
        or sum(weights.values()) <= 0
    ):
        raise ValueError("Source weights must be finite, nonnegative, and sum positive")
    return weights


def read_record(ref):
    with ref.path.open("rb") as stream:
        stream.seek(ref.offset)
        return json.loads(stream.readline())


def learning_rate(step, total, warmup, peak):
    warmup = min(warmup, total)
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step + 1 - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def make_batch(rows):
    denominator = sum(sum(row["weights"]) for row in rows)
    data = []
    for row in rows:
        size = len(row["input_ids"])
        data.append(
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints(row["input_ids"]),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=row["target_ids"], dtype="int64", shape=[size]
                    ),
                    "weights": tinker.TensorData(
                        data=[value / denominator for value in row["weights"]],
                        dtype="float32",
                        shape=[size],
                    ),
                },
            )
        )
    return data


def mean_nll(output, rows):
    total = 0.0
    count = 0
    for loss, row in zip(output.loss_fn_outputs, rows, strict=True):
        logprobs = loss["logprobs"].to_numpy()
        total -= float(np.dot(logprobs, row["weights"]))
        count += sum(row["weights"])
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--source-weight", action="append", default=[])
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--sampling", choices=["epochs", "replacement"], default="epochs"
    )
    parser.add_argument("--save-epoch-one", action="store_true")
    args = parser.parse_args()
    if (
        args.batch_size < 1
        or not math.isfinite(args.lr)
        or args.lr <= 0
        or not math.isfinite(args.epochs)
        or args.epochs <= 0
        or args.warmup_steps < 0
        or args.save_every < 0
        or not math.isfinite(args.grad_clip)
        or args.grad_clip <= 0
        or (args.steps is not None and args.steps < 1)
    ):
        parser.error("Invalid optimization parameters")
    if args.sampling == "epochs" and args.source_weight:
        parser.error("Source weights require --sampling replacement")
    sources = index_datasets(args.data)
    weights = source_weights(sources, args.source_weight)
    count = sum(map(len, sources.values()))
    steps_per_epoch = math.ceil(count / args.batch_size)
    total_steps = args.steps or (
        sum(
            math.ceil(
                min(count, math.ceil((args.epochs - epoch) * count)) / args.batch_size
            )
            for epoch in range(math.ceil(args.epochs))
        )
        if args.sampling == "epochs"
        else math.ceil(count * args.epochs / args.batch_size)
    )
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        key: str(value)
        if isinstance(value, Path)
        else [str(item) for item in value]
        if key == "data"
        else value
        for key, value in vars(args).items()
    }
    config.update(
        rank=32,
        optimizer="Adam",
        total_steps=total_steps,
        total_datums=count,
        source_counts={key: len(rows) for key, rows in sources.items()},
        source_weights=weights,
        source_sampling=args.sampling,
        loss_reduction="mean over target tokens per batch",
        lr_schedule="linear warmup then cosine to 0.1*lr",
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    service = tinker.ServiceClient(
        api_key=os.environ.get("TINKER_API_KEY") or os.environ.get("TINKER_DEVIN_KEY"),
        display_name=args.name,
    )
    client = service.create_lora_training_client(
        base_model=args.model, rank=32, seed=args.seed
    ).result()
    rng = random.Random(args.seed)
    names = list(sources)
    saved = []
    start = time.monotonic()
    epoch_rows = []
    epoch_position = 0

    def save(name, step):
        result = client.save_weights_for_sampler(name=name).result()
        record = {"name": name, "step": step, "sampler_path": result.path}
        saved.append(record)
        (args.output / "sampler.json").write_text(
            json.dumps(record | {"checkpoints": saved}, indent=2)
        )
        print(json.dumps(record), flush=True)
        return result.path

    with (args.output / "metrics.jsonl").open("w") as stream:
        for step in range(total_steps):
            if args.sampling == "epochs":
                if not epoch_rows or epoch_position >= len(epoch_rows):
                    epoch_rows = [
                        (name, ref) for name, refs in sources.items() for ref in refs
                    ]
                    rng.shuffle(epoch_rows)
                    epoch_position = 0
                batch = epoch_rows[epoch_position : epoch_position + args.batch_size]
                epoch_position += len(batch)
                selected = [name for name, _ in batch]
                rows = [read_record(ref) for _, ref in batch]
            else:
                selected = rng.choices(
                    names, weights=[weights[name] for name in names], k=args.batch_size
                )
                rows = [read_record(rng.choice(sources[name])) for name in selected]
            lr = learning_rate(step, total_steps, args.warmup_steps, args.lr)
            forward = client.forward_backward(make_batch(rows), loss_fn="cross_entropy")
            optim = client.optim_step(
                tinker.AdamParams(
                    learning_rate=lr,
                    beta1=0.9,
                    beta2=0.95,
                    eps=1e-8,
                    grad_clip_norm=args.grad_clip,
                )
            )
            result = forward.result()
            optim.result()
            nll = mean_nll(result, rows)
            if not math.isfinite(nll):
                raise ValueError(f"Nonfinite train NLL at step {step + 1}")
            metric = {
                "step": step + 1,
                "train_nll": nll,
                "learning_rate": lr,
                "batch_size": len(rows),
                "target_tokens": sum(sum(row["weights"]) for row in rows),
                "input_tokens": sum(len(row["input_ids"]) for row in rows),
                "sources": dict(collections.Counter(selected)),
                "elapsed_seconds": time.monotonic() - start,
            }
            stream.write(json.dumps(metric) + "\n")
            stream.flush()
            print(json.dumps(metric), flush=True)
            if args.save_epoch_one and step + 1 == steps_per_epoch:
                save(f"{args.name}-e1", step + 1)
            if (
                args.save_every
                and (step + 1) % args.save_every == 0
                and step + 1 < total_steps
            ):
                save(f"{args.name}-step{step + 1}", step + 1)
    path = save(args.name, total_steps)
    result = {
        "name": args.name,
        "sampler_path": path,
        "steps": total_steps,
        "rank": 32,
        "model": args.model,
        "elapsed_seconds": time.monotonic() - start,
        "checkpoints": saved,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
