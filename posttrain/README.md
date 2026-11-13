# IOI/Qwen post-training harness

Source only: no datasets, statements, scoring fixtures, model weights, references,
credentials, trajectories or run artifacts are distributed here. `test_*.py` files
are unit-test source and generate tiny arithmetic fixtures at runtime.

## Setup and boundaries

Requires Python 3.12, uv, g++/C++20, git-lfs, pdftotext, and sibling checkouts of
`model-proxy`, `hackathon-olympiad-agent`, `ioi-hackathon-agent`, and
`ioi-benchmark-service`. Run `bash setup.sh`; provide TINKER_API_KEY securely via
the environment. `IOI2_REPOS` overrides the sibling-checkout parent directory;
`IOI2_ROOT` overrides the artifact root (default: this directory). `IOI2_CGROUP`
selects the resource monitor's cgroup directory; `IOI2_IOI_PYTHON` and
`IOI2_TRAIN_PYTHON` can select existing compatible interpreters. Run commands
below from this directory. The external hackathon agents are invoked unchanged.

- **Pool stage:** `build_pool.py`, `build_ioi_archive_exclusions.py`
  download public olympiad tasks/references, validate them and exclude protected
  IOI/dev statements with title and shingle checks. `run_olympiad.py` collects
  train-only data; `--eval-heldout` is evaluation only.
- **Historical IOI stage:** `build_ioi_train.py` materializes only Batch tasks
  from years <=2023, validates known-correct references using
  `validate_ioi_reference.py`, and publishes only 100-point tasks.
  `run_ioi_train.py` supports manifests/task lists, repeated samples, concurrency,
  incremental artifacts and a 600-second maximum wall budget.
- **Synthetic stage:** `build_synthetic_ioi.py` drafts typed function interfaces,
  statements, graders and adapted references with the 397B teacher. All original
  train tests must pass at 100; added-main and missing-function link negatives
  must fail. Only `synthetic_ioi/manifest.json` is accepted data; do not glob the
  conversion tree. `run_synthetic_ioi.py` consumes this growing manifest.
- **SFT stage:** `train/build_sft.py` renders prefix-compatible trajectory segments
  with loss on all assistant spans and zero weight on system/user/tool tokens.
  Prefix breaks split segments. Prompt counts, masks and length limits are checked.
  `train/stage6_mix.py` balances whole-trajectory repetitions by target tokens;
  `train/train_sft.py` runs rank-32 LoRA on Tinker and saves sampler weights.
  `train/build_v2.py` freezes score-file hashes and eligible train-only cohorts,
  rejects final scoring submissions defining main, renders six source components,
  audits every assistant-span mask and balances either side to ~55% IOI targets
  without dropping unique trajectories. Oversampling uses whole trajectories.
- **Evaluation/operations:** `run_ioi_dev.py` evaluates explicitly protected dev
  assets without admitting them to training. `progress.py`, admission controls,
  process slots, resource watchers and verdict audits retain incremental evidence.
  Generated subprocesses/compilers have 2-GiB limits; worker memory limits are
  separate. Four shared production process slots were used. `watch_stage9.py`
  schedules the early/final freezes, training, three dev samples and held-out
  evaluation, preserves dev priority, and drops the optional 3-epoch run if its
  estimated training plus dev budget will not fit the deadline. Direct Tinker has
  no gateway-wide 429 coverage; local logs are not a provider-wide clean signal.

The generic `ioi_run_tests.sh` is implementation code, not a scoring fixture.
Protected statements/assets remain in their external repositories and are never
included in model training or copied into this package.

## V1-mix reproduction

The run used a frozen cohort. These commands reproduce the selection rules on
the chosen snapshot; changing snapshot contents changes the dataset.

```bash
python train/build_sft.py --run-dir runs/ioi-teacher-1 --output train/v1-ioi.jsonl --min-ioi-score 0 --positive-ioi-score --max-ioi-per-problem 4 --selection score-shortest --distinct
python train/build_sft.py --run-dir runs/coll-teacher-1 --output train/v1-efficient.jsonl --selection shortest --max-per-problem 1 --max-first-submission-tokens 8000 --max-turn-output-tokens 8000
python train/build_sft.py --run-dir runs/coll-base-1 --output train/v1-base.jsonl --max-per-problem 1000000
python train/stage6_mix.py --ioi train/v1-ioi.jsonl --other train/v1-efficient.jsonl --other train/v1-base.jsonl --output train/v1-mix.jsonl
python train/train_sft.py --data train/v1-mix.jsonl --name ioi2-v1-mix --output train/ioi2-v1-mix --model Qwen/Qwen3.5-9B --lr 1e-4 --epochs 2 --batch-size 16 --warmup-steps 2 --sampling epochs
python run_ioi_dev.py --checkpoint "$CHECKPOINT" --concurrency 6 --wall-seconds 600 --output runs/dev-v1-mix
python run_olympiad.py --checkpoint "$CHECKPOINT" --pool-dir pool_heldout --eval-heldout --concurrency 6 --wall-seconds 600 --output runs/heldout-v1-mix
```

## V2 recipe and commands

Freeze the early cohort at 14:00 UTC; freeze the final cohort only after 15:45 UTC
admission stops and in-flight work drains. Keep IOI-format target tokens near 55%:
synthetic teacher full solves (shortest, up to two/task), all synthetic on-policy
full solves, historical teacher positive scores (rounds 1+2, best four/task), and
historical on-policy positive scores (best two/task). Reject IOI trajectories whose
last completed scoring submission defines main. The other ~45% is efficient
teacher olympiad full solves plus all available base/v1-mix-era 9B olympiad full
solves. Preserve frozen cohort manifests and mask/prefix audits outside Git.

V2 has not run at PR-prep time: these are the scheduled commands, not a claim of
completed training. The scheduler records the exact executed command arrays and
timestamps in `logs/stage9_scheduler.json`; each frozen dataset records its build
commands in `train/<cohort>.data/commands.json`. These artifacts stay outside Git.

```bash
python watch_stage9.py
# The scheduler executes this early build at the specified time:
python train/build_v2.py --root "$IOI2_ROOT" --name v2-early --freeze-at 2026-11-13T14:00:00+00:00
# After collection cutoff/drain, FINAL_FREEZE_UTC is the recorded drain timestamp:
python train/build_v2.py --root "$IOI2_ROOT" --name v2-final --freeze-at "$FINAL_FREEZE_UTC"
```

```bash
# DATA=train/v2-early.jsonl, NAME=ioi2-v2-early, EPOCHS=2 for the 14:00 freeze.
# Final variants: DATA=train/v2-final.jsonl; NAME=ioi2-v2/EPOCHS=2 or ioi2-v2-3ep/EPOCHS=3.
WARMUP=$(python -c 'import math,sys; n=sum(1 for _ in open(sys.argv[1])); print(max(1,round(math.ceil(n/16)*int(sys.argv[2])*.05)))' "$DATA" "$EPOCHS")
python train/train_sft.py --data "$DATA" --name "$NAME" --output "train/$NAME" --model Qwen/Qwen3.5-9B --lr 1e-4 --epochs "$EPOCHS" --batch-size 16 --warmup-steps "$WARMUP" --sampling epochs
for SAMPLE in 1 2 3; do
  python run_ioi_dev.py --checkpoint "$CHECKPOINT" --concurrency 6 --wall-seconds 600 --output "runs/dev-$NAME-$SAMPLE" &
done
wait
python run_olympiad.py --checkpoint "$CHECKPOINT" --pool-dir pool_heldout --eval-heldout --concurrency 6 --wall-seconds 600 --output "runs/heldout-$NAME"
```

Adam betas are (0.9,0.95), clipping 1.0; warmup is followed by cosine decay to 10%
of peak LR. Dev samples have priority over held-out runs and the optional 3-epoch
variant. Never train on a protected evaluation trajectory.

## Local source checks

```bash
ruff check --select F,I .
PYTHONPATH=.:train python -m unittest discover -s . -p 'test_*.py'
PYTHONPATH=.:train python -m unittest discover -s train -p 'test_*.py'
```

Local agent-loop tests use a mock HTTP endpoint, not paid model inference. Native
grading tests require the external agent dependencies and compiler. No recovery,
repair, provider credentials or machine-specific launch state is shipped.
