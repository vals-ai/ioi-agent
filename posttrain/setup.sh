#!/usr/bin/env bash
set -euo pipefail
CODE=$(cd -- "$(dirname -- "$0")" && pwd)
REPOS=${IOI2_REPOS:-$(dirname "$(dirname "$CODE")")}
uv venv "$CODE/.venv"
uv pip install --python "$CODE/.venv/bin/python" -r "$CODE/requirements.txt" -e "$REPOS/model-proxy"
for name in hackathon-olympiad-agent ioi-hackathon-agent; do
    uv sync --project "$REPOS/$name"
    uv pip install --python "$REPOS/$name/.venv/bin/python" -e "$REPOS/model-proxy" openai==3.45.0
done
printf '%s\n' 'Set TINKER_API_KEY securely in your environment. No datasets are downloaded by setup.'
