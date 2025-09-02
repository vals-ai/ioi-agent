# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an IOI (International Olympiad in Informatics) Agent - an AI system designed to solve competitive programming problems from IOI competitions. It's part of the larger Vals AI benchmarking platform, located at `~/vals/benchmarks/benchmark-be/benchmarks/ioi/ioi_agent`.

## Key Commands

### Running Tests

```bash
# Basic test run
python test_agent.py --test 2024/nile --model grok/grok-code-fast-1

# With solutions included (for testing)
python test_agent.py --test 2024/sphinx --cheat --verbose

# Save results to JSON
python test_agent.py --test 2025/obstacles --save-results

# Available problems:
# 2024: hieroglyphs, message, mosaic, nile, sphinx, tree
# 2025: festival, migrations, obstacles, souvenirs, triples, worldmap
```

### Environment Setup

```bash
# Install dependencies (requires vals_model_proxy)
pip install -r requirements.txt  # if available

# Requires g++ compiler with C++20 support
g++ --version
```

## Architecture

### Core Components

**`agent.py`** - Main orchestration
- **Agent class**: ABC for running IOI problems with conversation flow
- **Session management**: Tracks submissions, scores, metadata (max 50 submissions, 100 turns)
- **Statistics tracking**: Token usage, tool calls, submission scores

**`tool.py`** - Execution tools
- **CppExecutor**: Compiles/runs C++ code with timeout and memory limits
- **Submission**: Evaluates against IOI test cases with subtask-based scoring
- **IOI evaluation**: Only awards points when ALL tests in a subtask pass

**`custom_model.py`** - Model integration
- **Problem loading**: Reads from `submission_scripts/` directory structure
- **Model wrapping**: Integrates with `vals_model_proxy` for LLM calls
- **PDF support**: Can extract problem statements from PDF files

**`utils.py`** - Utilities
- **Code extraction**: Parses C++ code from model output using triple backticks
- **Error handling**: Custom retry logic for context length and auth errors
- **Statistics merging**: Aggregates turn-level data into session statistics

**`logger.py`** - Logging system
- **Color-coded output**: Different colors for different log levels
- **File logging**: Automatic timestamped logs in `logs/` directory
- **Message truncation**: 1000 character limit for console output

### Problem Structure

```
submission_scripts/
├── 2024/              # 2024 IOI problems
│   ├── nile/
│   ├── sphinx/
│   └── ...
├── 2025/              # 2025 IOI problems
│   ├── obstacles/
│   │   ├── problem.json
│   │   ├── run_tests.sh
│   │   ├── tests/       # Test cases (.in/.out files)
│   │   ├── subtasks/    # Subtask definitions
│   │   └── graders/     # C++ graders and headers
│   └── ...
└── run_tests/         # Batch testing scripts
```

## How It Works

1. **Problem Loading**: Reads problem statements from `submission_scripts/` directory
2. **Conversation Flow**: LLM reasons through the problem in turns
3. **Code Testing**: `CppExecutor` tool compiles and runs C++ for experimentation  
4. **Official Submission**: `Submission` tool evaluates against IOI test cases
5. **IOI Scoring**: Subtask-based scoring where all tests in a subtask must pass
6. **Session Limits**: Maximum 50 submissions and 100 conversation turns

## Integration with Vals Platform

- Uses `vals_model_proxy` for LLM access and authentication
- Part of the `benchmarks/benchmark-be/benchmarks/ioi/` framework
- Results are compatible with benchmark aggregation and export systems
- Follows Vals AI evaluation patterns and metadata standards

## Generated Directories

- `logs/` - Execution logs with timestamps (auto-generated)
- `results_tests/` - Test results in JSON format (auto-generated)

Both directories are ignored in `.gitignore` and created automatically during execution.