# IOI Agent

An AI agent for solving International Olympiad in Informatics (IOI) competitive programming problems. This is Vals AI's evaluation harness for measuring large language model performance on the IOI. It's based on our [Finance Agent](https://www.vals.ai/benchmarks/finance_agent) harness, which we've also [open-sourced](https://github.com/vals-ai/finance-agent).

## Overview

The IOI Agent evaluates AI models on competitive programming problems from the International Olympiad in Informatics, testing their ability to:

- Understand complex algorithmic problem statements
- Design efficient solutions with appropriate data structures and algorithms
- Implement correct C++ code that passes all test cases
- Work within IOI constraints (subtask-based scoring, time/memory limits)

## How It Works

1. **Problem Loading**: The agent loads IOI problem statements and test cases
2. **Conversation Flow**: The AI model reasons through the problem in structured turns
3. **Code Testing**: Built-in C++ executor allows experimentation and debugging
4. **Submission**: Submitted solutions are evaluated against official IOI test cases
5. **Scoring**: Uses IOI's subtask-based scoring system (all tests in a subtask must pass)

## Evaluation Limits

- Maximum 50 submissions per problem
- Maximum 100 conversation turns per session
- C++20 compilation with standard IOI time/memory constraints

## Quick Start

### Requirements

- Python 3.11+
- g++ compiler with support for
    - c++ v20
    - `bits/stdc++`
- Access to Vals model proxy for LLM integration
- [Git LFS](https://git-lfs.com/) for test cases

#### Installing the Model Proxy
<!-- TODO: public repo? -->

### Test Agent
The `test_agent.py` file runs a demo

```bash
# Run a test
python test_agent.py

# ... with a specific model
python test_agent.py --model openai/gpt-5-2025-08-07

# ... on a specific question
python test_agent.py --test 2024/sphinx

# ... with verbose output
python test_agent.py --verbose

# Save detailed results
python test_agent.py --save-results
```

We've also included a `--cheat` flag that allows the model access to the official solution. Use this to test the infrastructure - most models we tested achieved a full score while cheating (by submitting the provided solution code).

### Output

The final score is printed in `test_agent.py`.
Results are automatically saved to `logs` directory.

## Available Problems

The IOI is an annual competition split into 2 days. Within each day, higher-numbered problems are harder. So Problems 1 and 4 are easier than problems 2 and 5 are easier than problems 3 and 6. Our results also corroborate evidence from student scores that the 2024 exam was slightly more difficult across the board.

**2024 IOI Problems:**

Day 1:
1. Nile
2. Message
3. Tree

Day 2:
4. Hieroglyphs
5. Mosaic
6. Sphinx

**2025 IOI Problems:**

Day 1:
1. Souvenirs
2. Triples
3. Worldmap

Day 2:
4. Festival
5. Migrations
6. Obstacles

## Results

IOI benchmark results are published on [vals.ai](https://www.vals.ai/benchmarks/ioi), where you can see how different AI models perform on competitive programming tasks. Recent evaluations show significant variation in model capabilities, with top performers achieving ~25% of the maximum score.