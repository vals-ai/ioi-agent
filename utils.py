from datetime import datetime
import json
import PyPDF2
import re
from typing import Callable, Awaitable, Any
import logging

INSTRUCTIONS_PROMPT = """
You will solve programming problems from the IOI competition.

Provide all submissions and execution requests in a c++ (v20) program with stdlib imports. Submit all code enclosed with three backticks. For example:

```
[your code here]
```

When you would like to make a submission, use the appropriate tool call. After you make a submission, you will see the score you received for each of the subtasks of that problem.

You may make at most 50 submissions in at most 100 turns. Your score will be calculated based on the number of categories passed acorss all submissions.

When you are done working and would not like to make more submissions you should respond with 'EXIT'.

The following includes the problem statement and any relevant contextual files:
{question}
"""

def _merge_statistics(metadata: dict) -> dict:
    """
    Merge turn-level statistics into session-level statistics.

    Args:
        metadata (dict): The metadata with turn-level statistics

    Returns:
        dict: Updated metadata with merged statistics
    """
    # Reset aggregate values to recalculate
    metadata["total_tokens"] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    metadata["tool_usage"] = {}
    metadata["tool_calls_count"] = 0
    metadata["api_calls_count"] = len(metadata["turns"])
    metadata["error_count"] = 0

    # Aggregate statistics from all turns
    for turn in metadata["turns"]:
        # Aggregate token usage
        in_tokens = turn["in_tokens"]
        out_tokens = turn["out_tokens"]
        metadata["total_tokens"]["prompt_tokens"] += in_tokens
        metadata["total_tokens"]["completion_tokens"] += out_tokens
        metadata["total_tokens"]["total_tokens"] += in_tokens + out_tokens

        # Count errors
        metadata["error_count"] += len(turn["errors"])

        # Aggregate tool usage
        for tool_call in turn["tool_calls"]:
            tool_name = tool_call["tool_name"]
            if tool_name not in metadata["tool_usage"]:
                metadata["tool_usage"][tool_name] = 0
            metadata["tool_usage"][tool_name] += 1
            metadata["tool_calls_count"] += 1

    # Calculate total duration
    if metadata["start_time"] and metadata["end_time"]:
        start = datetime.fromisoformat(metadata["start_time"])
        end = datetime.fromisoformat(metadata["end_time"])
        metadata["total_duration_seconds"] = (end - start).total_seconds()

    return metadata

def simple_extract_code(model_output: str):
    """
    Extract code from model output by finding the last code block between triple backticks.
    
    Args:
        model_output (str): The model output containing code blocks
        
    Returns:
        str: The extracted code or empty string if no valid code block found
    """
    outputlines = model_output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    # return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])
    return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def has_compilation_error(stdout: str, stderr: str) -> bool:
    """
    Check if there are compilation errors in the output.
    
    Args:
        stdout (str): Standard output from the test script
        stderr (str): Standard error from the test script
        
    Returns:
        bool: True if compilation errors are detected, False otherwise
    """
    # Common compilation error indicators
    compilation_error_patterns = [
        r'compilation failed',
        r'error:.*\berror\b',  # C++ compiler errors usually contain "error:"
        r'fatal error:',
        r'undefined reference',
        r'cannot find -l',  # linking errors
        r'ld returned.*exit status',  # linker errors
        r'collect2:.*error:',  # GCC collect2 errors
        r'make.*\[.*\].*Error',  # Make errors
        r'g\+\+.*error',  # G++ specific errors
        r'clang.*error',  # Clang specific errors
    ]
    
    combined_output = stdout + "\n" + stderr
    combined_output_lower = combined_output.lower()
    
    for pattern in compilation_error_patterns:
        if re.search(pattern, combined_output_lower, re.IGNORECASE):
            return True
            
    return False

