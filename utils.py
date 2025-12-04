import PyPDF2
import re


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
        r"compilation failed",
        r"error:.*\berror\b",  # C++ compiler errors usually contain "error:"
        r"fatal error:",
        r"undefined reference",
        r"cannot find -l",  # linking errors
        r"ld returned.*exit status",  # linker errors
        r"collect2:.*error:",  # GCC collect2 errors
        r"make.*\[.*\].*Error",  # Make errors
        r"g\+\+.*error",  # G++ specific errors
        r"clang.*error",  # Clang specific errors
    ]

    combined_output = stdout + "\n" + stderr
    combined_output_lower = combined_output.lower()

    for pattern in compilation_error_patterns:
        if re.search(pattern, combined_output_lower, re.IGNORECASE):
            return True

    return False
