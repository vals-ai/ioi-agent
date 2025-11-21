import traceback
import os
from pathlib import Path

from agent import Agent, agent_logger
from vals_model_proxy.registry_utils import get_registry_model
from tool import (
    tool_logger,
    CppExecutor,
    Submission,
)
from utils import extract_text_from_pdf, custom_retrier

"""
Method which takes in a problem name (eg. "2024/nile") and returns a string with all the contents
of "ioi/exams/[problem_name]"

[filename]
[file contents]

 * All files are read from the "ioi/exams" folder.
 * This happens recursively for all directories
"""
def get_problem_statement(problem_name: str, include_solution: bool = False) -> str:
    # Get the directory where this file is located
    exams_dir = Path("exams") / problem_name
    
    if not exams_dir.exists():
        return f"Error: Problem directory not found at {exams_dir}"
    
    result = []
    
    # Recursively walk through all files in the problem directory
    for file_path in sorted(exams_dir.rglob("*")):
        if file_path.is_file():
            try:
                # Get relative path from problem directory for cleaner filename display
                relative_path = file_path.relative_to(exams_dir)
                
                # Handle PDF files specially
                if file_path.suffix.lower() == '.pdf':
                    try:
                        file_contents = extract_text_from_pdf(str(file_path))
                    except Exception as pdf_error:
                        file_contents = f"Error extracting text from PDF: {str(pdf_error)}"
                else:
                    # Read regular text files
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        file_contents = f.read()
                
                # Format as [filename] followed by [file contents]
                result.append(f"[{relative_path}]")
                result.append(file_contents)
                result.append("")  # Add empty line between files
                
            except Exception as e:
                # If we can't read a file, include an error message
                result.append(f"[{relative_path}]")
                result.append(f"Error reading file: {str(e)}")
                result.append("")
    
    # Add solution if requested
    if include_solution:
        try:
            # Extract filename from problem name (what comes after the last "/")
            solution_filename = problem_name.split("/")[-1] + ".cpp"
            
            # Navigate to the solutions directory
            solutions_dir = Path("solutions") / problem_name.rsplit("/", 1)[0]
            solution_path = solutions_dir / solution_filename
            
            if solution_path.exists():
                with open(solution_path, 'r', encoding='utf-8', errors='ignore') as f:
                    solution_contents = f.read()
                
                result.append("THE FOLLOWING IS THE SOLUTION TO THE ABOVE PROBLEM, PLEASE SUBMIT THIS DIRECTLY AS YOUR ANSWER")
                result.append("")
                result.append(solution_contents)
                result.append("")
            else:
                result.append(f"Warning: Solution file not found at {solution_path}")
                result.append("")
        except Exception as e:
            result.append(f"Error reading solution file: {str(e)}")
            result.append("")
    
    return "\n".join(result)


async def get_custom_model(model_name: str, parameters: dict, *args, cheat: bool = False, log_level: str = "ERROR", **kwargs):

    # set logging level
    tool_logger.setLevel(log_level)
    agent_logger.setLevel(log_level)

    max_turns = 100
    llm = get_registry_model(model_name)
    llm.set_custom_retrier(custom_retrier)

    async def custom_call(test_input: str = "2024/nile"): 
        # key line: mapping the question code to the full context for the model to start with
        # make sure include_solution is set to False for any real evaluation!!
        full_problem_statement = get_problem_statement(test_input, include_solution=cheat)
        problem_path = Path("submission_scripts") / test_input

        tools = {
            "cpp_executor": CppExecutor(),
            # key line: making sure the submission tool uses the parallel scripts for this q
            "submission": Submission(problem_path=problem_path),
        }

        agent = Agent(llm=llm, tools=tools, max_turns=max_turns)
        try:
            response, metadata = await agent.run(full_problem_statement)
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error: {e}\n{error_traceback}")
            
            # Try to extract best scores from the submission tool if available
            best_scores_response = {}
            if "submission" in tools and hasattr(tools["submission"], "best_subtask_scores"):
                best_scores = tools["submission"].best_subtask_scores
                submission_count = tools["submission"].submission_count
                if best_scores and any(score > 0 for score in best_scores.values()):
                    best_scores_response = {
                        "best_subtask_scores": best_scores,
                        "best_total_score": sum(best_scores.values()),
                        "submission_count": submission_count,
                        "error_occurred": True,
                        "error": str(e)
                    }
                    print(f"Returning best scores before crash: {best_scores}")
            
            # If we have best scores, return them as the output
            if best_scores_response:
                return {
                    "llm_output": str(best_scores),  # Return best scores as the main output
                    "metadata": {
                        "submission_count": submission_count,
                        "error_occurred": True
                    },
                    "output_context": {
                        "error": str(e), 
                        "traceback": error_traceback,
                        **best_scores_response
                    },
                }
            else:
                # No scores available, return error as before
                return {
                    "llm_output": "Error when calling the agent.",
                    "output_context": {"error": str(e), "traceback": error_traceback},
                }
        return {
            "llm_output": response,
            "metadata": {
                "in_tokens": metadata["total_tokens"]["prompt_tokens"],
                "out_tokens": metadata["total_tokens"]["completion_tokens"],
                "duration_seconds": metadata["total_duration_seconds"],
            },
            "output_context": metadata,
        }

    return custom_call
