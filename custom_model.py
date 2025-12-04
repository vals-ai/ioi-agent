import traceback
from pathlib import Path

from submodules.ioi_agent.agent import Agent, agent_logger
from model_library.registry_utils import get_registry_model
from submodules.ioi_agent.tool import (
    tool_logger,
    CppExecutor,
    Submission,
)

"""
Method which takes in a problem name (eg. "2024/nile") and returns a string with all the contents
of "ioi/exams/[problem_name]"

[filename]
[file contents]

 * All files are read from the "ioi/exams" folder.
 * This happens recursively for all directories
"""


async def get_custom_model(
    model_name: str,
    parameters: dict,
    *args,
    cheat: bool = False,
    log_level: str = "ERROR",
    **kwargs,
):
    # set logging level
    tool_logger.setLevel(log_level)
    agent_logger.setLevel(log_level)

    max_turns = 100
    llm = get_registry_model(model_name)

    # hijack model logger
    llm.logger = agent_logger

    async def custom_call(test_input: str = "2024/nile"):
        # key line: mapping the question code to the full context for the model to start with
        # make sure include_solution is set to False for any real evaluation!!
        full_problem_statement = get_problem_statement(test_input)
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
            if "submission" in tools and hasattr(
                tools["submission"], "best_subtask_scores"
            ):
                best_scores = tools["submission"].best_subtask_scores
                submission_count = tools["submission"].submission_count
                if best_scores and any(score > 0 for score in best_scores.values()):
                    best_scores_response = {
                        "best_subtask_scores": best_scores,
                        "best_total_score": sum(best_scores.values()),
                        "submission_count": submission_count,
                        "error_occurred": True,
                        "error": str(e),
                    }
                    print(f"Returning best scores before crash: {best_scores}")

            # If we have best scores, return them as the output
            if best_scores_response:
                return {
                    "llm_output": str(
                        best_scores
                    ),  # Return best scores as the main output
                    "metadata": {
                        "submission_count": submission_count,
                        "error_occurred": True,
                    },
                    "output_context": {
                        "error": str(e),
                        "traceback": error_traceback,
                        **best_scores_response,
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
