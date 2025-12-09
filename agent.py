from typing import Any
import json
from datetime import datetime
from abc import ABC
import re
import traceback

from submodules.ioi_agent.tool import Tool
from model_library.base import (
    LLM,
    QueryResult,
    QueryResultMetadata,
    TextInput,
    ToolCall,
    ToolResult,
    InputItem,
)

from submodules.ioi_agent.logger import get_logger

agent_logger = get_logger(__name__)


class ModelException(Exception):
    """
    Raised on model errors
    not retried by default
    """

    pass


class ToolCallException(Exception):
    """
    raised when tool call str doesn't parse to json
    prev message is deleted and retried
    """

    pass


class Agent(ABC):
    _query_result_metadata: list[QueryResultMetadata] = []

    def __init__(
        self,
        tools: dict[str, Tool],
        llm: LLM,
        max_turns: int = 20,
    ):
        self.tools = tools
        self.llm = llm
        self.max_turns = max_turns
        self.input_items = []

    def _merge_statistics(
        self, metadata: dict[str, Any], query_result_metadata: list[QueryResultMetadata]
    ) -> dict[str, Any]:
        """
        Merge turn-level statistics into session-level statistics.

        Args:
            metadata (dict): The metadata with turn-level statistics

        Returns:
            dict: Updated metadata with merged statistics
        """
        # Reset aggregate values to recalculate
        metadata["tool_usage"] = {}
        metadata["tool_calls_count"] = 0
        metadata["api_calls_count"] = len(metadata["turns"])
        metadata["error_count"] = 0

        # Aggregate statistics from all turns
        for turn in metadata["turns"]:
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

        # Aggregate query result metadata (need to start with the first one to mantain the cost per token information)
        total_metadata = query_result_metadata[0]
        for qr_metadata in query_result_metadata[1:]:
            total_metadata = total_metadata.__add__(qr_metadata)

        total_metadata_dict = total_metadata.model_dump()
        metadata["total_metadata"] = total_metadata_dict

        return metadata

    async def _process_turn(
        self,
        turn_count: int,
        data_storage: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[str, dict[str, Any], bool]:
        """
        Process a single turn in the agent's conversation.

        Args:
            turn_count (int): The current turn number
            data_storage (dict): Storage for conversation data
            metadata (dict): Session metadata

        Returns:
            tuple: (final_answer, turn_metadata, should_continue)
        """
        agent_logger.info(f"\033[1;34m[TURN {turn_count}]\033[0m")

        # Get response from LLM
        try:
            response: QueryResult = await self.llm.query(
                input=self.input_items,
                tools=[tool.get_tool_repr() for tool in self.tools.values()],
            )

            self._query_result_metadata.append(response.metadata)
        except Exception as e:
            raise ModelException(e)

        # record response
        self.input_items = response.history

        # parse QueryResult
        response_text = response.output_text
        reasoning_text = response.reasoning
        tool_calls: list[ToolCall] = response.tool_calls

        # record turn metadata
        turn_metadata = response.metadata.model_dump()
        turn_metadata["tool_calls"] = []
        turn_metadata["errors"] = []

        # Log the thinking content if available
        if reasoning_text:
            agent_logger.info(f"\033[1;33m[LLM REASONING]\033[0m {reasoning_text}")

        if response_text:
            agent_logger.info(f"\033[1;33m[LLM THINKING]\033[0m {response_text}")

        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call.name

                # unpacks tool call arguments
                arguments = tool_call.args
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        agent_logger.warning(
                            f"Could not parse tool call arguments: {arguments}"
                        )
                        raise ToolCallException(
                            f"Could not parse tool call arguments: {arguments}"
                        )

                # Track tool call in turn metadata
                tool_call_metadata = {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "success": False,
                    "error": None,
                }
                if tool_name not in self.tools:
                    error_msg = f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"

                    # Update error tracking
                    tool_call_metadata["error"] = error_msg
                    turn_metadata["errors"].append(error_msg)

                    # Add error to messages
                    tool_result = ToolResult(tool_call=tool_call, result=error_msg)
                    self.input_items.append(tool_result)
                    continue

                tool_result = await self.tools[tool_name](arguments)

                if tool_result["success"]:
                    # Add tool result to messages
                    tool_call_metadata["success"] = True

                    # Special handling for submission tool
                    if tool_name == "submission":
                        try:
                            # Parse the submission result
                            result_data = json.loads(tool_result["result"])
                            if isinstance(result_data, dict):
                                # Update metadata with submission information
                                if "submission_count" in result_data:
                                    metadata["submission_count"] = result_data[
                                        "submission_count"
                                    ]

                                # Track best scores from submission
                                if "best_subtask_scores" in result_data:
                                    metadata["best_subtask_scores"] = result_data[
                                        "best_subtask_scores"
                                    ]
                                    metadata["has_submissions"] = True

                                # Check if maximum submissions reached
                                if (
                                    "error" in result_data
                                    and "Maximum submissions" in result_data["error"]
                                ):
                                    metadata["max_submissions_reached"] = True
                                    agent_logger.info(
                                        f"\033[1;31m[SUBMISSION LIMIT REACHED]\033[0m {result_data['error']}"
                                    )

                                    turn_metadata["tool_calls"].append(
                                        tool_call_metadata
                                    )

                                    # Exit due to submission limit
                                    return None, turn_metadata, False

                                # Log submission information
                                submission_info = f"Submission #{result_data.get('submission_count', '?')} - Score: {result_data.get('total_score', '?')} - Remaining: {result_data.get('submissions_remaining', '?')}"
                                agent_logger.info(
                                    f"\033[1;36m[SUBMISSION INFO]\033[0m {submission_info}"
                                )
                        except (json.JSONDecodeError, TypeError):
                            # If we can't parse the result, continue normally
                            pass
                else:
                    tool_call_metadata["error"] = tool_result["result"]
                    turn_metadata["errors"].append(tool_result["result"])

                tool_result = ToolResult(
                    tool_call=tool_call, result=tool_result["result"]
                )
                self.input_items.append(tool_result)

                # Add tool call metadata to turn
                turn_metadata["tool_calls"].append(tool_call_metadata)

        else:
            # Get text response when there are no tool calls

            # Use regex to check for "EXIT" pattern
            exit_pattern = re.compile(r"\bEXIT\b", re.IGNORECASE)

            if isinstance(response_text, str) and exit_pattern.search(response_text):
                # Agent requested to exit
                agent_logger.info(
                    "\033[1;31m[EXIT REQUESTED]\033[0m Agent requested exit"
                )
                return None, turn_metadata, False
            else:
                agent_logger.info(f"\033[1;33m[LLM THINKING]\033[0m {response_text}")

        return None, turn_metadata, True

    async def run(self, input_items: list[InputItem]) -> tuple[str, dict[str, Any]]:
        """
        Run the agent on a question from the user.

        Args:
            question (str): The user's question
            session_id (str, optional): A unique identifier for this session

        Returns:
            tuple[str, dict]: The final answer and metadata about the run
        """
        self.input_items = input_items

        # Initialize metadata
        metadata = {
            "model": self.llm.model_name,
            "user_input": str(input_items),
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_duration_seconds": 0,
            "turns": [],
            "tool_usage": {},
            "tool_calls_count": 0,
            "api_calls_count": 0,
            "error_count": 0,
            "submission_count": 0,
            "max_submissions_reached": False,
            "best_subtask_scores": None,
            "has_submissions": False,
        }

        # Initialize data storage for this conversation
        data_storage = {}

        turn_count = 0

        while turn_count < self.max_turns:
            turn_count += 1
            # Process the current turn
            try:
                result, turn_metadata, should_continue = await self._process_turn(
                    turn_count, data_storage, metadata
                )

                # Add turn metadata to session metadata
                metadata["turns"].append(turn_metadata)

            # Handle DoNotRetryException
            except ModelException as e:
                agent_logger.critical(f"\033[1;31m[MODEL EXCEPTION]\033[0m {e}")
                should_continue = False

            # for malformed tool calls
            except ToolCallException:
                last_message = self.input_items.pop(-1)
                agent_logger.warning(
                    f"\033[1;37m[RETRYING TOOL CALL]\033[0m Removed last message: {last_message}"
                )

            except Exception as e:
                # Log the error
                agent_logger.error(f"\033[1;31m[ERROR]\033[0m {e}")
                agent_logger.error(
                    f"\033[1;31m[traceback]\033[0m {traceback.format_exc()}"
                )

                # Explain the error to the agent and give them a chance to recover
                error_message = TextInput(
                    text=f"An error occurred: {str(e)}. Please review what happened and try a different approach."
                )
                self.input_items.append(error_message)

                # continue in spite of the error
                should_continue = True

            # Check if we should continue
            if not should_continue:
                break

        # Finalize session metadata
        metadata["end_time"] = datetime.now().isoformat()

        # Determine final answer based on submission results
        if metadata["has_submissions"]:
            # Return the best subtask scores as a string
            final_answer = str(metadata["best_subtask_scores"])
            metadata["final_answer"] = final_answer
        else:
            # No submissions were made
            final_answer = "Agent never called Submission"
            metadata["final_answer"] = final_answer

        # Merge turn-level statistics into session-level statistics
        metadata = self._merge_statistics(metadata, self._query_result_metadata)

        return final_answer, metadata
