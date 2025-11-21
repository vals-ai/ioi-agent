import os
import uuid
import json
from datetime import datetime
from abc import ABC
import re
import traceback

from tool import Tool
from model_library.base import *

from logger import get_logger
from utils import INSTRUCTIONS_PROMPT, _merge_statistics
from model_library.exceptions import DoNotRetryException

agent_logger = get_logger(__name__)


class Agent(ABC):
    def __init__(
        self,
        tools: dict[str, Tool],
        llm: LLM,
        max_turns: int = 20,
        instructions_prompt: str = INSTRUCTIONS_PROMPT,
    ):
        self.tools = tools
        self.llm = llm
        self.max_turns = max_turns
        self.instructions_prompt = instructions_prompt

    async def _process_turn(self, turn_count, data_storage, metadata):
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
        response: QueryResult = await self.llm.query(
            input=self.messages,
            tools=[tool.get_tool_repr() for tool in self.tools.values()]
        )

        # record response
        # TODO: make this less hacky
        self.messages = response.history

        # parse QueryResult
        response_text = response.output_text
        reasoning_text = response.reasoning
        tool_calls: list[ToolCall] = response.tool_calls

        # record turn metadata
        turn_metadata = response.metadata.model_dump()
        turn_metadata['tool_calls'] = []
        turn_metadata['errors'] = []

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
                    arguments = json.loads(arguments)

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
                    tool_result = ToolResult(
                        tool_call=tool_call,
                        result=error_msg
                    )
                    self.messages.append(tool_result)
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
                                    metadata["submission_count"] = result_data["submission_count"]
                                
                                # Track best scores from submission
                                if "best_subtask_scores" in result_data:
                                    metadata["best_subtask_scores"] = result_data["best_subtask_scores"]
                                    metadata["has_submissions"] = True
                                
                                # Check if maximum submissions reached
                                if "error" in result_data and "Maximum submissions" in result_data["error"]:
                                    metadata["max_submissions_reached"] = True
                                    agent_logger.info(f"\033[1;31m[SUBMISSION LIMIT REACHED]\033[0m {result_data['error']}")
                                    
                                    turn_metadata["tool_calls"].append(tool_call_metadata)
                                    
                                    # Exit due to submission limit
                                    return None, turn_metadata, False
                                
                                # Log submission information
                                submission_info = f"Submission #{result_data.get('submission_count', '?')} - Score: {result_data.get('total_score', '?')} - Remaining: {result_data.get('submissions_remaining', '?')}"
                                agent_logger.info(f"\033[1;36m[SUBMISSION INFO]\033[0m {submission_info}")
                        except (json.JSONDecodeError, TypeError):
                            # If we can't parse the result, continue normally
                            pass
                else:
                    tool_call_metadata["error"] = tool_result["result"]
                    turn_metadata["errors"].append(tool_result["result"])

                tool_result = ToolResult(
                    tool_call=tool_call,
                    result=tool_result["result"]
                )
                self.messages.append(tool_result)

                # Add tool call metadata to turn
                turn_metadata["tool_calls"].append(tool_call_metadata)

        else:
            # Get text response when there are no tool calls

            # Use regex to check for "EXIT" pattern
            exit_pattern = re.compile(r"\bEXIT\b", re.IGNORECASE)

            if isinstance(response_text, str) and exit_pattern.search(response_text):
                # Agent requested to exit
                agent_logger.info(f"\033[1;31m[EXIT REQUESTED]\033[0m Agent requested exit")
                return None, turn_metadata, False
            else:
                agent_logger.info(f"\033[1;33m[LLM THINKING]\033[0m {response_text}")

        return None, turn_metadata, True

    async def run(self, question: str, session_id: str = None) -> tuple[str, dict]:
        """
        Run the agent on a question from the user.

        Args:
            question (str): The user's question
            session_id (str, optional): A unique identifier for this session

        Returns:
            tuple[str, dict]: The final answer and metadata about the run
        """
        # Initialize metadata
        session_id = session_id or str(uuid.uuid4())
        metadata = {
            "session_id": session_id,
            "model": self.llm.model_name,
            "user_input": question,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_duration_seconds": 0,
            "total_tokens": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
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

        # Prepare initial message with instructions
        initial_prompt = self.instructions_prompt.format(question=question)

        initial_message = TextInput(
            text = initial_prompt
        )
        self.messages: list[InputItem] = [initial_message]

        agent_logger.info(
            f"\033[1;34m[USER INSTRUCTIONS]\033[0m {initial_prompt}"
        )

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
            except DoNotRetryException as e:
                agent_logger.error(f"\033[1;31m[DO NOT RETRY]\033[0m {e}")
                should_continue = False

            except Exception as e:
                # Log the error
                agent_logger.error(f"\033[1;31m[ERROR]\033[0m {e}")
                agent_logger.error(f"\033[1;31m[traceback]\033[0m {traceback.format_exc()}")

                # Explain the error to the agent and give them a chance to recover
                error_message = TextInput(
                    text=f"An error occurred: {str(e)}. Please review what happened and try a different approach."
                )
                self.messages.append(error_message)

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
        metadata = _merge_statistics(metadata)

        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Save metadata to logs/{session_id}.json
        log_path = os.path.join("logs", f"{session_id}.json")
        with open(log_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return final_answer, metadata
