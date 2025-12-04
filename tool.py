import json
import re
import os
import traceback
import tempfile
import asyncio
import shutil
import glob

from abc import ABC, abstractmethod
from openai.types.chat import ChatCompletionMessageToolCall
from anthropic.types.tool_use_block import ToolUseBlock

# model proxy
from model_library.base import ToolDefinition, ToolBody
from submodules.ioi_agent.logger import get_logger
from submodules.ioi_agent.utils import simple_extract_code, has_compilation_error

tool_logger = get_logger(__name__)


class Tool(ABC):
    """
    Abstract base class for tools.
    """

    name: str
    description: str
    input_arguments: dict
    required_arguments: list[str]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    def parse_tool_message(
        self,
        provider="openai",
        message: str | ChatCompletionMessageToolCall | ToolUseBlock = None,
    ):
        """
        Get the tool format for different providers.

        Args:
            provider (str): The provider to format the tool for ('openai' or 'anthropic')

        Returns:
            dict: Formatted tool definition
        """
        if provider.lower() != "anthropic":
            arguments = message.function.arguments
        elif provider.lower() == "anthropic":
            arguments = message.input
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        return arguments

    def get_tool_repr(self, provider: str = "openai") -> ToolDefinition:
        if provider != "openai":
            raise ValueError(f"{provider} is not supported")

        body = ToolBody(
            name=self.name,
            description=self.description,
            properties=self.input_arguments,
            required=self.required_arguments,
            # TODO: add kwargs
        )

        definition = ToolDefinition(name=self.name, body=body)

        return definition

    @abstractmethod
    def call_tool(self, arguments: dict, *args, **kwargs) -> list[str]:
        pass

    async def __call__(self, arguments: dict = None, *args, **kwargs) -> list[str]:
        tool_logger.info(
            f"\033[1;34m[TOOL: {self.name.upper()}]\033[0m Calling with arguments: {arguments}"
        )

        try:
            tool_result = await self.call_tool(arguments, *args, **kwargs)
            tool_logger.info(
                f"\033[1;32m[TOOL: {self.name.upper()}]\033[0m Returned: {tool_result}"
            )
            if self.name == "retrieve_information":
                return {
                    "success": True,
                    "result": tool_result["retrieval"],
                    "usage": tool_result["usage"],
                }
            else:
                return {"success": True, "result": json.dumps(tool_result)}
        except Exception as e:
            # logging
            tool_logger.error(f"\033[1;31m[TOOL: {self.name.upper()}]")
            tool_logger.error(f"\033[1;31m[ERROR]\033[0m {e}")
            tool_logger.warning(
                f"\033[1;31m[traceback]\033[0m {traceback.format_exc()}"
            )

            # result
            error_msg = str(e)
            return {"success": False, "result": error_msg}


class CppExecutor(Tool):
    """
    Tool for executing C++ code fragments and returning the result.
    Extracts code from LLM output and compiles/runs it.
    """

    name: str = "cpp_executor"
    description: str = (
        "Execute a C++ code fragment and return the result. "
        "The tool will extract C++ code from the provided text, compile it, and run it, "
        "returning the output or any compilation/runtime errors."
        "Do not use cin. Include any execution parameters hardcoded within your main function for testing."
    )
    input_arguments: dict = {
        "cpp_code": {
            "type": "string",
            "description": "The C++ code fragment to execute. Can contain full LLM output with code blocks or just the code.",
        }
    }
    required_arguments: list[str] = ["cpp_code"]

    def __init__(
        self,
        compiler: str = "g++",
        compiler_flags: list[str] = None,
        timeout: int = 180,
        *args,
        **kwargs,
    ):
        super().__init__(
            self.name,
            self.description,
            self.input_arguments,
            self.required_arguments,
            *args,
            **kwargs,
        )
        self.timeout = timeout
        self.compiler = compiler
        # girlboss mode: add bits/stdc++ flag for maximum slay
        self.compiler_flags = compiler_flags or [
            "-std=c++20",
            "-O2",
            "-include",
            "bits/stdc++.h",
        ]

    async def _compile_and_run_cpp(self, cpp_code: str) -> dict:
        """
        Compile and run C++ code, returning the result.

        Args:
            cpp_code (str): The C++ code to compile and run

        Returns:
            dict: Contains 'success', 'output', 'error', and 'exit_code'
        """
        # Create temporary files for the source and executable
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cpp", delete=False
        ) as src_file:
            src_file.write(cpp_code)
            src_path = src_file.name

        with tempfile.NamedTemporaryFile(suffix=".out", delete=False) as exe_file:
            exe_path = exe_file.name

        try:
            # Compile the C++ code
            compile_cmd = (
                [self.compiler] + self.compiler_flags + [src_path, "-o", exe_path]
            )

            compile_process = await asyncio.create_subprocess_exec(
                *compile_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            compile_stdout, compile_stderr = await asyncio.wait_for(
                compile_process.communicate(), timeout=self.timeout
            )

            if compile_process.returncode != 0:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Compilation failed:\n{compile_stderr.decode()}",
                    "exit_code": compile_process.returncode,
                }

            # Run the compiled executable
            run_process = await asyncio.create_subprocess_exec(
                exe_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            run_stdout, run_stderr = await asyncio.wait_for(
                run_process.communicate(), timeout=self.timeout
            )

            output = run_stdout.decode()
            error = run_stderr.decode()

            return {
                "success": run_process.returncode == 0,
                "output": output,
                "error": error if error else None,
                "exit_code": run_process.returncode,
            }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "output": "",
                "error": f"Execution timed out after {self.timeout} seconds",
                "exit_code": -1,
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Execution error: {str(e)}",
                "exit_code": -1,
            }
        finally:
            # Clean up temporary files
            try:
                os.unlink(src_path)
                os.unlink(exe_path)
            except OSError:
                pass  # Files might not exist or be accessible

    async def call_tool(self, arguments: dict) -> dict:
        """
        Execute C++ code and return the result.

        Args:
            arguments (dict): Contains 'cpp_code' key with the code to execute

        Returns:
            dict: The execution result
        """
        cpp_input = arguments.get("cpp_code", "")

        # Extract code from the input (handles both raw code and LLM output with code blocks)
        extracted_code = simple_extract_code(cpp_input)

        # If extraction failed, try to use the input as-is (might be raw code)
        if not extracted_code.strip():
            extracted_code = cpp_input

        if not extracted_code.strip():
            return {
                "success": False,
                "output": "",
                "error": "No C++ code found in the input",
                "exit_code": -1,
            }

        # TODO: consider handling the case in which the model doesn't provide a main function

        result = await self._compile_and_run_cpp(extracted_code)
        return result


class Submission(Tool):
    """
    Tool for submitting C++ code and getting a score based on IOI subtasks.
    Runs the code against test cases and returns scores for completed subtasks.
    Tracks submission count and enforces a maximum of 50 submissions.
    """

    name: str = "submission"
    description: str = (
        "Submit C++ code for evaluation against IOI test cases. "
        "Compiles and runs the provided C++ code against all available test cases. "
        "Returns scores based on completed subtasks - you get points for a subtask only if ALL tests in that subtask pass. "
        "Also tracks and returns the best score achieved for each subtask across all submissions. "
        "Maximum of 50 submissions allowed per session."
    )
    input_arguments: dict = {
        "cpp_code": {
            "type": "string",
            "description": "The C++ code to submit and evaluate.",
        }
    }
    required_arguments: list[str] = ["cpp_code"]

    def __init__(
        self,
        max_submissions: int = 50,
        problem_path: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            self.name,
            self.description,
            self.input_arguments,
            self.required_arguments,
            *args,
            **kwargs,
        )
        self.max_submissions = max_submissions
        self.submission_count = 0

        # Set default problem path if not provided
        if problem_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.problem_path = os.path.join(
                script_dir, "submission_scripts", "2024", "nile"
            )
        else:
            tool_logger.info(f"\033[1;33musing the problem path: {problem_path}\033[0m")
            self.problem_path = problem_path

        # Load problem configuration and subtasks
        self._load_problem_config()
        self._load_subtasks()

    def _load_problem_config(self):
        """Load problem configuration from problem.json"""
        try:
            problem_json_path = os.path.join(self.problem_path, "problem.json")
            with open(problem_json_path, "r") as f:
                self.problem_config = json.load(f)

            self.time_limit = self.problem_config.get("time_limit", 2.0)
            # Convert memory limit from bytes to kilobytes for ulimit
            memory_limit_bytes = self.problem_config.get("memory_limit", 2147483648)
            self.memory_limit_kb = memory_limit_bytes // 1024

        except Exception as e:
            tool_logger.error(f"Failed to load problem config: {e}")
            # Use defaults
            self.time_limit = 2.0
            self.memory_limit_kb = 2097152  # 2GB in KB

    def _load_subtasks(self):
        """Load all subtask definitions"""
        self.subtasks = {}
        subtasks_dir = os.path.join(self.problem_path, "subtasks")

        try:
            for subtask_file in glob.glob(os.path.join(subtasks_dir, "*.json")):
                subtask_name = os.path.basename(subtask_file).replace(".json", "")
                with open(subtask_file, "r") as f:
                    subtask_data = json.load(f)
                    self.subtasks[subtask_name] = {
                        "score": subtask_data["score"],
                        "testcases": subtask_data["testcases"],
                    }
        except Exception as e:
            tool_logger.error(f"Failed to load subtasks: {e}")
            self.subtasks = {}

        # Initialize best scores for each subtask to 0
        self.best_subtask_scores = {name: 0 for name in self.subtasks.keys()}

    def _update_best_scores(self, current_subtask_scores: dict):
        """
        Update the best scores by taking the maximum of current and best scores for each subtask.

        Args:
            current_subtask_scores (dict): Current submission's subtask scores
        """
        for subtask_name, current_score in current_subtask_scores.items():
            if subtask_name in self.best_subtask_scores:
                self.best_subtask_scores[subtask_name] = max(
                    self.best_subtask_scores[subtask_name], current_score
                )

    async def _run_tests_with_solution(self, cpp_code: str) -> dict:
        """
        Run all tests with the provided solution and return results.

        Args:
            cpp_code (str): The C++ solution code

        Returns:
            dict: Test results including which tests passed/failed
        """
        # Create a temporary working directory
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Copy all necessary files to temp directory
                solution_path = os.path.join(temp_dir, "solution.cpp")
                with open(solution_path, "w") as f:
                    f.write(cpp_code)

                # Copy graders, checker, tests, and script
                shutil.copytree(
                    os.path.join(self.problem_path, "graders"),
                    os.path.join(temp_dir, "graders"),
                )

                # only copy it if it's there (sphinx doesn't have it for instance)
                checker_src = os.path.join(self.problem_path, "checker")
                checker_dst = os.path.join(temp_dir, "checker")
                if os.path.exists(checker_src) and os.path.isdir(checker_src):
                    shutil.copytree(checker_src, checker_dst)

                shutil.copytree(
                    os.path.join(self.problem_path, "tests"),
                    os.path.join(temp_dir, "tests"),
                )

                # Copy and make script executable
                script_src = os.path.join(self.problem_path, "run_tests.sh")
                script_dst = os.path.join(temp_dir, "run_tests.sh")
                shutil.copy2(script_src, script_dst)
                os.chmod(script_dst, 0o755)

                # Run the test script
                cmd = [
                    "./run_tests.sh",
                    str(self.memory_limit_kb),
                    str(self.time_limit),
                ]

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=temp_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=3000,  # 5 minute timeout for all tests
                )

                stdout_str = stdout.decode()
                stderr_str = stderr.decode()

                tool_logger.debug(f"Test script stdout:\n{stdout_str}")
                tool_logger.info(f"Test script stderr:\n{stderr_str}")

                # Check for compilation errors first
                if has_compilation_error(stdout_str, stderr_str):
                    return {
                        "success": False,
                        "all_tests_passed": False,
                        "passed_tests": [],
                        "stdout": stdout_str,
                        "stderr": stderr_str,
                    }

                # Parse the output to find which tests passed
                passed_tests = self._parse_passed_tests(stdout_str, stderr_str)

                if process.returncode == 0:
                    # All tests passed
                    return {
                        "success": True,
                        "all_tests_passed": True,
                        "passed_tests": passed_tests,
                        "stdout": stdout_str,
                        "stderr": stderr_str,
                    }
                else:
                    # Some tests failed - we have the list of passed tests
                    return {
                        "success": False,
                        "all_tests_passed": False,
                        "passed_tests": passed_tests,
                        "stdout": stdout_str,
                        "stderr": stderr_str,
                    }

            except asyncio.TimeoutError:
                tool_logger.info(
                    "\033[1;33mTIMEOUT: Tests took longer than 5 minutes\033[0m"
                )
                return {
                    "success": False,
                    "all_tests_passed": False,
                    "passed_tests": [],
                    "stdout": "",
                    "stderr": "Tests timed out after 5 minutes",
                }
            except Exception as e:
                tool_logger.info(f"\033[1;33mEXCEPTION caught: {str(e)}\033[0m")
                return {
                    "success": False,
                    "all_tests_passed": False,
                    "passed_tests": [],
                    "stdout": "",
                    "stderr": f"Error running tests: {str(e)}",
                }

    def _parse_passed_tests(self, stdout: str, stderr: str) -> list:
        """Parse the output to find which tests passed"""
        passed_tests = []

        # Look for "Passed tests: test1 test2 test3" in stdout
        passed_pattern = r"Passed tests:\s*(.+)"
        match = re.search(passed_pattern, stdout)
        if match:
            # Extract the test names (space-separated)
            test_names = match.group(1).strip().split()
            passed_tests.extend(test_names)

        # Remove duplicates while preserving order
        seen = set()
        unique_passed = []
        for test in passed_tests:
            if test not in seen:
                seen.add(test)
                unique_passed.append(test)

        return unique_passed

    def _calculate_subtask_scores(self, test_results: dict) -> dict:
        """
        Calculate scores based on subtask completion.
        Only award points for subtasks where ALL tests pass.

        Args:
            test_results (dict): Results from test execution

        Returns:
            dict: Subtask scores and total score
        """
        passed_tests = set(test_results["passed_tests"])
        completed_subtasks = []
        subtask_scores = {}

        tool_logger.info(f"\033[1;33m Passed tests: {passed_tests} \033[0m")

        for subtask_name, subtask_data in self.subtasks.items():
            # Check if all tests in this subtask passed
            subtask_tests = set(subtask_data["testcases"])
            tool_logger.info(
                f"\033[1;33m Subtask {subtask_name} requires tests: {subtask_tests} \033[0m"
            )

            if subtask_tests.issubset(passed_tests):  # All required tests passed
                completed_subtasks.append(subtask_name)
                subtask_scores[subtask_name] = subtask_data["score"]
                tool_logger.info(
                    f"\033[1;33m Subtask {subtask_name} completed! Awarded {subtask_data['score']} points \033[0m"
                )
            else:
                subtask_scores[subtask_name] = 0
                missing_tests = subtask_tests - passed_tests
                tool_logger.info(
                    f"\033[1;33m Subtask {subtask_name} incomplete. Missing tests: {missing_tests} \033[0m"
                )

        total_score = sum(subtask_scores.values())

        return {
            "total_score": total_score,
            "subtask_scores": subtask_scores,
            "completed_subtasks": completed_subtasks,
        }

    async def call_tool(self, arguments: dict) -> dict:
        """
        Submit C++ code and return IOI-style subtask scores.

        Args:
            arguments (dict): Contains 'cpp_code' key with the code to submit

        Returns:
            dict: Contains score breakdown, submission info, and test results
        """
        # Increment submission count
        self.submission_count += 1

        # Check if maximum submissions reached
        if self.submission_count > self.max_submissions:
            best_total_score = sum(self.best_subtask_scores.values())
            return {
                "total_score": 0,
                "subtask_scores": {},
                "completed_subtasks": [],
                "best_total_score": best_total_score,
                "best_subtask_scores": self.best_subtask_scores.copy(),
                "submission_count": self.submission_count,
                "max_submissions": self.max_submissions,
                "submissions_remaining": 0,
                "error": f"Maximum submissions ({self.max_submissions}) exceeded. This is submission #{self.submission_count}.",
            }

        cpp_input = arguments.get("cpp_code", "")

        # Extract code from the input (handles both raw code and LLM output with code blocks)
        extracted_code = simple_extract_code(cpp_input)

        # If extraction failed, try to use the input as-is (might be raw code)
        if not extracted_code.strip():
            extracted_code = cpp_input

        if not extracted_code.strip():
            best_total_score = sum(self.best_subtask_scores.values())
            return {
                "total_score": 0,
                "subtask_scores": {},
                "completed_subtasks": [],
                "best_total_score": best_total_score,
                "best_subtask_scores": self.best_subtask_scores.copy(),
                "submission_count": self.submission_count,
                "max_submissions": self.max_submissions,
                "submissions_remaining": self.max_submissions - self.submission_count,
                "error": "No C++ code found in the input",
            }

        # Run tests with the solution
        test_results = await self._run_tests_with_solution(extracted_code)

        # Calculate subtask scores
        scoring_results = self._calculate_subtask_scores(test_results)

        # Update best scores with current submission
        self._update_best_scores(scoring_results["subtask_scores"])
        best_total_score = sum(self.best_subtask_scores.values())

        return {
            "total_score": scoring_results["total_score"],
            "subtask_scores": scoring_results["subtask_scores"],
            "completed_subtasks": scoring_results["completed_subtasks"],
            "best_total_score": best_total_score,
            "best_subtask_scores": self.best_subtask_scores.copy(),
            "submission_count": self.submission_count,
            "max_submissions": self.max_submissions,
            "submissions_remaining": self.max_submissions - self.submission_count,
        }
