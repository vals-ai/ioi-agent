"""Inspect a saved ATIF trajectory using Tinker's Qwen3.5 renderer."""

import argparse
import json
import os
from pathlib import Path

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from train.rendering import EndpointQwen35Renderer


def load_messages(trajectory):
    messages = []
    for step in trajectory["steps"]:
        source = step["source"]
        if source in ("user", "system"):
            messages.append({"role": source, "content": step.get("message", "")})
        elif source == "agent":
            content = []
            if step.get("reasoning_content"):
                content.append(
                    {"type": "thinking", "thinking": step["reasoning_content"]}
                )
            content.append({"type": "text", "text": step.get("message", "")})
            message = {"role": "assistant", "content": content}
            if step.get("tool_calls"):
                message["tool_calls"] = [
                    renderers.ToolCall(
                        id=call["tool_call_id"],
                        function=renderers.ToolCall.FunctionBody(
                            name=call["function_name"],
                            arguments=json.dumps(call["arguments"]),
                        ),
                    )
                    for call in step["tool_calls"]
                ]
            messages.append(message)
            for result in step.get("observation", {}).get("results", []):
                messages.append(
                    {
                        "role": "tool",
                        "content": result["content"],
                        "tool_call_id": result["source_call_id"],
                    }
                )
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--verify-endpoint", action="store_true")
    args = parser.parse_args()
    trajectory = json.loads(args.trajectory.read_text())
    messages = load_messages(trajectory)
    tokenizer = get_tokenizer(args.model)
    renderer = EndpointQwen35Renderer(tokenizer)
    tool_specs = trajectory.get("agent", {}).get("tool_definitions", [])
    specs = []
    for tool in tool_specs:
        if "function" in tool:
            specs.append(tool)
            continue
        body = tool.get("body", tool)
        parameters = body.get(
            "parameters",
            {
                "type": "object",
                "properties": body.get("properties", {}),
                "required": body.get("required", []),
            },
        )
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": body.get("description", ""),
                    "parameters": parameters,
                },
            }
        )
    for tool in specs:
        # Tinker drops OpenAI's function.strict, but retains the parameter schema.
        tool["function"].pop("strict", None)
        tool["function"]["parameters"].setdefault("additionalProperties", False)
    system = (
        messages.pop(0)["content"]
        if messages and messages[0]["role"] == "system"
        else ""
    )
    messages = renderer.create_conversation_prefix_with_tools(specs, system) + messages
    tokens = renderer.build_generation_prompt(messages).to_ints()
    decoded = tokenizer.decode(tokens)
    args.trajectory.with_suffix(".rendered.txt").write_text(decoded)
    endpoint_check = None
    if args.verify_endpoint:
        import httpx

        first_user = next(m for m in messages if m["role"] == "user")
        prompt = renderer.build_generation_prompt(
            renderer.create_conversation_prefix_with_tools(specs, system) + [first_user]
        ).to_ints()
        response = httpx.post(
            os.environ.get(
                "TINKER_BASE_URL",
                "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
            )
            + "/chat/completions",
            headers={
                "Authorization": "Bearer "
                + (os.environ.get("TINKER_API_KEY") or os.environ["TINKER_DEVIN_KEY"])
            },
            json={
                "model": args.model,
                "messages": [first_user],
                "tools": specs,
                "max_tokens": 1,
                "reasoning_effort": True,
            },
            timeout=120,
        )
        response.raise_for_status()
        server_tokens = response.json()["usage"]["prompt_tokens"]
        endpoint_check = {
            "cookbook_prompt_tokens": len(prompt),
            "endpoint_prompt_tokens": server_tokens,
        }
        assert len(prompt) == server_tokens, endpoint_check
    checked = 0
    thinking = 0
    calls = 0
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = renderer.build_generation_prompt(messages[:index]).to_ints()
        example, _ = renderer.build_supervised_example(messages[: index + 1])
        complete = example.to_ints()
        if complete[: len(prefix)] != prefix:
            raise AssertionError("Supervised prompt differs from inference prefix")
        parsed, termination = renderer.parse_response(complete[len(prefix) :])
        expected_calls = [
            (c.function.name, json.loads(c.function.arguments))
            for c in message.get("tool_calls", [])
        ]
        parsed_calls = [
            (c.function.name, json.loads(c.function.arguments))
            for c in parsed.get("tool_calls", [])
        ]
        assert parsed_calls == expected_calls, (expected_calls, parsed_calls)
        expected_thinking = [
            p["thinking"] for p in message["content"] if p["type"] == "thinking"
        ]
        actual_thinking = (
            [p["thinking"] for p in parsed["content"] if p["type"] == "thinking"]
            if isinstance(parsed["content"], list)
            else []
        )
        assert [x.strip() for x in actual_thinking] == [
            x.strip() for x in expected_thinking
        ]
        checked += 1
        thinking += bool(expected_thinking)
        calls += len(expected_calls)
    print(
        json.dumps(
            {
                "renderer": "qwen3_5",
                "generation_prompt_tokens": len(tokens),
                "assistant_roundtrips": checked,
                "thinking_roundtrips": thinking,
                "tool_call_roundtrips": calls,
                "tool_specs_in_atif": len(tool_specs),
                "endpoint_check": endpoint_check,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
