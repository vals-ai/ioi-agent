import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

import build_sft
import rendering
import train_sft
from rendering import EndpointQwen35Renderer, render_turn
from train_sft import (
    index_datasets,
    learning_rate,
    make_batch,
    mean_nll,
    source_weights,
)


class RenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = get_tokenizer("Qwen/Qwen3.5-9B")
        cls.renderer = EndpointQwen35Renderer(cls.tokenizer)

    def history(self, reasoning="target thoughts", content="answer"):
        return [
            {"kind": "text", "text": "question"},
            {
                "kind": "raw_response",
                "response": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                },
            },
        ]

    def test_target_only_mask_and_thinking(self):
        row = render_turn(self.renderer, self.history(), [])
        self.assertEqual(len(row["input_ids"]), len(row["target_ids"]))
        self.assertEqual(
            row["weights"][: row["prompt_tokens"] - 1], [0] * (row["prompt_tokens"] - 1)
        )
        self.assertTrue(
            all(value == 1 for value in row["weights"][row["prompt_tokens"] - 1 :])
        )
        text = self.tokenizer.decode(row["input_ids"] + [row["target_ids"][-1]])
        self.assertIn("target thoughts", text)
        self.assertIn("answer", text)

    def test_current_tool_chain_thinking_preserved(self):
        history = self.history("CHAIN REASONING")
        history.extend(
            [
                {
                    "kind": "tool_result",
                    "tool_call": {"id": "call"},
                    "result": "test result",
                },
                {
                    "kind": "raw_response",
                    "response": {
                        "role": "assistant",
                        "content": "next answer",
                        "reasoning_content": "next thought",
                    },
                },
            ]
        )
        row = render_turn(self.renderer, history, [])
        prefix = self.tokenizer.decode(row["input_ids"][: row["prompt_tokens"]])
        self.assertIn("CHAIN REASONING", prefix)
        history.insert(-1, {"kind": "text", "text": "new user question"})
        row = render_turn(self.renderer, history, [])
        prefix = self.tokenizer.decode(row["input_ids"][: row["prompt_tokens"]])
        self.assertNotIn("CHAIN REASONING", prefix)

    def test_tool_call_and_empty_thinking(self):
        history = self.history("", "")
        history[-1]["response"]["tool_calls"] = [
            {
                "id": "call",
                "type": "function",
                "function": {"name": "run", "arguments": '{"code":"hello"}'},
            }
        ]
        row = render_turn(self.renderer, history, [])
        self.assertIn("<function=run>", self.tokenizer.decode(row["target_ids"]))
        with self.assertRaisesRegex(ValueError, "No assistant"):
            render_turn(self.renderer, [], [])
        with self.assertRaisesRegex(ValueError, "Unsupported history"):
            render_turn(self.renderer, [{"kind": "image"}, history[-1]], [])

    def test_matches_hf_template_with_history_gate(self):
        messages = [
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": "old answer",
                "reasoning_content": "OLD THOUGHT",
            },
            {"role": "user", "content": "second"},
            {
                "role": "assistant",
                "content": "experiment",
                "reasoning_content": "CHAIN THOUGHT",
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "run", "arguments": {"code": "hello"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c", "content": "result"},
        ]
        expected = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )["input_ids"]
        converted = [
            {
                "role": m["role"],
                "content": m["content"],
                **({"tool_call_id": m["tool_call_id"]} if "tool_call_id" in m else {}),
            }
            for m in messages
        ]
        for original, converted_message in zip(messages, converted):
            if original["role"] == "assistant":
                converted_message["content"] = [
                    {"type": "thinking", "thinking": original["reasoning_content"]},
                    {"type": "text", "text": original["content"]},
                ]
                if "tool_calls" in original:
                    converted_message["tool_calls"] = [
                        renderers.ToolCall(
                            id="c",
                            function=renderers.ToolCall.FunctionBody(
                                name="run", arguments='{"code":"hello"}'
                            ),
                        )
                    ]
        self.assertEqual(
            self.renderer.build_generation_prompt(converted).to_ints(), expected
        )


class BuildTests(unittest.TestCase):
    def make_run(self, root, name, problem="p", score=1, error=None):
        directory = root / name / "000-p"
        directory.mkdir(parents=True)
        (directory / "score.json").write_text(
            json.dumps({"task_id": problem, "score": score, "returncode": 0})
        )
        (directory / "result.json").write_text(json.dumps({"final_error": error}))
        return directory.parent

    def test_filters_caps_partial_and_heldout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runs = [
                self.make_run(root, "one", error={"type": "MaxTurnsExceeded"}),
                self.make_run(root, "two", error={"type": "MaxTimeExceeded"}),
                self.make_run(root, "three"),
                self.make_run(
                    root, "bad", problem="bad", error={"type": "RateLimitError"}
                ),
                self.make_run(root, "partial", problem="partial", score=0.5),
                self.make_run(root, "dev", problem="heldout"),
            ]
            (root / "one/unfinished").mkdir()
            (root / "one/unfinished/score.json").write_text("{")
            kept, reasons = build_sft.candidates(runs, 1, 2, {"heldout"})
            self.assertEqual(len(kept), 2)
            self.assertEqual(reasons["per_problem_cap"], 1)
            self.assertEqual(reasons["terminal_error"], 1)
            self.assertEqual(reasons["score"], 1)
            self.assertEqual(reasons["heldout"], 1)
            self.assertEqual(reasons["invalid_score"], 1)
            kept, _ = build_sft.candidates(runs, 0.5, 2, {"heldout"})
            self.assertEqual(len(kept), 3)

    def test_empty_and_overlong_build(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = self.make_run(root, "run")
            turn = run / "000-p/logs/a/turns/turn_001"
            init = turn.parent / "init"
            turn.mkdir(parents=True)
            init.mkdir()
            (init / "config.json").write_text('{"tools":[]}')
            (turn / "history.json").write_text("[]")
            (turn / "result.json").write_text("{}")
            out = root / "out.jsonl"
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "build_sft",
                        "--run-dir",
                        str(run),
                        "--output",
                        str(out),
                        "--max-tokens",
                        "2",
                    ],
                ),
                patch.object(
                    build_sft,
                    "render_turn",
                    return_value={
                        "sequence_tokens": 3,
                        "input_ids": [1, 2],
                        "target_ids": [2, 3],
                        "weights": [0, 1],
                        "prompt_tokens": 2,
                        "target_tokens": 1,
                    },
                ),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(SystemExit, "No datums"),
            ):
                build_sft.main()
            stats = json.loads(out.with_suffix(".stats.json").read_text())
            self.assertEqual(stats["datums"], 0)
            self.assertEqual(stats["skipped_turns"]["over_length"], 1)
            self.assertEqual(build_sft.candidates([], 1, 2, set())[0], [])

    def test_trajectory_masks_and_prefix_breaks(self):
        def record(tokens, prompt):
            return {
                "input_ids": tokens[:-1],
                "target_ids": tokens[1:],
                "prompt_tokens": prompt,
            }

        rows = [record([1, 2, 3, 4], 2), record([1, 2, 3, 4, 5, 6, 7], 5)]
        merged = rendering.merge_turns(rows)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["weights"], [0, 1, 1, 0, 1, 1])
        self.assertEqual(merged[0]["prefix_verified_turns"], 2)
        split = rendering.merge_turns(rows + [record([1, 8, 9, 10], 3)])
        self.assertEqual(len(split), 2)
        self.assertEqual(split[1]["weights"], [0, 0, 1])
        self.assertEqual(rendering.merge_turns([]), [])
        with self.assertRaises(ValueError):
            rendering.merge_turns([record([1, 2], 2)])

    def test_ioi_points_best_trajectory_and_dev_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runs = [
                self.make_run(root, "partial", problem="2020/plants", score=70),
                self.make_run(root, "best", problem="2020/plants", score=90),
                self.make_run(
                    root,
                    "error",
                    problem="2020/plants",
                    score=100,
                    error={"type": "RateLimitError"},
                ),
                self.make_run(root, "dev", problem="2024/mosaic", score=100),
                self.make_run(root, "future", problem="2026/new", score=100),
                self.make_run(root, "full", problem="2021/dna", score=100),
            ]
            kept, rejected = build_sft.candidates(runs, 1, 2, set(), minimum_ioi=60)
            self.assertEqual([source for _, source, _ in kept], ["full", "best"])
            self.assertEqual(rejected["heldout_ioi"], 2)
            self.assertEqual(rejected["terminal_error"], 1)
            self.assertEqual(rejected["per_problem_cap"], 1)
            kept, _ = build_sft.candidates(runs, 1, 2, set())
            self.assertEqual([source for _, source, _ in kept], ["full"])


class TrainerTests(unittest.TestCase):
    def test_epoch_batches_clipping_and_checkpoint_names(self):
        class Future:
            def __init__(self, value=None):
                self.value = value

            def result(self):
                return self.value

        batches, params, saves = [], [], []

        class Client:
            def forward_backward(self, rows, **kwargs):
                batches.append(len(rows))
                return Future()

            def optim_step(self, value):
                params.append(value)
                return Future()

            def save_weights_for_sampler(self, name):
                saves.append(name)
                return Future(type("Saved", (), {"path": "fake://" + name})())

        class Service:
            def create_lora_training_client(self, **kwargs):
                return Future(Client())

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.jsonl"
            row = {
                "source": "teacher",
                "input_ids": [1, 2],
                "target_ids": [2, 3],
                "weights": [0, 1],
            }
            path.write_text((json.dumps(row) + "\n") * 3)
            argv = [
                "train_sft",
                "--data",
                str(path),
                "--name",
                "test",
                "--output",
                str(Path(temp) / "out"),
                "--epochs",
                "2",
                "--batch-size",
                "2",
                "--save-epoch-one",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(train_sft.tinker, "ServiceClient", return_value=Service()),
                patch.object(train_sft, "mean_nll", return_value=1),
            ):
                train_sft.main()
        self.assertEqual(batches, [2, 1, 2, 1])
        self.assertEqual(saves, ["test-e1", "test"])
        self.assertTrue(all(value.grad_clip_norm == 1 for value in params))

    def test_schedule_and_weights(self):
        self.assertEqual(learning_rate(0, 10, 2, 1), 0.5)
        self.assertEqual(learning_rate(1, 10, 2, 1), 1)
        self.assertAlmostEqual(learning_rate(9, 10, 2, 1), 0.1)
        self.assertEqual(source_weights({"a": [1, 2], "b": [3]}, []), {"a": 2, "b": 1})
        self.assertEqual(
            source_weights({"a": [1], "b": [2]}, ["a=4"]), {"a": 4, "b": 1}
        )
        for weights in [["a=-1"], ["a=nan"], ["missing=1"], ["a=0", "b=0"]]:
            with self.assertRaises(ValueError):
                source_weights({"a": [1], "b": [2]}, weights)

    def test_batch_and_nll(self):
        rows = [
            {
                "source": "a",
                "input_ids": [1, 2, 3],
                "target_ids": [2, 3, 4],
                "weights": [0, 1, 1],
            }
        ]
        batch = make_batch(rows)
        self.assertEqual(batch[0].loss_fn_inputs["weights"].data, [0, 0.5, 0.5])
        output = type(
            "Output",
            (),
            {
                "loss_fn_outputs": [
                    {
                        "logprobs": tinker.TensorData(
                            data=[-100, -1, -3], dtype="float32", shape=[3]
                        )
                    }
                ]
            },
        )()
        self.assertEqual(mean_nll(output, rows), 2)
        self.assertTrue(math.isfinite(mean_nll(output, rows)))
        self.assertEqual(
            float(np.array(batch[0].loss_fn_inputs["weights"].data).sum()), 1
        )

    def test_index_empty_bad_and_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.jsonl"
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "No SFT"):
                index_datasets([path])
            row = {
                "source": "teacher",
                "input_ids": [1, 2],
                "target_ids": [2, 3],
                "weights": [0, 1],
            }
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(len(index_datasets([path])["teacher"]), 1)
            row["weights"] = [0, 0]
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "loss mask"):
                index_datasets([path])


if __name__ == "__main__":
    unittest.main()
