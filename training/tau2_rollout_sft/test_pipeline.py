#!/usr/bin/env python3
"""Offline regression tests for the Tau2 rollout/SFT utilities."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import convert_tau2_results_to_sft
import report_pass1_pass4


HERE = Path(__file__).resolve().parent


def simulation(task_id: str, trial: int, reward: float) -> dict:
    return {
        "id": f"sim-{task_id}-{trial}",
        "task_id": task_id,
        "trial": trial,
        "termination_reason": "user_stop",
        "reward_info": {"reward": reward},
        "messages": [
            {"role": "assistant", "content": "Hello", "tool_calls": None},
            {"role": "user", "content": "Look up my reservation", "tool_calls": None},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "get_reservation_details",
                        "arguments": {"reservation_id": "ABC123"},
                    }
                ],
                "raw_data": {"message": {"reasoning_content": None}},
            },
            {
                "role": "tool",
                "id": "call-1",
                "content": '{"status":"available"}',
                "error": False,
            },
            {"role": "assistant", "content": "It is available", "tool_calls": None},
        ],
    }


class PipelineTests(unittest.TestCase):
    def test_areal_sampling_matches_tau2_row_count(self) -> None:
        paths = []
        for suffix in (".areal.jsonl", ".tau2.jsonl", ".sampled.jsonl"):
            handle = tempfile.NamedTemporaryFile(dir=HERE, suffix=suffix, delete=False)
            handle.close()
            paths.append(Path(handle.name))
        areal, reference, output = paths
        manifest = output.with_suffix(".manifest.json")
        try:
            source_rows = [
                {
                    "messages": [{"role": "user", "content": f"request-{index}"}],
                    "answer": {"role": "assistant", "content": f"answer-{index}"},
                    "metadata": {"source_dialog_id": f"dialog-{index}"},
                }
                for index in range(8)
            ]
            areal.write_text(
                "".join(json.dumps(row) + "\n" for row in source_rows),
                encoding="utf-8",
            )
            reference.write_text("{}\n{}\n{}\n", encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "sample_areal_to_match_tau2.py"),
                    "--areal",
                    str(areal),
                    "--tau2-reference",
                    str(reference),
                    "--output",
                    str(output),
                    "--seed",
                    "7",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            sampled = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(sampled), 3)
            self.assertEqual(
                len({row["metadata"]["source_dialog_id"] for row in sampled}), 3
            )
            details = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(details["tau2_reference_rows"], 3)
            self.assertEqual(details["output_rows"], 3)
        finally:
            for path in (*paths, manifest):
                path.unlink(missing_ok=True)

    @unittest.skipUnless(
        os.getenv("RUN_QWEN_INTEGRATION") == "1",
        "set RUN_QWEN_INTEGRATION=1 to load the Qwen tokenizer",
    )
    def test_qwen_tool_template(self) -> None:
        tokenizer = convert_tau2_results_to_sft.load_tokenizer(
            "Qwen/Qwen3-4B-Instruct-2507"
        )
        messages = [
            {"role": "system", "content": "Follow policy."},
            {"role": "user", "content": "Look up ABC123"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "get_reservation_details",
                        "arguments": '{"reservation_id":"ABC123"}',
                    }
                ],
            },
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_reservation_details",
                    "description": "Get reservation details",
                    "parameters": {
                        "type": "object",
                        "properties": {"reservation_id": {"type": "string"}},
                        "required": ["reservation_id"],
                    },
                },
            }
        ]
        length = convert_tau2_results_to_sft.token_length(
            tokenizer, messages, tools
        )
        self.assertGreater(length, 0)
        self.assertLess(length, 16_384)

    def test_official_pass1_pass4_formula(self) -> None:
        simulations = []
        for trial in range(4):
            simulations.append(simulation("0", trial, 1.0))
            simulations.append(simulation("1", trial, 1.0 if trial < 2 else 0.0))
        self.assertEqual(
            report_pass1_pass4.compute(simulations),
            {"pass^1": 0.75, "pass^4": 0.5},
        )

    def test_successful_train_conversion(self) -> None:
        paths = []
        for suffix in (".results.json", ".context.json", ".sft.jsonl"):
            handle = tempfile.NamedTemporaryFile(dir=HERE, suffix=suffix, delete=False)
            handle.close()
            paths.append(Path(handle.name))
        results, context, output = paths
        manifest = output.with_suffix(".manifest.json")
        try:
            results.write_text(
                json.dumps(
                    {
                        "info": {
                            "git_commit": "test-commit",
                            "agent_info": {"llm": "deepseek/deepseek-v4-flash"},
                            "user_info": {"llm": "deepseek/deepseek-v4-pro"},
                            "environment_info": {"domain_name": "airline"},
                        },
                        "simulations": [
                            simulation("0", 0, 1.0),
                            simulation("0", 1, 0.0),
                        ],
                    }
                ),
                encoding="utf-8",
            )
            context.write_text(
                json.dumps(
                    {
                        "domain": "airline",
                        "system_prompt": "Follow the airline policy.",
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_reservation_details",
                                    "description": "Get reservation details",
                                    "parameters": {"type": "object", "properties": {}},
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "convert_tau2_results_to_sft.py"),
                    "--results",
                    str(results),
                    "--context",
                    str(context),
                    "--output",
                    str(output),
                    "--skip-token-count",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["answer"]["tool_calls"][0]["name"], "get_reservation_details")
            self.assertEqual(rows[1]["answer"]["content"], "It is available")
            self.assertTrue(all(row["tools"] for row in rows))
            serialized = json.dumps(rows).lower()
            self.assertNotIn('"reasoning"', serialized)
            self.assertNotIn('"thinking"', serialized)
        finally:
            for path in (*paths, manifest):
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
