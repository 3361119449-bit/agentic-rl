"""Run the real Qwen-template -> Hermes-parser -> Tau2-tool round trip."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from tau2_agentic_rl.environment.tau2_gym import Tau2GymAdapter
from tau2_agentic_rl.tooling import validate_tool_call


async def verify(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from verl.experimental.agent_loop.tool_parser import ToolParser
    from verl.tools.base_tool import OpenAIFunctionToolSchema

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    environment = Tau2GymAdapter(
        task_id=args.task_id,
        user_model=args.user_model,
        user_llm_args={"api_base": args.base_url},
        user_cache_dir=args.cache_dir,
        max_steps=10,
    )
    incoming = await environment.reset(seed=0)
    try:
        schemas = environment.tool_schemas
        parser_schemas = [OpenAIFunctionToolSchema(**item) for item in schemas]
        schemas_by_name = {
            item["function"]["name"]: item["function"]["parameters"] for item in schemas
        }
        conversation = [
            {"role": "system", "content": environment.policy},
            *incoming,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_all_airports",
                            "arguments": {},
                        },
                    }
                ],
            },
        ]
        rendered = tokenizer.apply_chat_template(
            conversation,
            tools=schemas,
            tokenize=False,
            add_generation_prompt=False,
        )
        start = rendered.rfind("<tool_call>")
        end = rendered.find("</tool_call>", start)
        if start < 0 or end < 0:
            raise AssertionError(
                "Qwen chat template did not render the expected JSON tool-call tags"
            )
        completion = rendered[start : end + len("</tool_call>")]
        token_ids = tokenizer.encode(completion, add_special_tokens=False)
        parser = ToolParser.get_tool_parser("hermes", tokenizer)
        _, calls = await parser.extract_tool_calls(token_ids, parser_schemas)
        if len(calls) != 1:
            raise AssertionError(f"expected exactly one parsed call, got {len(calls)}")
        checked = validate_tool_call(
            calls[0].name,
            calls[0].arguments,
            schemas_by_name,
            environment.tool_names,
        )
        if not checked.valid:
            raise AssertionError(checked)
        before = environment.db_hash()
        step = await environment.step_tool(checked.name, checked.arguments)
        after = environment.db_hash()
        if step.tool_success is not True:
            raise AssertionError(f"Tau2 tool failed: {step.tool_result}")
        if before != after:
            raise AssertionError("read-only parser test unexpectedly changed the DB")
        print(
            json.dumps(
                {
                    "parsed_calls": 1,
                    "tool": checked.name,
                    "tau2_tool_success": True,
                    "db_unchanged": True,
                },
                indent=2,
            )
        )
    finally:
        await environment.force_cleanup_stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--task-id", default="0")
    parser.add_argument("--user-model", default=os.environ.get("DEEPSEEK_USER_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("outputs/tool_test_user_cache")
    )
    args = parser.parse_args()
    if (
        not args.user_model
        or not args.base_url
        or not os.environ.get("DEEPSEEK_API_KEY")
    ):
        raise RuntimeError(
            "set DEEPSEEK_USER_MODEL, DEEPSEEK_BASE_URL, and DEEPSEEK_API_KEY"
        )
    asyncio.run(verify(args))


if __name__ == "__main__":
    main()
