#!/usr/bin/env python3
"""Export the exact official Tau2 airline agent system prompt and tool schemas.

Run this file through the official Tau2 environment, for example:
    uv run python /path/to/export_tau2_airline_context.py --output context.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

from tau2.agent.llm_agent import LLMAgent
from tau2.runner.build import build_environment
from tau2.utils.utils import get_commit_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environment = build_environment("airline")
    tools = sorted(environment.get_tools(), key=lambda tool: tool.name)
    agent = LLMAgent(
        tools=tools,
        domain_policy=environment.get_policy(),
        llm="context-export-only",
        llm_args={},
    )
    payload = {
        "domain": "airline",
        "tau2_version": importlib.metadata.version("tau2"),
        "tau2_git_commit": get_commit_hash(),
        "system_prompt": agent.system_prompt,
        "tools": [tool.openai_schema for tool in tools],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
