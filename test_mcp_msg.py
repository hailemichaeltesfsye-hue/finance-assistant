#!/usr/bin/env python
"""
test_mcp_msg.py — Standalone MCP debug script.

This is NOT part of the main app (agent.py / server.py). It's an isolated
tool for inspecting the raw message/part structure that pydantic-ai
produces when talking to our MCP server, useful when debugging issues
like "Connection closed", malformed tool calls, or unexpected outputs.

Run it directly:
    uv run python test_mcp_msg.py
"""
import asyncio
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatCompatibleProvider

# ---- Pick which security role to test as ----
# Swap this to test different privilege levels (must match TOKEN_REGISTRY in server.py)
TOKEN = "tkn_auditor_abc123"   # Auditor (Read-Only)
# TOKEN = "tkn_teller_def456"  # Teller (Standard)
# TOKEN = "tkn_manager_xyz789" # Manager (Admin)

os.environ["MCP_AUTH_TOKEN"] = TOKEN

# Optional: dump the active token to a local file for quick inspection
with open("agent_token.txt", "w") as f:
    f.write(TOKEN)

# ---- Initialize MCP toolset pointing to our local server.py ----
# Under the hood, MCPToolset launches server.py as a stdio subprocess.
toolset = MCPToolset("server.py", max_retries=5)

# Use the small/fast model here since this script is just for structural debugging
provider = OpenAIChatCompatibleProvider(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)
model = OpenAIChatModel("llama-3.1-8b-instant", provider=provider)
agent = Agent(model=model, toolsets=[toolset])


async def main():
    print(f"Active MCP_AUTH_TOKEN: {TOKEN}\n")
    print("Sending test prompt: 'Check the balance of account 1.'\n")

    result = await agent.run("Check the balance of account 1.")

    print("=" * 70)
    print("RAW MESSAGE STRUCTURE")
    print("=" * 70)

    for j, msg in enumerate(result.new_messages()):
        print(f"[{j}] Message type: {type(msg).__name__}")
        for i, part in enumerate(msg.parts):
            print(f"    [{i}] Part type: {type(part).__name__}")
            # Print whatever attributes are relevant for this part type
            for attr in ("tool_name", "args", "content", "text"):
                if hasattr(part, attr):
                    value = getattr(part, attr)
                    value_str = str(value)
                    if len(value_str) > 200:
                        value_str = value_str[:200] + "..."
                    print(f"        {attr}: {value_str}")
        print()

    print("=" * 70)
    print("FINAL AGENT OUTPUT")
    print("=" * 70)
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())