#!/usr/bin/env python
"""Simple test to ask the agent 'What is Alice's balance?'"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up env - avoid load_dotenv which fails in certain contexts
if not os.getenv("GROQ_API_KEY"):
    # Try to read from .env file manually
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().split('\n'):
            line = line.strip()
            if line and '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key] = value

# Verify API key
if not os.getenv("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY not found in .env file")
    sys.exit(1)

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.groq import GroqModel

os.environ["MCP_AUTH_TOKEN"] = "tkn_auditor_abc123"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONUTF8"] = "1"

# Get the path to server.py
script_dir = Path(__file__).parent
server_path = script_dir / "server.py"

print(f"Using server: {server_path}")
print(f"MCP Token: {os.getenv('MCP_AUTH_TOKEN')}")
print("-" * 70)

async def main():
    # Create agent with MCPToolset
    model = GroqModel("llama-3.3-70b-versatile")
    # Increase timeout to 10 seconds since Windows subprocess communication can be slow
    toolset = MCPToolset(str(server_path), max_retries=2, session_timeout=10.0)
    
    agent = Agent(
        model,
        toolsets=[toolset],
        retries=2,
        system_prompt=(
            "You are a financial assistant. When asked about account balances, use the search_accounts tool "
            "to find the account by name, then report the balance. Be concise and direct."
        )
    )
    
    query = "What is Alice's balance?"
    print(f"\n📩 Query: {query}")
    print("-" * 70)
    
    try:
        result = await agent.run(query)
        print(f"\n✅ SUCCESS!\n")
        print(result.output)
        print("\n" + "-" * 70)
        
        # Show tool calls if any
        tool_calls = [
            p for msg in result.new_messages() 
            for p in msg.parts 
            if hasattr(p, 'tool_name')
        ]
        if tool_calls:
            print(f"\n🛠️  Tools used: {[getattr(p, 'tool_name', '?') for p in tool_calls]}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
