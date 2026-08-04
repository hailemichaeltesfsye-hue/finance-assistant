#!/usr/bin/env python
"""Test agent answer by using direct database tools instead of MCPToolset."""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up env - avoid load_dotenv which fails in certain contexts
if not os.getenv("GROQ_API_KEY"):
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().split('\n'):
            line = line.strip()
            if line and '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key] = value

if not os.getenv("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY not found")
    sys.exit(1)

from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
import db

print(f"GROQ_API_KEY found: {'*' * 20}")
print("-" * 70)

# Define direct database tools  
def search_accounts(query: str) -> list[dict]:
    """Search for accounts by name or role."""
    accounts = db.search_accounts_db(query)
    return [a.model_dump() for a in accounts]

def get_balance(account_id: int) -> dict:
    """Get the balance for an account."""
    balance = db.get_account_balance(account_id)
    return {"account_id": account_id, "balance": balance}

async def main():
    # Create agent with direct tools (no MCPToolset)
    model = GroqModel("llama-3.3-70b-versatile")
    
    agent = Agent(
        model,
        tools=[search_accounts, get_balance],
        system_prompt=(
            "You are a financial assistant. When asked about account balances, use the search_accounts tool "
            "to find the account by name, then use get_balance to fetch the balance. Be concise."
        )
    )
    
    query = "What is Alice's balance?"
    print(f"📩 Query: {query}")
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
