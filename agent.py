#!/usr/bin/env python
import asyncio
import os
import sys
import signal
from typing import Any

from dotenv import load_dotenv
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key or groq_key.strip() == "your_groq_api_key_here":
    print("\033[91m[Error] GROQ_API_KEY is not set or has the placeholder value in .env.\033[0m")
    print("Please edit the \033[94m.env\033[0m file and replace it with your actual Groq API key.")
    print("You can get a Groq API key at: https://console.groq.com")
    sys.exit(1)

try:
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPToolset
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart
    from pydantic_ai.exceptions import ModelHTTPError
    from pydantic_ai.models.groq import GroqModel
    from fastmcp.client.transports import PythonStdioTransport
except ImportError as e:
    print(f"\033[91m[Error] Failed to import Pydantic AI dependencies: {e}\033[0m")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt
except ImportError:
    class FallbackConsole:
        def print(self, *args, **kwargs):
            print(*args)
    Console = FallbackConsole
    Panel = lambda content, *args, **kwargs: content
    Table = None
    Prompt = None

# ==================== PII REDACTION LAYER ====================
from pii_registry import PIIRegistry
import db  # read-only import, just to seed known names at startup — not a security boundary

pii_registry = PIIRegistry()


def _seed_pii_registry():
    """
    Pre-registers every known account owner as a token before the chat starts,
    so names are redactable from the user's very first message.
    """
    try:
        for account in db.list_accounts_db():
            pii_registry.redact(account.owner_name)
    except Exception as e:
        print(f"\033[93m[Warning] Could not seed PII registry: {e}\033[0m")


_seed_pii_registry()


def redact_text(text: str) -> str:
    """Replaces every known real name in `text` with its token, longest names first
    so a name that's a substring of another doesn't get partially corrupted."""
    for real_name in sorted(pii_registry.to_token, key=len, reverse=True):
        text = text.replace(real_name, pii_registry.to_token[real_name])
    return text


def redact_value(value: Any) -> Any:
    """Recursively redacts owner_name-like strings inside a tool result (dict or list of dicts)."""
    if isinstance(value, dict):
        new_val = dict(value)
        for key in ("owner_name",):
            if key in new_val and isinstance(new_val[key], str):
                new_val[key] = pii_registry.redact(new_val[key])
        return new_val
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    return value


def restore_value(value: Any) -> Any:
    """Restores any redaction tokens found in a string back to real names before hitting server.py."""
    if isinstance(value, str):
        return pii_registry.restore(value)
    return value
# ==================== END PII SECTION ====================

console = Console()

MODEL_NAME = "llama-3.3-70b-versatile"
model = GroqModel(MODEL_NAME)

# ==================== LEAST PRIVILEGE ACCESS (LPA) CONFIGURATION ====================
# NOTE: This registry is for the DEMO'S OWN reference/UI display only (e.g. print_hud).
# The actual enforcement lives ONLY in server.py's check_permission().
DEMO_TOKENS = {
    "auditor": {
        "name": "Auditor (Read-Only)",
        "token": "tkn_auditor_abc123",
        "scopes": ["read:accounts"],
        "desc": "Can check balances, search accounts, and search transactions. Cannot modify state."
    },
    "teller": {
        "name": "Teller (Standard)",
        "token": "tkn_teller_def456",
        "scopes": ["read:accounts", "write:transactions"],
        "desc": "Can do Auditor actions + deposit, withdraw, transfer, and pay loans."
    },
    "manager": {
        "name": "Manager (Admin)",
        "token": "tkn_manager_xyz789",
        "scopes": ["read:accounts", "write:transactions", "admin:loans", "admin:accounts"],
        "desc": "Full administrative access. Can perform all Teller actions, plus list all accounts, apply for loans, and approve/reject pending actions."
    }
}

ACTIVE_ROLE = "Auditor (Read-Only)"
ACTIVE_TOKEN = "tkn_auditor_abc123"

# Tracks whichever Rich console.status(...) spinner is currently animating.
_active_status = None

# ==================== IDEMPOTENCY GUARD ====================
# Tracks which write-tool calls have already executed THIS user turn (reset at
# the top of every fresh run_agent_with_fallback call, i.e. _retry_count == 0).
# A 429/leak retry restarts the ENTIRE prompt from scratch, which can otherwise
# re-issue a tool call whose side effect already happened once.
_executed_this_turn: dict = {}

WRITE_TOOL_NAMES = {
    "deposit_funds", "withdraw_funds", "transfer_funds",
    "apply_for_loan", "pay_loan", "approve_transaction", "reject_transaction",
}


def _call_signature(name: str, args: dict) -> tuple:
    return (name, tuple(sorted(args.items())))
# ==================== END IDEMPOTENCY GUARD ====================


async def hitl_process_tool_call(ctx, call_tool, name: str, tool_args: dict) -> Any:
    """
    MCPToolset's process_tool_call hook.
    PII: restore tokens to real names before dispatch; redact real names in the result.
    HITL: enforcement lives in server.py now (gated tools return pending_approval on
    their own) — this hook just notices that and prints a friendly panel about it.
    Idempotency: write-tool calls are cached per-turn so a mid-turn retry/fallback
    can't accidentally execute the same side effect twice.
    """
    restored_args = {k: restore_value(v) for k, v in tool_args.items()}

    if name in WRITE_TOOL_NAMES:
        sig = _call_signature(name, restored_args)
        if sig in _executed_this_turn:
            return _executed_this_turn[sig]

    result = await call_tool(name, restored_args)
    result = redact_value(result)

    if name in WRITE_TOOL_NAMES:
        _executed_this_turn[_call_signature(name, restored_args)] = result

    if isinstance(result, dict) and result.get("status") == "pending_approval":
        if _active_status is not None:
            _active_status.stop()
        console.print()
        console.print(Panel(
            f"Tool: [bold white]{name}[/bold white]\n"
            f"Pending ID: [bold cyan]{result.get('pending_id')}[/bold cyan]\n"
            f"{result.get('message', '')}",
            title="⏸️  AWAITING HUMAN APPROVAL",
            border_style="yellow",
        ))
        if _active_status is not None:
            _active_status.start()

    return result


SYSTEM_PROMPT = (
    "You are a helpful, secure, and professional Financial Assistant Agent. "
    "You have direct access to financial tools via a Model Context Protocol (MCP) server.\n\n"
    "### PRIVACY & SECURITY NOTE:\n"
    "You will only see masked identifiers such as '[REDACTED_NAME_A]' and '[REDACTED_NAME_B]', so you never "
    "learn a real customer's name. You are to interact with tools using these exact redacted strings. "
    "Never try to guess, bypass, or invent new names or account IDs.\n\n"
    "### AVAILABLE TOOLS:\n"
    "- search_accounts(query): find accounts by owner name or role. Returns id, owner_name, balance, role, status.\n"
    "- get_balance(account_id): get the balance for a numeric account id.\n"
    "- search_transactions(query): search transactions by type/status/account number.\n"
    "- deposit_funds(account_id, amount): deposit into an account.\n"
    "- withdraw_funds(account_id, amount): withdraw from an account.\n"
    "- transfer_funds(from_account_id, to_account_id, amount): move funds between accounts.\n"
    "- apply_for_loan(account_id, amount): apply for a new loan.\n"
    "- pay_loan(loan_id, amount): pay towards an existing loan.\n"
    "- approve_transaction(pending_id): finalize a pending action. Manager credentials only.\n"
    "- reject_transaction(pending_id): reject a pending action. Manager credentials only.\n"
    "- list_accounts(filter_status): list all accounts (admin only).\n"
    "- search_audit_log(query): search the security audit log — every permission check and HITL event. Manager credentials only.\n\n"
    "### CRITICAL OPERATIONAL RULES:\n"
    "1. **Account Name to ID Resolution**: If a query mentions a redacted person's name (e.g. '[REDACTED_NAME_A]'), "
    "you MUST perform this sequential multi-step process, waiting for each tool's result before calling the next:\n"
    "   - Step 1: Call search_accounts natively with the redacted name to find the account.\n"
    "   - Step 2: Extract the correct numeric account_id from what was returned.\n"
    "   - Step 3: Call the appropriate action tool (e.g. get_balance, deposit_funds, apply_for_loan) using that "
    "retrieved account_id.\n"
    "   - If TWO people are involved (e.g. a transfer), resolve BOTH names first, as two separate sequential "
    "search_accounts calls, before calling the action tool.\n"
    "   - Never attempt to make up or guess an ID.\n"
    "   - THIS RULE APPLIES TO EVERY MULTI-STEP REQUEST, not just transfers: loans, deposits, withdrawals, and "
    "payments all require the same name-to-ID lookup first. NEVER nest one tool call's syntax inside another tool "
    "call's arguments, and NEVER emit a tool call as literal text in your response — always use the real tool-calling "
    "mechanism, one call at a time.\n"
    "2. **Native Tool Calling Only**: Never output raw tool-calling syntax, XML tags, or code-like strings in your text responses.\n"
    "3. **Permission Denials Are Expected**: If a tool call returns {'status': 'error', ...}, DO NOT retry the same call — "
    "explain to the user, in plain language, that their current role lacks the required privilege, stating which scope "
    "was needed and which scopes they currently hold.\n"
    "4. **Human-in-the-Loop for Transactions Exceeding $1000**:\n"
    "   - Any deposit, withdrawal, transfer, or loan request over $1000 will come back with status 'pending_approval' "
    "instead of executing immediately.\n"
    "   - When you see 'pending_approval', tell the user plainly that the request is flagged and awaiting human approval, "
    "and give them the pending_id. Do NOT call any other tool or make assumptions about whether it succeeded.\n"
    "   - You are STRICTLY PROHIBITED from calling approve_transaction or reject_transaction on your own initiative. "
    "Only call one of them when a human operator explicitly instructs you to approve or reject a specific pending_id.\n"
    "   - Once instructed, call the correct tool immediately with that pending_id, then summarize the updated status "
    "for the user.\n"
    "5. **One Tool Call At A Time**: Call at most ONE tool per turn.\n"
    "6. **Summarize Clearly**: Always state plainly what you did (or attempted) and summarize results for the user."
)


def clean_message_history(history: list | None) -> list | None:
    """Removes any model messages that contained raw leaked function-call syntax, to prevent chat contamination."""
    if not history:
        return history
    cleaned = []
    for msg in history:
        has_leak = False
        if hasattr(msg, "parts"):
            for part in msg.parts:
                part_text = getattr(part, "text", "") or ""
                if "<function=" in part_text:
                    has_leak = True
                    break
        if not has_leak:
            cleaned.append(msg)
    return cleaned


def make_toolset_for_token(token: str) -> MCPToolset:
    """Spawns a fresh server.py subprocess with MCP_AUTH_TOKEN baked into its env at launch time."""
    transport = PythonStdioTransport(
        script_path="server.py",
        python_cmd=sys.executable,
        env={**os.environ, "MCP_AUTH_TOKEN": token},
        keep_alive=False,
    )
    return MCPToolset(transport, process_tool_call=hitl_process_tool_call)


def get_agent_for_token(token: str, temperature: float = 0.0, active_model: Any = None) -> Agent:
    """Creates a fresh Agent + fresh MCP subprocess scoped to a single token/role."""
    toolset = make_toolset_for_token(token)
    return Agent(
        active_model or model,
        toolsets=[toolset],
        retries=2,
        model_settings={'parallel_tool_calls': False, 'temperature': temperature},
        system_prompt=SYSTEM_PROMPT,
    )


def render_tool_execution(messages) -> Table:
    """Creates a rich table to visualize tool invocations, with PII kept redacted in the display."""
    table = Table(
        title="🛠️ MCP Tool Execution Trace (PII Redaction Shield Active)",
        show_header=True, header_style="bold magenta", expand=True
    )
    table.add_column("Type", width=12)
    table.add_column("Details", style="cyan")

    for msg in messages:
        for part in msg.parts:
            if isinstance(part, ToolCallPart):
                args_str = redact_text(str(part.args))
                table.add_row(
                    "[bold yellow]CALL[/bold yellow]",
                    f"Tool: [bold white]{part.tool_name}[/bold white]\nArgs: [italic green]{args_str}[/italic green]"
                )
            elif isinstance(part, ToolReturnPart):
                content_str = redact_text(str(part.content))
                if len(content_str) > 150:
                    content_str = content_str[:147] + "..."

                is_denied = "Access Denied" in content_str or "PermissionError" in content_str
                is_pending = "pending_approval" in content_str
                if is_denied:
                    table.add_row("[bold red]DENIED[/bold red]", f"[bold red]Tool: {part.tool_name}\nResult: {content_str}[/bold red]")
                elif is_pending:
                    table.add_row("[bold yellow]PENDING[/bold yellow]", f"[bold yellow]Tool: {part.tool_name}\nResult: {content_str}[/bold yellow]")
                else:
                    table.add_row("[bold green]RETURN[/bold green]", f"Tool: [bold white]{part.tool_name}[/bold white]\nResult: {content_str}")
    return table


async def run_agent_with_fallback(prompt: str, message_history: list = None, _retry_count: int = 0) -> Any:
    global MODEL_NAME, model, ACTIVE_TOKEN

    if _retry_count == 0:
        _executed_this_turn.clear()  # fresh turn — forget last turn's idempotency cache

    message_history = clean_message_history(message_history)
    prompt = redact_text(prompt)

    retry_temperature = min(0.2 * _retry_count, 0.6)
    # If MODEL_NAME is already the lightweight fallback (because an earlier 429 downgraded
    # it) and we're now retrying because IT leaked syntax, don't retry on the same weak
    # model 3 times in a row. Alternate back toward the primary model on odd retries instead.
    active_model = model
    if MODEL_NAME == 'llama-3.1-8b-instant' and _retry_count in (1, 3):
        active_model = GroqModel('llama-3.3-70b-versatile')
    elif _retry_count >= 2 and MODEL_NAME != 'llama-3.1-8b-instant':
        active_model = GroqModel('llama-3.1-8b-instant')

    active_agent = get_agent_for_token(ACTIVE_TOKEN, temperature=retry_temperature, active_model=active_model)

    try:
        async with active_agent:
            result = await active_agent.run(prompt, message_history=message_history)

        if "<function=" in result.output and _retry_count < 3:
            console.print()
            console.print(Panel(
                "[bold yellow]⚠️ Model leaked raw tool-call syntax into its final answer. Retrying...[/bold yellow]",
                title="🔁 Auto-Retry", border_style="yellow"
            ))
            return await run_agent_with_fallback(prompt, message_history=message_history, _retry_count=_retry_count + 1)

        return result

    except ModelHTTPError as e:
        if e.status_code == 429:
            fallback_model_name = 'llama-3.1-8b-instant'
            console.print()
            console.print(Panel(
                f"[bold yellow]⚠️ Groq Rate Limit (429) Reached for {MODEL_NAME}![/bold yellow]\n"
                f"Automatically falling back to [bold cyan]{fallback_model_name}[/bold cyan].",
                title="🔄 Dynamic Rate Limit Fallback",
                border_style="yellow"
            ))
            MODEL_NAME = fallback_model_name
            model = GroqModel(fallback_model_name)
            fallback_agent = get_agent_for_token(ACTIVE_TOKEN, active_model=model)
            async with fallback_agent:
                result = await fallback_agent.run(prompt, message_history=message_history)
            return result
        elif _retry_count < 3 and ("<function=" in str(e) or "tool_use_failed" in str(e).lower() or "failed_generation" in str(e).lower()):
            console.print()
            console.print(Panel(
                "[bold yellow]⚠️ Model produced malformed tool-call syntax. Retrying...[/bold yellow]",
                title="🔁 Auto-Retry", border_style="yellow"
            ))
            return await run_agent_with_fallback(prompt, message_history=message_history, _retry_count=_retry_count + 1)
        else:
            raise e

    except Exception as e:
        if _retry_count < 3 and ("leaked" in str(e).lower() or "retries" in str(e).lower() or "<function=" in str(e) or "invalid_args" in str(e).lower()):
            console.print()
            console.print(Panel(
                "[bold yellow]⚠️ Model leaked raw tool-call syntax or hit retry limit. Retrying...[/bold yellow]",
                title="🔁 Auto-Retry", border_style="yellow"
            ))
            return await run_agent_with_fallback(prompt, message_history=message_history, _retry_count=_retry_count + 1)
        raise e


async def run_scenario(prompt: str, scenario_num: int, title: str):
    console.print("\n" + "=" * 80, style="bold blue")
    console.print(f"[bold yellow]Scenario {scenario_num}:[/bold yellow] {title}")
    console.print(f"[bold dim]Prompt:[/bold dim] \"{prompt}\"\n")

    global _active_status
    status = console.status(f"[bold green]Agent reasoning using {MODEL_NAME}...[/bold green]", spinner="dots")
    status.start()
    _active_status = status
    try:
        try:
            result = await run_agent_with_fallback(prompt)
        except Exception as e:
            console.print(f"[bold red]Error running scenario: {e}[/bold red]")
            return
    finally:
        status.stop()
        _active_status = None

    new_msgs = result.new_messages()
    has_tool_calls = any(isinstance(p, (ToolCallPart, ToolReturnPart)) for msg in new_msgs for p in msg.parts)

    if has_tool_calls and Table:
        console.print(render_tool_execution(new_msgs))
        console.print()

    console.print(Panel(result.output, title="🤖 Financial Assistant Response", border_style="green", expand=True))


def print_hud():
    matched_key = None
    for k, v in DEMO_TOKENS.items():
        if v["token"] == ACTIVE_TOKEN:
            matched_key = k
            break

    if matched_key:
        role_info = DEMO_TOKENS[matched_key]
        scopes_str = ", ".join([f"[bold green]{s}[/bold green]" for s in role_info["scopes"]])
        desc_str = role_info["desc"]
        title_str = f"🛡️ Active Session: [bold yellow]{role_info['name']}[/bold yellow]"
        border_color = "yellow"
    else:
        scopes_str = "[bold red]None (Untrusted Token)[/bold red]"
        desc_str = "[italic red]Custom security credentials. The MCP server will reject all operations.[/italic red]"
        title_str = "🛡️ Active Session: [bold red]Untrusted Token[/bold red]"
        border_color = "red"

    hud_table = Table.grid(padding=1)
    hud_table.add_column(style="bold cyan", width=15)
    hud_table.add_column()
    hud_table.add_row("Active Token:", f"[green]{ACTIVE_TOKEN}[/green]")
    hud_table.add_row("Granted Scopes:", scopes_str)
    hud_table.add_row("Description:", desc_str)

    console.print(Panel(hud_table, title=title_str, border_style=border_color, expand=True))


def print_help():
    help_table = Table(title="Available Interactive Commands", show_header=True, header_style="bold magenta", width=60)
    help_table.add_column("Command", style="cyan")
    help_table.add_column("Description", style="white")
    help_table.add_row("/help", "Show this help table")
    help_table.add_row("/role auditor|teller|manager", "Swap active security credentials")
    help_table.add_row("/token [value]", "Enter a custom untrusted token to test security controls")
    help_table.add_row("exit / quit", "Terminates the interactive session")
    console.print(help_table)


def change_role(role_key: str):
    global ACTIVE_TOKEN, ACTIVE_ROLE
    if role_key not in DEMO_TOKENS:
        console.print(f"[bold red]Invalid role '{role_key}'. Choose from: auditor, teller, manager[/bold red]")
        return
    role_info = DEMO_TOKENS[role_key]
    ACTIVE_ROLE = role_info["name"]
    ACTIVE_TOKEN = role_info["token"]
    console.print(Panel(
        f"Role set to [bold yellow]{ACTIVE_ROLE}[/bold yellow]. The next tool call will spawn a fresh MCP "
        f"subprocess authenticated with these credentials.",
        title="🔄 Credentials Rotated Successfully", border_style="green"
    ))
    print_hud()


def change_custom_token(token_val: str):
    global ACTIVE_TOKEN, ACTIVE_ROLE
    ACTIVE_ROLE = "Untrusted Token"
    ACTIVE_TOKEN = token_val
    console.print(Panel(
        f"Applied untrusted credentials: [bold red]{ACTIVE_TOKEN}[/bold red].\n"
        "Since this token has zero scopes registered in server.py, all tool executions will be blocked.",
        title="⚠️ Untrusted Credentials Loaded", border_style="red"
    ))
    print_hud()


async def run_automated_demo():
    global ACTIVE_TOKEN, ACTIVE_ROLE

    console.print(Panel(
        "[bold green]Starting Automated Pydantic AI + MCP Secure Financial Agent Demo[/bold green]\n"
        f"Model: [cyan]{MODEL_NAME}[/cyan]\n"
        "Interface: [magenta]Model Context Protocol (MCP) via fresh Stdio subprocess per role[/magenta]\n"
        "State Store: [yellow]SQLite accounts.db[/yellow]\n"
        "Security Layers: [magenta]LPA (server.py) + HITL pending/approve gate (server.py) + PII redaction (agent.py)[/magenta]\n"
        "This demo showcases how different API clients are restricted before their operations hit the database.",
        title="🔐 Secure Financial Agent Initialized", border_style="cyan"
    ))

    ACTIVE_ROLE = "Auditor (Read-Only)"
    ACTIVE_TOKEN = "tkn_auditor_abc123"
    console.print("\n" + "=" * 80, style="bold magenta")
    console.print(Panel(
        f"Configuring agent with role: [bold yellow]{ACTIVE_ROLE}[/bold yellow]\n"
        f"Active Token: [italic green]{ACTIVE_TOKEN}[/italic green]\n"
        f"Scopes: [dim]{DEMO_TOKENS['auditor']['scopes']}[/dim]",
        title="🔐 Session Credentials Configured", border_style="magenta"
    ))
    await run_scenario(
        prompt="Please check the current balance of Alice's account (Account ID: 1).",
        scenario_num=1, title="[AUDITOR] Account Balance Inquiry (Should SUCCEED)"
    )

    ACTIVE_ROLE = "Teller (Standard)"
    ACTIVE_TOKEN = "tkn_teller_def456"
    console.print("\n" + "=" * 80, style="bold cyan")
    console.print(Panel(
        f"Upgrading agent to role: [bold yellow]{ACTIVE_ROLE}[/bold yellow]\n"
        f"Active Token: [italic green]{ACTIVE_TOKEN}[/italic green]\n"
        f"Scopes: [dim]{DEMO_TOKENS['teller']['scopes']}[/dim]",
        title="🔐 Session Credentials Upgraded", border_style="cyan"
    ))
    await run_scenario(
        prompt="Transfer $250.00 from Alice's account (Account 1) to Bob's account (Account 2).",
        scenario_num=2, title="[TELLER] Fund Transfer (Should SUCCEED)"
    )
    await run_scenario(
        prompt="Apply for a new loan of $5000.00 for Bob's account (Account 2).",
        scenario_num=3, title="[TELLER] Loan Application Attempt (Should FAIL — missing admin:loans scope)"
    )

    ACTIVE_ROLE = "Manager (Admin)"
    ACTIVE_TOKEN = "tkn_manager_xyz789"
    console.print("\n" + "=" * 80, style="bold yellow")
    console.print(Panel(
        f"Upgrading agent to role: [bold yellow]{ACTIVE_ROLE}[/bold yellow]\n"
        f"Active Token: [italic green]{ACTIVE_TOKEN}[/italic green]\n"
        f"Scopes: [dim]{DEMO_TOKENS['manager']['scopes']}[/dim]",
        title="🔐 Session Credentials Upgraded", border_style="yellow"
    ))
    await run_scenario(
        prompt="Apply for a new loan of $5000.00 for Bob's account (Account 2).",
        scenario_num=4, title="[MANAGER] Loan Application (Should return pending_approval)"
    )
    await run_scenario(
        prompt="List all accounts in the database to see their roles and statuses.",
        scenario_num=5, title="[MANAGER] List All Accounts (Should SUCCEED)"
    )

    console.print("\n" + "=" * 80, style="bold green")
    console.print("[bold green]Automated Demo Completed Successfully![/bold green]")
    console.print("[bold cyan]Note: Scenario 4's loan is still pending — run 'python agent.py chat' as Manager and ask to approve it.[/bold cyan]")
    console.print("[bold green]Demonstrated scope-based security, async HITL approval, and end-to-end PII redaction.[/bold green]\n")


async def run_interactive_chat():
    console.print(Panel(
        "[bold cyan]Welcome to the Secure Financial Assistant Chat![/bold cyan]\n"
        "You are chatting with a Pydantic AI agent connected to the MCP financial server.\n"
        "The user interface uses end-to-end PII masking to protect customer identities.\n"
        "To help you explore security, you can use the following slash commands:\n"
        "  - [bold yellow]/role auditor[/bold yellow] : Swap credentials to Read-Only mode\n"
        "  - [bold yellow]/role teller[/bold yellow] : Swap credentials to Standard Write mode\n"
        "  - [bold yellow]/role manager[/bold yellow] : Swap credentials to Admin mode\n"
        "  - [bold yellow]/token [value][/bold yellow] : Set a custom, untrusted API token\n"
        "  - [bold yellow]/help[/bold yellow] : Show all commands\n"
        "Type your requests naturally (e.g., 'What is Alice's balance?', 'Apply for a $5000 loan for Bob').\n"
        "Type [bold red]exit[/bold red] or [bold red]quit[/bold red] to end the session.",
        title="💬 Secure Interactive Chat Mode", border_style="cyan"
    ))

    print_hud()
    message_history = []

    def get_user_input():
        try:
            if sys.stdin.isatty():
                return input("\n\033[1;32mYou\033[0m: ")
            else:
                line = sys.stdin.readline()
                if not line:
                    return "exit"
                return line.rstrip('\n')
        except (EOFError, KeyboardInterrupt):
            return "exit"

    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(None, get_user_input)
            if user_input.lower().strip() in ("exit", "quit"):
                console.print("[bold yellow]Goodbye![/bold yellow]")
                break
            if not user_input.strip():
                continue

            if user_input.startswith("/"):
                parts = user_input.strip().split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd == "/help":
                    print_help()
                elif cmd == "/role":
                    if not arg:
                        console.print("[dim italic]/role requires an argument (auditor, teller, or manager)[/dim italic]")
                    else:
                        change_role(arg.lower().strip())
                        message_history = []
                        console.print("[dim italic]Chat history cleared to prevent context mixing across roles[/dim italic]")
                elif cmd == "/token":
                    if not arg:
                        console.print("[bold red]Error: Please specify a token value. Example: /token my_secret_token[/bold red]")
                    else:
                        change_custom_token(arg.strip())
                        message_history = []
                        console.print("[dim italic]Chat history cleared to prevent prompt context leakage across tokens[/dim italic]")
                else:
                    console.print(f"[bold red]Unknown command: '{cmd}'. Type /help for available commands.[/bold red]")
                continue

            global _active_status
            status = console.status("[bold green]Agent reasoning...[/bold green]", spinner="dots")
            status.start()
            _active_status = status
            try:
                result = await run_agent_with_fallback(user_input, message_history=message_history)
                message_history = result.all_messages()
            finally:
                status.stop()
                _active_status = None

            new_msgs = result.new_messages()
            has_tool_calls = any(isinstance(p, (ToolCallPart, ToolReturnPart)) for msg in new_msgs for p in msg.parts)

            if has_tool_calls and Table:
                console.print(render_tool_execution(new_msgs))
                console.print()

            console.print(Panel(result.output, title="🔐 Secure Agent", border_style="green", expand=True))

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Session interrupted by user. Goodbye![/bold yellow]")
            raise
        except Exception as e:
            console.print(f"[bold red]An error occurred: {e}[/bold red]")


def main():
    def signal_handler(signum, frame):
        console.print("\n[bold yellow]Session interrupted. Goodbye![/bold yellow]")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        if len(sys.argv) > 1 and sys.argv[1].lower() == "chat":
            asyncio.run(run_interactive_chat())
        else:
            asyncio.run(run_automated_demo())
            console.print("To start secure interactive chat mode, run: [bold cyan]python agent.py chat[/bold cyan]\n")
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Session interrupted. Goodbye![/bold yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Fatal error: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()