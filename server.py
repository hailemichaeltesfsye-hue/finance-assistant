import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

from mcp.server.fastmcp import FastMCP
from schemas import (
    Transaction,
    Loan,
    Account,
    DepositRequest,
    WithdrawRequest,
    TransferRequest,
    LoanApplicationRequest,
    LoanPaymentRequest,
)
import db

mcp = FastMCP("Accounts MCP", json_response=True)

# ===================== LEAST PRIVILEGE ACCESS (LPA) SECURITY LAYER =====================
TOKEN_REGISTRY = {
    "tkn_auditor_abc123": {
        "role": "Auditor (Read-Only)",
        "scopes": {"read:accounts"}
    },
    "tkn_teller_def456": {
        "role": "Teller (Standard)",
        "scopes": {"read:accounts", "write:transactions"}
    },
    "tkn_manager_xyz789": {
        "role": "Manager (Admin)",
        "scopes": {"read:accounts", "write:transactions", "admin:loans", "admin:accounts"}
    }
}


def check_permission(required_scope: str, tool_name: str, args: dict | None = None) -> dict | None:
    """
    Validates that the active client has the required scope to execute the tool.
    Every call — success or denial — is written to the audit log here, since this
    is the one place every tool invocation is guaranteed to pass through.
    """
    args = args or {}
    token = os.getenv("MCP_AUTH_TOKEN")

    if not token:
        db.write_audit_log_db("Unknown (no token)", tool_name, args, "denied", "Missing MCP_AUTH_TOKEN")
        return {
            "status": "error",
            "error": "Access Denied",
            "message": f"Missing authentication token. To call '{tool_name}', the client must set a valid 'MCP_AUTH_TOKEN'."
        }

    if token not in TOKEN_REGISTRY:
        db.write_audit_log_db("Unknown (invalid token)", tool_name, args, "denied", "Invalid MCP_AUTH_TOKEN")
        return {
            "status": "error",
            "error": "Access Denied",
            "message": "Invalid authentication token. The provided token is not recognized by the MCP server security registry."
        }

    client_info = TOKEN_REGISTRY[token]
    allowed_scopes = client_info["scopes"]
    role = client_info["role"]

    if required_scope not in allowed_scopes:
        db.write_audit_log_db(role, tool_name, args, "denied", f"Missing required scope '{required_scope}'")
        return {
            "status": "error",
            "error": "Access Denied",
            "message": (
                f"Insufficient privileges. Tool '{tool_name}' requires the '{required_scope}' scope. "
                f"Your active role '{role}' only possesses scopes: {sorted(allowed_scopes)}."
            )
        }

    db.write_audit_log_db(role, tool_name, args, "success", "Permission granted")
    return None
# =========================================================================================


# ===================== HUMAN-IN-THE-LOOP (HITL) SECURITY LAYER =====================
HITL_THRESHOLD = 1000.0

GATED_TOOL_EXECUTORS = {
    "deposit_funds": lambda args: db.deposit_funds_db(args["account_id"], args["amount"]),
    "withdraw_funds": lambda args: db.withdraw_funds_db(args["account_id"], args["amount"]),
    "transfer_funds": lambda args: db.transfer_funds_db(args["from_account_id"], args["to_account_id"], args["amount"]),
    "apply_for_loan": lambda args: db.apply_for_loan_db(args["account_id"], args["amount"]),
    "pay_loan": lambda args: db.pay_loan_db(args["loan_id"], args["amount"]),
}


def gate_if_over_threshold(tool_name: str, args: dict, amount: float) -> dict | None:
    """Call AFTER check_permission passes, BEFORE the real DB write. Returns a
    pending_approval payload if sign-off is required, or None to proceed normally."""
    if amount > HITL_THRESHOLD:
        pending_id = db.create_pending_action_db(tool_name, args, amount)
        token = os.getenv("MCP_AUTH_TOKEN")
        role = TOKEN_REGISTRY.get(token, {}).get("role", "Unknown")
        db.write_audit_log_db(role, tool_name, args, "pending", f"Created pending_id {pending_id}")
        return {
            "status": "pending_approval",
            "pending_id": pending_id,
            "message": (
                f"This '{tool_name}' request for ${amount:,.2f} exceeds the ${HITL_THRESHOLD:,.2f} "
                f"auto-approval threshold and requires human sign-off. Pending ID: {pending_id}."
            ),
        }
    return None
# =========================================================================================


@mcp.tool()
def get_balance(account_id: int) -> dict:
    """Retrieves the balance of a specific account given its numeric account_id."""
    denial = check_permission("read:accounts", "get_balance", {"account_id": account_id})
    if denial:
        return denial
    return {"account_id": account_id, "balance": db.get_account_balance(account_id)}


@mcp.tool()
def search_transactions(query: str) -> list[dict] | dict:
    """Searches for transactions matching a query string."""
    denial = check_permission("read:accounts", "search_transactions", {"query": query})
    if denial:
        return denial
    if not query.strip():
        raise ValueError("Query cannot be empty")
    transactions = db.search_transactions_db(query)
    return [t.model_dump() for t in transactions]


@mcp.tool()
def deposit_funds(account_id: int, amount: float) -> dict:
    """Deposits funds into an account given its numeric account_id."""
    args = {"account_id": account_id, "amount": amount}
    denial = check_permission("write:transactions", "deposit_funds", args)
    if denial:
        return denial
    pending = gate_if_over_threshold("deposit_funds", args, amount)
    if pending:
        return pending
    req = DepositRequest(**args)
    db.deposit_funds_db(req.account_id, req.amount)
    return {"status": "success", "message": f"Deposited {req.amount} into account {req.account_id}"}


@mcp.tool()
def withdraw_funds(account_id: int, amount: float) -> dict:
    """Withdraws funds from an account given its numeric account_id."""
    args = {"account_id": account_id, "amount": amount}
    denial = check_permission("write:transactions", "withdraw_funds", args)
    if denial:
        return denial
    pending = gate_if_over_threshold("withdraw_funds", args, amount)
    if pending:
        return pending
    req = WithdrawRequest(**args)
    db.withdraw_funds_db(req.account_id, req.amount)
    return {"status": "success", "message": f"Withdrawn {req.amount} from account {req.account_id}"}


@mcp.tool()
def transfer_funds(from_account_id: int, to_account_id: int, amount: float) -> dict:
    """Transfers funds from one account to another using numeric account IDs."""
    args = {"from_account_id": from_account_id, "to_account_id": to_account_id, "amount": amount}
    denial = check_permission("write:transactions", "transfer_funds", args)
    if denial:
        return denial
    pending = gate_if_over_threshold("transfer_funds", args, amount)
    if pending:
        return pending
    req = TransferRequest(**args)
    db.transfer_funds_db(req.from_account_id, req.to_account_id, req.amount)
    return {"status": "success", "message": f"Transferred {req.amount} from account {req.from_account_id} to account {req.to_account_id}"}


@mcp.tool()
def apply_for_loan(account_id: int, amount: float) -> dict:
    """Applies for a loan for the specified account ID."""
    args = {"account_id": account_id, "amount": amount}
    denial = check_permission("admin:loans", "apply_for_loan", args)
    if denial:
        return denial
    pending = gate_if_over_threshold("apply_for_loan", args, amount)
    if pending:
        return pending
    req = LoanApplicationRequest(**args)
    db.apply_for_loan_db(req.account_id, req.amount)
    return {"status": "success", "message": f"Applied for a loan of {req.amount} for account {req.account_id}"}


@mcp.tool()
def pay_loan(loan_id: int, amount: float) -> dict:
    """Pays towards an active loan specified by loan_id."""
    args = {"loan_id": loan_id, "amount": amount}
    denial = check_permission("write:transactions", "pay_loan", args)
    if denial:
        return denial
    pending = gate_if_over_threshold("pay_loan", args, amount)
    if pending:
        return pending
    req = LoanPaymentRequest(**args)
    db.pay_loan_db(req.loan_id, req.amount)
    return {"status": "success", "message": f"Paid {req.amount} of loan {req.loan_id}"}


@mcp.tool()
def approve_transaction(pending_id: int) -> dict:
    """
    Finalizes a pending action, executing the underlying deposit/withdraw/transfer/loan
    operation. Restricted to admin:accounts (Manager), regardless of who submitted it.
    """
    denial = check_permission("admin:accounts", "approve_transaction", {"pending_id": pending_id})
    if denial:
        return denial

    pending = db.get_pending_action_db(pending_id)
    if not pending:
        return {"status": "error", "error": "Not Found", "message": f"No pending action with id {pending_id}."}
    if pending["status"] != "pending":
        return {
            "status": "error",
            "error": "Invalid State",
            "message": f"Pending action {pending_id} is already '{pending['status']}' and cannot be approved again."
        }

    executor = GATED_TOOL_EXECUTORS.get(pending["tool_name"])
    if not executor:
        return {"status": "error", "error": "Internal Error", "message": f"Unknown pending tool_name '{pending['tool_name']}'."}

    executor(pending["args"])
    db.resolve_pending_action_db(pending_id, "approved")
    role = TOKEN_REGISTRY.get(os.getenv("MCP_AUTH_TOKEN"), {}).get("role", "Unknown")
    db.write_audit_log_db(role, pending["tool_name"], pending["args"], "approved", f"Pending action {pending_id} approved and executed")
    return {"status": "success", "message": f"Pending action {pending_id} ({pending['tool_name']}) approved and executed."}


@mcp.tool()
def reject_transaction(pending_id: int) -> dict:
    """Rejects a pending action without executing it. Also restricted to admin:accounts."""
    denial = check_permission("admin:accounts", "reject_transaction", {"pending_id": pending_id})
    if denial:
        return denial

    pending = db.get_pending_action_db(pending_id)
    if not pending:
        return {"status": "error", "error": "Not Found", "message": f"No pending action with id {pending_id}."}
    if pending["status"] != "pending":
        return {
            "status": "error",
            "error": "Invalid State",
            "message": f"Pending action {pending_id} is already '{pending['status']}' and cannot be rejected again."
        }

    db.resolve_pending_action_db(pending_id, "rejected")
    role = TOKEN_REGISTRY.get(os.getenv("MCP_AUTH_TOKEN"), {}).get("role", "Unknown")
    db.write_audit_log_db(role, pending["tool_name"], pending["args"], "rejected", f"Pending action {pending_id} rejected")
    return {"status": "success", "message": f"Pending action {pending_id} rejected. No funds were moved."}


@mcp.tool()
def list_accounts(filter_status: str = "all") -> list[dict] | dict:
    """Lists accounts in the database, optionally filtered by status."""
    denial = check_permission("admin:accounts", "list_accounts", {"filter_status": filter_status})
    if denial:
        return denial
    accounts = db.list_accounts_db()
    if filter_status != "all":
        accounts = [a for a in accounts if a.status == filter_status]
    return [a.model_dump() for a in accounts]


@mcp.tool()
def search_accounts(query: str) -> list[dict] | dict:
    """Searches for accounts matching an owner name (e.g., 'Alice', 'Bob', 'Charlie') or role."""
    denial = check_permission("read:accounts", "search_accounts", {"query": query})
    if denial:
        return denial
    accounts = db.search_accounts_db(query)
    return [a.model_dump() for a in accounts]


@mcp.tool()
def search_audit_log(query: str = "") -> list[dict] | dict:
    """
    Searches the security audit log — every permission check, granted or denied,
    plus HITL pending/approved/rejected events. Manager credentials only.

    Args:
        query (str): Optional filter matched against role, tool_name, or outcome. Empty returns all entries.

    Returns:
        list[dict]: Audit entries, most recent first, or an error payload if access is denied.
    """
    denial = check_permission("admin:accounts", "search_audit_log", {"query": query})
    if denial:
        return denial
    return db.search_audit_log_db(query if query else None)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)