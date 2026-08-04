import json
import sqlite3
from datetime import datetime
from schemas import Transaction, Loan, Account

DB_PATH = "accounts.db"


def get_connection():
    """Returns a SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    """Initializes the database schema and seeds initial accounts, transactions, and loans if empty."""
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_name TEXT NOT NULL,
                balance REAL NOT NULL DEFAULT 0.0,
                role TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active'
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_account INTEGER,
                to_account INTEGER,
                amount REAL NOT NULL,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (from_account) REFERENCES accounts(id),
                FOREIGN KEY (to_account) REFERENCES accounts(id)
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS loans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER NOT NULL,
                remaining_balance REAL NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY (account_id) REFERENCES accounts(id)
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pending_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                args_json TEXT NOT NULL,
                amount REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                timestamp TEXT NOT NULL
            );
        """)

        cursor.execute("SELECT COUNT(*) FROM accounts")
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                "INSERT INTO accounts (id, owner_name, balance, role, status) VALUES (?, ?, ?, ?, ?)",
                (1, "Alice", 5000.0, "user", "active")
            )
            cursor.execute(
                "INSERT INTO accounts (id, owner_name, balance, role, status) VALUES (?, ?, ?, ?, ?)",
                (2, "Bob", 1500.0, "user", "active")
            )
            cursor.execute(
                "INSERT INTO accounts (id, owner_name, balance, role, status) VALUES (?, ?, ?, ?, ?)",
                (3, "Charlie", 12000.0, "admin", "active")
            )

            now_str = datetime.now().isoformat()
            cursor.execute(
                "INSERT INTO transactions (id, from_account, to_account, amount, type, status, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (1, None, 1, 5000.0, "deposit", "completed", now_str)
            )
            cursor.execute(
                "INSERT INTO transactions (id, from_account, to_account, amount, type, status, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (2, None, 2, 1500.0, "deposit", "completed", now_str)
            )
            cursor.execute(
                "INSERT INTO transactions (id, from_account, to_account, amount, type, status, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (3, None, 3, 12000.0, "deposit", "completed", now_str)
            )
            cursor.execute(
                "INSERT INTO transactions (id, from_account, to_account, amount, type, status, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (4, 1, 2, 200.0, "transfer", "completed", now_str)
            )

            cursor.execute(
                "INSERT INTO loans (id, account_id, remaining_balance, status) VALUES (?, ?, ?, ?)",
                (1, 1, 1000.0, "active")
            )

        conn.commit()


init_db()


def get_account_balance(account_id: int) -> float:
    """Retrieves the balance of a specific account."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT balance FROM accounts WHERE id = ?", (account_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Account {account_id} not found")
        return row[0]


def search_accounts_db(query: str) -> list[Account]:
    """Searches accounts by owner name or role."""
    if not query:
        raise ValueError("Query cannot be empty")
    with get_connection() as conn:
        cursor = conn.cursor()
        like_query = f"%{query}%"
        cursor.execute(
            """
            SELECT id, owner_name, balance, role, status
            FROM accounts
            WHERE owner_name LIKE ? OR role LIKE ?
            ORDER BY id
            """,
            (like_query, like_query),
        )
        rows = cursor.fetchall()
        return [
            Account(id=row[0], owner_name=row[1], balance=row[2], role=row[3], status=row[4])
            for row in rows
        ]


def list_accounts_db(status: str | None = None) -> list[Account]:
    """Lists all accounts, optionally filtered by status (e.g. 'active')."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if status:
            cursor.execute(
                "SELECT id, owner_name, balance, role, status FROM accounts WHERE status = ? ORDER BY id",
                (status,),
            )
        else:
            cursor.execute(
                "SELECT id, owner_name, balance, role, status FROM accounts ORDER BY id"
            )
        rows = cursor.fetchall()
        return [
            Account(id=row[0], owner_name=row[1], balance=row[2], role=row[3], status=row[4])
            for row in rows
        ]


def search_transactions_db(query: str) -> list[Transaction]:
    """Searches for transactions that match the query in their type, status, or account numbers."""
    if not query:
        raise ValueError("Query cannot be empty")
    with get_connection() as conn:
        cursor = conn.cursor()
        like_query = f"%{query}%"
        cursor.execute(
            """
            SELECT id, from_account, to_account, amount, type, status, timestamp
            FROM transactions
            WHERE type LIKE ? OR status LIKE ? OR CAST(from_account AS TEXT) LIKE ? OR CAST(to_account AS TEXT) LIKE ?
            """,
            (like_query, like_query, like_query, like_query)
        )
        rows = cursor.fetchall()
        return [
            Transaction(
                id=row[0], from_account=row[1], to_account=row[2], amount=row[3],
                type=row[4], status=row[5], timestamp=datetime.fromisoformat(row[6])
            )
            for row in rows
        ]


def deposit_funds_db(account_id: int, amount: float) -> None:
    """Deposits funds into an account and creates a transaction record."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM accounts WHERE id = ?", (account_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Account {account_id} not found")
        if row[0] != "active":
            raise ValueError(f"Account {account_id} is currently suspended/inactive")

        cursor.execute("UPDATE accounts SET balance = balance + ? WHERE id = ?", (amount, account_id))
        cursor.execute(
            "INSERT INTO transactions (from_account, to_account, amount, type, status, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (None, account_id, amount, "deposit", "completed", datetime.now().isoformat())
        )
        conn.commit()


def withdraw_funds_db(account_id: int, amount: float) -> None:
    """Withdraws funds from an account and creates a transaction record."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT balance, status FROM accounts WHERE id = ?", (account_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Account {account_id} not found")
        balance, status = row
        if status != "active":
            raise ValueError(f"Account {account_id} is currently suspended/inactive")
        if balance < amount:
            raise ValueError(f"Insufficient funds in account {account_id} (current balance: {balance})")

        cursor.execute("UPDATE accounts SET balance = balance - ? WHERE id = ?", (amount, account_id))
        cursor.execute(
            "INSERT INTO transactions (from_account, to_account, amount, type, status, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (account_id, None, amount, "withdrawal", "completed", datetime.now().isoformat())
        )
        conn.commit()


def transfer_funds_db(from_account_id: int, to_account_id: int, amount: float) -> None:
    """Transfers funds from one account to another and creates a transaction record."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT balance, status FROM accounts WHERE id = ?", (from_account_id,))
        row_from = cursor.fetchone()
        if not row_from:
            raise ValueError(f"Account {from_account_id} not found")
        balance_from, status_from = row_from
        if status_from != "active":
            raise ValueError(f"Source account {from_account_id} is currently suspended/inactive")
        if balance_from < amount:
            raise ValueError(f"Insufficient funds in account {from_account_id} (current balance: {balance_from})")

        cursor.execute("SELECT status FROM accounts WHERE id = ?", (to_account_id,))
        row_to = cursor.fetchone()
        if not row_to:
            raise ValueError(f"Account {to_account_id} not found")
        status_to = row_to[0]
        if status_to != "active":
            raise ValueError(f"Destination account {to_account_id} is currently suspended/inactive")

        cursor.execute("UPDATE accounts SET balance = balance - ? WHERE id = ?", (amount, from_account_id))
        cursor.execute("UPDATE accounts SET balance = balance + ? WHERE id = ?", (amount, to_account_id))
        cursor.execute(
            "INSERT INTO transactions (from_account, to_account, amount, type, status, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (from_account_id, to_account_id, amount, "transfer", "completed", datetime.now().isoformat())
        )
        conn.commit()


def apply_for_loan_db(account_id: int, amount: float) -> None:
    """Applies for a loan, creating a loan record, depositing the funds, and creating a transaction."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM accounts WHERE id = ?", (account_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Account {account_id} not found")
        if row[0] != "active":
            raise ValueError(f"Account {account_id} is currently suspended/inactive")

        cursor.execute(
            "INSERT INTO loans (account_id, remaining_balance, status) VALUES (?, ?, ?)",
            (account_id, amount, "active")
        )
        cursor.execute("UPDATE accounts SET balance = balance + ? WHERE id = ?", (amount, account_id))
        cursor.execute(
            "INSERT INTO transactions (from_account, to_account, amount, type, status, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (None, account_id, amount, "loan_payout", "completed", datetime.now().isoformat())
        )
        conn.commit()


def pay_loan_db(loan_id: int, amount: float) -> None:
    """Pays off a loan, updating the loan balance, deducting account balance, and creating a transaction."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT account_id, remaining_balance, status FROM loans WHERE id = ?", (loan_id,))
        row_loan = cursor.fetchone()
        if not row_loan:
            raise ValueError(f"Loan {loan_id} not found")
        account_id, remaining_balance, status = row_loan
        if status != "active":
            raise ValueError(f"Loan {loan_id} is already fully paid off")
        if remaining_balance < amount:
            raise ValueError(f"Payment amount {amount} exceeds remaining loan balance {remaining_balance}")

        cursor.execute("SELECT balance, status FROM accounts WHERE id = ?", (account_id,))
        row_acct = cursor.fetchone()
        if not row_acct:
            raise ValueError(f"Associated account {account_id} not found")
        balance_acct, status_acct = row_acct
        if status_acct != "active":
            raise ValueError(f"Associated account {account_id} is currently suspended/inactive")
        if balance_acct < amount:
            raise ValueError(f"Insufficient funds in account {account_id} to make loan payment (current balance: {balance_acct})")

        new_remaining = remaining_balance - amount
        new_status = "active" if new_remaining > 0 else "paid"
        cursor.execute(
            "UPDATE loans SET remaining_balance = ?, status = ? WHERE id = ?",
            (new_remaining, new_status, loan_id)
        )
        cursor.execute("UPDATE accounts SET balance = balance - ? WHERE id = ?", (amount, account_id))
        cursor.execute(
            "INSERT INTO transactions (from_account, to_account, amount, type, status, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (account_id, None, amount, "loan_payment", "completed", datetime.now().isoformat())
        )
        conn.commit()


def create_pending_action_db(tool_name: str, args: dict, amount: float) -> int:
    """Records a gated action awaiting human approval. Returns its pending_id."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO pending_actions (tool_name, args_json, amount, status, timestamp) VALUES (?, ?, ?, ?, ?)",
            (tool_name, json.dumps(args), amount, "pending", datetime.now().isoformat())
        )
        conn.commit()
        return cursor.lastrowid


def get_pending_action_db(pending_id: int) -> dict | None:
    """Fetches a pending action by id, or None if it doesn't exist."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, tool_name, args_json, amount, status, timestamp FROM pending_actions WHERE id = ?",
            (pending_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "tool_name": row[1],
            "args": json.loads(row[2]),
            "amount": row[3],
            "status": row[4],
            "timestamp": row[5],
        }


def resolve_pending_action_db(pending_id: int, new_status: str) -> None:
    """Marks a pending action as 'approved' or 'rejected'."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE pending_actions SET status = ? WHERE id = ?", (new_status, pending_id))
        conn.commit()