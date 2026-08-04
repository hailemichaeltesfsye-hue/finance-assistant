from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from typing import Literal


class Transaction(BaseModel):
    id: int
    from_account: int | None = None
    to_account: int | None = None
    amount: float = Field(gt=0, description="The transaction amount, must be greater than 0")
    type: Literal["deposit", "withdrawal", "transfer", "loan_payout", "loan_payment"]
    status: Literal["completed", "pending", "failed"]
    timestamp: datetime


class Loan(BaseModel):
    id: int
    account_id: int
    remaining_balance: float = Field(ge=0, description="Remaining balance of the loan, must be greater than or equal to 0")
    status: Literal["active", "paid"]


class Account(BaseModel):
    id: int
    owner_name: str
    balance: float
    role: str
    status: Literal["active", "suspended"]


class PendingAction(BaseModel):
    id: int
    tool_name: str
    args: dict
    amount: float
    status: Literal["pending", "approved", "rejected"]
    timestamp: datetime


class DepositRequest(BaseModel):
    account_id: int
    amount: float = Field(gt=0, description="Amount to deposit, must be greater than 0")


class WithdrawRequest(BaseModel):
    account_id: int
    amount: float = Field(gt=0, description="Amount to withdraw, must be greater than 0")


class AuditLogEntry(BaseModel):
    id: int
    timestamp: datetime
    role: str
    tool_name: str
    args: dict
    outcome: Literal["success", "denied", "pending", "approved", "rejected"]
    detail: str

class TransferRequest(BaseModel):
    from_account_id: int
    to_account_id: int
    amount: float = Field(gt=0, description="Amount to transfer, must be greater than 0")

    @field_validator("to_account_id")
    @classmethod
    def prevent_self_transfer(cls, v: int, info) -> int:
        if "from_account_id" in info.data and v == info.data["from_account_id"]:
            raise ValueError("Source and destination accounts must be different.")
        return v


class LoanApplicationRequest(BaseModel):
    account_id: int
    amount: float = Field(gt=0, description="Amount to apply for a loan, must be greater than 0")


class LoanPaymentRequest(BaseModel):
    loan_id: int
    amount: float = Field(gt=0, description="Amount to pay off the loan, must be greater than 0")