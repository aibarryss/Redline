"""Constants, error types, and hard limits for FinAudit-RL."""

from __future__ import annotations

SCHEMA_VERSION = "1.0.0"
CONFIG_VERSION = "1.0.0"

# Environment & submission limits
MAX_STEPS = 100
MAX_PAGE_LIMIT = 50
MIN_PAGE_LIMIT = 1
MAX_FINDINGS = 30
MAX_AFFECTED_IDS = 20
MAX_EVIDENCE_IDS = 10
MAX_ID_LEN = 128
MAX_EXPLANATION_LEN = 1000
MAX_CALC_EXPR_LEN = 256
EPISODE_TIMEOUT_SECONDS = 60.0
STEP_PENALTY = 0.002

# Four independently configurable base error types
ERROR_TYPE_DUPLICATE_INVOICE = "duplicate_invoice"
ERROR_TYPE_PAYMENT_ALLOCATION = "payment_allocation_mismatch"
ERROR_TYPE_FX_ERROR = "fx_rate_or_date_error"
ERROR_TYPE_CUTOFF_PERIOD = "cutoff_period_error"

ERROR_TYPES: tuple[str, ...] = (
    ERROR_TYPE_DUPLICATE_INVOICE,
    ERROR_TYPE_PAYMENT_ALLOCATION,
    ERROR_TYPE_FX_ERROR,
    ERROR_TYPE_CUTOFF_PERIOD,
)

# Two interacting multi-error scenarios
SCENARIO_FX_PARTIAL_CUTOFF = "interacting_fx_partial_cutoff"
SCENARIO_DUP_FX_ALLOCATION = "interacting_dup_fx_allocation"

INTERACTING_SCENARIOS: tuple[str, ...] = (
    SCENARIO_FX_PARTIAL_CUTOFF,
    SCENARIO_DUP_FX_ALLOCATION,
)

VALID_TABLES: tuple[str, ...] = ("ledger", "invoices", "payments", "fx")
VALID_ACTIONS: tuple[str, ...] = (
    "list_records",
    "get_record",
    "get_document",
    "calculator",
    "submit",
)

ROUNDING_MODES: tuple[str, ...] = ("ROUND_HALF_UP", "ROUND_HALF_EVEN")
FX_RATE_DATE_RULES: tuple[str, ...] = ("invoice_issue_date", "payment_date")

CURRENCY_EXPONENTS: dict[str, int] = {
    "USD": 2,
    "EUR": 2,
    "GBP": 2,
    "KZT": 2,
    "CHF": 2,
    "JPY": 0,
}

# Standard chart of accounts
ACCOUNT_CASH = "1010_CASH"
ACCOUNT_AR = "1200_ACCOUNTS_RECEIVABLE"
ACCOUNT_AP = "2000_ACCOUNTS_PAYABLE"
ACCOUNT_REVENUE = "4000_REVENUE"
ACCOUNT_EXPENSE = "5000_OPERATING_EXPENSE"
ACCOUNT_REFUND = "4100_SALES_RETURNS"
ACCOUNT_UNAPPLIED_CASH = "2150_UNAPPLIED_CASH"
ACCOUNT_FX_DIFF = "6100_FX_VARIANCE"
