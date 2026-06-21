from __future__ import annotations

"""
loan_intake.py
==============

This is the "simple table" layer Ravi asked for.

The idea: a human (or an AI doing NL extraction in chat) fills in
RawLoanInput with only the facts a normal person actually knows about
their loan. Everything else - end date, overdue amount, annualized
interest rate, total debt today - gets computed here.

Flow:
    plain English  -->  RawLoanInput (this module's "simple table")
                    -->  validate_raw_loan()  -->  list of questions to ask
                    -->  enrich_loan()         -->  EnrichedLoan (full table)

The plain-English -> RawLoanInput step is intentionally NOT done with
regex/NLP in this file. That step is genuinely an "understand free text"
problem, which is what an LLM (Claude, in chat, or a Claude API call
inside Streamlit) is good at and brittle code is not. This module is the
deterministic math layer that sits right after that step.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Optional, Literal

from payoff_simulator import add_months, months_between

LoanType = Literal["emi", "interest_only", "custom"]
Frequency = Literal["daily", "weekly", "monthly"]

PERIODS_PER_YEAR = {
    "daily": 365.0,
    "weekly": 52.0,
    "monthly": 12.0,
}


# ---------------------------------------------------------------------
# 1. THE SIMPLE TABLE - what a non-technical user (or the NL-extraction
#    step) needs to fill in. Nine real fields, matching what Ravi asked
#    for, plus the couple of fields that turn out to be unavoidable
#    (loan_type and either a term or an end date for EMI loans).
# ---------------------------------------------------------------------
@dataclass
class RawLoanInput:
    name: str
    principal_amount: float
    loan_type: LoanType                       # "emi" | "interest_only" | "custom"
    payment_amount: float                     # the EMI or the interest amount per cycle
    payment_frequency: Frequency = "monthly"

    start_date: Optional[date] = None         # ask if missing
    payment_day_rule: str = ""                # display only, e.g. "1st of every month"

    payments_made: int = 0                    # "Total EMI Paid" - how many cycles already paid
    annual_interest_rate: Optional[float] = None   # only needed if known explicitly

    term_count: Optional[int] = None          # total scheduled payments, if known
    end_date: Optional[date] = None           # explicit end date, if known

    priority: Optional[int] = None            # 1-10, defaults to 5 if not given
    notes: str = ""


# ---------------------------------------------------------------------
# 2. MISSING-FIELD DETECTION - this is the "AI asks the user" step.
#    Returns plain-English questions, not error codes, so they can be
#    shown directly in chat or in a Streamlit form.
# ---------------------------------------------------------------------
def validate_raw_loan(raw: RawLoanInput) -> List[str]:
    questions: List[str] = []

    if raw.start_date is None:
        questions.append(f"{raw.name}: what date did this loan start?")

    if raw.principal_amount <= 0:
        questions.append(f"{raw.name}: what is the principal (loan) amount?")

    if raw.payment_amount <= 0:
        questions.append(f"{raw.name}: what is the EMI / interest amount per payment?")

    if raw.loan_type == "emi":
        if raw.end_date is None and raw.term_count is None:
            questions.append(
                f"{raw.name}: this is an EMI loan - how many total payments (e.g. '10 months', "
                f"'100 days'), or what is the end date / closing date?"
            )

    if raw.loan_type == "interest_only" and raw.annual_interest_rate is None:
        # Not actually blocking - we can derive it from payment_amount / principal_amount -
        # but we flag it so the user can confirm we got the rate right.
        implied = (raw.payment_amount / raw.principal_amount) if raw.principal_amount else 0.0
        questions.append(
            f"{raw.name}: no interest rate given - I'll assume the {implied*100:.2f}% per "
            f"{raw.payment_frequency.rstrip('ly')} payment IS the interest rate. Confirm or correct?"
        )

    return questions


# ---------------------------------------------------------------------
# 3. THE FULL TABLE - what the engine actually needs.
# ---------------------------------------------------------------------
@dataclass
class EnrichedLoan:
    name: str
    loan_type: LoanType

    principal_amount: float
    start_date: date
    end_date: Optional[date]                  # None = open-ended (interest-only, ongoing)
    payment_frequency: Frequency
    payment_amount: float
    payment_day_rule: str

    term_count: Optional[int]
    payments_made: int
    expected_payments_by_now: int
    missed_payments: int
    overdue: bool
    overdue_amount: float

    annual_interest_rate: float
    rate_was_explicit: bool

    principal_paid_to_date: float
    outstanding_principal: float
    total_debt_now: float                     # outstanding_principal + overdue_amount

    priority: int                              # 1-10, as given/defaulted by the user
    sim_priority: int                          # priority with overdue boost baked in, for avalanche ranking

    next_due_date: Optional[date]
    expected_close_date: Optional[date]
    repayment_progress_pct: float

    status: str                                 # "overdue" | "current" | "completing_soon" | "ongoing"
    risk_flags: List[str] = field(default_factory=list)
    notes: str = ""


def _advance(d: date, n: int, frequency: Frequency) -> date:
    if n <= 0:
        return d
    if frequency == "daily":
        return d + timedelta(days=n)
    if frequency == "weekly":
        return d + timedelta(weeks=n)
    return add_months(d, n)


def _periods_elapsed(start: date, current: date, frequency: Frequency) -> int:
    if current <= start:
        return 0
    delta_days = (current - start).days
    if frequency == "daily":
        return delta_days
    if frequency == "weekly":
        return delta_days // 7
    # monthly
    months = (current.year - start.year) * 12 + (current.month - start.month)
    if current.day < start.day:
        months -= 1
    return max(0, months)


def _annualize(periodic_rate: float, frequency: Frequency) -> float:
    """Compounding annualization, consistent with debt_engine.annualized_rate."""
    n = PERIODS_PER_YEAR[frequency]
    return (1 + periodic_rate) ** n - 1


def enrich_loan(raw: RawLoanInput, current_date: date) -> EnrichedLoan:
    missing = validate_raw_loan(raw)
    hard_blockers = [
        q for q in missing
        if "interest rate given" not in q  # that one is a soft confirmation, not a blocker
    ]
    if raw.start_date is None or (
        raw.loan_type == "emi" and raw.end_date is None and raw.term_count is None
    ) or raw.principal_amount <= 0 or raw.payment_amount <= 0:
        raise ValueError(
            f"Cannot enrich '{raw.name}' - missing required info: {hard_blockers}"
        )

    start_date = raw.start_date
    risk_flags: List[str] = []

    # --- resolve term_count / end_date ---
    if raw.loan_type == "emi":
        if raw.end_date is not None:
            end_date = raw.end_date
            term_count = (
                months_between(start_date, end_date)
                if raw.payment_frequency == "monthly"
                else _periods_between_dates(start_date, end_date, raw.payment_frequency)
            )
        else:
            term_count = raw.term_count
            end_date = _advance(start_date, term_count, raw.payment_frequency)
    else:
        # interest_only / custom with no fixed term: open-ended until paid off
        term_count = raw.term_count
        end_date = raw.end_date  # usually None

    # --- resolve annual interest rate ---
    rate_was_explicit = raw.annual_interest_rate is not None
    if rate_was_explicit:
        annual_interest_rate = raw.annual_interest_rate
    elif raw.loan_type == "interest_only":
        periodic_rate = raw.payment_amount / raw.principal_amount if raw.principal_amount else 0.0
        annual_interest_rate = _annualize(periodic_rate, raw.payment_frequency) * 100.0
    elif raw.loan_type == "emi" and term_count:
        total_payable = raw.payment_amount * term_count
        implied_cost = max(0.0, total_payable - raw.principal_amount)
        periodic_rate = (implied_cost / raw.principal_amount / term_count) if raw.principal_amount else 0.0
        annual_interest_rate = _annualize(periodic_rate, raw.payment_frequency) * 100.0
    else:
        annual_interest_rate = 0.0
        risk_flags.append("Could not determine an interest rate - treated as 0%. Please confirm.")

    if annual_interest_rate > 60:
        risk_flags.append(
            f"Effective annual rate is very high ({annual_interest_rate:.1f}%) - double-check the "
            f"payment amount and principal are correct."
        )

    # --- overdue calculation ---
    expected_payments_by_now = _periods_elapsed(start_date, current_date, raw.payment_frequency)
    if term_count is not None:
        expected_payments_by_now = min(expected_payments_by_now, term_count)
    missed_payments = max(0, expected_payments_by_now - raw.payments_made)

    # Also mark overdue if the next scheduled payment date is today or in the past and unpaid.
    # Payment k is due at _advance(start, k), so the next unpaid payment (payments_made+1)
    # is due at _advance(start, payments_made).
    # Use <= so "due today and not yet paid" is also treated as overdue.
    next_payment_date = _advance(start_date, raw.payments_made, raw.payment_frequency) \
        if (term_count is None or raw.payments_made < term_count) else None
    if missed_payments == 0 and next_payment_date is not None and next_payment_date < current_date:
        # The next due date has strictly passed (not just today) and it hasn't been paid yet
        missed_payments = 1

    overdue = missed_payments > 0
    overdue_amount = missed_payments * raw.payment_amount

    # --- principal paid to date ---
    if raw.loan_type == "interest_only":
        principal_paid_to_date = 0.0
    elif raw.loan_type == "emi" and term_count:
        total_payable = raw.payment_amount * term_count
        implied_cost = max(0.0, total_payable - raw.principal_amount)
        cost_per_payment = implied_cost / term_count
        principal_per_payment = max(0.0, raw.payment_amount - cost_per_payment)
        principal_paid_to_date = min(raw.principal_amount, principal_per_payment * raw.payments_made)
    else:
        principal_paid_to_date = min(raw.principal_amount, raw.payment_amount * raw.payments_made)

    outstanding_principal = max(0.0, raw.principal_amount - principal_paid_to_date)
    total_debt_now = outstanding_principal + overdue_amount

    # --- priority ---
    priority = raw.priority if raw.priority is not None else 5
    priority = max(1, min(10, priority))
    sim_priority = priority + (50 if overdue else 0)  # forces overdue loans to the top of avalanche ranking

    # --- next due date ---
    # Payment k is due at _advance(start, k-1+1) = _advance(start, k) ... actually:
    # payment #1 due at _advance(start, 1), payment #2 at _advance(start, 2), etc.
    # BUT _advance(start, 0) = start itself, so if we index from 0:
    # slot 0 = start, slot 1 = start+1 period, ...
    # next_payment_date already computed above = _advance(start, payments_made).
    # For overdue: that's the first unpaid date (show it so user knows what they owe).
    # For current: payments_made slot is future (since <= check didn't fire), so show it.
    _next_slot = max(raw.payments_made, expected_payments_by_now)
    next_due_date = (
        _advance(start_date, _next_slot, raw.payment_frequency)
        if (term_count is None or _next_slot < term_count)
        else None
    )

    if term_count:
        repayment_progress_pct = min(100.0, round(100.0 * raw.payments_made / term_count, 1))
    else:
        repayment_progress_pct = 0.0

    if overdue:
        status = "overdue"
    elif term_count and raw.payments_made >= term_count - 1:
        status = "completing_soon"
    elif raw.loan_type == "interest_only":
        status = "ongoing"
    else:
        status = "current"

    if raw.loan_type == "interest_only":
        risk_flags.append("Interest-only: regular payments do not reduce principal. Needs a lump sum or extra payments to close.")

    return EnrichedLoan(
        name=raw.name,
        loan_type=raw.loan_type,
        principal_amount=raw.principal_amount,
        start_date=start_date,
        end_date=end_date,
        payment_frequency=raw.payment_frequency,
        payment_amount=raw.payment_amount,
        payment_day_rule=raw.payment_day_rule,
        term_count=term_count,
        payments_made=raw.payments_made,
        expected_payments_by_now=expected_payments_by_now,
        missed_payments=missed_payments,
        overdue=overdue,
        overdue_amount=overdue_amount,
        annual_interest_rate=annual_interest_rate,
        rate_was_explicit=rate_was_explicit,
        principal_paid_to_date=principal_paid_to_date,
        outstanding_principal=outstanding_principal,
        total_debt_now=total_debt_now,
        priority=priority,
        sim_priority=sim_priority,
        next_due_date=next_due_date,
        expected_close_date=end_date,
        repayment_progress_pct=repayment_progress_pct,
        status=status,
        risk_flags=risk_flags,
        notes=raw.notes,
    )


def _periods_between_dates(start: date, end: date, frequency: Frequency) -> int:
    days = (end - start).days
    if frequency == "daily":
        return max(0, days)
    if frequency == "weekly":
        return max(0, days // 7)
    return months_between(start, end)


def enrich_portfolio(raw_loans: List[RawLoanInput], current_date: date) -> List[EnrichedLoan]:
    return [enrich_loan(r, current_date) for r in raw_loans]


def print_enriched_table(loans: List[EnrichedLoan]) -> None:
    print("\n=== ENRICHED LOAN TABLE ===")
    for L in loans:
        print("------")
        print(f"{L.name}  [{L.loan_type}, {L.status}]")
        print(f"  Principal: {L.principal_amount:,.2f}   Outstanding: {L.outstanding_principal:,.2f}")
        print(f"  Start: {L.start_date}   End/closes: {L.end_date}   Term payments: {L.term_count}")
        print(f"  {L.payment_amount:,.2f} / {L.payment_frequency} ({L.payment_day_rule or 'no rule given'})")
        print(f"  Payments made: {L.payments_made}   Expected by now: {L.expected_payments_by_now}   Missed: {L.missed_payments}")
        if L.overdue:
            print(f"  ⚠️ OVERDUE by {L.overdue_amount:,.2f}")
        print(f"  Annual rate: {L.annual_interest_rate:.2f}%  ({'explicit' if L.rate_was_explicit else 'derived'})")
        print(f"  Total debt now (outstanding + overdue): {L.total_debt_now:,.2f}")
        print(f"  Priority: {L.priority}/10   Sim priority (avalanche key): {L.sim_priority}")
        print(f"  Progress: {L.repayment_progress_pct}%   Next due: {L.next_due_date}")
        for flag in L.risk_flags:
            print(f"  ⚑ {flag}")
