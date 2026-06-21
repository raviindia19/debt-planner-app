from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional, Dict, Any

import pandas as pd
import streamlit as st

from loan_intake import RawLoanInput, EnrichedLoan, validate_raw_loan
from debt_planner import build_plan, PlanResult

LOAN_TYPES = ["emi", "interest_only", "custom"]
FREQUENCIES = ["daily", "weekly", "monthly"]

RAW_COLUMNS = [
    "name", "loan_type", "principal_amount", "payment_amount", "payment_frequency",
    "start_date", "payment_day_rule", "payments_made", "term_count", "end_date",
    "annual_interest_rate", "priority", "notes",
]


def default_rows() -> List[Dict[str, Any]]:
    """Ravi's three example loans, pre-filled so there's something to click 'Calculate' on immediately."""
    return [
        dict(
            name="Family Commitment", loan_type="emi", principal_amount=50000.0,
            payment_amount=5000.0, payment_frequency="monthly", start_date=date(2026, 5, 1),
            payment_day_rule="1st of every month", payments_made=1, term_count=10, end_date=None,
            annual_interest_rate=None, priority=10, notes="Family pressure / immediate obligation",
        ),
        dict(
            name="Khatu", loan_type="emi", principal_amount=100000.0,
            payment_amount=1200.0, payment_frequency="daily", start_date=date(2026, 5, 1),
            payment_day_rule="daily", payments_made=51, term_count=100, end_date=None,
            annual_interest_rate=None, priority=5, notes="Daily finance loan",
        ),
        dict(
            name="Radhe", loan_type="interest_only", principal_amount=100000.0,
            payment_amount=7000.0, payment_frequency="monthly", start_date=date(2026, 2, 1),
            payment_day_rule="5th of every month", payments_made=5, term_count=None, end_date=None,
            annual_interest_rate=None, priority=8, notes="Interest-only lender",
        ),
    ]


# ---------------------------------------------------------------------
# Pure conversion / formatting logic - no st.* calls below this line,
# so it can be imported and unit-tested without a running Streamlit
# session.
# ---------------------------------------------------------------------
def _to_date(value) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except ValueError:
            return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return None


def _to_int(value) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_optional_float(value) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in RAW_COLUMNS:
        if c not in df.columns:
            df[c] = None

    df["name"] = df["name"].fillna("").astype(str)
    df["loan_type"] = df["loan_type"].fillna("emi").astype(str)
    df["principal_amount"] = pd.to_numeric(df["principal_amount"], errors="coerce").fillna(0.0)
    df["payment_amount"] = pd.to_numeric(df["payment_amount"], errors="coerce").fillna(0.0)
    df["payment_frequency"] = df["payment_frequency"].fillna("monthly").astype(str)
    df["payment_day_rule"] = df["payment_day_rule"].fillna("").astype(str)
    df["payments_made"] = pd.to_numeric(df["payments_made"], errors="coerce").fillna(0).astype(int)
    df["term_count"] = pd.to_numeric(df["term_count"], errors="coerce")
    df["annual_interest_rate"] = pd.to_numeric(df["annual_interest_rate"], errors="coerce")
    df["priority"] = pd.to_numeric(df["priority"], errors="coerce").fillna(5).astype(int)
    df["notes"] = df["notes"].fillna("").astype(str)
    df["start_date"] = df["start_date"].apply(_to_date)
    df["end_date"] = df["end_date"].apply(_to_date)

    return df[RAW_COLUMNS]


def dataframe_to_raw_loans(df: pd.DataFrame) -> List[RawLoanInput]:
    loans: List[RawLoanInput] = []
    for _, row in df.iterrows():
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        loans.append(
            RawLoanInput(
                name=name,
                principal_amount=_to_float(row.get("principal_amount")),
                loan_type=str(row.get("loan_type") or "emi"),
                payment_amount=_to_float(row.get("payment_amount")),
                payment_frequency=str(row.get("payment_frequency") or "monthly"),
                start_date=_to_date(row.get("start_date")),
                payment_day_rule=str(row.get("payment_day_rule") or ""),
                payments_made=_to_int(row.get("payments_made")) or 0,
                annual_interest_rate=_to_optional_float(row.get("annual_interest_rate")),
                term_count=_to_int(row.get("term_count")),
                end_date=_to_date(row.get("end_date")),
                priority=_to_int(row.get("priority")),
                notes=str(row.get("notes") or ""),
            )
        )
    return loans


def enriched_to_dataframe(loans: List[EnrichedLoan]) -> pd.DataFrame:
    rows = []
    for L in loans:
        rows.append({
            "Loan": L.name,
            "Type": L.loan_type,
            "Status": L.status,
            "Principal": round(L.principal_amount, 2),
            "Outstanding": round(L.outstanding_principal, 2),
            "Overdue": f"{L.overdue_amount:,.0f}" if L.overdue else "-",
            "Total debt now": round(L.total_debt_now, 2),
            "Annual rate": f"{L.annual_interest_rate:.1f}%" + ("" if L.rate_was_explicit else " (derived)"),
            "Start": L.start_date,
            "Closes": str(L.end_date) if L.end_date else "ongoing",
            "Payment": f"{L.payment_amount:,.0f} / {L.payment_frequency}",
            "Paid so far": f"{L.payments_made}/{L.term_count}" if L.term_count else f"{L.payments_made}",
            "Progress": f"{L.repayment_progress_pct}%",
            "Priority": L.priority,
            "Next due": L.next_due_date,
        })
    return pd.DataFrame(rows)


def timeline_to_dataframe(sim_result) -> pd.DataFrame:
    rows = []
    for x in sim_result.timeline:
        rows.append({
            "Month": x.month_index,
            "Date": x.month_date,
            "Loan": x.debt_name,
            "Status": x.note,
            "Opening": round(x.opening_balance, 2),
            "Scheduled": round(x.scheduled_payment, 2),
            "Cost (interest)": round(x.interest_or_cost, 2),
            "Principal paid": round(x.principal_paid, 2),
            "Extra payment": round(x.extra_payment, 2),
            "Closing": round(x.closing_balance, 2),
        })
    return pd.DataFrame(rows)


def split_questions(raw_loans: List[RawLoanInput]) -> tuple[List[str], List[str]]:
    all_questions: List[str] = []
    for r in raw_loans:
        all_questions.extend(validate_raw_loan(r))
    blocking = [q for q in all_questions if "interest rate given" not in q]
    soft = [q for q in all_questions if "interest rate given" in q]
    return blocking, soft


# ---------------------------------------------------------------------
# UI - only runs when launched via `streamlit run streamlit_app.py`,
# not on a plain import (so the functions above stay unit-testable).
# ---------------------------------------------------------------------
def render_app() -> None:
    st.set_page_config(page_title="Debt Planner", layout="wide")
    st.title("Debt Planner")
    st.caption("Describe your loans in plain numbers - the engine works out the rest.")

    with st.sidebar:
        st.header("Monthly cash")
        monthly_income = st.number_input("Monthly income", min_value=0.0, value=40000.0, step=1000.0)
        monthly_fixed_expenses = st.number_input("Fixed monthly expenses (rent, groceries, etc.)", min_value=0.0, value=0.0, step=1000.0)
        current_date = st.date_input("Today's date", value=date.today())

        st.divider()
        st.write("This app will:")
        st.write("1. work out end dates, overdue amounts, and effective interest rates per loan")
        st.write("2. show this month's surplus after scheduled payments")
        st.write("3. run an avalanche-within-priority payoff simulation and your debt-free date")

    st.subheader("Your loans")
    st.write(
        "Leave **total payments** blank if you know the **end date** instead, or vice versa. "
        "Leave **annual interest rate** blank if you don't know it - it's worked out automatically "
        "for interest-only loans, and estimated for EMI loans from principal/payment/term."
    )

    if "loans_df" not in st.session_state:
        st.session_state.loans_df = pd.DataFrame(default_rows())

    edited = st.data_editor(
        normalize_dataframe(st.session_state.loans_df),
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={
            "name": st.column_config.TextColumn("Loan name", required=True),
            "loan_type": st.column_config.SelectboxColumn("Type", options=LOAN_TYPES, required=True),
            "principal_amount": st.column_config.NumberColumn("Principal", min_value=0.0, format="%.2f"),
            "payment_amount": st.column_config.NumberColumn("Payment amount", min_value=0.0, format="%.2f"),
            "payment_frequency": st.column_config.SelectboxColumn("Frequency", options=FREQUENCIES, required=True),
            "start_date": st.column_config.DateColumn("Start date"),
            "payment_day_rule": st.column_config.TextColumn("Due date rule (e.g. '5th of month')"),
            "payments_made": st.column_config.NumberColumn("Payments made so far", min_value=0, step=1),
            "term_count": st.column_config.NumberColumn("Total payments (term)", min_value=0, step=1, help="Leave blank if you gave an end date instead"),
            "end_date": st.column_config.DateColumn("End date (optional)", help="Leave blank if you gave a term instead"),
            "annual_interest_rate": st.column_config.NumberColumn("Annual interest rate % (optional)", help="Leave blank to let the engine work it out"),
            "priority": st.column_config.NumberColumn("Priority (1-10)", min_value=1, max_value=10, step=1),
            "notes": st.column_config.TextColumn("Notes"),
        },
    )
    st.session_state.loans_df = edited

    run = st.button("Calculate my debt-free plan", type="primary")

    if not run:
        st.info("Fill in your loans above and click 'Calculate my debt-free plan'.")
        return

    raw_loans = dataframe_to_raw_loans(edited)
    if not raw_loans:
        st.warning("Add at least one loan first.")
        return

    blocking, soft = split_questions(raw_loans)
    if blocking:
        st.error("Need a bit more info before this can be calculated:")
        for q in blocking:
            st.write(f"- {q}")
        return

    try:
        plan: PlanResult = build_plan(
            raw_loans=raw_loans,
            monthly_income=monthly_income,
            monthly_fixed_expenses=monthly_fixed_expenses,
            current_date=current_date,
            verbose=False,
        )
    except ValueError as e:
        st.error(str(e))
        return

    if soft:
        st.info("Assumed - please confirm these:")
        for q in soft:
            st.write(f"- {q}")

    st.markdown("## Your loans, fully worked out")
    st.dataframe(enriched_to_dataframe(plan.enriched_loans), width="stretch", hide_index=True)

    for L in plan.enriched_loans:
        for flag in L.risk_flags:
            st.warning(f"{L.name}: {flag}")

    st.markdown("## This month")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Income", f"{monthly_income:,.0f}")
    c2.metric("Fixed expenses", f"{monthly_fixed_expenses:,.0f}")
    c3.metric("Scheduled loan payments", f"{plan.monthly_scheduled_outflow:,.0f}")
    c4.metric("Surplus", f"{plan.monthly_surplus:,.0f}")

    st.markdown("## Payoff plan")
    st.write(
        f"**Overall debt-free date:** {plan.overall_debt_free_date} "
        f"&nbsp;&middot;&nbsp; **{plan.simulation.total_months} months from now**"
    )
    st.write(f"**Total interest/cost across all loans:** {plan.simulation.total_interest_or_cost:,.2f}")

    payoff_rows = [
        {"Loan": name, "Clears on": str(plan.simulation.payoff_dates[name]) if name in plan.simulation.payoff_dates else "not within horizon"}
        for name in plan.simulation.payoff_order
    ]
    st.dataframe(pd.DataFrame(payoff_rows), width="stretch", hide_index=True)

    st.markdown("## Suggested actions")
    for a in plan.actions:
        st.write(f"- {a}")

    with st.expander("Show month-by-month simulation timeline"):
        st.dataframe(timeline_to_dataframe(plan.simulation), width="stretch", hide_index=True)


if __name__ == "__main__":
    render_app()
