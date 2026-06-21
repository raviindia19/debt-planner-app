
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from payoff_simulator import Debt as SimDebt, PayoffSimulator

try:
    from decision_engine import IncomeContext
except Exception:

    @dataclass
    class IncomeContext:
        monthly_income: float
        monthly_expenses: float = 0.0
        other_fixed_obligations: float = 0.0

        @property
        def available_cash(self) -> float:
            return max(0.0, self.monthly_income - self.monthly_expenses - self.other_fixed_obligations)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def parse_date(value: Any) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(value.strip(), fmt).date()
            except ValueError:
                continue
    return None


def add_months(d: date, months: int) -> date:
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    day = min(d.day, [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30,
                      31, 31, 30, 31, 30, 31][month - 1])
    return date(year, month, day)


def parse_percent(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().upper()
    if not text or text in {"NA", "N/A", "NONE", "NULL", "-"}:
        return None
    text = text.replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def compute_end_date(start_date: Optional[date], frequency: str, total_emi_paid: Optional[int]) -> Optional[date]:
    if start_date is None or total_emi_paid is None or total_emi_paid <= 0:
        return None

    freq = (frequency or "").strip().lower()
    steps = max(0, total_emi_paid - 1)

    if "daily" in freq:
        return start_date + timedelta(days=steps)
    if "weekly" in freq:
        return start_date + timedelta(weeks=steps)
    if "month" in freq or "interest" in freq:
        return add_months(start_date, steps)
    if "year" in freq:
        return date(start_date.year + steps, start_date.month, start_date.day)

    return None


# -----------------------------------------------------------------------------
# Base table (user-facing)
# -----------------------------------------------------------------------------

TABLE_SCHEMA_VERSION = 2

DISPLAY_COLUMNS = [
    "Name",
    "Principal Amount",
    "Start_Date",
    "EMI_Date",
    "EMI Frequency",
    "EMI Amount",
    "Total EMI Paid",
    "Priority",
    "Annual Interest Rate",
]

REQUIRED_COLUMNS = DISPLAY_COLUMNS + ["Notes"]


def default_rows() -> list[dict[str, Any]]:
    return [
        {
            "Name": "Family Commitment",
            "Principal Amount": 50000,
            "Start_Date": date(2026, 5, 1),
            "EMI_Date": "1st of Every Month",
            "EMI Frequency": "Monthly",
            "EMI Amount": 5000,
            "Total EMI Paid": 10,
            "Priority": 10,
            "Annual Interest Rate": "0%",
            "Notes": "Family obligation",
        },
        {
            "Name": "Khatu",
            "Principal Amount": 100000,
            "Start_Date": date(2026, 5, 1),
            "EMI_Date": "Daily",
            "EMI Frequency": "Daily",
            "EMI Amount": 1200,
            "Total EMI Paid": 100,
            "Priority": 5,
            "Annual Interest Rate": "NA",
            "Notes": "Daily repayment loan",
        },
        {
            "Name": "Radhe",
            "Principal Amount": 100000,
            "Start_Date": date(2026, 2, 1),
            "EMI_Date": "5th of Every Month",
            "EMI Frequency": "Only Interest",
            "EMI Amount": 7000,
            "Total EMI Paid": 12,
            "Priority": 8,
            "Annual Interest Rate": "84%",
            "Notes": "Interest-only at 7% monthly",
        },
    ]


def build_dataframe() -> pd.DataFrame:
    # Reset any old saved table that came from the previous technical schema.
    if st.session_state.get("debts_schema_version") != TABLE_SCHEMA_VERSION:
        st.session_state.debts_df = pd.DataFrame(default_rows())
        st.session_state.debts_schema_version = TABLE_SCHEMA_VERSION

    if "debts_df" not in st.session_state:
        st.session_state.debts_df = pd.DataFrame(default_rows())
        st.session_state.debts_schema_version = TABLE_SCHEMA_VERSION

    df = st.session_state.debts_df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[REQUIRED_COLUMNS].copy()
    return df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    for col in REQUIRED_COLUMNS:
        if col not in work.columns:
            work[col] = None

    work["Name"] = work["Name"].fillna("").astype(str)
    work["Principal Amount"] = pd.to_numeric(work["Principal Amount"], errors="coerce").fillna(0.0)
    work["Start_Date"] = work["Start_Date"].apply(parse_date)
    work["EMI_Date"] = work["EMI_Date"].fillna("").astype(str)
    work["EMI Frequency"] = work["EMI Frequency"].fillna("").astype(str)
    work["EMI Amount"] = pd.to_numeric(work["EMI Amount"], errors="coerce").fillna(0.0)
    work["Total EMI Paid"] = work["Total EMI Paid"].apply(parse_int)
    work["Priority"] = pd.to_numeric(work["Priority"], errors="coerce").fillna(0).astype(int)
    work["Annual Interest Rate"] = work["Annual Interest Rate"].apply(parse_percent)
    work["Notes"] = work["Notes"].fillna("").astype(str)

    # Internal fields used by the engine (hidden from the user table)
    work["balance"] = work["Principal Amount"]
    work["emi"] = work["EMI Amount"]
    work["start_date"] = work["Start_Date"].astype(object)
    work["emi_date"] = work["EMI_Date"]
    work["emi_frequency"] = work["EMI Frequency"]
    work["emi_count"] = work["Total EMI Paid"].fillna(0).astype(int)
    work["end_date"] = [compute_end_date(s, f, c) for s, f, c in zip(work["Start_Date"], work["EMI Frequency"], work["Total EMI Paid"])]
    work["annual_interest_rate"] = work["Annual Interest Rate"]
    work["priority"] = work["Priority"]

    # Basic internal defaults so the engine still works.
    work["due_now"] = work["EMI Amount"]
    work["minimum_due"] = work["EMI Amount"]
    work["required_to_settle"] = work["EMI Amount"]
    work["strictness"] = work["Priority"]
    work["commitment_priority"] = work["Priority"].where(work["Priority"] >= 8, 0)
    work["user_priority"] = 0
    work["due_in_days"] = None
    work["overdue"] = False
    work["partial_ok"] = False
    work["partial_threshold_pct"] = 0.85
    work["track_in_simulator"] = True
    work["order"] = 0
    work["notes"] = work["Notes"]

    return work


def find_missing_fields(df: pd.DataFrame) -> list[str]:
    missing: list[str] = []
    for idx, row in df.iterrows():
        row_missing: list[str] = []
        if not str(row.get("Name", "")).strip():
            row_missing.append("Name")
        if float(row.get("Principal Amount", 0) or 0) <= 0:
            row_missing.append("Principal Amount")
        if row.get("Start_Date") is None:
            row_missing.append("Start_Date")
        if not str(row.get("EMI_Date", "")).strip():
            row_missing.append("EMI_Date")
        if not str(row.get("EMI Frequency", "")).strip():
            row_missing.append("EMI Frequency")
        if float(row.get("EMI Amount", 0) or 0) <= 0:
            row_missing.append("EMI Amount")
        if row.get("Total EMI Paid") is None:
            row_missing.append("Total EMI Paid")
        if pd.isna(row.get("Priority", None)):
            row_missing.append("Priority")
        if row_missing:
            missing.append(f"Row {idx + 1} ({row.get('Name', 'Unnamed')}): {', '.join(row_missing)}")
    return missing


# -----------------------------------------------------------------------------
# Decision / simulation helpers
# -----------------------------------------------------------------------------


def debt_risk_score(row: pd.Series) -> float:
    score = float(row.get("Priority", 0) or 0) * 100.0
    rate = parse_percent(row.get("Annual Interest Rate")) or 0.0
    score += rate
    if row.get("overdue", False):
        score += 1000.0
    return score


def decision_plan(df: pd.DataFrame, available_cash: float):
    work = df.copy()
    work["_risk"] = work.apply(debt_risk_score, axis=1)
    ranked = work.sort_values(by=["Priority", "_risk", "Name"], ascending=[False, False, True]).copy()

    remaining_cash = available_cash
    rows = []
    paid_map: Dict[str, float] = {}
    needs_user_choice = False

    for _, row in ranked.iterrows():
        name = str(row["Name"])
        target = float(row["EMI Amount"] or 0.0)
        if not name.strip() or target <= 0:
            continue

        if remaining_cash <= 0:
            rows.append({"name": name, "target": target, "paid": 0.0, "unpaid": target, "action": "skipped", "reason": "no cash left", "remaining_cash_after": remaining_cash})
            continue

        if remaining_cash >= target:
            paid = target
            unpaid = 0.0
            remaining_cash -= paid
            action = "full_settlement"
            reason = "fully settled"
        else:
            paid = remaining_cash
            unpaid = target - paid
            remaining_cash = 0.0
            action = "partial_payment"
            reason = "partial payment because cash ran out"
            needs_user_choice = True

        paid_map[name] = paid_map.get(name, 0.0) + paid
        rows.append({"name": name, "target": target, "paid": paid, "unpaid": unpaid, "action": action, "reason": reason, "remaining_cash_after": remaining_cash})

        if unpaid > 0:
            break

    result_df = pd.DataFrame(rows)
    recommendation = "Used priority order. If cash is short, the app will ask for the next decision or allow partial payment."
    return result_df, needs_user_choice, recommendation, remaining_cash, paid_map


def convert_to_sim_debts(df: pd.DataFrame, paid_map: Dict[str, float]) -> List[SimDebt]:
    sim_debts: List[SimDebt] = []
    for _, row in df.iterrows():
        name = str(row["Name"])
        balance = float(row["Principal Amount"] or 0.0)
        paid_now = float(paid_map.get(name, 0.0))
        adjusted_balance = max(0.0, balance - paid_now)
        if adjusted_balance <= 0:
            continue

        emi = float(row["EMI Amount"] or 0.0)
        if emi <= 0:
            continue

        start_dt = parse_date(row["Start_Date"])
        end_dt = compute_end_date(start_dt, str(row["EMI Frequency"] or ""), parse_int(row["Total EMI Paid"]))
        annual_rate = parse_percent(row["Annual Interest Rate"])

        sim_debts.append(
            SimDebt(
                name=name,
                principal=adjusted_balance,
                emi=emi,
                start_date=(start_dt or date.today()).isoformat(),
                end_date=(end_dt or (start_dt or date.today())).isoformat(),
                annual_interest_rate=annual_rate,
                priority=int(row["Priority"] or 0),
                notes=str(row.get("Notes", "")),
            )
        )
    return sim_debts


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Debt Planner Engine", layout="wide")
st.title("Debt Planner Engine")
st.caption("A simple intake table first. Hidden engine fields are derived automatically.")

with st.sidebar:
    st.header("Monthly cash")
    monthly_income = st.number_input("Monthly income", min_value=0.0, value=40000.0, step=1000.0)
    monthly_expenses = st.number_input("Monthly expenses", min_value=0.0, value=0.0, step=1000.0)
    other_fixed = st.number_input("Other fixed obligations", min_value=0.0, value=0.0, step=1000.0)
    current_date = st.date_input("Current date", value=date.today())

    st.divider()
    st.write("The app will:")
    st.write("1. accept a plain-user debt table")
    st.write("2. derive the internal engine columns")
    st.write("3. run decision logic and payoff simulation")

st.subheader("Debt intake table")
st.write("Use simple language here. The engine will calculate end date, annualized rate, and remaining balance internally.")

base_df = build_dataframe()
edited = st.data_editor(
    base_df,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Name": st.column_config.TextColumn("Name"),
        "Principal Amount": st.column_config.NumberColumn("Principal Amount", min_value=0.0),
        "Start_Date": st.column_config.DateColumn("Start_Date"),
        "EMI_Date": st.column_config.TextColumn("EMI_Date"),
        "EMI Frequency": st.column_config.TextColumn("EMI Frequency"),
        "EMI Amount": st.column_config.NumberColumn("EMI Amount", min_value=0.0),
        "Total EMI Paid": st.column_config.NumberColumn("Total EMI Paid", min_value=0, step=1, help="Use this as the payment count / tenure so the engine can calculate end date."),
        "Priority": st.column_config.NumberColumn("Priority", min_value=1, max_value=10, step=1),
        "Annual Interest Rate": st.column_config.TextColumn("Annual Interest Rate", help="Examples: 84%, 0%, NA"),
        "Notes": st.column_config.TextColumn("Notes"),
    },
)

work = normalize_dataframe(edited)
st.session_state.debts_df = work[REQUIRED_COLUMNS].copy()
missing = find_missing_fields(work)
if missing:
    with st.expander("Missing info the AI should ask for", expanded=True):
        for item in missing:
            st.write(f"- {item}")

col1, col2 = st.columns([1, 2])
with col1:
    run = st.button("Run decision + simulation", type="primary")
with col2:
    st.info("Tip: keep this table simple. The engine will create the hidden internal fields automatically.")

if run:
    available_cash = max(0.0, monthly_income - monthly_expenses - other_fixed)

    st.markdown("## Normalized internal table")
    st.dataframe(
        work[["Name", "Principal Amount", "Start_Date", "EMI_Date", "EMI Frequency", "EMI Amount", "Total EMI Paid", "Priority", "Annual Interest Rate", "balance", "end_date", "priority"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("## Decision output")
    decision_df, needs_choice, recommendation, remaining_cash, paid_map = decision_plan(work, available_cash)
    st.dataframe(decision_df, use_container_width=True, hide_index=True)
    st.write(f"**Available cash:** {available_cash:,.2f}")
    st.write(f"**Remaining cash after decision:** {remaining_cash:,.2f}")
    st.write(f"**Needs user choice:** {needs_choice}")
    st.write(f"**Recommendation:** {recommendation}")

    st.markdown("## Balances after decision")
    balance_rows = []
    for _, row in work.iterrows():
        name = str(row["Name"])
        original = float(row["Principal Amount"] or 0.0)
        paid_now = float(paid_map.get(name, 0.0))
        balance_rows.append({
            "name": name,
            "original_balance": original,
            "paid_now": paid_now,
            "remaining_balance": max(0.0, original - paid_now),
        })
    st.dataframe(pd.DataFrame(balance_rows), use_container_width=True, hide_index=True)

    st.markdown("## Payoff simulation")
    sim_debts = convert_to_sim_debts(work, paid_map)
    if not sim_debts:
        st.warning("No debts were left for simulation after the decision step.")
    else:
        try:
            simulator = PayoffSimulator(
                income=monthly_income,
                monthly_expenses=monthly_expenses,
                debts=sim_debts,
                current_date=current_date,
            )
        except TypeError:
            simulator = PayoffSimulator(
                income=monthly_income,
                monthly_expenses=monthly_expenses,
                debts=sim_debts,
            )

        sim_result = simulator.simulate(verbose=False)
        total_cost = getattr(sim_result, "total_cost", getattr(sim_result, "total_interest_or_cost", 0.0))
        simulation_date = getattr(sim_result, "simulation_date", current_date)
        st.dataframe(
            pd.DataFrame([{
                "simulation_date": simulation_date,
                "months_to_debt_freedom": getattr(sim_result, "total_months", None),
                "total_paid": round(getattr(sim_result, "total_paid", 0.0), 2),
                "total_cost": round(total_cost, 2),
                "total_principal_paid": round(getattr(sim_result, "total_principal_paid", 0.0), 2),
                "payoff_order": ", ".join(getattr(sim_result, "payoff_order", [])),
            }]),
            use_container_width=True,
            hide_index=True,
        )
        payoff_dates = getattr(sim_result, "payoff_dates", {})
        if payoff_dates:
            st.markdown("### Loan-wise clear dates")
            st.dataframe(pd.DataFrame([{"debt": k, "payoff_date": v} for k, v in payoff_dates.items()]), use_container_width=True, hide_index=True)
