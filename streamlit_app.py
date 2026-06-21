from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from datetime import date
from typing import Optional, List, Dict, Any

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



# -----------------------------
# Data model for the UI
# -----------------------------
@dataclass
class UiDebt:
    name: str
    balance: float
    due_now: float = 0.0
    minimum_due: float = 0.0
    required_to_settle: Optional[float] = None
    strictness: int = 5
    commitment_priority: int = 0
    user_priority: int = 0
    due_in_days: Optional[int] = None
    overdue: bool = False
    partial_ok: bool = False
    partial_threshold_pct: float = 0.85

    emi: float = 0.0
    start_date: str = "2025-01-01"
    end_date: str = "2026-01-01"
    annual_interest_rate: Optional[float] = None
    track_in_simulator: bool = True
    notes: str = ""
    order: int = 0  # lower first; 0 means auto/risk order


def default_rows() -> list[dict[str, Any]]:
    return [
        dict(
            name="Family Commitment",
            balance=50000.0,
            due_now=5000.0,
            minimum_due=5000.0,
            required_to_settle=5000.0,
            strictness=9,
            commitment_priority=10,
            user_priority=0,
            due_in_days=0,
            overdue=False,
            partial_ok=False,
            partial_threshold_pct=0.85,
            emi=0.0,
            start_date="2025-01-01",
            end_date="2026-01-01",
            annual_interest_rate=None,
            track_in_simulator=False,
            notes="Family pressure / immediate obligation",
            order=1,
        ),
        dict(
            name="Daily EMI Loan",
            balance=165000.0,
            due_now=36525.0,
            minimum_due=36525.0,
            required_to_settle=36525.0,
            strictness=6,
            commitment_priority=0,
            user_priority=0,
            due_in_days=2,
            overdue=False,
            partial_ok=False,
            partial_threshold_pct=0.85,
            emi=1200.0,
            start_date="2025-01-01",
            end_date="2027-01-01",
            annual_interest_rate=None,
            track_in_simulator=True,
            notes="Daily EMI / fast-draining loan",
            order=2,
        ),
        dict(
            name="Interest Only 7%",
            balance=100000.0,
            due_now=7000.0,
            minimum_due=7000.0,
            required_to_settle=7000.0,
            strictness=8,
            commitment_priority=0,
            user_priority=0,
            due_in_days=5,
            overdue=False,
            partial_ok=False,
            partial_threshold_pct=0.90,
            emi=7000.0,
            start_date="2025-01-01",
            end_date="2030-01-01",
            annual_interest_rate=7.0,
            track_in_simulator=True,
            notes="Interest-only style monthly payment",
            order=3,
        ),
    ]


def debt_risk_score(row: pd.Series) -> float:
    urgency = 0.0
    if bool(row.get("overdue", False)):
        urgency += 1000.0
    else:
        due = row.get("due_in_days", None)
        if pd.notna(due):
            due = int(due)
            if due <= 0:
                urgency += 800.0
            elif due <= 3:
                urgency += 500.0
            elif due <= 7:
                urgency += 300.0
            elif due <= 15:
                urgency += 120.0
            else:
                urgency += 40.0
        else:
            urgency += 20.0

    commitment = float(row.get("commitment_priority", 0) or 0) * 80.0
    strictness = float(row.get("strictness", 0) or 0) * 50.0
    user = float(row.get("user_priority", 0) or 0) * 70.0

    balance = float(row.get("balance", 0) or 0)
    target = row.get("required_to_settle", None)
    if pd.isna(target) or target is None or float(target) <= 0:
        target = float(row.get("minimum_due", 0) or 0)
        if target <= 0:
            target = float(row.get("due_now", 0) or 0)

    ratio = (float(target) / balance) if balance > 0 else 1.0
    settle_eff = (1.0 - min(ratio, 1.0)) * 10.0

    return urgency + commitment + strictness + user + settle_eff * 10.0


def row_target_amount(row: pd.Series) -> float:
    target = row.get("required_to_settle", None)
    if pd.isna(target) or target is None:
        target = row.get("minimum_due", 0)
        if pd.isna(target) or target is None or float(target) <= 0:
            target = row.get("due_now", 0)
    return max(0.0, float(target or 0.0))


def build_dataframe() -> pd.DataFrame:
    if "debts_df" not in st.session_state:
        st.session_state.debts_df = pd.DataFrame(default_rows())
    return st.session_state.debts_df.copy()


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure columns exist
    required_cols = [
        "name", "balance", "due_now", "minimum_due", "required_to_settle", "strictness",
        "commitment_priority", "user_priority", "due_in_days", "overdue", "partial_ok",
        "partial_threshold_pct", "emi", "start_date", "end_date", "annual_interest_rate",
        "track_in_simulator", "notes", "order"
    ]
    for c in required_cols:
        if c not in df.columns:
            df[c] = None

    # Clean types / fill blanks
    df["name"] = df["name"].fillna("").astype(str)
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce").fillna(0.0)
    df["due_now"] = pd.to_numeric(df["due_now"], errors="coerce").fillna(0.0)
    df["minimum_due"] = pd.to_numeric(df["minimum_due"], errors="coerce").fillna(0.0)
    df["required_to_settle"] = pd.to_numeric(df["required_to_settle"], errors="coerce")
    df["strictness"] = pd.to_numeric(df["strictness"], errors="coerce").fillna(5).astype(int)
    df["commitment_priority"] = pd.to_numeric(df["commitment_priority"], errors="coerce").fillna(0).astype(int)
    df["user_priority"] = pd.to_numeric(df["user_priority"], errors="coerce").fillna(0).astype(int)
    df["due_in_days"] = pd.to_numeric(df["due_in_days"], errors="coerce")
    df["overdue"] = df["overdue"].fillna(False).astype(bool)
    df["partial_ok"] = df["partial_ok"].fillna(False).astype(bool)
    df["partial_threshold_pct"] = pd.to_numeric(df["partial_threshold_pct"], errors="coerce").fillna(0.85)
    df["emi"] = pd.to_numeric(df["emi"], errors="coerce").fillna(0.0)
    df["start_date"] = df["start_date"].fillna("2025-01-01").astype(str)
    df["end_date"] = df["end_date"].fillna("2026-01-01").astype(str)
    df["annual_interest_rate"] = pd.to_numeric(df["annual_interest_rate"], errors="coerce")
    df["track_in_simulator"] = df["track_in_simulator"].fillna(True).astype(bool)
    df["notes"] = df["notes"].fillna("").astype(str)
    df["order"] = pd.to_numeric(df["order"], errors="coerce").fillna(0).astype(int)

    # Auto-fill required_to_settle if blank
    for idx, row in df.iterrows():
        if pd.isna(row["required_to_settle"]):
            target = row["minimum_due"]
            if pd.isna(target) or float(target) <= 0:
                target = row["due_now"]
            df.at[idx, "required_to_settle"] = max(0.0, float(target or 0.0))

    return df


def decision_plan(df: pd.DataFrame, available_cash: float) -> tuple[pd.DataFrame, bool, str, float, Dict[str, float]]:
    """
    Returns:
    - decision table
    - needs_user_choice
    - recommendation
    - remaining_cash
    - paid_map
    """
    work = df.copy()

    # Sort by manual order if any positive order exists; otherwise by risk
    manual_rows = work[work["order"] > 0].copy()
    auto_rows = work[work["order"] <= 0].copy()

    manual_rows = manual_rows.sort_values(["order", "name"], ascending=[True, True])
    if len(manual_rows) > 0:
        ranked = pd.concat([manual_rows, auto_rows.sort_values(by=["name"])], ignore_index=True)
        ordering_mode = "manual order"
    else:
        ranked = work.sort_values(by=["_risk"], ascending=False) if "_risk" in work.columns else work.copy()
        ordering_mode = "risk order"

    remaining_cash = available_cash
    paid_map: Dict[str, float] = {}
    rows = []
    needs_user_choice = False

    for _, row in ranked.iterrows():
        name = str(row["name"])
        balance = float(row["balance"])
        target = row_target_amount(row)

        if not name.strip() or target <= 0:
            continue

        if remaining_cash <= 0:
            rows.append({
                "name": name,
                "target": target,
                "paid": 0.0,
                "unpaid": target,
                "action": "skipped",
                "reason": "no cash left",
                "remaining_cash_after": remaining_cash,
            })
            continue

        if remaining_cash >= target:
            paid = target
            unpaid = 0.0
            remaining_cash -= paid
            action = "full_settlement"
            reason = "fully settled"
        else:
            ratio = remaining_cash / target if target > 0 else 0.0
            partial_ok = bool(row.get("partial_ok", False))
            threshold = float(row.get("partial_threshold_pct", 1.0) or 1.0)

            if partial_ok or ratio >= threshold:
                paid = remaining_cash
                unpaid = target - paid
                remaining_cash = 0.0
                action = "partial_payment" if partial_ok else "partial_effective"
                reason = f"partial accepted ({ratio:.0%})"
            else:
                paid = 0.0
                unpaid = target
                action = "manual_choice"
                reason = "not enough cash; needs user choice"
                needs_user_choice = True

        paid_map[name] = paid_map.get(name, 0.0) + paid

        rows.append({
            "name": name,
            "target": target,
            "paid": paid,
            "unpaid": unpaid,
            "action": action,
            "reason": reason,
            "remaining_cash_after": remaining_cash,
        })

        # If we hit a debt that needs user choice, stop the automatic allocation.
        if action == "manual_choice":
            break

    result_df = pd.DataFrame(rows)
    recommendation = (
        f"Used {ordering_mode}. "
        "If cash is short, change the order or allow partial payment for the debt you want to prioritize."
    )

    return result_df, needs_user_choice, recommendation, remaining_cash, paid_map


def convert_to_sim_debts(df: pd.DataFrame, paid_map: Dict[str, float]) -> List[SimDebt]:
    sim_debts: List[SimDebt] = []
    for _, row in df.iterrows():
        if not bool(row["track_in_simulator"]):
            continue

        name = str(row["name"])
        balance = float(row["balance"])
        paid_now = float(paid_map.get(name, 0.0))
        adjusted_balance = max(0.0, balance - paid_now)

        if adjusted_balance <= 0:
            continue

        emi = float(row["emi"] or 0.0)
        if emi <= 0:
            continue

        sim_debts.append(
            SimDebt(
                name=name,
                principal=adjusted_balance,
                emi=emi,
                start_date=str(row["start_date"]),
                end_date=str(row["end_date"]),
                annual_interest_rate=(None if pd.isna(row["annual_interest_rate"]) else float(row["annual_interest_rate"])),
                priority=max(int(row["strictness"]), int(row["commitment_priority"]), int(row["user_priority"])),
                notes=str(row["notes"]),
            )
        )
    return sim_debts


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Debt Planner Engine", layout="wide")

st.title("Debt Planner Engine")
st.caption("Decision engine + payoff simulator in one app")

with st.sidebar:
    st.header("Monthly cash")
    monthly_income = st.number_input("Monthly income", min_value=0.0, value=40000.0, step=1000.0)
    monthly_expenses = st.number_input("Monthly expenses", min_value=0.0, value=0.0, step=1000.0)
    other_fixed = st.number_input("Other fixed obligations", min_value=0.0, value=0.0, step=1000.0)
    current_date = st.date_input("Current date", value=date.today())

    st.divider()
    st.write("The app will:")
    st.write("1. rank debts")
    st.write("2. create this month's decision plan")
    st.write("3. run the payoff simulator using remaining balances")

df = build_dataframe()
df["_risk"] = df.apply(debt_risk_score, axis=1)

st.subheader("Edit debts")
st.write("You can add rows, change order, and adjust partial-payment behavior.")
edited = st.data_editor(
    normalize_dataframe(df),
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "order": st.column_config.NumberColumn("order", help="Lower numbers run first. Use 0 for auto risk order."),
        "partial_ok": st.column_config.CheckboxColumn("partial_ok", help="Allow partial payment"),
        "track_in_simulator": st.column_config.CheckboxColumn("track_in_simulator", help="Include in payoff simulation"),
        "overdue": st.column_config.CheckboxColumn("overdue"),
    },
)

col1, col2 = st.columns([1, 2])
with col1:
    run = st.button("Run decision + simulation", type="primary")
with col2:
    st.info("Tip: use order 1,2,3 to force the month-by-month priority you want.")

if run:
    work = normalize_dataframe(edited)
    work["_risk"] = work.apply(debt_risk_score, axis=1)

    available_cash = max(0.0, monthly_income - monthly_expenses - other_fixed)
    income = IncomeContext(
        monthly_income=monthly_income,
        monthly_expenses=monthly_expenses,
        other_fixed_obligations=other_fixed,
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
        name = str(row["name"])
        balance = float(row["balance"])
        paid_now = float(paid_map.get(name, 0.0))
        remaining = max(0.0, balance - paid_now)
        balance_rows.append({
            "name": name,
            "original_balance": balance,
            "paid_now": paid_now,
            "remaining_balance": remaining,
            "track_in_simulator": bool(row["track_in_simulator"]),
        })
    balance_df = pd.DataFrame(balance_rows)
    st.dataframe(balance_df, use_container_width=True, hide_index=True)

    sim_debts = convert_to_sim_debts(work, paid_map)

    st.markdown("## Payoff simulation")
    if not sim_debts:
        st.warning("No debts were left for simulation after the decision step.")
    else:
        simulator = PayoffSimulator(
            income=monthly_income,
            monthly_expenses=monthly_expenses,
            debts=sim_debts,
        )
        sim_result = simulator.simulate(verbose=False)

        sim_date = getattr(sim_result, "simulation_date", current_date)
        total_cost = getattr(sim_result, "total_cost", getattr(sim_result, "total_interest_or_cost", 0.0))
        sim_summary = pd.DataFrame([{
            "simulation_date": sim_date,
            "months_to_debt_freedom": sim_result.total_months,
            "total_paid": round(sim_result.total_paid, 2),
            "total_cost": round(total_cost, 2),
            "total_principal_paid": round(sim_result.total_principal_paid, 2),
            "payoff_order": ", ".join(sim_result.payoff_order),
        }])
        st.dataframe(sim_summary, use_container_width=True, hide_index=True)

        st.write("### Payoff dates")
        if sim_result.payoff_dates:
            payoff_df = pd.DataFrame(
                [{"debt": k, "payoff_date": v} for k, v in sim_result.payoff_dates.items()]
            )
            st.dataframe(payoff_df, use_container_width=True, hide_index=True)
        else:
            st.info("No payoff dates yet in the simulation output.")

        with st.expander("Show simulation timeline"):
            tl_rows = []
            for x in sim_result.timeline:
                tl_rows.append({
                    "month_index": getattr(x, "month_index", None),
                    "month_date": getattr(x, "month_date", None),
                    "debt_name": getattr(x, "debt_name", ""),
                    "status": getattr(x, "status", getattr(x, "note", "")),
                    "opening_balance": round(getattr(x, "opening_balance", 0.0), 2),
                    "scheduled_payment": round(getattr(x, "scheduled_payment", 0.0), 2),
                    "cost_component": round(getattr(x, "cost_component", getattr(x, "interest_or_cost", 0.0)), 2),
                    "principal_paid": round(getattr(x, "principal_paid", 0.0), 2),
                    "extra_payment": round(getattr(x, "extra_payment", 0.0), 2),
                    "closing_balance": round(getattr(x, "closing_balance", 0.0), 2),
                    "note": getattr(x, "note", getattr(x, "status", "")),
                })
            st.dataframe(pd.DataFrame(tl_rows), use_container_width=True, hide_index=True)
else:
    st.info("Edit the debts table and click 'Run decision + simulation'.")
