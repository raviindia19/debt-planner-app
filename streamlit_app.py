from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from payoff_simulator_v3 import Debt as SimDebt, PayoffSimulator


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


def target_amount(row: pd.Series) -> float:
    target = row.get("required_to_settle", None)
    if pd.isna(target) or target is None:
        target = row.get("minimum_due", 0)
        if pd.isna(target) or target is None or float(target) <= 0:
            target = row.get("due_now", 0)
    return max(0.0, float(target or 0.0))


def risk_score(row: pd.Series) -> float:
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

    balance = float(row.get("balance", 0) or 0)
    target = target_amount(row)
    ratio = target / balance if balance > 0 else 1.0
    settle_eff = (1.0 - min(ratio, 1.0)) * 10.0

    commitment = float(row.get("commitment_priority", 0) or 0) * 80.0
    strictness = float(row.get("strictness", 0) or 0) * 50.0
    user = float(row.get("user_priority", 0) or 0) * 70.0

    return urgency + commitment + strictness + user + settle_eff * 10.0


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    columns = [
        "name", "balance", "due_now", "minimum_due", "required_to_settle", "strictness",
        "commitment_priority", "user_priority", "due_in_days", "overdue", "partial_ok",
        "partial_threshold_pct", "emi", "start_date", "end_date", "annual_interest_rate",
        "track_in_simulator", "notes", "order"
    ]
    for c in columns:
        if c not in df.columns:
            df[c] = None

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

    for idx, row in df.iterrows():
        if pd.isna(row["required_to_settle"]):
            df.at[idx, "required_to_settle"] = target_amount(row)
    return df


def decision_plan(df: pd.DataFrame, available_cash: float):
    work = df.copy()
    work["_risk"] = work.apply(risk_score, axis=1)

    manual = work[work["order"] > 0].sort_values(["order", "_risk", "name"], ascending=[True, False, True])
    auto = work[work["order"] <= 0].sort_values(["_risk", "name"], ascending=[False, True])
    ranked = pd.concat([manual, auto], ignore_index=True)

    remaining_cash = available_cash
    paid_map: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []
    needs_choice = False

    for _, row in ranked.iterrows():
        name = str(row["name"])
        target = target_amount(row)
        if not name.strip() or target <= 0:
            continue

        if remaining_cash <= 0:
            rows.append({
                "name": name, "target": target, "paid": 0.0, "unpaid": target,
                "action": "skipped", "reason": "no cash left", "remaining_cash_after": remaining_cash
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
                needs_choice = True

        paid_map[name] = paid_map.get(name, 0.0) + paid
        rows.append({
            "name": name, "target": target, "paid": paid, "unpaid": unpaid,
            "action": action, "reason": reason, "remaining_cash_after": remaining_cash
        })

        if action == "manual_choice":
            break

    recommendation = "Use the order column to set priorities. Keep calculations below as background."
    return pd.DataFrame(rows), needs_choice, recommendation, remaining_cash, paid_map


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


def friendly_summary(decision_df: pd.DataFrame, remaining_cash: float, sim_result) -> str:
    manual = decision_df[decision_df["action"] == "manual_choice"]
    if not manual.empty:
        first = manual.iloc[0]
        return f"You cannot fully settle '{first['name']}' this month. Change the order or allow a partial payment if you want to reduce it."
    if remaining_cash > 0:
        return f"You still have ₹{remaining_cash:,.0f} left after the current plan."
    if getattr(sim_result, "payoff_dates", None):
        return f"Estimated debt-free date: {max(sim_result.payoff_dates.values())}"
    return "Review the plan below."


def build_summary_cards(available_cash: float, decision_df: pd.DataFrame, sim_result, summary_text: str):
    shortfall = float(decision_df["unpaid"].sum()) if not decision_df.empty and "unpaid" in decision_df.columns else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Available cash", f"₹{available_cash:,.0f}")
    c2.metric("Unpaid in plan", f"₹{shortfall:,.0f}")
    c3.metric("Debt-free months", f"{getattr(sim_result, 'total_months', 0)}")
    if getattr(sim_result, "payoff_dates", None):
        c4.metric("Debt-free date", f"{max(sim_result.payoff_dates.values())}")
    else:
        c4.metric("Debt-free date", "—")
    st.info(summary_text)


st.set_page_config(page_title="Debt Planner Engine", layout="wide")
st.title("Debt Planner Engine")
st.caption("Human-friendly summary on top, calculations below")

with st.sidebar:
    st.header("Monthly cash")
    monthly_income = st.number_input("Monthly income", min_value=0.0, value=40000.0, step=1000.0)
    monthly_expenses = st.number_input("Monthly expenses", min_value=0.0, value=0.0, step=1000.0)
    other_fixed = st.number_input("Other fixed obligations", min_value=0.0, value=0.0, step=1000.0)
    current_date = st.date_input("Current date", value=date.today())
    show_details = st.checkbox("Show technical calculations", value=False)

st.markdown("## Edit debts")
st.write("Keep this page simple for the user. The detailed math can stay hidden.")

base_df = normalize_dataframe(pd.DataFrame(default_rows()))
edited = st.data_editor(
    base_df,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "order": st.column_config.NumberColumn("order", help="Use 1, 2, 3 to force priority. Use 0 for auto-rank."),
        "partial_ok": st.column_config.CheckboxColumn("partial_ok", help="Allow partial payment"),
        "track_in_simulator": st.column_config.CheckboxColumn("track_in_simulator", help="Include in payoff simulation"),
        "overdue": st.column_config.CheckboxColumn("overdue"),
    },
)

work = normalize_dataframe(edited)
work["_risk"] = work.apply(risk_score, axis=1)
available_cash = max(0.0, monthly_income - monthly_expenses - other_fixed)

decision_df, needs_choice, recommendation, remaining_cash, paid_map = decision_plan(work, available_cash)

st.markdown("## Optional manual override")
st.write("If you want to use leftover cash, choose a debt and an extra amount.")
manual_enabled = st.checkbox("Use leftover cash on a specific debt", value=False)
extra_manual: Dict[str, float] = {}
if manual_enabled and remaining_cash > 0:
    candidates = list(work["name"].astype(str))
    chosen_debt = st.selectbox("Choose debt", candidates, index=0)
    extra_amount = st.number_input(
        "Extra amount to apply",
        min_value=0.0,
        max_value=float(remaining_cash),
        value=0.0,
        step=1000.0,
    )
    if extra_amount > 0:
        extra_manual[chosen_debt] = float(extra_amount)

sim_debts = convert_to_sim_debts(work, paid_map)
if extra_manual:
    for i, d in enumerate(sim_debts):
        if d.name in extra_manual:
            sim_debts[i] = SimDebt(
                name=d.name,
                principal=max(0.0, d.principal - extra_manual[d.name]),
                emi=d.emi,
                start_date=str(work.loc[work["name"] == d.name, "start_date"].iloc[0]),
                end_date=str(work.loc[work["name"] == d.name, "end_date"].iloc[0]),
                annual_interest_rate=(None if pd.isna(work.loc[work["name"] == d.name, "annual_interest_rate"].iloc[0]) else float(work.loc[work["name"] == d.name, "annual_interest_rate"].iloc[0])),
                priority=max(
                    int(work.loc[work["name"] == d.name, "strictness"].iloc[0]),
                    int(work.loc[work["name"] == d.name, "commitment_priority"].iloc[0]),
                    int(work.loc[work["name"] == d.name, "user_priority"].iloc[0]),
                ),
                notes=str(work.loc[work["name"] == d.name, "notes"].iloc[0]),
            )

if sim_debts:
    simulator = PayoffSimulator(
        income=monthly_income,
        monthly_expenses=monthly_expenses,
        debts=sim_debts,
        current_date=current_date,
    )
    sim_result = simulator.simulate(verbose=False)
else:
    class Empty:
        total_months = 0
        payoff_dates = {}
        total_paid = 0.0
        total_cost = 0.0
        total_principal_paid = 0.0
        payoff_order: list[str] = []
        timeline = []
    sim_result = Empty()

summary_text = friendly_summary(decision_df, remaining_cash - sum(extra_manual.values()), sim_result)
build_summary_cards(available_cash, decision_df, sim_result, summary_text)

simple_rows = []
for _, row in decision_df.iterrows():
    label = "Pay now" if row["action"] in {"full_settlement", "partial_payment", "partial_effective"} else "Needs your choice"
    simple_rows.append({
        "Debt": row["name"],
        "Status": label,
        "Pay this month": f"₹{float(row['paid']):,.0f}",
        "Left unpaid": f"₹{float(row['unpaid']):,.0f}",
    })
st.dataframe(pd.DataFrame(simple_rows), use_container_width=True, hide_index=True)

st.markdown("## Debt-free schedule")
if getattr(sim_result, "payoff_dates", None):
    schedule_df = pd.DataFrame([{"Debt": k, "Payoff date": v} for k, v in sim_result.payoff_dates.items()]).sort_values("Payoff date")
    st.dataframe(schedule_df, use_container_width=True, hide_index=True)
else:
    st.info("No payoff dates yet.")

if show_details:
    with st.expander("Technical calculations", expanded=False):
        st.markdown("### Decision output")
        st.dataframe(decision_df, use_container_width=True, hide_index=True)

        st.markdown("### Balances after decision")
        balance_rows = []
        for _, row in work.iterrows():
            name = str(row["name"])
            paid_now = float(paid_map.get(name, 0.0)) + float(extra_manual.get(name, 0.0))
            balance = float(row["balance"])
            balance_rows.append({
                "name": name,
                "original_balance": balance,
                "paid_now": paid_now,
                "remaining_balance": max(0.0, balance - paid_now),
                "track_in_simulator": bool(row["track_in_simulator"]),
            })
        st.dataframe(pd.DataFrame(balance_rows), use_container_width=True, hide_index=True)

        st.markdown("### Payoff simulation summary")
        sim_summary = pd.DataFrame([{
            "simulation_date": current_date,
            "months_to_debt_freedom": getattr(sim_result, "total_months", 0),
            "total_paid": round(getattr(sim_result, "total_paid", 0.0), 2),
            "total_cost": round(getattr(sim_result, "total_cost", 0.0), 2),
            "total_principal_paid": round(getattr(sim_result, "total_principal_paid", 0.0), 2),
            "payoff_order": ", ".join(getattr(sim_result, "payoff_order", [])),
        }])
        st.dataframe(sim_summary, use_container_width=True, hide_index=True)

        st.markdown("### Timeline")
        tl_rows = []
        for x in getattr(sim_result, "timeline", []):
            tl_rows.append({
                "month_index": x.month_index,
                "month_date": x.month_date,
                "debt_name": x.debt_name,
                "status": x.status,
                "opening_balance": round(x.opening_balance, 2),
                "scheduled_payment": round(x.scheduled_payment, 2),
                "cost_component": round(x.cost_component, 2),
                "principal_paid": round(x.principal_paid, 2),
                "extra_payment": round(x.extra_payment, 2),
                "closing_balance": round(x.closing_balance, 2),
                "note": x.note,
            })
        st.dataframe(pd.DataFrame(tl_rows), use_container_width=True, hide_index=True)
else:
    st.caption("Turn on 'Show technical calculations' in the sidebar to see the math, balance table, and simulator timeline.")

st.markdown("## AI-ready structured context")
st.write(
    "The future AI assistant should read the normalized debt rows and the calculated decision/simulation outputs, "
    "not the raw user text. That keeps math deterministic and lets AI focus on conversation and explanations."
)
