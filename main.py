from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Dict

from decision_engine import (
    Debt as DecisionDebt,
    IncomeContext,
    decide_payments,
)

from payoff_simulator import (
    Debt as SimDebt,
    PayoffSimulator,
)


@dataclass
class MasterDebt:
    """
    Single source of truth for both engines.
    - Decision engine uses the immediate-payment fields.
    - Payoff simulator uses the loan schedule fields.
    """
    name: str
    balance: float

    # Immediate decision fields
    due_now: float = 0.0
    minimum_due: float = 0.0
    required_to_settle: Optional[float] = None
    strictness: int = 5
    commitment_priority: int = 0
    user_priority: int = 0
    due_in_days: Optional[int] = None
    overdue: bool = False

    # Loan schedule fields for simulator
    emi: float = 0.0
    start_date: str = "2025-01-01"
    end_date: str = "2026-01-01"
    annual_interest_rate: Optional[float] = None
    track_in_simulator: bool = True

    notes: str = ""


def build_sample_data() -> tuple[list[MasterDebt], IncomeContext, date]:
    """
    Edit these values for your own testing.
    """
    current_date = date(2026, 6, 20)

    debts = [
        MasterDebt(
            name="Family Commitment",
            balance=50000,
            due_now=5000,
            minimum_due=5000,
            required_to_settle=5000,
            strictness=9,
            commitment_priority=10,
            due_in_days=0,
            overdue=False,
            emi=0.0,
            track_in_simulator=False,
            notes="Family pressure / immediate obligation",
        ),
        MasterDebt(
            name="Daily EMI Loan",
            balance=165000,
            due_now=36525,
            minimum_due=36525,
            required_to_settle=36525,
            strictness=6,
            commitment_priority=0,
            due_in_days=2,
            overdue=False,
            emi=1200,
            start_date="2025-01-01",
            end_date="2027-01-01",
            annual_interest_rate=None,
            track_in_simulator=True,
            notes="Daily EMI / fast-draining loan",
        ),
        MasterDebt(
            name="Interest Only 7%",
            balance=100000,
            due_now=7000,
            minimum_due=7000,
            required_to_settle=7000,
            strictness=8,
            commitment_priority=0,
            due_in_days=5,
            overdue=False,
            emi=7000,
            start_date="2025-01-01",
            end_date="2030-01-01",
            annual_interest_rate=7.0,
            track_in_simulator=True,
            notes="Interest-only style monthly payment",
        ),
    ]

    income = IncomeContext(
        monthly_income=40000,
        monthly_expenses=0,
        other_fixed_obligations=0,
    )

    return debts, income, current_date


def to_decision_debt(d: MasterDebt) -> DecisionDebt:
    return DecisionDebt(
        name=d.name,
        balance=d.balance,
        due_now=d.due_now,
        minimum_due=d.minimum_due,
        required_to_settle=d.required_to_settle,
        strictness=d.strictness,
        commitment_priority=d.commitment_priority,
        user_priority=d.user_priority,
        due_in_days=d.due_in_days,
        overdue=d.overdue,
        notes=d.notes,
    )


def sum_paid_by_debt(decision_lines) -> Dict[str, float]:
    paid_map: Dict[str, float] = {}
    for line in decision_lines:
        paid_map[line.debt_name] = paid_map.get(line.debt_name, 0.0) + float(line.paid)
    return paid_map


def bridge_to_simulator(master_debts: List[MasterDebt], paid_map: Dict[str, float]) -> List[SimDebt]:
    sim_debts: List[SimDebt] = []

    for d in master_debts:
        if not d.track_in_simulator:
            continue

        paid = paid_map.get(d.name, 0.0)
        adjusted_balance = max(0.0, d.balance - paid)

        if adjusted_balance <= 0.0:
            continue

        if d.emi <= 0:
            continue

        sim_debts.append(
            SimDebt(
                name=d.name,
                principal=adjusted_balance,
                emi=d.emi,
                start_date=d.start_date,
                end_date=d.end_date,
                annual_interest_rate=d.annual_interest_rate,
                priority=max(d.strictness, d.commitment_priority, d.user_priority),
                notes=d.notes,
            )
        )

    return sim_debts


def print_master_balances(master_debts: List[MasterDebt], paid_map: Dict[str, float]) -> None:
    print("\n=== BALANCES AFTER DECISION ENGINE ===")
    for d in master_debts:
        paid = paid_map.get(d.name, 0.0)
        remaining = max(0.0, d.balance - paid)
        print(
            f"{d.name}: original_balance={d.balance:.2f}, paid_now={paid:.2f}, "
            f"remaining_balance={remaining:.2f}, track_in_simulator={d.track_in_simulator}"
        )


def main() -> None:
    debts, income, current_date = build_sample_data()

    print("\n==============================")
    print("STEP 1: DECISION ENGINE")
    print("==============================")

    decision_debts = [to_decision_debt(d) for d in debts]
    decision_result = decide_payments(decision_debts, income, interactive=True)

    paid_map = sum_paid_by_debt(decision_result.lines)
    print_master_balances(debts, paid_map)

    print("\n==============================")
    print("STEP 2: PAYOFF SIMULATOR")
    print("==============================")

    sim_debts = bridge_to_simulator(debts, paid_map)

    if not sim_debts:
        print("No simulation debts left after the decision step.")
        return

    simulator = PayoffSimulator(
        income=income.monthly_income,
        monthly_expenses=income.monthly_expenses,
        debts=sim_debts,
        current_date=current_date,
    )

    sim_result = simulator.simulate(verbose=True)

    print("\n==============================")
    print("FINAL CONNECTED SUMMARY")
    print("==============================")
    print(f"Decision engine needs user choice: {decision_result.needs_user_choice}")
    print(f"Decision engine recommendation: {decision_result.recommendation}")
    print(f"Remaining cash after decisions: {decision_result.remaining_cash:.2f}")
    print(f"Simulation months to debt freedom: {sim_result.total_months}")
    print(f"Simulation total paid: {sim_result.total_paid:.2f}")
    print(f"Simulation total cost: {sim_result.total_cost:.2f}")
    print(f"Simulation payoff order: {sim_result.payoff_order}")

    if decision_result.lines:
        print("\nDecision lines:")
        for line in decision_result.lines:
            print(
                f"- {line.debt_name}: target={line.target:.2f}, paid={line.paid:.2f}, "
                f"unpaid={line.unpaid:.2f}, action={line.action}"
            )


if __name__ == "__main__":
    main()
