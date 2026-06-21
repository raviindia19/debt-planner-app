from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# This script expects payoff_simulator.py to be in the same folder.
from payoff_simulator import Debt, PayoffSimulator, SimulationResult


@dataclass
class Scenario:
    name: str
    income: float
    monthly_expenses: float
    debts: List[Debt]
    notes: str = ""


def apply_lump_sum_payment(debts: List[Debt], debt_name: str, amount: float) -> List[Debt]:
    """
    Reduces the principal of a specific debt before simulation starts.
    Useful for "what if I pay an extra lump sum this month?" analysis.
    """
    if amount <= 0:
        return deepcopy(debts)

    updated = deepcopy(debts)
    for d in updated:
        if d.name == debt_name:
            d.principal = max(0.0, d.principal - amount)
            break
    return updated


def run_scenario(scenario: Scenario, verbose: bool = False) -> SimulationResult:
    sim = PayoffSimulator(
        income=scenario.income,
        monthly_expenses=scenario.monthly_expenses,
        debts=deepcopy(scenario.debts),
    )
    return sim.simulate(verbose=verbose)


def compare_results(base_name: str, base: SimulationResult, modified_name: str, modified: SimulationResult) -> Dict[str, Any]:
    months_saved = base.total_months - modified.total_months
    paid_diff = base.total_paid - modified.total_paid
    cost_saved = base.total_interest_or_cost - modified.total_interest_or_cost
    principal_diff = base.total_principal_paid - modified.total_principal_paid

    return {
        "base_name": base_name,
        "modified_name": modified_name,
        "base_months": base.total_months,
        "modified_months": modified.total_months,
        "months_saved": months_saved,
        "base_total_paid": base.total_paid,
        "modified_total_paid": modified.total_paid,
        "total_paid_difference": paid_diff,
        "base_cost": base.total_interest_or_cost,
        "modified_cost": modified.total_interest_or_cost,
        "cost_saved": cost_saved,
        "base_principal_paid": base.total_principal_paid,
        "modified_principal_paid": modified.total_principal_paid,
        "principal_paid_difference": principal_diff,
    }


def print_comparison(title: str, comparison: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(f"Base scenario months:      {comparison['base_months']}")
    print(f"Modified scenario months:  {comparison['modified_months']}")
    print(f"Months saved:              {comparison['months_saved']}")

    print(f"Base total paid:           {comparison['base_total_paid']:.2f}")
    print(f"Modified total paid:       {comparison['modified_total_paid']:.2f}")
    print(f"Difference in total paid:  {comparison['total_paid_difference']:.2f}")

    print(f"Base interest/cost:        {comparison['base_cost']:.2f}")
    print(f"Modified interest/cost:    {comparison['modified_cost']:.2f}")
    print(f"Cost saved:                {comparison['cost_saved']:.2f}")

    print(f"Base principal paid:       {comparison['base_principal_paid']:.2f}")
    print(f"Modified principal paid:   {comparison['modified_principal_paid']:.2f}")
    print(f"Principal diff:            {comparison['principal_paid_difference']:.2f}")


def analyze_what_if(
    base_income: float,
    base_monthly_expenses: float,
    debts: List[Debt],
    expense_change: float = 0.0,
    income_change: float = 0.0,
    lump_sum_debt_name: Optional[str] = None,
    lump_sum_amount: float = 0.0,
    verbose: bool = False,
) -> None:
    """
    Compare a base scenario with a modified scenario.

    Examples:
    - reduce monthly expenses by 10000
    - increase income by 5000
    - pay lump sum 50000 toward one debt before simulation
    """
    base_scenario = Scenario(
        name="Base",
        income=base_income,
        monthly_expenses=base_monthly_expenses,
        debts=deepcopy(debts),
        notes="Original scenario",
    )

    modified_debts = deepcopy(debts)

    if lump_sum_debt_name and lump_sum_amount > 0:
        modified_debts = apply_lump_sum_payment(modified_debts, lump_sum_debt_name, lump_sum_amount)

    modified_scenario = Scenario(
        name="Modified",
        income=base_income + income_change,
        monthly_expenses=max(0.0, base_monthly_expenses - expense_change),
        debts=modified_debts,
        notes="Changed income/expenses/lump sum",
    )

    print("\nWHAT-IF INPUTS")
    print("=" * 70)
    print(f"Base income:             {base_scenario.income:.2f}")
    print(f"Base monthly expenses:    {base_scenario.monthly_expenses:.2f}")
    print(f"Income change:            {income_change:.2f}")
    print(f"Expense change:           {expense_change:.2f}")
    print(f"Lump sum debt:            {lump_sum_debt_name}")
    print(f"Lump sum amount:          {lump_sum_amount:.2f}")
    print(f"Modified income:          {modified_scenario.income:.2f}")
    print(f"Modified monthly expenses: {modified_scenario.monthly_expenses:.2f}")

    if lump_sum_debt_name and lump_sum_amount > 0:
        print("\nLump sum effect on debts:")
        for d in modified_scenario.debts:
            if d.name == lump_sum_debt_name:
                print(f"- {d.name}: new principal = {d.principal:.2f}")
                break

    print("\nRUNNING BASE SCENARIO...")
    base_result = run_scenario(base_scenario, verbose=verbose)

    print("\nRUNNING MODIFIED SCENARIO...")
    modified_result = run_scenario(modified_scenario, verbose=verbose)

    comparison = compare_results(base_scenario.name, base_result, modified_scenario.name, modified_result)
    print_comparison("WHAT-IF COMPARISON", comparison)

    print("\nPAYOFF ORDER (BASE):")
    print(base_result.payoff_order)

    print("\nPAYOFF ORDER (MODIFIED):")
    print(modified_result.payoff_order)

    if base_result.payoff_dates and modified_result.payoff_dates:
        print("\nPAYOFF DATE CHANGES:")
        all_names = sorted(set(base_result.payoff_dates) | set(modified_result.payoff_dates))
        for name in all_names:
            base_dt = base_result.payoff_dates.get(name)
            mod_dt = modified_result.payoff_dates.get(name)
            print(f"- {name}: base={base_dt}, modified={mod_dt}")


if __name__ == "__main__":
    # Sample debts matching our earlier conversation
    debts = [
        Debt(
            name="Loan 1 (2L / 15k EMI / 24 months)",
            principal=200000,
            emi=15000,
            start_date="2025-01-01",
            end_date="2027-01-01",
            notes="Shorter term loan",
        ),
        Debt(
            name="Loan 2 (5L / 20k EMI / 60 months)",
            principal=500000,
            emi=20000,
            start_date="2025-01-01",
            end_date="2030-01-01",
            notes="Longer term loan",
        ),
    ]

    # Example 1: reduce monthly expenses by 10k
    analyze_what_if(
        base_income=100000,
        base_monthly_expenses=30000,
        debts=debts,
        expense_change=10000,
        income_change=0,
        verbose=False,
    )
    
    # Example 2: add a 50k lump sum to Loan 1
    analyze_what_if(
        base_income=100000,
        base_monthly_expenses=30000,
        debts=debts,
        expense_change=0,
        income_change=0,
        lump_sum_debt_name="Loan 1 (2L / 15k EMI / 24 months)",
        lump_sum_amount=50000,
        verbose=False,
    )
