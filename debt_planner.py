from __future__ import annotations

"""
debt_planner.py
================

Wires loan_intake.py into the existing payoff_simulator.py without
touching debt_engine.py, decision_engine.py, or streamlit_app.py.

PayoffSimulator.rank_debts() already does avalanche-within-priority-tier
(priority first, interest rate as tie-break) - so the only job left here
is building the bridge: turn each EnrichedLoan into a payoff_simulator.Debt
with the right principal (outstanding + overdue, per Ravi's spec),
the right priority (overdue gets boosted above everything else), and a
remaining-term window so the simulator's cost heuristic is computed on
what's actually left to pay, not the original loan-from-day-one numbers.
"""

from dataclasses import dataclass
from datetime import date
from typing import List, Dict

from loan_intake import RawLoanInput, EnrichedLoan, enrich_portfolio, validate_raw_loan
from payoff_simulator import Debt as SimDebt, PayoffSimulator, SimulationResult, add_months


@dataclass
class PlanResult:
    enriched_loans: List[EnrichedLoan]
    monthly_income: float
    monthly_fixed_expenses: float
    monthly_scheduled_outflow: float
    monthly_surplus: float
    simulation: SimulationResult
    overall_debt_free_date: date
    actions: List[str]


def _remaining_term_periods(loan: EnrichedLoan) -> int:
    if loan.term_count is None:
        return 0
    return max(1, loan.term_count - loan.payments_made)


def _virtual_window(loan: EnrichedLoan, current_date: date) -> tuple[date, date]:
    """
    Gives the simulator a start/end window sized to what's actually left,
    so its cost heuristic (implied_cost / principal / term) is computed
    against the REMAINING principal and REMAINING term, not the original
    loan-from-day-one numbers.
    """
    if loan.loan_type == "interest_only" or loan.term_count is None:
        # open-ended: give it a long horizon, it'll close early once the
        # avalanche extra-payment allocation pays off the principal
        return current_date, add_months(current_date, 360)

    remaining = _remaining_term_periods(loan)
    if loan.payment_frequency == "monthly":
        end = add_months(current_date, remaining)
    elif loan.payment_frequency == "weekly":
        end = current_date + __import__("datetime").timedelta(weeks=remaining)
    else:
        end = current_date + __import__("datetime").timedelta(days=remaining)
    return current_date, end


def to_sim_debt(loan: EnrichedLoan, current_date: date) -> SimDebt:
    start, end = _virtual_window(loan, current_date)
    return SimDebt(
        name=loan.name,
        principal=loan.total_debt_now,
        emi=loan.payment_amount,
        start_date=start,
        end_date=end,
        # loan_intake already derives this correctly with frequency-aware compounding
        # (daily/weekly/monthly). Always pass it through rather than letting the
        # simulator's own calendar-month heuristic recompute it - that heuristic
        # assumes monthly granularity and silently breaks for daily/weekly loans.
        annual_interest_rate=loan.annual_interest_rate,
        priority=loan.sim_priority,
        notes=loan.notes,
    )


def monthly_equivalent(loan: EnrichedLoan) -> float:
    if loan.payment_frequency == "daily":
        return loan.payment_amount * 30.4375
    if loan.payment_frequency == "weekly":
        return loan.payment_amount * 4.345
    return loan.payment_amount


def build_plan(
    raw_loans: List[RawLoanInput],
    monthly_income: float,
    monthly_fixed_expenses: float,
    current_date: date,
    verbose: bool = True,
) -> PlanResult:
    # 1. validate first - fail loud with plain-English questions, don't guess silently
    all_questions: List[str] = []
    for raw in raw_loans:
        all_questions.extend(validate_raw_loan(raw))
    blocking = [q for q in all_questions if "interest rate given" not in q]
    if blocking:
        raise ValueError("Need more info before this can be calculated:\n- " + "\n- ".join(blocking))

    # 2. enrich
    enriched = enrich_portfolio(raw_loans, current_date)

    if verbose:
        from loan_intake import print_enriched_table
        print_enriched_table(enriched)
        if all_questions:
            print("\n=== SOFT CONFIRMATIONS (assumed, please verify) ===")
            for q in all_questions:
                print(f"- {q}")

    # 3. monthly surplus snapshot (this month, scheduled payments only)
    scheduled_outflow = sum(monthly_equivalent(L) for L in enriched if L.status != "completing_soon" or L.missed_payments == 0)
    surplus = max(0.0, monthly_income - monthly_fixed_expenses - scheduled_outflow)

    # 4. bridge into the existing avalanche-aware simulator
    sim_debts = [to_sim_debt(L, current_date) for L in enriched]
    simulator = PayoffSimulator(
        income=monthly_income,
        monthly_expenses=monthly_fixed_expenses,
        debts=sim_debts,
        current_date=current_date,
    )
    sim_result = simulator.simulate(verbose=verbose)

    overall_date = max(sim_result.payoff_dates.values()) if sim_result.payoff_dates else current_date

    # 5. plain-English actions
    actions: List[str] = []
    overdue_loans = [L for L in enriched if L.overdue]
    if overdue_loans:
        names = ", ".join(f"{L.name} ({L.overdue_amount:,.0f} overdue)" for L in overdue_loans)
        actions.append(f"Clear overdue arrears first - this always outranks everything else: {names}.")

    ranked_names = sim_result.payoff_order
    if len(ranked_names) > 1:
        actions.append(
            f"Avalanche order for any extra cash this month: {' -> '.join(ranked_names)} "
            f"(same priority tier broken by highest effective interest rate first)."
        )

    if surplus <= 0:
        actions.append("No surplus left after this month's scheduled payments - nothing extra to throw at any loan yet.")
    else:
        actions.append(f"Monthly surplus available for extra payments: {surplus:,.2f}.")

    interest_only_open = [L for L in enriched if L.loan_type == "interest_only"]
    if interest_only_open:
        names = ", ".join(L.name for L in interest_only_open)
        actions.append(
            f"{names}: interest-only, so the principal will only shrink via extra/avalanche payments "
            f"once higher-priority and overdue loans are clear."
        )

    return PlanResult(
        enriched_loans=enriched,
        monthly_income=monthly_income,
        monthly_fixed_expenses=monthly_fixed_expenses,
        monthly_scheduled_outflow=scheduled_outflow,
        monthly_surplus=surplus,
        simulation=sim_result,
        overall_debt_free_date=overall_date,
        actions=actions,
    )


def print_plan_summary(plan: PlanResult) -> None:
    print("\n=== MONTHLY SNAPSHOT ===")
    print(f"Income: {plan.monthly_income:,.2f}")
    print(f"Fixed expenses: {plan.monthly_fixed_expenses:,.2f}")
    print(f"Scheduled loan outflow: {plan.monthly_scheduled_outflow:,.2f}")
    print(f"Surplus: {plan.monthly_surplus:,.2f}")

    print("\n=== PER-LOAN CLEAR DATES ===")
    for name in plan.simulation.payoff_order:
        d = plan.simulation.payoff_dates.get(name)
        print(f"- {name}: {d if d else 'not closed within simulation horizon'}")

    print(f"\nOverall debt-free date: {plan.overall_debt_free_date}")
    print(f"Total months to debt freedom: {plan.simulation.total_months}")
    print(f"Total cost (interest/fees) across all loans: {plan.simulation.total_interest_or_cost:,.2f}")

    print("\n=== SUGGESTED ACTIONS ===")
    for a in plan.actions:
        print(f"- {a}")


if __name__ == "__main__":
    # Ravi's exact example, dates filled in as he gave them
    current_date = date(2026, 6, 20)

    raw_loans = [
        RawLoanInput(
            name="Family Commitment",
            principal_amount=50000,
            loan_type="emi",
            payment_amount=5000,
            payment_frequency="monthly",
            start_date=date(2026, 5, 1),
            payment_day_rule="1st of every month",
            payments_made=1,
            term_count=10,
            priority=10,
            notes="Family pressure / immediate obligation",
        ),
        RawLoanInput(
            name="Khatu",
            principal_amount=100000,
            loan_type="emi",
            payment_amount=1200,
            payment_frequency="daily",
            start_date=date(2026, 5, 1),
            payment_day_rule="daily",
            payments_made=51,
            term_count=100,
            priority=5,
            notes="Daily finance loan",
        ),
        RawLoanInput(
            name="Radhe",
            principal_amount=100000,
            loan_type="interest_only",
            payment_amount=7000,
            payment_frequency="monthly",
            start_date=date(2026, 2, 1),
            payment_day_rule="5th of every month",
            payments_made=5,
            priority=8,
            notes="Interest-only lender",
        ),
    ]

    plan = build_plan(
        raw_loans=raw_loans,
        monthly_income=40000,
        monthly_fixed_expenses=0,
        current_date=current_date,
        verbose=True,
    )

    print_plan_summary(plan)
