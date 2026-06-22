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
from datetime import date, timedelta
from typing import List, Dict, Optional

from loan_intake import RawLoanInput, EnrichedLoan, enrich_portfolio, validate_raw_loan
from payoff_simulator import Debt as SimDebt, PayoffSimulator, SimulationResult, PaymentLine, add_months


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
    # The simulator steps monthly, so EMI must be the calendar-month equivalent.
    # For interest_only, the emi stores the per-period amount used to derive periodic_rate.
    emi_monthly = monthly_equivalent(loan)
    return SimDebt(
        name=loan.name,
        principal=loan.total_debt_now,
        emi=emi_monthly,
        start_date=start,
        end_date=end,
        annual_interest_rate=loan.annual_interest_rate,  # always use derived/explicit rate
        priority=loan.sim_priority,
        interest_type="interest_only" if loan.loan_type == "interest_only" else "emi",
        # Fix 3: don't process scheduled payments until the loan is actually due.
        # next_due_date is guaranteed to be >= today after Fix 1 in loan_intake.
        first_payment_date=loan.next_due_date,
        notes=loan.notes,
    )


def monthly_equivalent(loan: EnrichedLoan) -> float:
    if loan.payment_frequency == "daily":
        return loan.payment_amount * 30.4375
    if loan.payment_frequency == "weekly":
        return loan.payment_amount * 4.345
    return loan.payment_amount


def _expand_timeline(
    timeline: List[PaymentLine],
    freq_map: Dict[str, str],
    per_period_map: Dict[str, float],
    remaining_periods_map: Optional[Dict[str, int]] = None,
) -> List[PaymentLine]:
    """
    The simulator steps monthly.  For daily/weekly loans, each monthly row is
    expanded into one row per actual payment period.

    freq_map:              loan_name -> "daily" | "weekly" | "monthly"
    per_period_map:        loan_name -> actual per-period payment (e.g. 1200 for Khatu)
    remaining_periods_map: loan_name -> total remaining periods at simulation start
                           (used to stop expansion at the correct date instead of always 30 days)
    """
    # Track how many sub-periods have been emitted per loan across all monthly rows.
    periods_used: Dict[str, int] = {}
    expanded: List[PaymentLine] = []

    for line in timeline:
        freq = freq_map.get(line.debt_name, "monthly")
        if freq == "monthly":
            expanded.append(line)
            continue

        step = timedelta(days=1) if freq == "daily" else timedelta(weeks=1)
        default_max_per_month = 30 if freq == "daily" else 4

        # How many sub-periods should this monthly row expand into?
        name = line.debt_name
        used_so_far = periods_used.get(name, 0)
        if remaining_periods_map and name in remaining_periods_map:
            left = remaining_periods_map[name] - used_so_far
            n_periods = max(1, min(left, default_max_per_month))
        else:
            n_periods = default_max_per_month

        # Fix 4: use the ACTUAL per-period payment amount (e.g. 1200), not monthly/30.
        # Interest is spread evenly from the monthly aggregate; principal = payment - interest.
        per_period_pmt = per_period_map.get(name)  # e.g. 1200 for Khatu
        total_interest = line.interest_or_cost
        total_extra = line.extra_payment
        total_principal = line.principal_paid + total_extra

        sub_interest = round(total_interest / n_periods, 2)

        balance = line.opening_balance
        actual_rows = 0
        for i in range(n_periods):
            is_last = (i == n_periods - 1)
            sub_date = line.month_date + step * i

            # Interest component — last row absorbs rounding
            c = (max(0.0, round(total_interest - sub_interest * (n_periods - 1), 2))
                 if is_last else sub_interest)

            # Scheduled payment = actual per-period amount if known, else pro-rate
            if per_period_pmt is not None:
                sched = per_period_pmt if not is_last else max(0.0, min(per_period_pmt, balance + c))
            else:
                sched = round(line.scheduled_payment / n_periods, 2)

            # Principal from scheduled payment
            p_sched = max(0.0, sched - c)

            # Extra payment — spread pro-rata, last absorbs rounding
            e = (max(0.0, round(total_extra - round(total_extra / n_periods, 2) * (n_periods - 1), 2))
                 if is_last else round(total_extra / n_periods, 2))
            p_extra = e

            p_total = min(p_sched + p_extra, balance)  # can't pay more than balance
            closing = max(0.0, round(balance - p_total, 2))

            note = line.note if is_last else ("daily EMI" if freq == "daily" else "weekly EMI")

            expanded.append(PaymentLine(
                month_index=line.month_index,
                month_date=sub_date,
                debt_name=name,
                opening_balance=round(balance, 2),
                scheduled_payment=round(sched, 2),
                interest_or_cost=round(c, 2),
                principal_paid=round(p_sched, 2),
                extra_payment=round(e, 2),
                closing_balance=closing,
                note=note,
            ))
            balance = closing
            actual_rows += 1

            if closing <= 0.005:
                break

        periods_used[name] = used_so_far + actual_rows

    # Re-sort by date so monthly and daily rows interleave correctly
    expanded.sort(key=lambda r: (r.month_date, r.debt_name))
    return expanded


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

    # 4b. Expand monthly-stepped timeline into per-period rows for daily/weekly loans,
    #     using remaining_periods_map so Khatu stops at day 49 not day 60.
    freq_map = {L.name: L.payment_frequency for L in enriched}
    per_period_map = {L.name: L.payment_amount for L in enriched}
    remaining_periods_map = {
        L.name: max(1, L.term_count - L.payments_made)
        for L in enriched
        if L.payment_frequency in ("daily", "weekly") and L.term_count is not None
    }
    sim_result.timeline = _expand_timeline(
        sim_result.timeline, freq_map, per_period_map, remaining_periods_map
    )
    # Update payoff_dates for daily/weekly loans to the actual date the balance hits 0
    # (the monthly simulator's date is only an approximation for sub-monthly loans).
    for line in sim_result.timeline:
        if line.closing_balance <= 0.005 and freq_map.get(line.debt_name, "monthly") != "monthly":
            sim_result.payoff_dates[line.debt_name] = line.month_date

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
