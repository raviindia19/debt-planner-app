from __future__ import annotations

"""
debt_planner.py
================

Wires loan_intake.py into the existing payoff_simulator.py without
touching debt_engine.py, decision_engine.py, or streamlit_app.py.

PayoffSimulator.rank_debts() already does avalanche-within-priority-tier
(priority first, interest rate as tie-break) - so the only job left here
is building the bridge: turn each EnrichedLoan into a payoff_simulator.Debt
with the right principal (outstanding only - NOT total_debt_now, which
double-counts the overdue amount), the right priority (overdue gets boosted
above everything else), and a remaining-term window so the simulator's cost
heuristic is computed on what's actually left to pay, not the original
loan-from-day-one numbers.

BUG FIXES (vs previous version)
--------------------------------
Fix A – principal double-count (Bug 1):
    to_sim_debt() now passes loan.outstanding_principal, not loan.total_debt_now.
    outstanding_principal already covers all remaining payments (including the
    overdue one), so adding overdue_amount on top was double-counting.

Fix B – payment-day ignored in timeline (Bugs 2 & 3):
    Added _shift_payment_dates(): after _expand_timeline(), monthly loan rows
    are moved from the simulator's step-date (e.g. 22nd) to their actual
    payment_day (e.g. 1st for Family, 5th for Radhe).

Fix C – Khatu reducing-balance instead of flat interest (Bug 4):
    _expand_timeline() now accepts flat_cost_per_period_map.  For daily/weekly
    EMI loans whose cost is implied from principal+term (like Khatu's ₹200/day),
    we use that fixed amount instead of prorating the monthly reducing-balance
    interest total.  Result: every Khatu day shows exactly ₹200 interest
    regardless of remaining balance.
"""

from calendar import monthrange as _cal_monthrange
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
        # FIX A: use outstanding_principal, NOT total_debt_now.
        # outstanding_principal already includes all remaining payments (overdue ones too),
        # so adding overdue_amount on top was double-counting and inflating the opening balance.
        principal=loan.outstanding_principal,
        emi=emi_monthly,
        start_date=start,
        end_date=end,
        annual_interest_rate=loan.annual_interest_rate,  # always use derived/explicit rate
        priority=loan.sim_priority,
        interest_type="interest_only" if loan.loan_type == "interest_only" else "emi",
        # Don't process scheduled payments until the loan is actually due.
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
    flat_cost_per_period_map: Optional[Dict[str, float]] = None,
) -> List[PaymentLine]:
    """
    The simulator steps monthly.  For daily/weekly loans, each monthly row is
    expanded into one row per actual payment period.

    freq_map:               loan_name -> "daily" | "weekly" | "monthly"
    per_period_map:         loan_name -> actual per-period payment (e.g. 1200 for Khatu)
    remaining_periods_map:  loan_name -> total remaining periods at simulation start
                            (used to stop expansion at the correct date instead of always 30 days)
    flat_cost_per_period_map: loan_name -> fixed interest/cost per period (e.g. 200 for Khatu).
                            When present, this overrides the prorated monthly interest so that
                            flat-rate loans (like daily finance / Khatu) show a constant interest
                            component regardless of remaining balance.  (FIX C)
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

        # FIX C: for flat-rate loans, use the fixed per-period cost instead of
        # prorating the monthly reducing-balance interest total.
        flat_cost = (flat_cost_per_period_map or {}).get(name)

        total_interest = line.interest_or_cost
        total_extra = line.extra_payment
        total_principal = line.principal_paid + total_extra

        if flat_cost is not None:
            # flat-rate: every sub-period gets exactly flat_cost interest
            sub_interest = flat_cost
        else:
            sub_interest = round(total_interest / n_periods, 2)

        per_period_pmt = per_period_map.get(name)  # e.g. 1200 for Khatu
        balance = line.opening_balance
        actual_rows = 0

        for i in range(n_periods):
            is_last = (i == n_periods - 1)
            sub_date = line.month_date + step * i

            # Interest component
            if flat_cost is not None:
                # Fixed amount every period; last row absorbs any floating-point residue
                c = flat_cost
            else:
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


def _shift_payment_dates(
    timeline: List[PaymentLine],
    enriched: List[EnrichedLoan],
) -> List[PaymentLine]:
    """
    FIX B – move monthly loan rows to their actual payment day.

    The simulator fires all loans on the same monthly step date (e.g. 22nd).
    This post-processing step replaces each monthly-frequency loan's month_date
    with its correct payment_day within that same calendar month.

    Examples:
      Family (payment_day=1):  row dated 2026-07-22 → 2026-07-01
      Radhe  (payment_day=5):  row dated 2026-07-22 → 2026-07-05
    """
    payment_day_map: Dict[str, int] = {
        L.name: L.payment_day
        for L in enriched
        if L.payment_frequency == "monthly" and L.payment_day is not None
    }
    if not payment_day_map:
        return timeline

    shifted: List[PaymentLine] = []
    for line in timeline:
        pday = payment_day_map.get(line.debt_name)
        if pday is None:
            shifted.append(line)
            continue
        d = line.month_date
        max_day = _cal_monthrange(d.year, d.month)[1]
        new_date = d.replace(day=min(pday, max_day))
        shifted.append(PaymentLine(
            month_index=line.month_index,
            month_date=new_date,
            debt_name=line.debt_name,
            opening_balance=line.opening_balance,
            scheduled_payment=line.scheduled_payment,
            interest_or_cost=line.interest_or_cost,
            principal_paid=line.principal_paid,
            extra_payment=line.extra_payment,
            closing_balance=line.closing_balance,
            note=line.note,
        ))

    shifted.sort(key=lambda r: (r.month_date, r.debt_name))
    return shifted


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

    # 4b. Build per-loan maps for timeline expansion and date shifting.
    freq_map = {L.name: L.payment_frequency for L in enriched}
    per_period_map = {L.name: L.payment_amount for L in enriched}

    remaining_periods_map = {
        L.name: max(1, L.term_count - L.payments_made)
        for L in enriched
        if L.payment_frequency in ("daily", "weekly") and L.term_count is not None
    }

    # FIX C: pre-compute flat cost-per-period for daily/weekly EMI loans where
    # the interest is implied from principal and term (e.g. Khatu = ₹200/day flat).
    # This prevents _expand_timeline from inheriting the simulator's reducing-balance
    # interest split, which would give a different (wrong) interest figure each month.
    flat_cost_per_period_map: Dict[str, float] = {}
    for L in enriched:
        if (
            L.payment_frequency in ("daily", "weekly")
            and L.loan_type == "emi"
            and L.term_count is not None
            and L.principal_amount > 0
        ):
            total_payable = L.payment_amount * L.term_count
            implied_cost = max(0.0, total_payable - L.principal_amount)
            if implied_cost > 0:
                flat_cost_per_period_map[L.name] = implied_cost / L.term_count

    # Expand monthly-stepped timeline into per-period rows for daily/weekly loans.
    sim_result.timeline = _expand_timeline(
        sim_result.timeline,
        freq_map,
        per_period_map,
        remaining_periods_map,
        flat_cost_per_period_map=flat_cost_per_period_map,
    )

    # Update payoff_dates for daily/weekly loans to the actual date the balance hits 0.
    for line in sim_result.timeline:
        if line.closing_balance <= 0.005 and freq_map.get(line.debt_name, "monthly") != "monthly":
            sim_result.payoff_dates[line.debt_name] = line.month_date

    # FIX B: shift monthly loan rows from the simulator step-date to the loan's
    # actual payment_day (e.g. Family → 1st, Radhe → 5th).
    sim_result.timeline = _shift_payment_dates(sim_result.timeline, enriched)

    # Re-sync payoff_dates after the date shift so they reflect the real payment day.
    for line in sim_result.timeline:
        if line.closing_balance <= 0.005:
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
    current_date = date(2026, 6, 22)

    raw_loans = [
        RawLoanInput(
            name="Family Commitment",
            principal_amount=50000,
            loan_type="emi",
            payment_amount=5000,
            payment_frequency="monthly",
            start_date=date(2026, 5, 1),
            payment_day_rule="1st of every month",
            payment_day=1,
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
            payment_day=5,
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
