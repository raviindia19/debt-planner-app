from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from calendar import monthrange
from typing import List, Optional, Dict, Tuple


def parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def add_months(d: date, months: int) -> date:
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    day = min(d.day, monthrange(year, month)[1])
    return date(year, month, day)


def months_between(start: date, end: date) -> int:
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day > start.day:
        months += 1
    return max(0, months)


@dataclass
class Debt:
    name: str
    principal: float
    emi: float
    start_date: date | str
    end_date: date | str

    annual_interest_rate: Optional[float] = None  # e.g. 12.0 means 12% p.a.
    payment_day: Optional[int] = None  # unused in v1, kept for future expansion
    priority: int = 0
    interest_type: str = "emi"          # "emi" | "interest_only"
    first_payment_date: Optional[date] = None  # skip scheduled payments before this date
    notes: str = ""

    def __post_init__(self) -> None:
        self.start_date = parse_date(self.start_date)
        self.end_date = parse_date(self.end_date)

    @property
    def term_months(self) -> int:
        return months_between(self.start_date, self.end_date)

    @property
    def heuristic_monthly_cost_rate(self) -> float:
        """
        If explicit interest rate is missing, infer a simple monthly cost rate
        from principal, EMI, and term.
        """
        term = self.term_months
        if term <= 0 or self.principal <= 0:
            return 0.0

        total_paid = self.emi * term
        implied_cost = max(0.0, total_paid - self.principal)
        return implied_cost / self.principal / term

    @property
    def effective_monthly_rate(self) -> float:
        if self.annual_interest_rate is not None:
            return (1 + self.annual_interest_rate / 100.0) ** (1 / 12.0) - 1
        return self.heuristic_monthly_cost_rate

    @property
    def effective_annual_rate(self) -> float:
        if self.annual_interest_rate is not None:
            return self.annual_interest_rate / 100.0
        return (1 + self.heuristic_monthly_cost_rate) ** 12 - 1

    @property
    def total_scheduled_paid(self) -> float:
        return self.emi * self.term_months

    @property
    def implied_total_cost(self) -> float:
        return max(0.0, self.total_scheduled_paid - self.principal)


@dataclass
class DebtState:
    debt: Debt
    balance: float
    closed: bool = False
    closed_on: Optional[date] = None
    total_paid: float = 0.0
    total_interest_or_cost: float = 0.0
    total_principal_paid: float = 0.0


@dataclass
class PaymentLine:
    month_index: int
    month_date: date
    debt_name: str
    opening_balance: float
    scheduled_payment: float
    interest_or_cost: float
    principal_paid: float
    extra_payment: float
    closing_balance: float
    note: str

    @property
    def cost_component(self) -> float:
        return self.interest_or_cost

    @property
    def status(self) -> str:
        return self.note


@dataclass
class SimulationResult:
    total_months: int
    total_paid: float
    total_interest_or_cost: float
    total_principal_paid: float
    payoff_order: List[str]
    payoff_dates: Dict[str, date]
    timeline: List[PaymentLine] = field(default_factory=list)
    simulation_date: Optional[date] = None

    @property
    def total_cost(self) -> float:
        return self.total_interest_or_cost


class PayoffSimulator:
    def __init__(
        self,
        income: float,
        debts: List[Debt],
        monthly_expenses: float = 0.0,
        current_date: Optional[date | str] = None,
    ) -> None:
        self.income = income
        self.monthly_expenses = monthly_expenses
        self.debts = debts
        self.current_date = parse_date(current_date) if current_date is not None else None

    @property
    def disposable_cash(self) -> float:
        return max(0.0, self.income - self.monthly_expenses)

    def rank_debts(self) -> List[Debt]:
        def score(d: Debt) -> Tuple[float, float, float, float]:
            monthly_drain_ratio = d.emi / d.principal if d.principal > 0 else 0.0
            return (
                d.priority * 100.0,
                d.effective_annual_rate * 100.0,
                monthly_drain_ratio * 100.0,
                d.principal,
            )

        return sorted(self.debts, key=score, reverse=True)

    def _scheduled_interest_and_principal(self, debt: Debt, balance: float, remaining_months: int) -> Tuple[float, float]:
        """
        Returns (interest_or_cost, principal_component) for the scheduled EMI.
        - interest_only loans: interest = balance × periodic_rate, principal = 0 always.
        - If explicit interest exists, use amortization-like interest calculation.
        - Otherwise use a straight-line inferred cost spread across remaining months.
        """
        remaining_months = max(1, remaining_months)

        # Fix 5: interest_only loans never reduce principal via scheduled payments.
        # periodic_rate = emi / original_principal (set at debt creation time).
        if debt.interest_type == "interest_only":
            periodic_rate = debt.emi / debt.principal if debt.principal > 0 else 0.0
            interest = balance * periodic_rate
            return interest, 0.0

        if debt.annual_interest_rate is not None:
            monthly_rate = debt.effective_monthly_rate
            interest = balance * monthly_rate
            principal_component = max(0.0, debt.emi - interest)
            return interest, principal_component

        inferred_monthly_cost = debt.implied_total_cost / debt.term_months if debt.term_months > 0 else 0.0
        principal_component = max(0.0, debt.emi - inferred_monthly_cost)
        principal_component = min(principal_component, balance)
        interest_or_cost = max(0.0, debt.emi - principal_component)
        return interest_or_cost, principal_component

    def simulate(self, verbose: bool = True) -> SimulationResult:
        states: Dict[str, DebtState] = {
            d.name: DebtState(debt=d, balance=d.principal)
            for d in self.debts
        }

        ranked = self.rank_debts()
        payoff_order = [d.name for d in ranked]
        payoff_dates: Dict[str, date] = {}
        timeline: List[PaymentLine] = []

        month_index = 0
        current_date = self.current_date or min(d.start_date for d in self.debts)
        total_paid = 0.0
        total_interest_or_cost = 0.0
        total_principal_paid = 0.0

        if verbose:
            print("\n=== PAYOFF SIMULATION START ===")
            print(f"Income: {self.income:.2f}")
            print(f"Monthly expenses: {self.monthly_expenses:.2f}")
            print(f"Disposable cash: {self.disposable_cash:.2f}\n")

            print("=== INPUT DEBTS ===")
            for d in ranked:
                print("------")
                print(f"Loan: {d.name}")
                print(f"Principal: {d.principal:.2f}")
                print(f"EMI: {d.emi:.2f}")
                print(f"Start date: {d.start_date}")
                print(f"End date: {d.end_date}")
                print(f"Term months: {d.term_months}")
                print(f"Explicit annual interest: {None if d.annual_interest_rate is None else f'{d.annual_interest_rate:.2f}%'}")
                print(f"Heuristic annual rate: {d.effective_annual_rate * 100:.2f}%")
                print(f"Implied total cost: {d.implied_total_cost:.2f}")
                print(f"Priority: {d.priority}")

            print("\n=== RANK ORDER FOR EXTRA PAYMENTS ===")
            for i, d in enumerate(ranked, 1):
                print(f"{i}. {d.name}")

        while True:
            active = [s for s in states.values() if not s.closed and s.balance > 0.005]
            if not active:
                break

            month_index += 1
            month_date = add_months(current_date, month_index - 1)

            scheduled_cash_used = 0.0

            if verbose:
                print("\n--- MONTH", month_index, month_date, "---")
                print("Scheduled payments:")

            for d in ranked:
                st = states[d.name]
                if st.closed or st.balance <= 0.005:
                    continue

                # Fix 3: don't charge scheduled payment until the loan's first payment date.
                # (e.g. Family due July 1 should not appear in a June 21 simulation step.)
                if d.first_payment_date is not None and month_date < d.first_payment_date:
                    continue

                if month_date > d.end_date:
                    scheduled_payment = 0.0
                else:
                    if d.interest_type == "interest_only":
                        # For interest-only, scheduled = interest on current balance only
                        periodic_rate = d.emi / d.principal if d.principal > 0 else 0.0
                        scheduled_payment = st.balance * periodic_rate
                    elif d.annual_interest_rate is not None:
                        scheduled_payment = min(
                            d.emi,
                            st.balance + st.balance * d.effective_monthly_rate,
                        )
                    else:
                        scheduled_payment = min(d.emi, st.balance)

                opening_balance = st.balance
                interest_or_cost, principal_component = self._scheduled_interest_and_principal(
                    d, st.balance, max(1, d.term_months - (month_index - 1))
                )

                # Fix 5: interest_only → zero principal from scheduled payments.
                if d.interest_type == "interest_only":
                    principal_from_schedule = 0.0
                elif d.annual_interest_rate is not None:
                    principal_from_schedule = max(0.0, scheduled_payment - interest_or_cost)
                else:
                    principal_from_schedule = min(principal_component, opening_balance)

                actual_payment = interest_or_cost + principal_from_schedule
                if actual_payment > opening_balance + interest_or_cost:
                    actual_payment = opening_balance + interest_or_cost
                    principal_from_schedule = max(0.0, actual_payment - interest_or_cost)

                new_balance = max(0.0, opening_balance - principal_from_schedule)

                st.balance = new_balance
                st.total_paid += actual_payment
                st.total_interest_or_cost += interest_or_cost
                st.total_principal_paid += principal_from_schedule

                total_paid += actual_payment
                total_interest_or_cost += interest_or_cost
                total_principal_paid += principal_from_schedule
                scheduled_cash_used += actual_payment

                note = "scheduled EMI"
                if new_balance <= 0.005:
                    st.closed = True
                    st.closed_on = month_date
                    payoff_dates[d.name] = month_date
                    note = "closed by scheduled payment"

                line = PaymentLine(
                    month_index=month_index,
                    month_date=month_date,
                    debt_name=d.name,
                    opening_balance=opening_balance,
                    scheduled_payment=actual_payment,
                    interest_or_cost=interest_or_cost,
                    principal_paid=principal_from_schedule,
                    extra_payment=0.0,
                    closing_balance=new_balance,
                    note=note,
                )
                timeline.append(line)

                if verbose:
                    print(
                        f"{d.name}: opening={opening_balance:.2f}, "
                        f"scheduled={actual_payment:.2f}, cost={interest_or_cost:.2f}, "
                        f"principal={principal_from_schedule:.2f}, closing={new_balance:.2f}"
                    )

            extra_cash = max(0.0, self.disposable_cash - scheduled_cash_used)

            if verbose:
                print(f"Scheduled cash used: {scheduled_cash_used:.2f}")
                print(f"Extra cash available: {extra_cash:.2f}")

            if extra_cash > 0:
                if verbose:
                    print("Extra allocation:")

                for d in ranked:
                    st = states[d.name]
                    if st.closed or st.balance <= 0.005:
                        continue

                    if extra_cash <= 0:
                        break

                    extra = min(extra_cash, st.balance)
                    if extra <= 0:
                        continue

                    opening_balance = st.balance
                    st.balance = max(0.0, st.balance - extra)
                    st.total_paid += extra
                    st.total_principal_paid += extra

                    total_paid += extra
                    total_principal_paid += extra
                    extra_cash -= extra

                    note = "extra payment"
                    if st.balance <= 0.005:
                        st.closed = True
                        st.closed_on = month_date
                        payoff_dates[d.name] = month_date
                        note = "closed by extra payment"

                    line = PaymentLine(
                        month_index=month_index,
                        month_date=month_date,
                        debt_name=d.name,
                        opening_balance=opening_balance,
                        scheduled_payment=0.0,
                        interest_or_cost=0.0,
                        principal_paid=extra,
                        extra_payment=extra,
                        closing_balance=st.balance,
                        note=note,
                    )
                    timeline.append(line)

                    if verbose:
                        print(f"{d.name}: extra={extra:.2f}, closing_balance={st.balance:.2f} [{note}]")

                if verbose and extra_cash > 0:
                    print(f"Unused extra cash after all debts closed: {extra_cash:.2f}")

            if month_index > 600:
                raise RuntimeError("Simulation exceeded 600 months. Check inputs.")

        total_months = month_index

        if verbose:
            print("\n=== SIMULATION SUMMARY ===")
            print(f"Total months to debt freedom: {total_months}")
            print(f"Total paid: {total_paid:.2f}")
            print(f"Total interest/cost (approx): {total_interest_or_cost:.2f}")
            print(f"Total principal paid: {total_principal_paid:.2f}")
            print("Payoff order:", payoff_order)
            print("Payoff dates:")
            for name, dt in payoff_dates.items():
                print(f"- {name}: {dt}")

        return SimulationResult(
            total_months=total_months,
            total_paid=total_paid,
            total_interest_or_cost=total_interest_or_cost,
            total_principal_paid=total_principal_paid,
            payoff_order=payoff_order,
            payoff_dates=payoff_dates,
            timeline=timeline,
            simulation_date=current_date,
        )


if __name__ == "__main__":
    debts = [
        Debt(
            name="Loan 1 (2L / 15k EMI / 24 months)",
            principal=200000,
            emi=15000,
            start_date="2025-01-01",
            end_date="2027-01-01",
            priority=0,
            notes="Shorter term loan",
        ),
        Debt(
            name="Loan 2 (5L / 20k EMI / 60 months)",
            principal=500000,
            emi=20000,
            start_date="2025-01-01",
            end_date="2030-01-01",
            priority=0,
            notes="Longer term loan",
        ),
    ]

    sim = PayoffSimulator(
        income=100000,
        monthly_expenses=0,
        debts=debts,
    )

    result = sim.simulate(verbose=True)

    print("\n=== RESULT OBJECT (quick view) ===")
    print("Months:", result.total_months)
    print("Paid:", round(result.total_paid, 2))
    print("Interest/Cost:", round(result.total_interest_or_cost, 2))
    print("Payoff order:", result.payoff_order)
