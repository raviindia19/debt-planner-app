from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from calendar import monthrange
from typing import List, Optional, Dict, Tuple, Literal


LoanStatus = Literal["not_started", "active", "overdue", "closed"]


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
    """
    Approximate month count between two dates.

    Example:
    - 2025-01-01 to 2025-02-01 => 1
    - 2025-01-01 to 2025-01-15 => 0
    - 2025-01-01 to 2026-06-20 => 17 or 18 depending on day cut
    """
    if end < start:
        return 0

    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day >= start.day:
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
    priority: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        self.start_date = parse_date(self.start_date)
        self.end_date = parse_date(self.end_date)

    @property
    def term_months(self) -> int:
        return max(1, months_between(self.start_date, self.end_date))

    @property
    def scheduled_monthly_principal(self) -> float:
        """
        Simple baseline amortization assumption when explicit interest rate is missing.
        This assumes principal is spread evenly across the original term.
        """
        return self.principal / self.term_months

    @property
    def total_scheduled_paid(self) -> float:
        return self.emi * self.term_months

    @property
    def implied_total_cost(self) -> float:
        return max(0.0, self.total_scheduled_paid - self.principal)

    @property
    def heuristic_monthly_cost_rate(self) -> float:
        if self.principal <= 0 or self.term_months <= 0:
            return 0.0
        return self.implied_total_cost / self.principal / self.term_months

    @property
    def effective_monthly_rate(self) -> float:
        if self.annual_interest_rate is not None:
            return (1 + (self.annual_interest_rate / 100.0)) ** (1 / 12.0) - 1
        return self.heuristic_monthly_cost_rate

    @property
    def effective_annual_rate(self) -> float:
        if self.annual_interest_rate is not None:
            return self.annual_interest_rate / 100.0
        return (1 + self.heuristic_monthly_cost_rate) ** 12 - 1


@dataclass
class DebtState:
    debt: Debt
    opening_balance_today: float
    scheduled_months_already_elapsed: int
    closed: bool = False
    closed_on: Optional[date] = None
    total_paid_before_today: float = 0.0
    total_cost_before_today: float = 0.0
    total_principal_before_today: float = 0.0

    balance_today: float = 0.0
    status_today: LoanStatus = "active"

    total_paid_after_today: float = 0.0
    total_cost_after_today: float = 0.0
    total_principal_after_today: float = 0.0


@dataclass
class PaymentLine:
    month_index: int
    month_date: date
    debt_name: str
    status: LoanStatus
    opening_balance: float
    scheduled_payment: float
    cost_component: float
    principal_paid: float
    extra_payment: float
    closing_balance: float
    note: str


@dataclass
class SimulationResult:
    simulation_date: date
    total_months: int
    total_paid: float
    total_cost: float
    total_principal_paid: float
    payoff_order: List[str]
    payoff_dates: Dict[str, date]
    timeline: List[PaymentLine] = field(default_factory=list)
    starting_balances: Dict[str, float] = field(default_factory=dict)
    starting_statuses: Dict[str, LoanStatus] = field(default_factory=dict)


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
        self.current_date = parse_date(current_date) if current_date is not None else date.today()

    @property
    def disposable_cash(self) -> float:
        return max(0.0, self.income - self.monthly_expenses)

    def debt_status(self, debt: Debt, balance: float, as_of: date) -> LoanStatus:
        if balance <= 0.005:
            return "closed"
        if as_of < debt.start_date:
            return "not_started"
        if as_of > debt.end_date:
            return "overdue"
        return "active"

    def _effective_term_months_from_today(self, debt: Debt) -> int:
        """
        Months between today and end date, if any.
        """
        if self.current_date >= debt.end_date:
            return 0
        return months_between(self.current_date, debt.end_date)

    def _months_elapsed_from_start(self, debt: Debt) -> int:
        if self.current_date <= debt.start_date:
            return 0
        return min(debt.term_months, months_between(debt.start_date, self.current_date))

    def _fast_forward_balance_to_today(self, debt: Debt, verbose: bool = False) -> Tuple[float, float, float, int]:
        """
        Fast-forward a loan from its start date to the simulator's current date.

        Returns:
            balance_today,
            total_paid_before_today,
            total_cost_before_today,
            scheduled_months_elapsed
        """
        if self.current_date < debt.start_date:
            return debt.principal, 0.0, 0.0, 0

        scheduled_months = self._months_elapsed_from_start(debt)
        balance = debt.principal
        total_paid = 0.0
        total_cost = 0.0
        total_principal = 0.0

        for _ in range(scheduled_months):
            if balance <= 0.005:
                break

            if debt.annual_interest_rate is not None:
                monthly_rate = debt.effective_monthly_rate
                cost = balance * monthly_rate
                principal_paid = min(max(0.0, debt.emi - cost), balance)
                actual_payment = min(debt.emi, cost + principal_paid)
            else:
                # Simple linear amortization:
                # principal is spread across the original term.
                principal_target = debt.scheduled_monthly_principal
                principal_paid = min(principal_target, balance)

                # whatever remains of EMI becomes cost/interest
                cost = max(0.0, debt.emi - principal_paid)
                actual_payment = min(debt.emi, balance + cost)

            balance = max(0.0, balance - principal_paid)
            total_paid += actual_payment
            total_cost += cost
            total_principal += principal_paid

        if verbose:
            print(f"Fast-forward {debt.name}:")
            print(f"  months elapsed since start = {scheduled_months}")
            print(f"  balance today = {balance:.2f}")
            print(f"  paid before today = {total_paid:.2f}")
            print(f"  cost before today = {total_cost:.2f}")
            print(f"  principal before today = {total_principal:.2f}")

        return balance, total_paid, total_cost, scheduled_months

    def rank_debts(self, states: Dict[str, DebtState]) -> List[Debt]:
        def score(d: Debt) -> Tuple[float, float, float, float]:
            st = states[d.name]
            monthly_drain_ratio = d.emi / max(1.0, st.balance_today)
            overdue_bonus = 1.0 if st.status_today == "overdue" else 0.0
            active_bonus = 0.5 if st.status_today == "active" else 0.0
            return (
                overdue_bonus * 1000.0 + active_bonus * 100.0 + d.priority * 100.0,
                d.effective_annual_rate * 100.0,
                monthly_drain_ratio * 100.0,
                st.balance_today,
            )

        return sorted(self.debts, key=score, reverse=True)

    def _scheduled_cost_and_principal_after_today(self, debt: Debt, balance: float) -> Tuple[float, float]:
        """
        For projections after today, estimate the scheduled payment split.
        """
        if balance <= 0.005:
            return 0.0, 0.0

        if debt.annual_interest_rate is not None:
            monthly_rate = debt.effective_monthly_rate
            cost = balance * monthly_rate
            principal_component = min(max(0.0, debt.emi - cost), balance)
            return cost, principal_component

        principal_component = min(debt.scheduled_monthly_principal, balance)
        cost = max(0.0, debt.emi - principal_component)
        return cost, principal_component

    def _build_initial_states(self, verbose: bool = False) -> Dict[str, DebtState]:
        states: Dict[str, DebtState] = {}

        if verbose:
            print("\n=== FAST-FORWARD TO CURRENT DATE ===")
            print(f"Current date: {self.current_date}")

        for d in self.debts:
            balance_today, paid_before, cost_before, months_elapsed = self._fast_forward_balance_to_today(d, verbose=verbose)
            status_today = self.debt_status(d, balance_today, self.current_date)

            states[d.name] = DebtState(
                debt=d,
                opening_balance_today=d.principal,
                scheduled_months_already_elapsed=months_elapsed,
                closed=balance_today <= 0.005,
                balance_today=balance_today,
                status_today=status_today,
                total_paid_before_today=paid_before,
                total_cost_before_today=cost_before,
                total_principal_before_today=max(0.0, d.principal - balance_today),
            )

        return states

    def simulate(self, verbose: bool = True) -> SimulationResult:
        states = self._build_initial_states(verbose=verbose)
        ranked = self.rank_debts(states)
        payoff_order = [d.name for d in ranked]
        payoff_dates: Dict[str, date] = {}
        timeline: List[PaymentLine] = []

        total_paid = sum(st.total_paid_before_today for st in states.values())
        total_cost = sum(st.total_cost_before_today for st in states.values())
        total_principal_paid = sum(st.total_principal_before_today for st in states.values())

        if verbose:
            print("\n=== INPUT DEBTS (AS OF CURRENT DATE) ===")
            for d in ranked:
                st = states[d.name]
                print("------")
                print(f"Loan: {d.name}")
                print(f"Principal: {d.principal:.2f}")
                print(f"EMI: {d.emi:.2f}")
                print(f"Start date: {d.start_date}")
                print(f"End date: {d.end_date}")
                print(f"Status today: {st.status_today}")
                print(f"Balance today: {st.balance_today:.2f}")
                print(f"Months already elapsed: {st.scheduled_months_already_elapsed}")
                print(f"Annual rate: {None if d.annual_interest_rate is None else f'{d.annual_interest_rate:.2f}%'}")
                print(f"Heuristic annual rate: {d.effective_annual_rate * 100:.2f}%")
                print(f"Implied total scheduled paid: {d.total_scheduled_paid:.2f}")
                print(f"Implied total cost: {d.implied_total_cost:.2f}")
                print(f"Priority: {d.priority}")

            print("\n=== RANK ORDER FOR EXTRA PAYMENTS ===")
            for i, d in enumerate(ranked, 1):
                print(f"{i}. {d.name}")

        month_index = 0
        while True:
            active = [s for s in states.values() if not s.closed and s.balance_today > 0.005]
            if not active:
                break

            month_index += 1
            month_date = add_months(self.current_date, month_index - 1)

            scheduled_cash_used = 0.0

            if verbose:
                print(f"\n--- MONTH {month_index} {month_date} ---")
                print("Scheduled payments:")

            # Scheduled payments for debts that are already active today or overdue today.
            for d in ranked:
                st = states[d.name]
                if st.closed or st.balance_today <= 0.005:
                    continue

                # If the debt has not started by the simulation current date, do not schedule payment yet.
                if self.current_date < d.start_date:
                    if verbose:
                        print(f"{d.name}: not started, balance={st.balance_today:.2f}")
                    timeline.append(
                        PaymentLine(
                            month_index=month_index,
                            month_date=month_date,
                            debt_name=d.name,
                            status="not_started",
                            opening_balance=st.balance_today,
                            scheduled_payment=0.0,
                            cost_component=0.0,
                            principal_paid=0.0,
                            extra_payment=0.0,
                            closing_balance=st.balance_today,
                            note="not started yet",
                        )
                    )
                    continue

                opening_balance = st.balance_today
                cost_component, principal_component = self._scheduled_cost_and_principal_after_today(d, opening_balance)

                # If the original term is already over, we still keep the debt alive as overdue.
                # The scheduled payment becomes the EMI amount or whatever balance remains, whichever is lower.
                scheduled_payment = min(d.emi, opening_balance + cost_component)
                principal_paid = min(principal_component, opening_balance)
                actual_cost = min(cost_component, max(0.0, scheduled_payment - principal_paid))
                actual_payment = min(opening_balance + actual_cost, scheduled_payment)

                new_balance = max(0.0, opening_balance - principal_paid)

                st.balance_today = new_balance
                st.total_paid_after_today += actual_payment
                st.total_cost_after_today += actual_cost
                st.total_principal_after_today += principal_paid

                total_paid += actual_payment
                total_cost += actual_cost
                total_principal_paid += principal_paid
                scheduled_cash_used += actual_payment

                note = st.status_today
                if new_balance <= 0.005:
                    st.closed = True
                    st.closed_on = month_date
                    payoff_dates[d.name] = month_date
                    note = "closed by scheduled payment"

                timeline.append(
                    PaymentLine(
                        month_index=month_index,
                        month_date=month_date,
                        debt_name=d.name,
                        status=st.status_today,
                        opening_balance=opening_balance,
                        scheduled_payment=actual_payment,
                        cost_component=actual_cost,
                        principal_paid=principal_paid,
                        extra_payment=0.0,
                        closing_balance=new_balance,
                        note=note,
                    )
                )

                if verbose:
                    print(
                        f"{d.name}: status={st.status_today}, opening={opening_balance:.2f}, "
                        f"scheduled={actual_payment:.2f}, cost={actual_cost:.2f}, "
                        f"principal={principal_paid:.2f}, closing={new_balance:.2f}"
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
                    if st.closed or st.balance_today <= 0.005:
                        continue

                    if extra_cash <= 0:
                        break

                    extra = min(extra_cash, st.balance_today)
                    if extra <= 0:
                        continue

                    opening_balance = st.balance_today
                    st.balance_today = max(0.0, st.balance_today - extra)
                    st.total_paid_after_today += extra
                    st.total_principal_after_today += extra

                    total_paid += extra
                    total_principal_paid += extra
                    extra_cash -= extra

                    note = "extra payment"
                    if st.balance_today <= 0.005:
                        st.closed = True
                        st.closed_on = month_date
                        payoff_dates[d.name] = month_date
                        note = "closed by extra payment"

                    timeline.append(
                        PaymentLine(
                            month_index=month_index,
                            month_date=month_date,
                            debt_name=d.name,
                            status=st.status_today,
                            opening_balance=opening_balance,
                            scheduled_payment=0.0,
                            cost_component=0.0,
                            principal_paid=extra,
                            extra_payment=extra,
                            closing_balance=st.balance_today,
                            note=note,
                        )
                    )

                    if verbose:
                        print(f"{d.name}: extra={extra:.2f}, closing_balance={st.balance_today:.2f} [{note}]")

                if verbose and extra_cash > 0:
                    print(f"Unused extra cash after all debts closed: {extra_cash:.2f}")

            if month_index > 600:
                raise RuntimeError("Simulation exceeded 600 months. Check inputs.")

        if verbose:
            print("\n=== SIMULATION SUMMARY ===")
            print(f"Simulation date: {self.current_date}")
            print(f"Total months simulated forward: {month_index}")
            print(f"Total paid (including before today): {total_paid:.2f}")
            print(f"Total cost (approx): {total_cost:.2f}")
            print(f"Total principal paid (approx): {total_principal_paid:.2f}")
            print("Payoff order:", payoff_order)
            print("Payoff dates:")
            for name, dt in payoff_dates.items():
                print(f"- {name}: {dt}")

        return SimulationResult(
            simulation_date=self.current_date,
            total_months=month_index,
            total_paid=total_paid,
            total_cost=total_cost,
            total_principal_paid=total_principal_paid,
            payoff_order=payoff_order,
            payoff_dates=payoff_dates,
            timeline=timeline,
            starting_balances={name: st.balance_today if st.scheduled_months_already_elapsed == 0 else st.opening_balance_today for name, st in states.items()},
            starting_statuses={name: st.status_today for name, st in states.items()},
        )


if __name__ == "__main__":
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
        Debt(
            name="Overdue Loan Example",
            principal=100000,
            emi=7000,
            start_date="2024-01-01",
            end_date="2025-01-01",
            notes="Started in the past and is overdue today",
        ),
    ]

    sim = PayoffSimulator(
        income=100000,
        monthly_expenses=30000,
        debts=debts,
        current_date="2026-06-20",
    )

    result = sim.simulate(verbose=True)

    print("\n=== RESULT OBJECT (quick view) ===")
    print("Simulation date:", result.simulation_date)
    print("Months:", result.total_months)
    print("Paid:", round(result.total_paid, 2))
    print("Cost:", round(result.total_cost, 2))
    print("Principal paid:", round(result.total_principal_paid, 2))
    print("Payoff order:", result.payoff_order)
