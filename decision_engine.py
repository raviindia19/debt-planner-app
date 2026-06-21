from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import List, Optional, Literal


PaymentAction = Literal["full_settlement", "partial_payment", "skipped", "manual_choice"]


@dataclass
class Debt:
    name: str
    balance: float

    due_now: float = 0.0
    minimum_due: float = 0.0
    required_to_settle: Optional[float] = None

    strictness: int = 5              # 0..10
    commitment_priority: int = 0     # family/social/emotional importance
    user_priority: int = 0           # manual override
    due_in_days: Optional[int] = None
    overdue: bool = False

    notes: str = ""

    def target_amount(self) -> float:
        if self.required_to_settle is not None:
            return max(0.0, self.required_to_settle)
        return max(0.0, self.minimum_due if self.minimum_due > 0 else self.due_now)

    def urgency_label(self) -> str:
        if self.overdue:
            return "overdue"
        if self.due_in_days is None:
            return "unknown"
        if self.due_in_days <= 0:
            return "due_today"
        if self.due_in_days <= 3:
            return "critical"
        if self.due_in_days <= 7:
            return "high"
        if self.due_in_days <= 15:
            return "medium"
        return "low"


@dataclass
class IncomeContext:
    monthly_income: float
    monthly_expenses: float = 0.0
    other_fixed_obligations: float = 0.0

    @property
    def available_cash(self) -> float:
        return max(0.0, self.monthly_income - self.monthly_expenses - self.other_fixed_obligations)


@dataclass
class DecisionLine:
    debt_name: str
    target: float
    paid: float
    unpaid: float
    action: PaymentAction
    reason: str


@dataclass
class DecisionResult:
    available_cash: float
    total_required: float
    total_paid: float
    remaining_cash: float
    shortfall: float
    needs_user_choice: bool
    recommendation: str
    primary_target: Optional[str]
    lines: List[DecisionLine] = field(default_factory=list)
    notes: str = ""


def debt_risk_score(debt: Debt) -> float:
    """
    Higher score = handle earlier.
    This is only for display / sorting, not for auto-choosing when cash is short.
    """
    urgency = 0.0
    if debt.overdue:
        urgency += 1000.0
    elif debt.due_in_days is not None:
        if debt.due_in_days <= 0:
            urgency += 800.0
        elif debt.due_in_days <= 3:
            urgency += 500.0
        elif debt.due_in_days <= 7:
            urgency += 300.0
        elif debt.due_in_days <= 15:
            urgency += 120.0
        else:
            urgency += 40.0
    else:
        urgency += 20.0

    commitment = debt.commitment_priority * 80.0
    strictness = debt.strictness * 50.0
    user = debt.user_priority * 70.0

    target = debt.target_amount()
    ratio = target / debt.balance if debt.balance > 0 and isfinite(target) else 1.0
    settle_efficiency = (1.0 - min(ratio, 1.0)) * 10.0

    return urgency + commitment + strictness + user + settle_efficiency * 10.0


def sort_debts_by_risk(debts: List[Debt]) -> List[Debt]:
    return sorted(debts, key=debt_risk_score, reverse=True)


def print_debt_table(debts: List[Debt]) -> None:
    print("\n=== DEBT LIST ===")
    for i, d in enumerate(debts, 1):
        print(
            f"{i}. {d.name} | balance={d.balance:.2f} | target={d.target_amount():.2f} | "
            f"risk={debt_risk_score(d):.2f} | overdue={d.overdue} | due_in_days={d.due_in_days} | "
            f"strictness={d.strictness} | commitment={d.commitment_priority} | user_priority={d.user_priority}"
        )


def _parse_order_input(raw: str, debts: List[Debt]) -> List[Debt]:
    """
    Accepts input like: 2,1,3 or "Family Commitment, Interest Only 7%"
    """
    raw = raw.strip()
    if not raw:
        return []

    by_index = {str(i): d for i, d in enumerate(debts, 1)}
    by_name = {d.name.lower(): d for d in debts}

    chosen: List[Debt] = []
    used_names = set()

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for part in parts:
        d = None

        if part in by_index:
            d = by_index[part]
        else:
            d = by_name.get(part.lower())

        if d is not None and d.name not in used_names:
            chosen.append(d)
            used_names.add(d.name)

    return chosen


def _prompt_yes_no(message: str) -> bool:
    while True:
        ans = input(message).strip().lower()
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please type y or n.")


def decide_payments(debts: List[Debt], income: IncomeContext, interactive: bool = True) -> DecisionResult:
    available = income.available_cash
    ranked = sort_debts_by_risk(debts)

    total_required = sum(d.target_amount() for d in debts if d.target_amount() > 0)
    total_paid = 0.0
    remaining_cash = available
    shortfall = max(0.0, total_required - available)
    needs_user_choice = available < total_required

    lines: List[DecisionLine] = []
    primary_target: Optional[str] = None

    print("\n=== DECISION ENGINE v4 ===")
    print(f"Available cash this month: {available:.2f}")
    print(f"Monthly income: {income.monthly_income:.2f}")
    print(f"Monthly expenses: {income.monthly_expenses:.2f}")
    print(f"Other fixed obligations: {income.other_fixed_obligations:.2f}")

    print_debt_table(ranked)

    if available >= total_required:
        print("\nCash is enough to cover all target amounts.")
        print("The engine will pay in risk order and close all possible debts.")

        for d in ranked:
            target = d.target_amount()
            if target <= 0:
                continue

            paid = min(target, remaining_cash)
            unpaid = max(0.0, target - paid)
            remaining_cash -= paid
            total_paid += paid

            if primary_target is None and paid > 0:
                primary_target = d.name

            lines.append(
                DecisionLine(
                    debt_name=d.name,
                    target=target,
                    paid=paid,
                    unpaid=unpaid,
                    action="full_settlement" if unpaid == 0 else "partial_payment",
                    reason="auto-allocated because cash was sufficient",
                )
            )

            print(
                f"{d.name}: target={target:.2f}, paid={paid:.2f}, unpaid={unpaid:.2f}, "
                f"remaining_cash={remaining_cash:.2f}"
            )

        recommendation = "All debts can be handled this month."
        notes = "Cash was sufficient, so no user choice was needed."

        return DecisionResult(
            available_cash=available,
            total_required=total_required,
            total_paid=total_paid,
            remaining_cash=remaining_cash,
            shortfall=shortfall,
            needs_user_choice=False,
            recommendation=recommendation,
            primary_target=primary_target,
            lines=lines,
            notes=notes,
        )

    # Short cash case: ask the user which loans to close first.
    print("\nCash is not enough to cover all targets.")
    print("You must choose which debt to close first.")
    print("Enter the order using numbers or names, separated by commas.")
    print("Example: 2,1,3")
    print("Or press Enter to use the suggested risk order.")

    if interactive:
        order_raw = input("Your priority order: ").strip()
    else:
        order_raw = ""

    chosen_order = _parse_order_input(order_raw, ranked)
    if not chosen_order:
        chosen_order = ranked[:]  # default suggestion
        print("\nUsing suggested risk order:")
        print(", ".join(d.name for d in chosen_order))
    else:
        print("\nUsing your chosen order:")
        print(", ".join(d.name for d in chosen_order))

    print("\nNow we will allocate only in your chosen order.")
    print("If money runs out before a debt can be fully settled, the engine will ask whether partial payment is acceptable.")

    for d in chosen_order:
        target = d.target_amount()
        if target <= 0:
            continue

        if remaining_cash <= 0:
            lines.append(
                DecisionLine(
                    debt_name=d.name,
                    target=target,
                    paid=0.0,
                    unpaid=target,
                    action="skipped",
                    reason="no cash left",
                )
            )
            print(f"{d.name}: skipped, no cash left.")
            continue

        if remaining_cash >= target:
            paid = target
            unpaid = 0.0
            remaining_cash -= paid
            total_paid += paid

            if primary_target is None:
                primary_target = d.name

            lines.append(
                DecisionLine(
                    debt_name=d.name,
                    target=target,
                    paid=paid,
                    unpaid=unpaid,
                    action="full_settlement",
                    reason="fully settled in user-selected order",
                )
            )

            print(
                f"{d.name}: target={target:.2f}, paid={paid:.2f}, unpaid={unpaid:.2f}, "
                f"remaining_cash={remaining_cash:.2f}, action=full_settlement"
            )
            continue

        # Not enough cash to settle this debt. Ask the user what to do.
        print(f"\n{d.name}:")
        print(f"  target amount = {target:.2f}")
        print(f"  cash available = {remaining_cash:.2f}")
        print(f"  ratio covered  = {remaining_cash / target:.0%}")

        if _prompt_yes_no("  Do you want to make a partial payment to this debt? (y/n): "):
            paid = remaining_cash
            unpaid = target - paid
            remaining_cash = 0.0
            total_paid += paid

            if primary_target is None and paid > 0:
                primary_target = d.name

            lines.append(
                DecisionLine(
                    debt_name=d.name,
                    target=target,
                    paid=paid,
                    unpaid=unpaid,
                    action="partial_payment",
                    reason="user explicitly approved partial payment",
                )
            )

            print(
                f"  paid={paid:.2f}, unpaid={unpaid:.2f}, remaining_cash={remaining_cash:.2f}, "
                f"action=partial_payment"
            )
        else:
            lines.append(
                DecisionLine(
                    debt_name=d.name,
                    target=target,
                    paid=0.0,
                    unpaid=target,
                    action="skipped",
                    reason="user rejected partial payment",
                )
            )
            print("  skipped by user choice.")
            
            if remaining_cash > 0:
                print(f"\n⚠️ You still have {remaining_cash:.2f} cash left.")

                print("What do you want to do next?")
                print("1. Try next debt")
                print("2. Reconsider this debt")
                print("3. Stop and keep cash")

                choice = input("Enter 1 / 2 / 3: ").strip()

                if choice == "2":
                    # 🔁 retry same debt
                    if _prompt_yes_no("Do you want to make partial payment now? (y/n): "):
                        paid = remaining_cash
                        unpaid = target - paid
                        remaining_cash = 0.0
                        total_paid += paid

                        lines.append(
                            DecisionLine(
                                debt_name=d.name,
                                target=target,
                                paid=paid,
                                unpaid=unpaid,
                                action="partial_payment",
                                reason="user reconsidered and accepted partial",
                            )
                        )

                        print(f"  paid={paid:.2f}, unpaid={unpaid:.2f}, remaining_cash=0.00")
                        continue

                elif choice == "3":
                    print("Stopping allocation. Keeping remaining cash.")
                    break

            # If the user skips this debt, we continue to the next one.
            # This is important because the user might prefer another lender first.
            continue

    recommendation = (
        "Cash is short this month, so the user must choose which debt to close first. "
        "Use the entered order as the repayment priority."
    )

    notes = (
        "This version does not auto-decide conflict cases. "
        "When money is short, it asks the user for the payment order and asks again before any partial payment."
    )
    
    if remaining_cash > 0:
        print(f"\n⚠️ FINAL DECISION: You still have {remaining_cash:.2f} unused cash.")

        print("Options:")
        print("1. Allocate remaining cash to any debt")
        print("2. Keep cash for safety")
        print("3. Exit")

        choice = input("Enter 1 / 2 / 3: ").strip()

        if choice == "1":
            print("\nWhich debt do you want to allocate to?")
            for i, d in enumerate(debts, 1):
                print(f"{i}. {d.name}")

            sel = input("Enter number: ").strip()

            try:
                idx = int(sel) - 1
                d = debts[idx]

                paid = remaining_cash
                remaining_cash = 0
                total_paid += paid

                lines.append(
                    DecisionLine(
                        debt_name=d.name,
                        target=d.target_amount(),
                        paid=paid,
                        unpaid = d.target_amount() - paid,
                        action="manual_choice",
                        reason="user allocated remaining cash manually",
                    )
                )

                print(f"Allocated {paid:.2f} to {d.name}")

            except:
                print("Invalid selection. Keeping cash.")


    return DecisionResult(
        available_cash=available,
        total_required=total_required,
        total_paid=total_paid,
        remaining_cash=remaining_cash,
        shortfall=shortfall,
        needs_user_choice=True,
        recommendation=recommendation,
        primary_target=primary_target,
        lines=lines,
        notes=notes,
    )


if __name__ == "__main__":
    debts = [
        Debt(
            name="Family Commitment",
            balance=50000,
            due_now=5000,
            minimum_due=5000,
            required_to_settle=5000,
            commitment_priority=10,
            strictness=9,
            due_in_days=0,
            notes="family pressure",
        ),
        Debt(
            name="Daily EMI Loan",
            balance=165000,
            due_now=36525,
            minimum_due=36525,
            required_to_settle=36525,
            strictness=6,
            due_in_days=2,
            notes="daily emi",
        ),
        Debt(
            name="Interest Only 7%",
            balance=100000,
            due_now=7000,
            minimum_due=7000,
            required_to_settle=7000,
            strictness=8,
            due_in_days=5,
            notes="interest-only lender",
        ),
    ]

    income = IncomeContext(
        monthly_income=40000,
        monthly_expenses=0,
        other_fixed_obligations=0,
    )

    result = decide_payments(debts, income, interactive=True)

    print("\n=== RESULT SUMMARY ===")
    print(f"Available cash: {result.available_cash:.2f}")
    print(f"Total required: {result.total_required:.2f}")
    print(f"Total paid: {result.total_paid:.2f}")
    print(f"Remaining cash: {result.remaining_cash:.2f}")
    print(f"Shortfall: {result.shortfall:.2f}")
    print(f"Needs user choice: {result.needs_user_choice}")
    print(f"Recommendation: {result.recommendation}")
    print(f"Primary target: {result.primary_target}")

    print("\n=== DECISION LINES ===")
    for line in result.lines:
        print(
            f"- {line.debt_name}: target={line.target:.2f}, paid={line.paid:.2f}, "
            f"unpaid={line.unpaid:.2f}, action={line.action}, reason={line.reason}"
        )
