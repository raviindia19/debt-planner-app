from dataclasses import dataclass
from typing import List, Optional, Literal

PaymentFrequency = Literal["daily", "weekly", "monthly", "yearly"]
DebtKind = Literal["interest_only", "emi", "custom"]

FREQ_TO_MONTHS = {
    "daily": 30.4375,
    "weekly": 4.345,
    "monthly": 1.0,
    "yearly": 1 / 12.0,
}

FREQ_TO_YEAR = {
    "daily": 365.0,
    "weekly": 52.0,
    "monthly": 12.0,
    "yearly": 1.0,
}


@dataclass
class Debt:
    name: str
    principal: float

    kind: DebtKind = "custom"

    due_now: float = 0.0
    minimum_due: float = 0.0

    payment_amount: float = 0.0
    payment_frequency: PaymentFrequency = "monthly"

    interest_rate: Optional[float] = None
    interest_rate_period: PaymentFrequency = "monthly"

    required_to_settle: Optional[float] = None
    strictness: int = 5              # 0 to 10
    user_priority: int = 0           # manual override
    commitment_priority: int = 0     # family/social/emotional priority
    partial_ok: bool = False         # partial payment useful or not

    notes: str = ""

    def balance(self) -> float:
        return self.principal


@dataclass
class Income:
    monthly_income: float
    expenses: float = 0.0

    @property
    def available(self) -> float:
        return max(0.0, self.monthly_income - self.expenses)


def monthly_payment(debt: Debt) -> float:
    return debt.payment_amount * FREQ_TO_MONTHS[debt.payment_frequency]


def annualized_rate(debt: Debt) -> float:
    if debt.interest_rate is None:
        return 0.0

    r = debt.interest_rate / 100.0

    if debt.interest_rate_period == "monthly":
        return (1 + r) ** 12 - 1
    if debt.interest_rate_period == "weekly":
        return (1 + r) ** 52 - 1
    if debt.interest_rate_period == "daily":
        return (1 + r) ** 365 - 1

    return r


def cash_drain_ratio(debt: Debt) -> float:
    bal = debt.balance()
    if bal <= 0:
        return 0.0
    return monthly_payment(debt) / bal


def priority_score(debt: Debt) -> float:
    """
    Higher score = should be handled earlier.
    """
    return (
        debt.commitment_priority * 100.0 +
        debt.user_priority * 80.0 +
        debt.strictness * 50.0 +
        cash_drain_ratio(debt) * 1000.0 +
        annualized_rate(debt) * 100.0
    )


def rank_debts(debts: List[Debt]) -> List[Debt]:
    return sorted(debts, key=priority_score, reverse=True)


def settle_target(debt: Debt) -> float:
    """
    What amount should be considered for settlement this cycle.
    """
    if debt.required_to_settle is not None:
        return debt.required_to_settle

    base = max(debt.minimum_due, debt.due_now)
    return base


def allocate_money(debts: List[Debt], income: Income):
    remaining_cash = income.available
    ranked = rank_debts(debts)

    allocations = []
    paid_debts = set()

    print("\n=== AVAILABLE CASH ===")
    print(f"available_cash = {remaining_cash:.2f}")

    print("\n=== PRIORITY ORDER ===")
    for d in ranked:
        print(f"{d.name} | score = {round(priority_score(d), 2)}")

    print("\n=== ALLOCATION LOGIC ===")

    # First pass: mandatory / commitment / due debts
    for d in ranked:
        target = settle_target(d)

        if target <= 0:
            continue

        if d.name in paid_debts:
            continue

        if remaining_cash >= target:
            paid = target
            unpaid = 0.0
            remaining_cash -= paid
            reason = "FULL SETTLEMENT"
        else:
            # No automatic partial payment for strict lenders unless partial_ok is true
            if d.strictness >= 8:
                paid = 0.0
                unpaid = target
                reason = "STRICT LENDER - NEED USER DECISION"
            elif d.partial_ok:
                paid = remaining_cash
                unpaid = target - paid
                remaining_cash = 0.0
                reason = "PARTIAL PAYMENT ALLOWED"
            else:
                paid = 0.0
                unpaid = target
                reason = "SKIPPED - PARTIAL NOT SAFE"

        allocations.append({
            "debt": d.name,
            "required": target,
            "paid": paid,
            "unpaid": unpaid,
            "remaining_cash_after": remaining_cash,
            "reason": reason
        })

        print(
            f"{d.name}: required={target:.2f}, paid={paid:.2f}, "
            f"unpaid={unpaid:.2f}, remaining_cash={remaining_cash:.2f}, reason={reason}"
        )

        # Mark as paid only if fully settled
        if unpaid == 0.0 and paid > 0:
            paid_debts.add(d.name)

    # Second pass: use any leftover cash only for debts that can be fully settled now
    if remaining_cash > 0:
        settle_candidates = []
        for d in ranked:
            if d.name in paid_debts:
                continue

            target = settle_target(d)

            if target <= remaining_cash:
                settle_candidates.append(d)

        if settle_candidates:
            best = sorted(settle_candidates, key=priority_score, reverse=True)[0]
            target = settle_target(best)

            remaining_cash -= target
            allocations.append({
                "debt": best.name,
                "required": target,
                "paid": target,
                "unpaid": 0.0,
                "remaining_cash_after": remaining_cash,
                "reason": "FULL SETTLEMENT FROM REMAINING CASH"
            })

            print(
                f"{best.name}: required={target:.2f}, paid={target:.2f}, "
                f"unpaid=0.00, remaining_cash={remaining_cash:.2f}, reason=FULL SETTLEMENT FROM REMAINING CASH"
            )

            paid_debts.add(best.name)

    unresolved = [a for a in allocations if a["unpaid"] > 0]

    print("\n=== SUMMARY ===")
    print(f"remaining_cash = {remaining_cash:.2f}")

    if unresolved:
        print("\n⚠️ NEED USER CHOICE")
        for u in unresolved:
            print(f"- {u['debt']} unpaid = {u['unpaid']:.2f}")

        print("\nThe engine should ask the user which lender is more dangerous or less flexible.")
        print("Do not auto-spend leftover cash on a strict lender without confirmation.")

    return allocations


def explain_debts(debts: List[Debt]):
    print("\n=== DEBT CALCULATIONS ===")
    for d in debts:
        print("------")
        print(f"Loan: {d.name}")
        print(f"Principal: {d.principal:.2f}")
        print(f"Due now: {d.due_now:.2f}")
        print(f"Minimum due: {d.minimum_due:.2f}")
        print(f"Required to settle: {d.required_to_settle if d.required_to_settle is not None else 'None'}")
        print(f"Monthly payment equivalent: {monthly_payment(d):.2f}")
        print(f"Cash drain ratio: {cash_drain_ratio(d):.4f}")
        print(f"Annualized rate: {round(annualized_rate(d) * 100, 2)}%")
        print(f"Strictness: {d.strictness}")
        print(f"Commitment priority: {d.commitment_priority}")
        print(f"User priority: {d.user_priority}")
        print(f"Partial OK: {d.partial_ok}")
        print(f"Priority score: {round(priority_score(d), 2)}")
        print(f"Notes: {d.notes}")


if __name__ == "__main__":
    debts = [
        Debt(
            name="Family Commitment",
            principal=50000,
            due_now=5000,
            minimum_due=5000,
            required_to_settle=5000,
            commitment_priority=10,
            strictness=9,
            partial_ok=False,
            notes="Must be paid first due to family pressure"
        ),
        Debt(
            name="Daily EMI Loan",
            principal=100000,
            due_now=36525,
            minimum_due=36525,
            required_to_settle=36525,   # use the real required amount
            payment_amount=1200,
            payment_frequency="daily",
            strictness=6,
            partial_ok=False,
            notes="Daily EMI loan"
        ),
        Debt(
            name="Interest Only 7%",
            principal=100000,
            due_now=7000,
            minimum_due=7000,
            required_to_settle=7000,
            payment_amount=7000,
            payment_frequency="monthly",
            interest_rate=7,
            interest_rate_period="monthly",
            strictness=8,
            partial_ok=False,
            notes="Interest-only monthly lender"
        ),
    ]

    income = Income(
        monthly_income=40000,
        expenses=0
    )

    explain_debts(debts)
    allocate_money(debts, income)
