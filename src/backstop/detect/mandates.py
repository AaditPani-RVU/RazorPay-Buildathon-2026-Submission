"""Recurring revenue at risk from mandate state, not from a rate anomaly.

The anomaly detector answers "did something just break?". That is the wrong
question for most subscription revenue, because the usual failure is not an
event at all -- it is a *stock*. Mandates lapse quietly, one at a time, and a
merchant wakes up with several hundred dead authorisations and no incident to
point at. Nothing spiked. There is no baseline to deviate from. A z-test over
a rate would report all clear.

So this is a scan, not a detector: walk the book, price the dead mandates, and
rank them. No statistics, because there is no anomaly to establish -- an
expired mandate is not unusual, it is simply worthless until re-authorised.

The two paths are complementary and the pipeline runs both. A sudden collapse
in mandate presentations is an incident and belongs to the detector; the slow
accumulation of lapsed authorisations belongs here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from backstop.domain.entities import (
    BILLING_PERIODS_PER_YEAR,
    MandateStatus,
    Subscription,
)
from backstop.domain.money import Money

#: Why each non-active state costs money, and whether the customer can fix it.
#: Revoked is deliberately separated from expired: both stop the money, but one
#: is an administrative lapse and the other is a decision. Chasing them the
#: same way is how a merchant turns a cancellation into a complaint.
STATUS_REASON: dict[MandateStatus, str] = {
    MandateStatus.EXPIRED: "authorisation lapsed; re-registration would restore it",
    MandateStatus.PAUSED: "customer paused collection; may resume on its own",
    MandateStatus.REVOKED: "customer revoked authorisation; this is a decision, not a lapse",
    MandateStatus.NOT_REGISTERED: "no mandate was ever registered against this plan",
}

#: Rough odds the money comes back if the customer is asked. Revoked is low on
#: purpose -- somebody who cancelled a mandate mostly meant to.
REREGISTRATION_ODDS: dict[MandateStatus, float] = {
    MandateStatus.EXPIRED: 0.42,
    MandateStatus.PAUSED: 0.55,
    MandateStatus.NOT_REGISTERED: 0.30,
    MandateStatus.REVOKED: 0.08,
}


@dataclass(frozen=True)
class MandateRisk:
    """One subscription whose mandate cannot currently collect."""

    subscription_id: str
    customer_id: str
    status: MandateStatus
    charge_amount: Money
    consecutive_failures: int

    @property
    def reason(self) -> str:
        return STATUS_REASON.get(self.status, "mandate cannot collect")

    @property
    def annual_value(self) -> Money:
        """What the lapse costs over a year if never fixed.

        Priced by `Subscription.annual_value`, so the scan, the policy
        engine's value thresholds and the ledger cannot disagree about what a
        mandate is worth.
        """
        return self.charge_amount * BILLING_PERIODS_PER_YEAR

    @property
    def expected_recovery(self) -> Money:
        return self.annual_value * REREGISTRATION_ODDS.get(self.status, 0.2)

    def describe(self) -> str:
        return (
            f"{self.subscription_id:<24} {self.status.value:<16} "
            f"{self.charge_amount.format():>12}/mo  "
            f"{self.annual_value.format():>14}/yr  {self.reason}"
        )


@dataclass
class MandateRiskReport:
    at_risk: list[MandateRisk] = field(default_factory=list)
    active: int = 0

    @property
    def total_annual_value(self) -> Money:
        total = Money.zero()
        for r in self.at_risk:
            total += r.annual_value
        return total

    @property
    def expected_recovery(self) -> Money:
        total = Money.zero()
        for r in self.at_risk:
            total += r.expected_recovery
        return total

    def by_status(self) -> dict[MandateStatus, tuple[int, Money]]:
        counts: dict[MandateStatus, int] = defaultdict(int)
        value: dict[MandateStatus, Money] = defaultdict(Money.zero)
        for r in self.at_risk:
            counts[r.status] += 1
            value[r.status] = value[r.status] + r.annual_value
        return {s: (counts[s], value[s]) for s in counts}

    def render(self) -> str:
        lines = [
            f"active mandates    {self.active:,}",
            f"lapsed mandates    {len(self.at_risk):,}",
            f"recurring at risk  {self.total_annual_value} per year",
            f"expected recovery  {self.expected_recovery} if all are chased",
            "",
        ]
        for status, (n, value) in sorted(
            self.by_status().items(), key=lambda kv: -kv[1][1].paise
        ):
            odds = REREGISTRATION_ODDS.get(status, 0.2)
            lines.append(
                f"  {status.value:<16} {n:>5} mandates  {value.format():>16}/yr  "
                f"~{odds:.0%} recoverable  {STATUS_REASON[status]}"
            )
        return "\n".join(lines)


def scan(subscriptions: list[Subscription]) -> MandateRiskReport:
    """Price every mandate that cannot currently collect, worst first."""
    report = MandateRiskReport()
    for sub in subscriptions:
        if sub.cancelled_at is not None:
            continue  # a cancelled subscription is not revenue at risk
        if sub.mandate_status is MandateStatus.ACTIVE:
            report.active += 1
            continue
        report.at_risk.append(
            MandateRisk(
                subscription_id=sub.id,
                customer_id=sub.customer_id,
                status=sub.mandate_status,
                charge_amount=sub.amount,
                consecutive_failures=sub.consecutive_failures,
            )
        )
    report.at_risk.sort(key=lambda r: -r.annual_value.paise)
    return report
