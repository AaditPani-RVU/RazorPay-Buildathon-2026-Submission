"""Receivables at risk from age, not from a rate anomaly.

The third revenue surface, and the third shape of failure. A payment fails as
an *event* -- one authorisation, one decline code, one moment a detector can
find. A mandate fails as a *state* -- an authorisation that stopped being able
to collect and stays that way. A receivable fails as neither: it fails by
*ageing*. Nothing breaks and no state flips. An invoice is issued, falls due,
and then simply gets older, and every day it does the money is slightly less
likely to arrive.

So this is a scan like the mandate book, but the thing being priced is
different. A lapsed mandate is worth its annual stream and either comes back or
does not. An overdue invoice is worth its outstanding balance and *decays* --
which means the interesting quantity is not whether it is overdue but how long
it has been, and the right output is the aging report a finance team already
reads every week.

Two suppressions are built in rather than left to policy, because they change
what is *chaseable*, not merely what is permitted:

*   A **disputed** invoice is not a collections problem. The buyer contests
    that the money is owed, and no amount of dunning resolves a disagreement
    about the underlying invoice. It stays in the at-risk total -- the money is
    genuinely at risk -- but it is never counted as chaseable, and it routes to
    a person.
*   A **live promise to pay** is a commitment with a date on it. Chasing
    inside it is how a merchant turns a cooperative buyer into an annoyed one.

The engine enforces both anyway. Counting them here as well is deliberate
redundancy: a number that says "this is what we could collect" should not
include money the rules will refuse to let anyone chase.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from backstop.domain.entities import Invoice
from backstop.domain.money import Money


class AgingBucket(StrEnum):
    """Standard AR aging brackets. Not invented here on purpose -- a finance
    team already reads its ledger in these buckets, and a recovery agent that
    reports in its own private brackets cannot be reconciled against the
    ledger it claims to be improving."""

    CURRENT = "current"
    DAYS_1_30 = "1-30"
    DAYS_31_60 = "31-60"
    DAYS_61_90 = "61-90"
    DAYS_90_PLUS = "90+"


def bucket_for(days_overdue: int) -> AgingBucket:
    if days_overdue <= 0:
        return AgingBucket.CURRENT
    if days_overdue <= 30:
        return AgingBucket.DAYS_1_30
    if days_overdue <= 60:
        return AgingBucket.DAYS_31_60
    if days_overdue <= 90:
        return AgingBucket.DAYS_61_90
    return AgingBucket.DAYS_90_PLUS


#: What Backstop believes an invoice in each bracket is still worth collecting,
#: as a share of its outstanding balance.
#:
#: The decay is the whole point of aging a ledger. Receivables do not fail at a
#: moment, they fade, and a policy that treats a 20-day-old invoice and a
#: 200-day-old one as the same collections problem is wrong about both -- too
#: aggressive on the first, far too late on the second.
#:
#: These are the agent's published estimate, the numbers it reasons and reports
#: with. `simulate.recoverability` holds separate and deliberately different
#: figures for what actually happens, so the eval cannot grade the agent
#: against its own assumption.
COLLECTION_ODDS: dict[AgingBucket, float] = {
    AgingBucket.CURRENT: 0.97,
    AgingBucket.DAYS_1_30: 0.88,
    AgingBucket.DAYS_31_60: 0.71,
    AgingBucket.DAYS_61_90: 0.52,
    AgingBucket.DAYS_90_PLUS: 0.24,
}

#: Why each bracket is treated differently by the collections playbook.
BUCKET_REASON: dict[AgingBucket, str] = {
    AgingBucket.CURRENT: "not yet due; nothing is wrong",
    AgingBucket.DAYS_1_30: "recently late; usually an AP cycle, not a problem",
    AgingBucket.DAYS_31_60: "late enough to be deliberate or forgotten",
    AgingBucket.DAYS_61_90: "the buyer probably cannot clear it in one payment",
    AgingBucket.DAYS_90_PLUS: "collections or write-off; a person decides, not automation",
}


@dataclass(frozen=True)
class ReceivableRisk:
    """One overdue invoice, priced and aged."""

    invoice_id: str
    buyer_id: str
    outstanding: Money
    days_overdue: int
    disputed: bool
    promise_live: bool

    @property
    def bucket(self) -> AgingBucket:
        return bucket_for(self.days_overdue)

    @property
    def is_chaseable(self) -> bool:
        """Whether collections may act on this at all.

        A dispute is a disagreement about whether the money is owed and dunning
        cannot settle one. A live promise is a commitment the buyer has already
        made. Both are at risk; neither is chaseable.
        """
        return not self.disputed and not self.promise_live

    @property
    def expected_recovery(self) -> Money:
        """Nothing is expected from what may not be chased -- an expectation
        that quietly includes money no action is permitted to pursue is a
        forecast of work that will never happen."""
        if not self.is_chaseable:
            return Money.zero()
        return self.outstanding * COLLECTION_ODDS.get(self.bucket, 0.2)

    @property
    def reason(self) -> str:
        if self.disputed:
            return "disputed; a human owns this, not automation"
        if self.promise_live:
            return "promise to pay is live; the buyer has committed to a date"
        return BUCKET_REASON.get(self.bucket, "overdue")

    def describe(self) -> str:
        return (
            f"{self.invoice_id:<24} {self.bucket.value:<8} "
            f"{self.days_overdue:>4}d  {self.outstanding.format():>14}  {self.reason}"
        )


@dataclass
class ReceivablesReport:
    at_risk: list[ReceivableRisk] = field(default_factory=list)
    settled: int = 0
    current: int = 0
    """Issued, not yet due. Healthy, and counted so the total is a ledger
    rather than only its bad half."""
    current_value: Money = field(default_factory=Money.zero)

    @property
    def total_outstanding(self) -> Money:
        total = Money.zero()
        for r in self.at_risk:
            total += r.outstanding
        return total

    @property
    def chaseable_value(self) -> Money:
        total = Money.zero()
        for r in self.at_risk:
            if r.is_chaseable:
                total += r.outstanding
        return total

    @property
    def disputed_value(self) -> Money:
        total = Money.zero()
        for r in self.at_risk:
            if r.disputed:
                total += r.outstanding
        return total

    @property
    def expected_recovery(self) -> Money:
        total = Money.zero()
        for r in self.at_risk:
            total += r.expected_recovery
        return total

    @property
    def weighted_days_overdue(self) -> float:
        """Average age weighted by money, not by invoice count.

        Counting invoices equally would let a hundred small recent ones hide a
        single large ancient one, which is exactly the receivable that matters.
        """
        paise = sum(r.outstanding.paise for r in self.at_risk)
        if not paise:
            return 0.0
        return sum(r.days_overdue * r.outstanding.paise for r in self.at_risk) / paise

    def by_bucket(self) -> dict[AgingBucket, tuple[int, Money]]:
        counts: dict[AgingBucket, int] = defaultdict(int)
        value: dict[AgingBucket, Money] = defaultdict(Money.zero)
        for r in self.at_risk:
            counts[r.bucket] += 1
            value[r.bucket] = value[r.bucket] + r.outstanding
        return {b: (counts[b], value[b]) for b in AgingBucket if b in counts}

    def render(self) -> str:
        blocked = sum(1 for r in self.at_risk if not r.is_chaseable)
        lines = [
            f"settled invoices   {self.settled:,}",
            f"current (not due)  {self.current:,}  {self.current_value}",
            f"overdue invoices   {len(self.at_risk):,}",
            f"receivables at risk {self.total_outstanding}",
            (f"  of which chaseable {self.chaseable_value}  "
             f"[{blocked} suppressed: {self.disputed_value} disputed]"),
            f"expected recovery  {self.expected_recovery} if all chaseable are chased",
            f"weighted age       {self.weighted_days_overdue:.0f} days overdue, money-weighted",
            "",
        ]
        for bucket, (n, value) in self.by_bucket().items():
            odds = COLLECTION_ODDS.get(bucket, 0.2)
            lines.append(
                f"  {bucket.value:<8} {n:>5} invoices  {value.format():>16}  "
                f"~{odds:.0%} collectable  {BUCKET_REASON[bucket]}"
            )
        return "\n".join(lines)


def scan(invoices: list[Invoice], now: datetime) -> ReceivablesReport:
    """Age the ledger and price every overdue invoice, oldest money first.

    `now` is when the book is scanned, and unlike a payment there is no failure
    moment to work from -- an invoice has no event, only an issue date and a
    due date -- so age is measured from here.
    """
    report = ReceivablesReport()
    for inv in invoices:
        if inv.is_settled:
            report.settled += 1
            continue
        if not inv.is_overdue(now):
            report.current += 1
            report.current_value += inv.outstanding
            continue
        report.at_risk.append(
            ReceivableRisk(
                invoice_id=inv.id,
                buyer_id=inv.buyer_id,
                outstanding=inv.outstanding,
                days_overdue=inv.days_overdue(now),
                disputed=inv.disputed_at is not None,
                promise_live=inv.promise is not None and inv.promise.is_live(now),
            )
        )
    report.at_risk.sort(key=lambda r: (-r.outstanding.paise, -r.days_overdue))
    return report
