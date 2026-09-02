"""Who an action is about, and what has already been said to them.

Two things every caller that rules on an action needs, and neither belongs in
the engine: which entities the action's subject names, and how many messages
that subject and that *person* have already had. The engine reads both off a
`PolicyContext`; somebody has to build one.

The backtest built it inline, and so, in a smaller way, did the live
walkthrough. That was survivable while there was one caller. It stops being
survivable the moment a second surface rules on the same actions and reports
different verdicts, because the difference would not be in the rules -- it
would be in which entity the two callers thought an action was about. A
mandate presentation that failed is an order *and* belongs to a subscription;
resolve that one way in the measurement and the other way in a console, and
the two disagree about what the policy engine did.

So subject resolution lives here, once, and both read it.

`ContactBook` is the other half. Frequency and fatigue are the only rules that
depend on what the *arm itself* has already sent, which makes contact history
mutable state threaded through an evaluation loop rather than a property of
the batch. Keeping the per-subject and per-person indexes side by side is
deliberate: the per-person cap has to be answerable at the moment an action is
judged, and deriving it by scanning every subject at that moment would be
quadratic in exactly the loop that runs 35,000 times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from backstop.domain.actions import Action
from backstop.domain.declines import RootCause
from backstop.domain.entities import (
    ContactRecord,
    Customer,
    Invoice,
    Order,
    Subscription,
    new_id,
    utc,
)
from backstop.ledger.ledger import Surface
from backstop.policy.engine import PolicyConfig, PolicyContext


@dataclass(frozen=True)
class Subject:
    """One action's subject, resolved to a surface and its entities."""

    surface: Surface
    customer: Customer | None = None
    order: Order | None = None
    invoice: Invoice | None = None
    subscription: Subscription | None = None

    @property
    def known(self) -> bool:
        """Whether the subject is in the batch at all.

        A subject that resolves to nothing must not be ruled on: the rules
        would abstain for lack of evidence and the action would sail through
        an engine that had nothing to check it against.
        """
        return any((self.order, self.invoice, self.subscription))

    @property
    def id(self) -> str:
        for entity in (self.order, self.invoice, self.subscription):
            if entity is not None:
                return entity.id
        return ""


@dataclass
class SubjectIndex:
    """Resolves an action's `subject_id` to the entities the rules read."""

    orders: dict[str, Order] = field(default_factory=dict)
    subscriptions: dict[str, Subscription] = field(default_factory=dict)
    invoices: dict[str, Invoice] = field(default_factory=dict)
    customers: dict[str, Customer] = field(default_factory=dict)
    subscription_by_order: dict[str, str] = field(default_factory=dict)
    """Which subscription a mandate presentation belongs to. A failed mandate
    charge is an order, so an action naming that order is still working on the
    authorisation behind it, and the rules that speak about mandates have to
    be able to see it."""

    @classmethod
    def of(cls, scenario) -> SubjectIndex:
        return cls(
            orders={o.id: o for o in scenario.orders},
            subscriptions={s.id: s for s in scenario.subscriptions},
            invoices={i.id: i for i in scenario.invoices},
            customers=dict(scenario.customers),
            subscription_by_order=dict(scenario.subscription_by_order),
        )

    def resolve(self, subject_id: str) -> Subject:
        """Name the surface first, then the entities.

        Order of precedence matters and is not arbitrary. A subscription id
        names the recurring surface even though the mandate has orders behind
        it, because re-registering an authorisation acts on the authorisation.
        An order id names payments, and carries its subscription along for the
        mandate rules to read.
        """
        subscription = self.subscriptions.get(subject_id)
        if subscription is not None:
            return Subject(
                surface=Surface.RECURRING,
                customer=self.customers.get(subscription.customer_id),
                subscription=subscription,
            )
        invoice = self.invoices.get(subject_id)
        if invoice is not None:
            return Subject(
                surface=Surface.RECEIVABLE,
                customer=self.customers.get(invoice.buyer_id),
                invoice=invoice,
            )
        order = self.orders.get(subject_id)
        behind = self.subscription_by_order.get(subject_id)
        return Subject(
            surface=Surface.PAYMENT,
            customer=self.customers.get(order.customer_id) if order else None,
            order=order,
            subscription=self.subscriptions.get(behind) if behind else None,
        )

    def context(
        self,
        action: Action,
        *,
        now: datetime | None = None,
        contacts: ContactBook | None = None,
        diagnosis: RootCause | None = None,
        outage_until: datetime | None = None,
        config: PolicyConfig | None = None,
    ) -> PolicyContext:
        """A context for one action, judged at its own scheduled moment.

        `now` defaults to the action's own `scheduled_at` rather than to wall
        time, because that is the moment the rules are being asked about. A
        quiet-hours check run against the clock on the wall would report on the
        hour a plan was drawn instead of the hour it lands.
        """
        subject = self.resolve(action.subject_id)
        at = utc(now if now is not None else action.scheduled_at)
        book = contacts or ContactBook()
        return PolicyContext(
            now=at,
            customer=subject.customer,
            order=subject.order,
            invoice=subject.invoice,
            subscription=subject.subscription,
            contacts=book.for_subject(action.subject_id),
            customer_contacts=(
                book.for_customer(subject.customer.id) if subject.customer else []
            ),
            diagnosis=diagnosis,
            outage_until=outage_until,
            config=config or PolicyConfig(),
        )


@dataclass
class ContactBook:
    """What has already been sent, indexed the two ways the rules ask.

    Deliberately not derived from the entities: this is the history *this run*
    has created, and an arm that ignored its own sends would report a contact
    cap it was not actually keeping.
    """

    by_subject: dict[str, list[ContactRecord]] = field(default_factory=dict)
    by_customer: dict[str, list[ContactRecord]] = field(default_factory=dict)

    def for_subject(self, subject_id: str) -> list[ContactRecord]:
        return self.by_subject.get(subject_id, [])

    def for_customer(self, customer_id: str) -> list[ContactRecord]:
        return self.by_customer.get(customer_id, [])

    def record(self, action: Action, *, customer_id: str, at: datetime) -> ContactRecord | None:
        """Write down a contact that actually went out. Returns None for the
        actions that contact nobody, so a caller can hand every executed action
        to this without first asking what kind it was."""
        if not action.is_contact or action.channel is None:
            return None
        entry = ContactRecord(
            id=new_id("contact"),
            customer_id=customer_id,
            channel=action.channel,
            at=utc(at),
            subject_ref=action.subject_id,
        )
        self.by_subject.setdefault(action.subject_id, []).append(entry)
        if customer_id:
            self.by_customer.setdefault(customer_id, []).append(entry)
        return entry

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.by_subject.values())
