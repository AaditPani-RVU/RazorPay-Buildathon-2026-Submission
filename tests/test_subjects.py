"""Subject resolution: which entity an action is actually about.

This used to be written inline in the backtest and, differently, in the live
walkthrough. Two callers resolving the same action to different entities would
not disagree about the *rules* -- they would disagree about the evidence, and
report different verdicts from an engine that behaved identically. These tests
pin the resolution order, which is the part with a real decision in it.

Offline: hand-built entities, no generator, no model, no network.
"""

from datetime import UTC, datetime, timedelta

from backstop.domain.actions import Action, ActionType
from backstop.domain.declines import DeclineCode, Rail
from backstop.domain.entities import (
    AttemptStatus,
    Channel,
    Customer,
    Invoice,
    MandateStatus,
    Order,
    PaymentAttempt,
    Subscription,
)
from backstop.domain.money import Money
from backstop.ledger.ledger import Surface
from backstop.policy.subjects import ContactBook, SubjectIndex

NOW = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)


def order(oid: str = "order_1", customer: str = "cust_1") -> Order:
    return Order(
        id=oid, customer_id=customer, amount=Money.rupees(2000), created_at=NOW,
        attempts=[
            PaymentAttempt(
                id=f"att_{oid}", order_id=oid, customer_id=customer,
                amount=Money.rupees(2000), rail=Rail.CARD, at=NOW,
                status=AttemptStatus.FAILED,
                decline_code=DeclineCode.INSUFFICIENT_FUNDS,
            )
        ],
    )


def index_of(**kw) -> SubjectIndex:
    return SubjectIndex(
        orders={o.id: o for o in kw.get("orders", [])},
        subscriptions={s.id: s for s in kw.get("subscriptions", [])},
        invoices={i.id: i for i in kw.get("invoices", [])},
        customers={c.id: c for c in kw.get("customers", [])},
        subscription_by_order=kw.get("subscription_by_order", {}),
    )


def test_an_order_id_names_the_payments_surface():
    index = index_of(orders=[order()], customers=[Customer(id="cust_1")])
    subject = index.resolve("order_1")
    assert subject.surface is Surface.PAYMENT
    assert subject.order is not None
    assert subject.customer is not None


def test_a_subscription_id_names_recurring_even_though_it_has_orders():
    sub = Subscription(
        id="sub_1", customer_id="cust_1", amount=Money.rupees(499), rail=Rail.UPI,
        mandate_status=MandateStatus.EXPIRED, next_charge_at=NOW, orders=["order_1"],
    )
    index = index_of(orders=[order()], subscriptions=[sub],
                     customers=[Customer(id="cust_1")])
    assert index.resolve("sub_1").surface is Surface.RECURRING


def test_a_failed_mandate_presentation_carries_its_subscription():
    """A mandate charge fails as an *order*, and the rules that speak about
    mandates have to be able to see the authorisation behind it."""
    sub = Subscription(
        id="sub_1", customer_id="cust_1", amount=Money.rupees(499), rail=Rail.UPI,
        mandate_status=MandateStatus.REVOKED, next_charge_at=NOW,
    )
    index = index_of(
        orders=[order()], subscriptions=[sub], customers=[Customer(id="cust_1")],
        subscription_by_order={"order_1": "sub_1"},
    )
    subject = index.resolve("order_1")
    assert subject.surface is Surface.PAYMENT
    assert subject.subscription is sub


def test_an_unknown_subject_is_not_known():
    """A subject the batch does not contain must be refusable rather than
    ruled on: the rules would abstain for lack of evidence and let it through."""
    assert not index_of().resolve("nothing_at_all").known


def test_context_is_judged_at_the_actions_own_moment_by_default():
    index = index_of(orders=[order()], customers=[Customer(id="cust_1")])
    later = NOW + timedelta(days=3)
    action = Action(
        type=ActionType.RETRY_PAYMENT, subject_id="order_1",
        scheduled_at=later, rationale="x",
    )
    assert index.context(action).now == later
    assert index.context(action, now=NOW).now == NOW


def test_the_contact_book_answers_by_subject_and_by_person():
    book = ContactBook()
    for subject in ("order_1", "inv_1"):
        book.record(
            Action(type=ActionType.SEND_DUNNING, subject_id=subject,
                   scheduled_at=NOW, channel=Channel.EMAIL, rationale="x"),
            customer_id="cust_1", at=NOW,
        )
    assert len(book.for_subject("order_1")) == 1
    assert len(book.for_customer("cust_1")) == 2
    assert book.total == 2


def test_the_contact_book_ignores_actions_that_contact_nobody():
    book = ContactBook()
    written = book.record(
        Action(type=ActionType.RETRY_PAYMENT, subject_id="order_1",
               scheduled_at=NOW, rationale="x"),
        customer_id="cust_1", at=NOW,
    )
    assert written is None
    assert book.total == 0


def test_context_carries_the_book_the_two_ways_the_rules_ask():
    index = index_of(
        orders=[order()], invoices=[
            Invoice(id="inv_1", buyer_id="cust_1", amount=Money.rupees(9000),
                    issued_at=NOW, due_at=NOW)
        ],
        customers=[Customer(id="cust_1")],
    )
    book = ContactBook()
    book.record(
        Action(type=ActionType.SEND_DUNNING, subject_id="inv_1", scheduled_at=NOW,
               channel=Channel.EMAIL, rationale="x"),
        customer_id="cust_1", at=NOW,
    )
    action = Action(
        type=ActionType.SEND_DUNNING, subject_id="order_1", scheduled_at=NOW,
        channel=Channel.EMAIL, rationale="x",
    )
    ctx = index.context(action, contacts=book)
    # Nothing has been sent about this order, but the person has heard from us.
    assert ctx.contacts == []
    assert len(ctx.customer_contacts) == 1
