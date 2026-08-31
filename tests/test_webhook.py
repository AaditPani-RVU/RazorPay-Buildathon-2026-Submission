"""The webhook receiver: what it trusts, and what it books as recovery.

A receiver is the one place in this system where an *outsider* proposes a
change to the revenue number. Everything else is proposed by our own planner
and disposed of by our own rules; here somebody else's HTTP request says money
arrived. So these tests are aimed at the four ways such a thing goes wrong:

*   **It trusts an unsigned body.** The check has to happen on raw bytes,
    before parsing, and a missing secret must fail closed rather than open.
*   **It books revenue it did not cause.** A merchant's own payment links get
    paid all day. Crediting those would be the accounts-payable-cycle error
    with a webhook in front of it.
*   **It books the same money twice.** Razorpay redelivers, and two event types
    can describe one settlement.
*   **It disagrees with the poll.** If a mandate is worth a year down one path
    and one charge down the other, the ledger reports whichever raced first.

Everything runs offline. No key, no secret, no network.
"""

import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta

from backstop.domain.actions import Action, ActionType
from backstop.domain.declines import Rail
from backstop.domain.entities import (
    Channel,
    Customer,
    Invoice,
    MandateStatus,
    Order,
    Subscription,
)
from backstop.domain.money import Money
from backstop.execute.executor import Outcome
from backstop.execute.razorpay import (
    ApiResponse,
    RazorpayExecutor,
    RecordedTransport,
)
from backstop.execute.webhook import (
    SETTLING_EVENTS,
    Verdict,
    WebhookReceiver,
    parse,
    verify,
)

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
SECRET = "whsec_test_backstop"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def ok(body):
    return ApiResponse(status=200, body=body)


def executor(**kw):
    transport = RecordedTransport(routes={
        "POST /payment_links": [ok({"id": "plink_A", "short_url": "https://rzp.io/A"})],
        "GET /orders": [ok({"count": 0, "items": []})],
        "POST /orders": [ok({"id": "order_A"})],
        "POST /subscription_registration/auth_links": [
            ok({"id": "inv_A", "short_url": "https://rzp.io/B"})
        ],
    })
    order = Order(id="order_1", customer_id="cust_1",
                  amount=Money.rupees(1200), created_at=NOW)
    inv = Invoice(id="inv_1", buyer_id="cust_1", amount=Money.rupees(50000),
                  issued_at=NOW - timedelta(days=70), due_at=NOW - timedelta(days=40))
    sub = Subscription(id="sub_1", customer_id="cust_1", amount=Money.rupees(499),
                       rail=Rail.EMANDATE_NACH, mandate_status=MandateStatus.EXPIRED,
                       next_charge_at=NOW)
    cust = Customer(id="cust_1", email="a@b.test", phone="+919800000000",
                    consented_channels={Channel.EMAIL, Channel.SMS})
    return RazorpayExecutor(
        transport=transport,
        orders={"order_1": order}, invoices={"inv_1": inv},
        subscriptions={"sub_1": sub}, customers={"cust_1": cust},
        **kw,
    )


def act(kind=ActionType.SEND_DUNNING, *, subject="order_1", channel=Channel.EMAIL, at=NOW):
    return Action(type=kind, subject_id=subject, scheduled_at=at,
                  channel=channel, rationale="test")


def delivery(event, kind, entity, *, created=None):
    body = {
        "entity": "event",
        "account_id": "acc_test",
        "event": event,
        "contains": [kind],
        "payload": {kind: {"entity": entity}},
        "created_at": int((created or NOW).timestamp()),
    }
    return json.dumps(body).encode()


def signed(body, secret=SECRET):
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def receiver(ex=None, secret=SECRET):
    return WebhookReceiver(executor=ex or executor(), secret=secret)


def dispatch_a_link(ex):
    """Put a real dispatch in the executor so an event has something to match."""
    result = ex.execute(act(), NOW)
    assert result.outcome is Outcome.DISPATCHED
    return result


# --------------------------------------------------------------------------
# it verifies before it trusts
# --------------------------------------------------------------------------


def test_a_forged_signature_is_rejected():
    body = delivery("payment_link.paid", "payment_link",
                    {"id": "plink_A", "status": "paid", "amount_paid": 120000})
    r = receiver().receive(body, "deadbeef", at=NOW)
    assert r.verdict is Verdict.REJECTED
    assert r.status == 400
    assert r.result is None


def test_a_signature_from_the_wrong_secret_is_rejected():
    body = delivery("payment_link.paid", "payment_link", {"id": "plink_A"})
    r = receiver().receive(body, signed(body, "whsec_someone_else"), at=NOW)
    assert r.verdict is Verdict.REJECTED


def test_no_configured_secret_fails_closed():
    """A receiver with no secret must refuse everything rather than accept it.

    The dangerous version of this bug looks like it has a signature check.
    """
    body = delivery("payment_link.paid", "payment_link", {"id": "plink_A"})
    assert receiver(secret="").receive(body, signed(body), at=NOW).verdict is Verdict.REJECTED
    assert not verify(body, signed(body), "")


def test_the_signature_covers_the_raw_bytes_not_the_parsed_body():
    """Re-serialising before verifying would make these two agree. They must not."""
    body = delivery("payment_link.paid", "payment_link", {"id": "plink_A"})
    sig = signed(body)
    reordered = json.dumps(json.loads(body), sort_keys=True, indent=2).encode()
    assert reordered != body
    assert verify(body, sig, SECRET)
    assert not verify(reordered, sig, SECRET)


def test_an_unparseable_body_never_reaches_the_parser_unsigned():
    r = receiver().receive(b"not json at all", "deadbeef", at=NOW)
    assert r.verdict is Verdict.REJECTED
    assert r.event is None


def test_a_signed_but_unparseable_body_is_ignored_not_rejected():
    body = b"{ this is signed and still garbage"
    r = receiver().receive(body, signed(body), at=NOW)
    assert r.verdict is Verdict.IGNORED
    assert r.status == 200


# --------------------------------------------------------------------------
# it credits only what recovery dispatched
# --------------------------------------------------------------------------


def test_a_paid_link_we_dispatched_is_recovered():
    ex = executor()
    dispatch_a_link(ex)
    body = delivery("payment_link.paid", "payment_link",
                    {"id": "plink_A", "status": "paid", "amount_paid": 120000})
    r = receiver(ex).receive(body, signed(body), event_id="evt_1", at=NOW)
    assert r.verdict is Verdict.RECOVERED
    assert r.result.outcome is Outcome.RECOVERED
    assert r.result.recovered == Money.rupees(1200)


def test_a_link_the_merchant_created_themselves_is_not_recovery():
    """The whole counterfactual argument, arriving over HTTP.

    A dashboard-created payment link being paid is real revenue and recovery
    caused none of it. A receiver that credited every `payment_link.paid` on
    the account would report the merchant's own business as its work.
    """
    ex = executor()
    dispatch_a_link(ex)
    body = delivery("payment_link.paid", "payment_link",
                    {"id": "plink_NOT_OURS", "status": "paid", "amount_paid": 999900})
    r = receiver(ex).receive(body, signed(body), event_id="evt_2", at=NOW)
    assert r.verdict is Verdict.UNMATCHED
    assert r.result is None
    assert "not recovered revenue" in r.detail


def test_an_unmatched_event_is_answered_200_so_razorpay_stops_retrying():
    """Not ours is a correct outcome, not a failure. A non-2xx would have
    Razorpay redeliver it forever."""
    ex = executor()
    body = delivery("payment_link.paid", "payment_link", {"id": "plink_X", "amount_paid": 100})
    assert receiver(ex).receive(body, signed(body), at=NOW).status == 200


def test_an_event_type_that_is_not_a_settlement_is_ignored():
    ex = executor()
    dispatch_a_link(ex)
    body = delivery("payment_link.expired", "payment_link", {"id": "plink_A"})
    r = receiver(ex).receive(body, signed(body), at=NOW)
    assert r.verdict is Verdict.IGNORED
    assert r.result is None


def test_ours_but_unpaid_is_unsettled_rather_than_recovered():
    ex = executor()
    dispatch_a_link(ex)
    body = delivery("payment_link.paid", "payment_link",
                    {"id": "plink_A", "status": "created", "amount_paid": 0})
    r = receiver(ex).receive(body, signed(body), at=NOW)
    assert r.verdict is Verdict.UNSETTLED
    assert r.result is None


# --------------------------------------------------------------------------
# it credits once
# --------------------------------------------------------------------------


def test_a_redelivered_event_is_not_credited_twice():
    ex = executor()
    dispatch_a_link(ex)
    rec = receiver(ex)
    body = delivery("payment_link.paid", "payment_link",
                    {"id": "plink_A", "status": "paid", "amount_paid": 120000})
    first = rec.receive(body, signed(body), event_id="evt_9", at=NOW)
    second = rec.receive(body, signed(body), event_id="evt_9", at=NOW)
    assert first.verdict is Verdict.RECOVERED
    assert second.verdict is Verdict.DUPLICATE
    assert second.result is None


def test_two_different_events_about_one_settlement_credit_once():
    """The load-bearing layer of the dedupe.

    Event-id deduplication cannot catch this: `order.paid` and
    `payment_link.paid` are genuinely different deliveries describing one
    payment. Only deduplicating on the *dispatch* does.
    """
    ex = executor()
    dispatch_a_link(ex)
    rec = receiver(ex)
    paid = {"id": "plink_A", "status": "paid", "amount_paid": 120000}
    first = rec.receive(delivery("payment_link.paid", "payment_link", paid),
                        signed(delivery("payment_link.paid", "payment_link", paid)),
                        event_id="evt_a", at=NOW)
    again = delivery("payment_link.partially_paid", "payment_link", paid)
    second = rec.receive(again, signed(again), event_id="evt_b", at=NOW)
    assert first.verdict is Verdict.RECOVERED
    assert second.verdict is Verdict.DUPLICATE


def test_a_webhook_and_a_poll_do_not_both_credit():
    """The two reconciliation paths must not race into a double count."""
    ex = executor()
    dispatch_a_link(ex)
    body = delivery("payment_link.paid", "payment_link",
                    {"id": "plink_A", "status": "paid", "amount_paid": 120000})
    assert receiver(ex).receive(body, signed(body), at=NOW).verdict is Verdict.RECOVERED

    ex.transport.routes["GET /payment_links/:id"] = [
        ok({"id": "plink_A", "status": "paid", "amount_paid": 120000})
    ]
    assert ex.reconcile(at=NOW) == []


# --------------------------------------------------------------------------
# it agrees with the poll about what money is worth
# --------------------------------------------------------------------------


def test_a_reregistered_mandate_is_worth_a_year_over_a_webhook_too():
    """The unit of recurring revenue cannot depend on which path delivered
    the news. A year through the poll and one charge through the webhook
    would make the recurring number a function of network timing."""
    ex = executor()
    ex.execute(act(ActionType.REQUEST_MANDATE_REREGISTRATION, subject="sub_1"), NOW)
    body = delivery("invoice.paid", "invoice", {"id": "inv_A", "status": "paid",
                                                "amount_paid": 100})
    r = receiver(ex).receive(body, signed(body), at=NOW)
    assert r.verdict is Verdict.RECOVERED
    assert r.result.recovered == ex.subscriptions["sub_1"].annual_value


def test_a_partly_paid_invoice_credits_only_what_arrived():
    ex = executor()
    ex.execute(act(ActionType.OFFER_PART_PAYMENT, subject="inv_1"), NOW)
    body = delivery("payment_link.partially_paid", "payment_link",
                    {"id": "plink_A", "status": "partially_paid", "amount_paid": 2000000})
    r = receiver(ex).receive(body, signed(body), at=NOW)
    assert r.verdict is Verdict.RECOVERED
    assert r.result.recovered == Money.rupees(20000)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def test_parse_pulls_the_entity_out_of_the_payload_envelope():
    body = delivery("order.paid", "order", {"id": "order_A", "amount_paid": 5000})
    event = parse(body, event_id="evt_z")
    assert event.id == "evt_z"
    assert event.name == "order.paid"
    assert event.entity_kind == "order"
    assert event.entity_id == "order_A"
    assert event.at == NOW


def test_every_settling_event_names_a_key_that_exists_in_its_payload():
    """Guards the table against a typo that would silently ignore an event."""
    for name, kind in SETTLING_EVENTS.items():
        body = delivery(name, kind, {"id": "ent_1", "status": "paid", "amount_paid": 1})
        assert parse(body) is not None, name
