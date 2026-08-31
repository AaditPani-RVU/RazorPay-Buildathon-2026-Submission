"""The Razorpay adapter: what it sends, what it refuses, and what it claims.

A live executor is the one component that can do something irreversible to a
person -- everything else in this system either computes a number or refuses
an action. So these tests are aimed less at "does the payload parse" and more
at the four ways a real adapter goes wrong:

*   **It messages somebody twice.** A replayed plan, a retried run, a second
    process. Idempotency is tested from both ends: the client-side memo and
    Razorpay's own uniqueness error.
*   **It claims money it has not got.** A dispatch is not a recovery. Nothing
    the adapter does synchronously may report `RECOVERED`, because nothing it
    does synchronously moves money.
*   **It sends when nobody asked it to.** Test mode really delivers, so silence
    has to be the default and noise has to be a decision.
*   **It runs against a live key.** One token in a `.env` file separates a
    simulation from mailing real customers.

Everything runs offline against `RecordedTransport`; no key is needed.
"""

from datetime import UTC, datetime, timedelta

import pytest

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
from backstop.execute.executor import Outcome
from backstop.execute.razorpay import (
    MIN_CHARGEABLE_PAISE,
    ApiResponse,
    HttpTransport,
    RazorpayError,
    RazorpayExecutor,
    RecordedTransport,
    capabilities,
    reference_for,
)

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)

LINK = "POST /payment_links"
ORDERS = "POST /orders"
LOOKUP = "GET /orders"
AUTH = "POST /subscription_registration/auth_links"
NOTIFY = "POST /payment_links/:id/notify_by/email"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def customer(cid="cust_1", *, email="a@example.test", phone="+919800000000"):
    return Customer(id=cid, email=email, phone=phone,
                    consented_channels={Channel.EMAIL, Channel.SMS})


def order(oid="order_1", *, amount=1200):
    o = Order(id=oid, customer_id="cust_1", amount=Money.rupees(amount), created_at=NOW)
    o.attempts.append(
        PaymentAttempt(
            id="att_1", order_id=oid, customer_id="cust_1", amount=o.amount,
            rail=Rail.CARD, at=NOW, status=AttemptStatus.FAILED,
            decline_code=DeclineCode.INSUFFICIENT_FUNDS,
        )
    )
    return o


def invoice(iid="inv_1", *, amount=50000, days_overdue=40):
    return Invoice(
        id=iid, buyer_id="cust_1", amount=Money.rupees(amount),
        issued_at=NOW - timedelta(days=days_overdue + 30),
        due_at=NOW - timedelta(days=days_overdue),
    )


def subscription(sid="sub_1", *, amount=499, rail=Rail.EMANDATE_NACH):
    return Subscription(
        id=sid, customer_id="cust_1", amount=Money.rupees(amount), rail=rail,
        mandate_status=MandateStatus.EXPIRED, next_charge_at=NOW,
    )


def act(kind=ActionType.SEND_DUNNING, *, subject="order_1", at=NOW,
        channel=Channel.EMAIL, amount_paise=None, route=None, why="test"):
    return Action(type=kind, subject_id=subject, scheduled_at=at, channel=channel,
                  amount_paise=amount_paise, route=route, rationale=why)


def ok(body):
    return ApiResponse(status=200, body=body)


def transport(**routes):
    """A recorded transport pre-loaded with plausible success responses."""
    base = {
        LINK: [ok({"id": "plink_A", "short_url": "https://rzp.io/rzp/A"})],
        LOOKUP: [ok({"count": 0, "items": []})],
        ORDERS: [ok({"id": "order_A"})],
        AUTH: [ok({"id": "inv_A", "short_url": "https://rzp.io/rzp/B"})],
        NOTIFY: [ok({"success": True})],
        "POST /payment_links/:id/notify_by/sms": [ok({"success": True})],
    }
    base.update(routes)
    return RecordedTransport(routes=base)


def executor(t=None, *, notify=False, **kw):
    return RazorpayExecutor(
        transport=t or transport(),
        orders=kw.pop("orders", {"order_1": order()}),
        invoices=kw.pop("invoices", {"inv_1": invoice()}),
        subscriptions=kw.pop("subscriptions", {"sub_1": subscription()}),
        customers=kw.pop("customers", {"cust_1": customer()}),
        notify=notify,
        **kw,
    )


# --------------------------------------------------------------------------
# a dispatch is not a recovery
# --------------------------------------------------------------------------


def test_dunning_is_dispatched_never_recovered():
    """The single most important claim the adapter makes about itself.

    A payment link is paid when a human opens it. Reporting anything else the
    moment the link is created would put a number in the ledger that no money
    stands behind.
    """
    result = executor().execute(act(), NOW)
    assert result.outcome is Outcome.DISPATCHED
    assert not result.is_recovery
    assert result.is_pending
    assert result.recovered == Money.zero()


def test_dispatch_carries_a_checkable_handle():
    result = executor().execute(act(), NOW)
    assert result.external is not None
    assert result.external.entity == "payment_link"
    assert result.external.id == "plink_A"
    assert result.external.url.startswith("https://")


def test_inert_actions_touch_no_api():
    t = transport()
    ex = executor(t)
    for kind in (ActionType.WAIT, ActionType.DO_NOTHING,
                 ActionType.ESCALATE_TO_HUMAN, ActionType.ESCALATE_TO_RISK):
        result = ex.execute(act(kind, channel=None), NOW)
        assert result.outcome is Outcome.NOT_APPLICABLE
    assert t.calls == []


# --------------------------------------------------------------------------
# what actually gets sent
# --------------------------------------------------------------------------


def test_link_carries_the_amount_and_an_audit_trail():
    t = transport()
    executor(t).execute(act(), NOW)
    [payload] = t.payloads(LINK)
    assert payload["amount"] == Money.rupees(1200).paise
    assert payload["currency"] == "INR"
    assert payload["notes"]["backstop_subject"] == "order_1"
    assert payload["notes"]["backstop_action"] == "send_dunning"
    assert payload["customer"]["email"] == "a@example.test"


def test_receivables_are_chased_for_the_outstanding_balance_not_the_face_value():
    inv = invoice(amount=50000)
    inv.amount_paid = Money.rupees(20000)
    t = transport()
    executor(t, invoices={"inv_1": inv}).execute(act(subject="inv_1"), NOW)
    [payload] = t.payloads(LINK)
    assert payload["amount"] == Money.rupees(30000).paise


def test_part_payment_asks_for_the_instalment_the_planner_chose():
    """The offer is a smaller ask, not a softer reminder, and the size of the
    ask is the planner's decision -- the adapter passes it through."""
    t = transport()
    executor(t).execute(
        act(ActionType.OFFER_PART_PAYMENT, subject="inv_1",
            amount_paise=Money.rupees(15000).paise),
        NOW,
    )
    [payload] = t.payloads(LINK)
    assert payload["accept_partial"] is True
    assert payload["first_min_partial_amount"] == Money.rupees(15000).paise
    assert payload["amount"] == Money.rupees(50000).paise


def test_an_instalment_larger_than_the_balance_is_clamped():
    t = transport()
    executor(t).execute(
        act(ActionType.OFFER_PART_PAYMENT, subject="inv_1",
            amount_paise=Money.rupees(999999).paise),
        NOW,
    )
    [payload] = t.payloads(LINK)
    assert payload["first_min_partial_amount"] == payload["amount"]


def test_a_retry_records_an_intent_and_says_so():
    """Test mode has no saved instrument, so `POST /orders` is the honest
    floor. The result must not imply a charge was attempted."""
    t = transport()
    result = executor(t).execute(act(ActionType.RETRY_PAYMENT, channel=None), NOW)
    [payload] = t.payloads(ORDERS)
    assert payload["receipt"] == reference_for(act(ActionType.RETRY_PAYMENT, channel=None))
    assert result.outcome is Outcome.DISPATCHED
    assert "nothing charges until the customer pays" in result.detail


def test_a_retry_checks_for_itself_before_creating_a_second_order():
    """`/orders` does not enforce a unique receipt, so a re-run in a fresh
    process would duplicate a charging action -- the worst thing this adapter
    could do. It looks first, and pays a round trip for the privilege."""
    t = transport(**{LOOKUP: [ok({"count": 1, "items": [{"id": "order_earlier"}]})]})
    result = executor(t).execute(act(ActionType.RETRY_PAYMENT, channel=None), NOW)
    assert result.outcome is Outcome.NO_EFFECT
    assert t.payloads(ORDERS) == []


def test_switch_route_keeps_the_acquirer_in_the_notes():
    """`/orders` cannot express an acquirer preference. Dropping it silently
    would make a route switch indistinguishable from a plain retry."""
    t = transport()
    executor(t).execute(
        act(ActionType.SWITCH_ROUTE, channel=None, route="acq_beta"), NOW
    )
    [payload] = t.payloads(ORDERS)
    assert payload["notes"]["route"] == "acq_beta"


# --------------------------------------------------------------------------
# mandates re-register; they are not dunned
# --------------------------------------------------------------------------


def test_reregistration_uses_the_authorisation_endpoint():
    t = transport()
    result = executor(t).execute(
        act(ActionType.REQUEST_MANDATE_REREGISTRATION, subject="sub_1"), NOW
    )
    assert t.payloads(LINK) == []
    [payload] = t.payloads(AUTH)
    assert payload["subscription_registration"]["method"] == "emandate"
    assert result.external.id == "inv_A"


def test_emandate_authorises_at_zero_and_upi_at_one_rupee():
    """Probed against the live test API: UPI and card reject a zero-amount
    authorisation, e-mandate accepts one."""
    t = transport()
    ex = executor(t, subscriptions={"sub_1": subscription(rail=Rail.EMANDATE_NACH)})
    ex.execute(act(ActionType.REQUEST_MANDATE_REREGISTRATION, subject="sub_1"), NOW)

    t2 = transport()
    ex2 = executor(t2, subscriptions={"sub_1": subscription(rail=Rail.UPI_AUTOPAY)})
    ex2.execute(act(ActionType.REQUEST_MANDATE_REREGISTRATION, subject="sub_1"), NOW)

    assert t.payloads(AUTH)[0]["amount"] == 0
    assert t2.payloads(AUTH)[0]["amount"] == MIN_CHARGEABLE_PAISE


def test_an_order_billed_against_a_mandate_re_registers_that_mandate():
    """A mandate-rail payment fails as an *order*, so the playbook's answer
    names the order while the thing needing re-authorisation is the
    subscription behind it. Refusing that would be the adapter rejecting a
    good action for naming the subject the planner naturally names."""
    t = transport()
    ex = executor(t, subscription_by_order={"order_1": "sub_1"})
    result = ex.execute(
        act(ActionType.REQUEST_MANDATE_REREGISTRATION, subject="order_1"), NOW
    )
    assert result.outcome is Outcome.DISPATCHED
    [payload] = t.payloads(AUTH)
    assert payload["subscription_registration"]["max_amount"] == Money.rupees(499).paise


def test_that_mandate_is_still_credited_a_year_when_it_comes_back():
    t = transport(**{"GET /invoices/:id": [ok({"status": "paid", "amount_paid": 0})]})
    ex = executor(t, subscription_by_order={"order_1": "sub_1"})
    ex.execute(act(ActionType.REQUEST_MANDATE_REREGISTRATION, subject="order_1"), NOW)
    [settled] = ex.reconcile()
    assert settled.recovered == subscription().annual_value


def test_the_mandate_ceiling_is_the_mandate_amount_not_headroom():
    """Asking a customer to authorise more than the merchant is owed is asking
    for a larger permission than the mandate needs."""
    t = transport()
    executor(t, subscriptions={"sub_1": subscription(amount=499)}).execute(
        act(ActionType.REQUEST_MANDATE_REREGISTRATION, subject="sub_1"), NOW
    )
    reg = t.payloads(AUTH)[0]["subscription_registration"]
    assert reg["max_amount"] == Money.rupees(499).paise
    assert reg["expire_at"] > datetime.now(UTC).timestamp()


# --------------------------------------------------------------------------
# idempotency
# --------------------------------------------------------------------------


def test_the_same_action_twice_sends_once():
    t = transport()
    ex = executor(t)
    first = ex.execute(act(), NOW)
    second = ex.execute(act(), NOW)
    assert first.outcome is Outcome.DISPATCHED
    assert second.outcome is Outcome.NO_EFFECT
    assert len(t.payloads(LINK)) == 1


def test_a_replay_is_not_charged_for():
    """Nothing left the building, so the merchant is not billed for postage.
    The simulator charges for a wasted chase because one was really sent;
    here one was not."""
    ex = executor()
    ex.execute(act(), NOW)
    assert ex.execute(act(), NOW).cost == Money.zero()


DUPLICATE_LINK = ApiResponse(400, {"error": {"description": (
    "payment link with given reference_id: bkstp_x already exists. "
    "Please create a payment link with a different reference_id"
)}})


def test_razorpay_refusing_a_duplicate_reference_is_not_an_error():
    """A previous process already sent this one. That is a successful
    outcome for the customer -- one message -- and must not raise."""
    t = transport(**{LINK: [DUPLICATE_LINK],
                     "GET /payment_links": [ok({"payment_links": []})]})
    result = executor(t).execute(act(), NOW)
    assert result.outcome is Outcome.NO_EFFECT
    assert "not sent twice" in result.detail


def test_an_already_sent_link_is_recovered_so_it_stays_reconcilable():
    """The restart hole, closed.

    `dispatched` lives in memory. A process that sends a link and then dies
    has lost its only handle on the thing that went out, and a payment landing
    on that link could never be credited -- not by a poll, which has nothing
    to poll, and not by a webhook, which would find no matching dispatch and
    correctly refuse to credit a stranger. Looking the reference back up is
    what makes "already sent" recoverable rather than merely declined.
    """
    t = transport(**{
        LINK: [DUPLICATE_LINK],
        "GET /payment_links": [ok({"payment_links": [
            {"id": "plink_OLD", "short_url": "https://rzp.io/rzp/OLD"}
        ]})],
    })
    ex = executor(t)
    result = ex.execute(act(), NOW)

    assert result.outcome is Outcome.NO_EFFECT, "still not a second send"
    assert result.external.id == "plink_OLD"
    assert "reconcilable again" in result.detail
    assert len(ex.pending) == 1, "and now it can be reconciled"


def test_a_recovered_dispatch_can_then_be_credited():
    """The point of recovering it: the loop closes after a restart."""
    t = transport(**{
        LINK: [DUPLICATE_LINK],
        "GET /payment_links": [ok({"payment_links": [{"id": "plink_OLD"}]})],
    })
    ex = executor(t)
    ex.execute(act(), NOW)
    t.routes["GET /payment_links/:id"] = [
        ok({"id": "plink_OLD", "status": "paid", "amount_paid": 120000})
    ]
    settled = ex.reconcile(at=NOW)
    assert len(settled) == 1
    assert settled[0].recovered == Money.rupees(1200)


def test_an_already_sent_link_that_cannot_be_found_is_still_not_resent():
    """A failed lookup must not become a second message to a customer."""
    t = transport(**{LINK: [DUPLICATE_LINK],
                     "GET /payment_links": [ApiResponse(500, {"error": {}})]})
    result = executor(t).execute(act(), NOW)
    assert result.outcome is Outcome.NO_EFFECT
    assert result.external is None


def test_a_duplicate_auth_link_is_honestly_unrecoverable():
    """Auth links are invoices of type `link`; they do not come back from
    `GET /invoices` and there is no filter for their receipt. Guessing at one
    would risk reconciling against somebody else's invoice, so the adapter
    declines to and the limit is stated rather than papered over."""
    t = transport(**{AUTH: [ApiResponse(400, {"error": {
        "description": "receipt must be unique"}})]})
    ex = executor(t)
    result = ex.execute(act(ActionType.REQUEST_MANDATE_REREGISTRATION, subject="sub_1"), NOW)
    assert result.outcome is Outcome.NO_EFFECT
    assert result.external is None
    assert ex.pending == []


def test_an_unrelated_bad_request_still_raises():
    """The duplicate check is matched on prose, so it has to be narrow enough
    that a real failure is not silently swallowed as de-duplication."""
    t = transport(**{
        LINK: [ApiResponse(400, {"error": {"description": "amount must be at least 100"}})]
    })
    with pytest.raises(RazorpayError, match="amount must be at least"):
        executor(t).execute(act(), NOW)


def test_an_auth_link_duplicate_speaks_a_different_dialect():
    """Payment links complain about `reference_id`; auth links complain about
    `receipt`. Both mean "already sent", and only a live re-run found the
    second one."""
    t = transport(**{
        AUTH: [ApiResponse(400, {"error": {"description": (
            "receipt must be unique for each item : bkstp_e0bb7eff53198f71be0b"
        )}})]
    })
    result = executor(t).execute(
        act(ActionType.REQUEST_MANDATE_REREGISTRATION, subject="sub_1"), NOW
    )
    assert result.outcome is Outcome.NO_EFFECT
    assert "not sent twice" in result.detail


def test_a_failing_auth_link_still_raises():
    t = transport(**{
        AUTH: [ApiResponse(400, {"error": {"description": "max_amount is invalid"}})]
    })
    with pytest.raises(RazorpayError, match="max_amount"):
        executor(t).execute(
            act(ActionType.REQUEST_MANDATE_REREGISTRATION, subject="sub_1"), NOW
        )


@pytest.mark.parametrize(
    ("status", "description", "expected"),
    [
        (400, "payment link with given reference_id: x already exists", True),
        (400, "receipt must be unique for each item : x", True),
        (400, "the receipt is too long", False),
        (400, "reference_id is required", False),
        (409, "reference_id: x already exists", False),
    ],
)
def test_duplicate_detection_is_narrow(status, description, expected):
    """It is matched on prose, so it has to demand two independent tokens.
    A 400 that merely mentions one of the words is a real failure."""
    assert ApiResponse(status, {"error": {"description": description}}) \
        .is_duplicate_reference is expected


def test_the_reference_ignores_the_models_prose():
    """A model rewording its own justification must not buy an extra message
    to a customer, for the same reason the policy engine ignores rationale."""
    assert reference_for(act(why="first wording")) == reference_for(act(why="second wording"))


def test_a_genuinely_later_chase_gets_its_own_reference():
    assert reference_for(act()) != reference_for(act(at=NOW + timedelta(days=3)))


def test_references_fit_razorpays_field():
    assert len(reference_for(act())) <= 40


# --------------------------------------------------------------------------
# silence is the default
# --------------------------------------------------------------------------


def test_nothing_is_sent_unless_sending_was_asked_for():
    """Razorpay delivers in test mode. A link created for a demo must not
    arrive in somebody's inbox because a default was convenient."""
    t = transport()
    result = executor(t, notify=False).execute(act(), NOW)
    [payload] = t.payloads(LINK)
    assert payload["notify"] == {"email": False, "sms": False}
    assert payload["reminder_enable"] is False
    assert t.payloads(NOTIFY) == []
    assert "not sent" in result.detail


def test_notify_sends_on_the_channel_the_planner_chose():
    t = transport()
    result = executor(t, notify=True).execute(act(channel=Channel.SMS), NOW)
    assert t.payloads(LINK)[0]["notify"] == {"email": False, "sms": True}
    assert len(t.payloads("POST /payment_links/:id/notify_by/sms")) == 1
    assert "sent by sms" in result.detail


def test_an_order_is_never_notified():
    """An order is not a message. Nothing to send."""
    t = transport()
    executor(t, notify=True).execute(act(ActionType.RETRY_PAYMENT, channel=None), NOW)
    assert t.payloads(NOTIFY) == []


# --------------------------------------------------------------------------
# capability gaps are recorded, not raised
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (act(ActionType.RETRY_PAYMENT, subject="sub_1", channel=None),
         "nothing to present"),
        (act(ActionType.RETRY_PAYMENT, subject="inv_1", channel=None),
         "no instrument to re-present"),
        (act(ActionType.REQUEST_MANDATE_REREGISTRATION, subject="order_1"),
         "no mandate stands behind this subject"),
        (act(ActionType.SEND_DUNNING, subject="sub_1"),
         "restored by re-authorisation"),
        (act(ActionType.OFFER_PART_PAYMENT, subject="sub_1", amount_paise=100),
         "restored by re-authorisation"),
        (act(channel=Channel.VOICE), "no voice channel"),
        (act(subject="order_missing"), "no such subject"),
    ],
)
def test_impossible_actions_are_reported_and_never_dispatched(action, expected):
    """These mirror the simulator's refusals in the same words, so a merchant
    reading both ledgers does not have to learn two vocabularies."""
    t = transport()
    result = executor(t).execute(action, NOW)
    assert result.outcome is Outcome.NO_EFFECT
    assert expected in result.detail
    assert t.calls == []


def test_voice_is_refused_rather_than_quietly_emailed():
    """Substituting a channel the customer did not consent to is the failure
    mode, not the fallback."""
    t = transport()
    executor(t).execute(act(channel=Channel.VOICE), NOW)
    assert t.payloads(LINK) == []


def test_a_contact_with_nobody_to_contact_is_not_sent():
    t = transport()
    result = executor(t, customers={}).execute(act(), NOW)
    assert result.outcome is Outcome.NO_EFFECT
    assert "no contact details" in result.detail


# --------------------------------------------------------------------------
# reconciliation: the half a simulator does not need
# --------------------------------------------------------------------------


def test_an_unpaid_link_reconciles_to_nothing():
    t = transport(**{"GET /payment_links/:id": [ok({"status": "created", "amount_paid": 0})]})
    ex = executor(t)
    ex.execute(act(), NOW)
    assert ex.reconcile() == []
    assert len(ex.pending) == 1


def test_a_paid_link_becomes_a_recovery_worth_what_was_paid():
    t = transport(**{
        "GET /payment_links/:id": [ok({"status": "paid", "amount_paid": 120000})]
    })
    ex = executor(t)
    ex.execute(act(), NOW)
    [settled] = ex.reconcile()
    assert settled.outcome is Outcome.RECOVERED
    assert settled.recovered == Money.rupees(1200)
    assert settled.action.subject_id == "order_1"


def test_a_part_payment_credits_only_what_arrived():
    t = transport(**{
        "GET /payment_links/:id": [ok({"status": "partially_paid", "amount_paid": 1500000})]
    })
    ex = executor(t)
    ex.execute(
        act(ActionType.OFFER_PART_PAYMENT, subject="inv_1",
            amount_paise=Money.rupees(15000).paise),
        NOW,
    )
    [settled] = ex.reconcile()
    assert settled.recovered == Money.rupees(15000)
    assert "30% of the balance" in settled.detail


def test_a_reregistered_mandate_is_credited_a_year_not_a_charge():
    """The unit of recurring revenue is decided in one place. An adapter with
    its own opinion about it would put a second number in the system."""
    t = transport(**{"GET /invoices/:id": [ok({"status": "paid", "amount_paid": 0})]})
    ex = executor(t)
    ex.execute(act(ActionType.REQUEST_MANDATE_REREGISTRATION, subject="sub_1"), NOW)
    [settled] = ex.reconcile()
    assert settled.recovered == subscription().annual_value
    assert settled.recovered == Money.rupees(499 * 12)


def test_reconciling_twice_does_not_book_the_money_twice():
    t = transport(**{
        "GET /payment_links/:id": [ok({"status": "paid", "amount_paid": 120000})]
    })
    ex = executor(t)
    ex.execute(act(), NOW)
    assert len(ex.reconcile()) == 1
    assert ex.reconcile() == []
    assert ex.pending == []


# --------------------------------------------------------------------------
# the live-key guardrail
# --------------------------------------------------------------------------


def test_a_live_key_is_refused():
    with pytest.raises(RazorpayError, match="refusing to run against key"):
        HttpTransport(key_id="rzp_live_abcdef", key_secret="s")


def test_the_refusal_does_not_leak_the_secret():
    with pytest.raises(RazorpayError) as err:
        HttpTransport(key_id="rzp_live_abcdef", key_secret="supersecret")
    assert "supersecret" not in str(err.value)


def test_missing_keys_say_where_to_put_them():
    with pytest.raises(RazorpayError, match=".env"):
        HttpTransport(key_id="", key_secret="")


def test_a_test_key_is_accepted():
    assert HttpTransport(key_id="rzp_test_abcdef", key_secret="s").name == "razorpay-http"


# --------------------------------------------------------------------------
# capability probe
# --------------------------------------------------------------------------


def test_the_probe_reports_a_disabled_product_without_failing():
    """The account this was built against has Subscriptions switched off. The
    adapter does not need it, and the probe should say so rather than throw."""
    t = RecordedTransport(routes={
        "GET /payments?count=1": [ok({"items": []})],
        "GET /orders?count=1": [ok({"items": []})],
        "GET /payment_links?count=1": [ok({"items": []})],
        "GET /invoices?count=1": [ok({"items": []})],
        "GET /plans?count=1": [ApiResponse(401, {"error": "Unauthorized"})],
    })
    rows = capabilities(t)
    by_name = {name: okness for name, okness, _, _ in rows}
    assert by_name["payment_links"] is True
    assert by_name["plans"] is False
