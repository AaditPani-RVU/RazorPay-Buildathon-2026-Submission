"""The receivables surface: aging, and what collecting an invoice may claim.

Receivables are the easiest surface in the system on which to report a large,
completely false number, and the reason is the counterfactual. Most overdue
B2B invoices are paid whether or not anybody chases them -- an accounts-payable
department runs on a cycle, and an invoice that is twelve days late is usually
not a collections problem, it is a Tuesday. A tool that dunned the whole aged
debtors report and booked everything that subsequently arrived would show a
spectacular recovery rate while having caused almost none of it.

So these tests are aimed at that first, and then at the two things a
collections agent must refuse to do: chase a buyer who is disputing the
invoice, and chase one who has already committed to a date.
"""

from datetime import UTC, datetime, timedelta

import pytest

from backstop.decide.planner import (
    PART_PAYMENT_OFFER_SHARE,
    naive_invoice_chase,
    receivable_actions,
)
from backstop.detect.receivables import AgingBucket, bucket_for, scan
from backstop.domain.actions import Action, ActionType
from backstop.domain.entities import (
    Channel,
    ContactRecord,
    Customer,
    Invoice,
    InvoiceStatus,
    PromiseToPay,
    new_id,
)
from backstop.domain.money import Money
from backstop.evaluation.backtest import run
from backstop.execute.executor import Outcome, SimulatedExecutor
from backstop.ledger.ledger import Surface
from backstop.policy.engine import (
    Disposition,
    PolicyConfig,
    PolicyContext,
    PolicyEngine,
)
from backstop.simulate.generator import SimConfig, generate
from backstop.simulate.recoverability import InvoiceRecovery

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
ENGINE = PolicyEngine()


def invoice(
    *,
    days_overdue=45,
    amount=80000,
    paid=0,
    disputed=False,
    promise_days=None,
    iid="inv_1",
    buyer="buyer_1",
):
    due = NOW - timedelta(days=days_overdue)
    inv = Invoice(
        id=iid,
        buyer_id=buyer,
        amount=Money.rupees(amount),
        issued_at=due - timedelta(days=30),
        due_at=due,
        amount_paid=Money.rupees(paid),
    )
    if paid:
        inv.status = InvoiceStatus.PART_PAID
    if disputed:
        inv.disputed_at = due + timedelta(days=2)
    if promise_days is not None:
        inv.promise = PromiseToPay(
            promised_at=due + timedelta(days=1),
            promised_for=NOW + timedelta(days=promise_days),
            amount=inv.outstanding,
        )
    return inv


def truth(
    *,
    chased=True,
    unprompted=False,
    constrained=False,
    share=0.5,
    days=30.0,
    iid="inv_1",
):
    return InvoiceRecovery(
        invoice_id=iid,
        would_pay_if_chased=chased,
        pays_unprompted=unprompted,
        needs_part_payment=constrained,
        part_payment_share=share,
        responsive_until=NOW + timedelta(days=days) if chased else None,
    )


def executor(inv, rec):
    return SimulatedExecutor(
        orders={}, recoverability={},
        invoices={inv.id: inv}, invoice_recovery={inv.id: rec},
    )


def buyer(**kw):
    opts = {
        "id": "buyer_1",
        "email": "ap@example.test",
        "phone": "+919800000000",
        "consented_channels": {Channel.EMAIL, Channel.SMS},
    }
    opts.update(kw)
    return Customer(**opts)


# -- aging ------------------------------------------------------------------


@pytest.mark.parametrize(
    "days,expected",
    [
        (-3, AgingBucket.CURRENT),
        (0, AgingBucket.CURRENT),
        (1, AgingBucket.DAYS_1_30),
        (30, AgingBucket.DAYS_1_30),
        (31, AgingBucket.DAYS_31_60),
        (60, AgingBucket.DAYS_31_60),
        (61, AgingBucket.DAYS_61_90),
        (90, AgingBucket.DAYS_61_90),
        (91, AgingBucket.DAYS_90_PLUS),
        (400, AgingBucket.DAYS_90_PLUS),
    ],
)
def test_aging_boundaries(days, expected):
    """The brackets are the ones a finance team already reads. Off-by-one at a
    boundary would put an invoice in the wrong half of the collections ladder."""
    assert bucket_for(days) == expected


def test_a_settled_invoice_is_not_at_risk():
    inv = invoice()
    inv.status = InvoiceStatus.PAID
    inv.amount_paid = inv.amount
    report = scan([inv], NOW)
    assert report.at_risk == []
    assert report.settled == 1


def test_a_not_yet_due_invoice_is_counted_but_not_at_risk():
    """A healthy ledger is still part of the ledger. Reporting only the bad
    half makes the at-risk figure impossible to put in proportion."""
    report = scan([invoice(days_overdue=-10)], NOW)
    assert report.at_risk == []
    assert report.current == 1
    assert report.current_value == Money.rupees(80000)


def test_part_payment_leaves_only_the_balance_at_risk():
    report = scan([invoice(amount=80000, paid=30000)], NOW)
    assert report.total_outstanding == Money.rupees(50000)


def test_disputed_invoices_are_at_risk_but_never_chaseable():
    """The money is genuinely at risk, so it belongs in the total. Chasing it
    is not permitted, so it must not be in what we claim we could collect."""
    report = scan([invoice(disputed=True)], NOW)
    assert report.total_outstanding == Money.rupees(80000)
    assert report.chaseable_value == Money.zero()
    assert report.disputed_value == Money.rupees(80000)
    assert report.expected_recovery == Money.zero()


def test_a_live_promise_suppresses_chasing_but_a_lapsed_one_does_not():
    live = scan([invoice(promise_days=5)], NOW)
    assert live.chaseable_value == Money.zero()

    lapsed = scan([invoice(promise_days=-5)], NOW)
    assert lapsed.chaseable_value == Money.rupees(80000)


def test_collectability_decays_with_age():
    """The whole reason for aging a ledger. If a 20-day-old invoice and a
    200-day-old one were priced the same, the buckets would be decoration."""
    recent = scan([invoice(days_overdue=10)], NOW).expected_recovery
    old = scan([invoice(days_overdue=200)], NOW).expected_recovery
    assert recent > old * 3


def test_weighted_age_is_money_weighted_not_invoice_weighted():
    """One large ancient invoice is the receivable that matters. Counting
    invoices equally would let a crowd of small recent ones bury it."""
    invoices = [
        invoice(days_overdue=5, amount=1000, iid="inv_small_1"),
        invoice(days_overdue=5, amount=1000, iid="inv_small_2"),
        invoice(days_overdue=300, amount=500000, iid="inv_big"),
    ]
    report = scan(invoices, NOW)
    assert report.weighted_days_overdue > 290


# -- the collections ladder -------------------------------------------------


def test_a_disputed_invoice_goes_to_a_person_not_a_dunning_run():
    actions = receivable_actions(invoice(disputed=True), NOW)
    assert [a.type for a in actions] == [ActionType.ESCALATE_TO_HUMAN]


def test_a_live_promise_produces_a_wait_rather_than_silence():
    """Recording the wait is the point. Proposing nothing would be
    indistinguishable in the audit trail from never having looked."""
    actions = receivable_actions(invoice(promise_days=6), NOW)
    assert [a.type for a in actions] == [ActionType.WAIT]


def test_a_recently_late_invoice_gets_exactly_one_reminder():
    """Most of these are an AP cycle that has not turned over. A second and
    third chase into that bracket buys almost nothing and spends a buyer
    relationship to get it."""
    actions = receivable_actions(invoice(days_overdue=12), NOW)
    assert len(actions) == 1
    assert actions[0].type is ActionType.SEND_DUNNING


def test_middle_aged_invoices_are_offered_a_split():
    """By sixty days the buyers who have not paid are increasingly ones who
    cannot pay in one piece, and dunning harder cannot move that blocker."""
    for days in (45, 75):
        kinds = [a.type for a in receivable_actions(invoice(days_overdue=days), NOW)]
        assert ActionType.OFFER_PART_PAYMENT in kinds
        assert ActionType.SEND_DUNNING in kinds


def test_the_oldest_bracket_is_handed_to_a_person():
    """Ninety-day paper is where settlement and write-off decisions live, and
    those are not automation's to make."""
    actions = receivable_actions(invoice(days_overdue=140), NOW)
    assert [a.type for a in actions] == [ActionType.ESCALATE_TO_HUMAN]


def test_a_part_payment_offer_asks_for_part_of_the_balance():
    """An offer to split that asks for the whole amount is not an offer, and
    one that asks for a token is a discount nobody agreed to."""
    inv = invoice(days_overdue=75, amount=80000, paid=20000)
    offer = next(
        a for a in receivable_actions(inv, NOW)
        if a.type is ActionType.OFFER_PART_PAYMENT
    )
    assert offer.amount_paise == round(inv.outstanding.paise * PART_PAYMENT_OFFER_SHARE)
    assert 0 < offer.amount_paise < inv.outstanding.paise


def test_nothing_is_proposed_for_a_settled_or_current_invoice():
    settled = invoice()
    settled.status = InvoiceStatus.PAID
    assert receivable_actions(settled, NOW) == []
    assert receivable_actions(invoice(days_overdue=-4), NOW) == []


# -- what collecting an invoice may claim -----------------------------------


def test_an_invoice_that_would_have_been_paid_anyway_recovers_nothing():
    """The single most important test on this surface. An agent that mails a
    buyer whose AP cycle was about to pay, and then books the payment, has
    measured the AP cycle and called it recovery."""
    inv = invoice()
    rec = truth(chased=True, unprompted=True)
    action = receivable_actions(inv, NOW)[0]
    result = executor(inv, rec).execute(action, NOW + timedelta(hours=12))

    assert result.outcome is Outcome.NO_EFFECT
    assert result.recovered == Money.zero()
    assert "own cycle" in result.detail
    assert result.cost, "and it still cost the merchant a contact"


def test_an_incremental_invoice_recovers_its_balance():
    inv = invoice(amount=80000, paid=20000)
    action = receivable_actions(inv, NOW)[0]
    result = executor(inv, truth()).execute(action, NOW + timedelta(hours=12))

    assert result.outcome is Outcome.RECOVERED
    assert result.recovered == Money.rupees(60000), "the balance, not the face value"


def test_a_cashflow_constrained_buyer_answers_only_an_offer():
    """Dunning somebody harder for a balance they do not have is the
    collections equivalent of retrying a stolen card: more pressure applied to
    a blocker that pressure cannot move."""
    inv = invoice(days_overdue=75, amount=80000)
    rec = truth(constrained=True, share=0.6)

    actions = receivable_actions(inv, NOW)
    dunning = next(a for a in actions if a.type is ActionType.SEND_DUNNING)
    offer = next(a for a in actions if a.type is ActionType.OFFER_PART_PAYMENT)

    ex = executor(inv, rec)
    assert ex.execute(dunning, NOW + timedelta(hours=12)).outcome is Outcome.NO_EFFECT
    landed = ex.execute(offer, NOW + timedelta(days=4))
    assert landed.outcome is Outcome.RECOVERED
    # They could have found 60%, but nobody asked for more than half.
    assert landed.recovered == Money.rupees(40000)


def test_a_buyer_pays_no_more_than_they_can_find():
    """The other side of the same bound. Asking half of a buyer who has a
    third does not produce half."""
    inv = invoice(days_overdue=75, amount=90000)
    rec = truth(constrained=True, share=1 / 3)
    offer = next(
        a for a in receivable_actions(inv, NOW)
        if a.type is ActionType.OFFER_PART_PAYMENT
    )
    landed = executor(inv, rec).execute(offer, NOW + timedelta(days=4))
    assert landed.recovered == Money.rupees(30000)


def test_offering_a_split_to_a_buyer_who_could_pay_in_full_collects_less():
    """The offer is not free, and the merchant eats the remainder. Modelling it
    as strictly better than a reminder would make the playbook prefer it
    everywhere for no reason."""
    inv = invoice(days_overdue=75, amount=100000)
    rec = truth(constrained=False, share=0.9)
    actions = receivable_actions(inv, NOW)

    full = executor(inv, rec).execute(
        next(a for a in actions if a.type is ActionType.SEND_DUNNING),
        NOW + timedelta(hours=12),
    )
    split = executor(inv, rec).execute(
        next(a for a in actions if a.type is ActionType.OFFER_PART_PAYMENT),
        NOW + timedelta(days=4),
    )
    assert full.recovered == Money.rupees(100000)
    # They pay the instalment that was asked for, not what they could afford.
    assert split.recovered == Money.rupees(100000 * PART_PAYMENT_OFFER_SHARE)


def test_chasing_after_the_invoice_goes_cold_recovers_nothing():
    inv = invoice()
    rec = truth(days=5.0)
    action = receivable_actions(inv, NOW)[0]
    result = executor(inv, rec).execute(action, NOW + timedelta(days=40))

    assert result.outcome is Outcome.NO_EFFECT
    assert "cold" in result.detail


def test_an_invoice_cannot_be_collected_twice():
    inv = invoice(days_overdue=75)
    ex = executor(inv, truth())
    actions = receivable_actions(inv, NOW)

    first = ex.execute(actions[0], NOW + timedelta(hours=12))
    second = ex.execute(actions[1], NOW + timedelta(days=4))
    assert first.outcome is Outcome.RECOVERED
    assert second.outcome is Outcome.NO_EFFECT
    assert "wasted" in second.detail


def test_an_invoice_has_nothing_to_re_present():
    """A B2B receivable has no instrument on file. A retry against one is a
    category error, and reporting it as a decline would hide that."""
    inv = invoice()
    retry = Action(
        type=ActionType.RETRY_PAYMENT, subject_id=inv.id,
        scheduled_at=NOW, rationale="wrong tool for this surface",
    )
    result = executor(inv, truth()).execute(retry, NOW)
    assert result.outcome is Outcome.NO_EFFECT
    assert "no instrument" in result.detail


def test_a_disputed_invoice_has_no_latent_outcome_at_all():
    """Modelling a probability that dunning settles a dispute would invent a
    reward for exactly the behaviour dispute_freeze exists to refuse."""
    scenario = generate(SimConfig(seed=5, days=2, orders_per_day=400))
    disputed = [i.id for i in scenario.invoices if i.disputed_at is not None]
    assert disputed, "the generator should produce some disputes"
    for iid in disputed:
        assert iid not in scenario.invoice_recovery


# -- the rules ---------------------------------------------------------------


def test_the_engine_refuses_to_chase_a_dispute():
    inv = invoice(disputed=True)
    dunning = receivable_actions(invoice(days_overdue=45), NOW)[0]
    ruling = ENGINE.evaluate(dunning, PolicyContext(now=NOW, customer=buyer(), invoice=inv))
    assert ruling.disposition is Disposition.DENY
    assert ruling.blocking_rule == "dispute_freeze"


def test_the_engine_refuses_to_chase_inside_a_promise():
    inv = invoice(promise_days=6)
    dunning = receivable_actions(invoice(days_overdue=45), NOW)[0]
    ruling = ENGINE.evaluate(dunning, PolicyContext(now=NOW, customer=buyer(), invoice=inv))
    assert ruling.disposition is Disposition.DENY
    assert ruling.blocking_rule == "promise_to_pay"


def test_a_buyer_is_a_person_and_the_consent_rule_applies_to_them():
    """Buyers used to have no customer record, which meant every collections
    contact was denied for want of one. An AP contact is a named individual and
    the rules that protect a consumer protect them too."""
    inv = invoice()
    dunning = receivable_actions(inv, NOW)[0]

    allowed = ENGINE.evaluate(dunning, PolicyContext(now=NOW, customer=buyer(), invoice=inv))
    assert allowed.allowed

    opted_out = buyer(opted_out_at=NOW - timedelta(days=1))
    refused = ENGINE.evaluate(dunning, PolicyContext(now=NOW, customer=opted_out, invoice=inv))
    assert refused.disposition is Disposition.DENY
    assert refused.blocking_rule == "contact_consent"


def test_the_per_subject_cap_does_not_protect_a_buyer_with_many_invoices():
    """The hazard this surface introduces, stated as a test. Four invoices,
    three contacts each, and every per-invoice cap is scrupulously observed
    while one person is written to twelve times."""
    cfg = PolicyConfig()
    prior = [
        ContactRecord(
            id=new_id("contact"), customer_id="buyer_1", channel=Channel.EMAIL,
            at=NOW - timedelta(days=n), subject_ref=f"inv_{n}",
        )
        for n in range(cfg.max_contacts_per_customer)
    ]
    inv = invoice(iid="inv_new")
    dunning = receivable_actions(inv, NOW)[0]

    # Per-subject: this invoice has never been contacted about, so the
    # frequency rule sees nothing and would wave it through.
    ctx = PolicyContext(
        now=NOW, customer=buyer(), invoice=inv,
        contacts=[], customer_contacts=prior,
    )
    ruling = ENGINE.evaluate(dunning, ctx)
    assert ruling.disposition is Disposition.DENY
    assert ruling.blocking_rule == "contact_fatigue"


def test_the_fatigue_rule_ignores_contacts_outside_the_window():
    cfg = PolicyConfig()
    stale = [
        ContactRecord(
            id=new_id("contact"), customer_id="buyer_1", channel=Channel.EMAIL,
            at=NOW - timedelta(days=cfg.contact_window_days + 5 + n),
            subject_ref=f"inv_{n}",
        )
        for n in range(cfg.max_contacts_per_customer + 3)
    ]
    inv = invoice(iid="inv_new")
    dunning = receivable_actions(inv, NOW)[0]
    ctx = PolicyContext(now=NOW, customer=buyer(), invoice=inv, customer_contacts=stale)
    assert ENGINE.evaluate(dunning, ctx).allowed


def test_an_escalation_is_never_blocked_by_a_contact_rule():
    """Handing a disputed or ancient invoice to a person contacts nobody, and
    a stopping rule that stopped the stopping action would be a bad joke."""
    inv = invoice(disputed=True)
    escalation = receivable_actions(inv, NOW)[0]
    ctx = PolicyContext(now=NOW, customer=buyer(opted_out_at=NOW), invoice=inv)
    assert ENGINE.evaluate(escalation, ctx).allowed


def test_naive_collections_chases_what_it_must_not():
    """The baseline is not a strawman -- pulling the aged debtors report and
    dunning everything on it is the default behaviour of most collections
    tooling. What it cannot do is read the two suppressions."""
    invoices = [
        invoice(iid="inv_ok", days_overdue=40),
        invoice(iid="inv_disputed", days_overdue=40, disputed=True),
        invoice(iid="inv_promised", days_overdue=40, promise_days=6),
    ]
    chased = {a.subject_id for a in naive_invoice_chase(invoices, NOW)}
    assert chased == {"inv_ok", "inv_disputed", "inv_promised"}


def test_naive_collections_leaves_settled_and_current_invoices_alone():
    """Even the naive arm is not a liability generator. It over-chases the
    overdue ledger, which is the realistic failure, rather than everything."""
    settled = invoice(iid="inv_paid")
    settled.status = InvoiceStatus.PAID
    invoices = [settled, invoice(iid="inv_current", days_overdue=-5)]
    assert naive_invoice_chase(invoices, NOW) == []


# -- end to end --------------------------------------------------------------


@pytest.fixture(scope="module")
def arms():
    scenario = generate(SimConfig(seed=7, days=3, orders_per_day=3000))
    result, _ = run(scenario, offline=True, model=None)
    return scenario, {a.name: a for a in result}


def test_the_receivables_surface_is_actually_measured(arms):
    _, by_name = arms
    assert by_name["backstop"].ledger.on(Surface.RECEIVABLE).recovered


def test_the_policed_arm_breaks_no_receivables_rules(arms):
    _, by_name = arms
    broken = [v for v in by_name["backstop"].violations if v.surface is Surface.RECEIVABLE]
    assert broken == []


def test_naive_collections_breaks_the_rules_the_surface_exists_to_enforce(arms):
    """Dunning the aged debtors report is what most collections tooling does.
    It chases disputes and it chases buyers who already committed to a date."""
    _, by_name = arms
    broken = {
        v.rule_id for v in by_name["naive-retry"].violations
        if v.surface is Surface.RECEIVABLE
    }
    assert "dispute_freeze" in broken
    assert "promise_to_pay" in broken


def test_backstop_collects_with_far_less_contact_and_no_illegal_revenue(arms):
    """What holds on this surface regardless of the batch.

    Deliberately *not* an assertion that the policed arm out-collects naive.
    It usually does, and it does not always, and the reason is the trade in
    the test below: refusing to let automation work ninety-day paper hands
    naive a bracket for free. Asserting a win here would be asserting
    something the design does not promise and the seeds do not support.
    """
    _, by_name = arms
    naive = by_name["naive-retry"].ledger.on(Surface.RECEIVABLE)
    policed = by_name["backstop"].ledger.on(Surface.RECEIVABLE)

    assert policed.contacts_sent < naive.contacts_sent / 2
    assert policed.recovered_in_violation == Money.zero()
    assert naive.recovered_in_violation, (
        "and part of what naive took was money a merchant could not keep"
    )
    assert policed.cost < naive.cost


def test_the_oldest_bracket_is_revenue_the_policy_engine_gives_up(arms):
    """The price of the ladder's last rung, stated as a test.

    Backstop hands ninety-day paper to a person and therefore books nothing
    from it, while naive dunns it and collects. That is the largest single
    thing policing costs on this surface, it is a deliberate choice about who
    decides on settlement and write-off, and it should be visible rather than
    buried in a net figure.
    """
    scenario, by_name = arms
    invoices = {i.id: i for i in scenario.invoices}

    def collected_from_oldest(arm):
        total = Money.zero()
        for e in arm.ledger.on(Surface.RECEIVABLE).executed:
            if not e.execution.is_recovery:
                continue
            inv = invoices[e.action.subject_id]
            if bucket_for(inv.days_overdue(scenario.ends_at)) is AgingBucket.DAYS_90_PLUS:
                total += e.execution.recovered
        return total

    assert collected_from_oldest(by_name["backstop"]) == Money.zero()
    assert collected_from_oldest(by_name["naive-retry"]), (
        "naive should be taking money out of the bracket Backstop refuses to work"
    )


def test_no_person_hears_from_backstop_more_than_the_cap(arms):
    """The per-person guarantee, measured end to end across all three surfaces.

    Not per invoice, not per order, not per mandate: per human being. A
    customer with a failed order, a lapsed mandate and two overdue invoices is
    one person, and every per-subject cap can be scrupulously observed while
    they are written to a dozen times.
    """
    scenario, by_name = arms
    cfg = PolicyConfig()
    owner = {}
    for order in scenario.orders:
        owner[order.id] = order.customer_id
    for sub in scenario.subscriptions:
        owner[sub.id] = sub.customer_id
    for inv in scenario.invoices:
        owner[inv.id] = inv.buyer_id

    def worst_burst(arm):
        by_person: dict[str, list[datetime]] = {}
        for e in arm.ledger.executed:
            if e.action.is_contact:
                by_person.setdefault(owner[e.action.subject_id], []).append(
                    e.action.scheduled_at
                )
        window = timedelta(days=cfg.contact_window_days)
        worst = 0
        for times in by_person.values():
            times.sort()
            start = 0
            for end in range(len(times)):
                while times[end] - times[start] > window:
                    start += 1
                worst = max(worst, end - start + 1)
        return worst

    assert worst_burst(by_name["backstop"]) <= cfg.max_contacts_per_customer
    assert worst_burst(by_name["naive-retry"]) > cfg.max_contacts_per_customer, (
        "the baseline should be visibly exceeding it, or the rule guards nothing"
    )


def test_no_invoice_is_collected_twice(arms):
    _, by_name = arms
    for arm in by_name.values():
        won = [
            e.action.subject_id for e in arm.ledger.on(Surface.RECEIVABLE).executed
            if e.execution.is_recovery
        ]
        assert len(won) == len(set(won))


def test_collections_never_out_recovers_what_was_incrementally_available(arms):
    """The ceiling. No arm may collect more than the money that would not have
    arrived on its own -- if one does, the counterfactual has stopped being
    applied and every number on this surface is inflated."""
    scenario, by_name = arms
    outstanding = {i.id: i.outstanding for i in scenario.invoices}
    ceiling = Money.zero()
    for rec in scenario.invoice_recovery.values():
        if rec.is_incremental:
            ceiling += outstanding[rec.invoice_id]

    for arm in by_name.values():
        assert arm.ledger.on(Surface.RECEIVABLE).recovered <= ceiling


def test_the_naive_arm_pays_for_every_wasted_chase(arms):
    """An arm that dunned a ledger of invoices that were going to be paid
    anyway should be charged for the postage, or the comparison flatters it."""
    _, by_name = arms
    naive = by_name["naive-retry"].ledger.on(Surface.RECEIVABLE)
    assert naive.wasted_actions > naive.orders_recovered
    assert naive.cost > by_name["backstop"].ledger.on(Surface.RECEIVABLE).cost
