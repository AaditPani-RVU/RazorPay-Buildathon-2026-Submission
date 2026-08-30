"""Turning a diagnosis into proposed actions.

Four paths, because revenue at risk has four shapes and only one of them
needs a model.

*   **Incident-driven.** A detected cluster with a diagnosis gets a strategy
    from the LLM: for each class of failure inside it, what to do, when, and
    over how many attempts. This is judgment work -- the same decline code
    warrants a different response during an issuer outage than during an
    authentication drop-off -- so a model earns its place.
*   **The long tail.** Most failed payments belong to no incident at all. A
    card expired on an ordinary Tuesday. There is nothing to diagnose and
    nothing to reason about, so these are handled straight off the decline
    taxonomy. Calling a model 14,000 times to re-derive a lookup table would be
    slower, costlier and less reliable.
*   **The book.** Lapsed mandates are neither. Nothing broke and nothing was
    declined -- an authorisation stopped being able to collect, and it stays
    that way until somebody re-registers it. There is no event to reason about
    at all, so this is a scan and a lookup. What varies is not what to do but
    whether it should be done, which is a policy question rather than a
    planning one.
*   **The ledger.** Overdue invoices are a third non-event. They fail by
    *ageing* rather than by breaking, so what drives the response is not a
    cause but a duration, and the ladder from a reminder through a
    part-payment offer to a human is set by how old the money is.

The model plans per *cluster*, never per order. It proposes a shape; expansion
to concrete actions is deterministic. Both paths then go through the policy
engine one action at a time, so neither can escape the rules -- and the tail
path being deterministic is not a loophole, it is the same rules applied to a
planner that happens not to be a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from backstop.detect.correlate import RiskCluster
from backstop.detect.detector import segments_for
from backstop.detect.receivables import AgingBucket, bucket_for
from backstop.diagnose.diagnoser import Diagnosis
from backstop.domain.actions import Action, ActionType
from backstop.domain.declines import DeclineCode, RetryClass, RootCause
from backstop.domain.entities import (
    Channel,
    Invoice,
    MandateStatus,
    Order,
    Subscription,
)
from backstop.llm import LLMClient, LLMError

# --------------------------------------------------------------------------
# What the model is allowed to propose
# --------------------------------------------------------------------------


class DeclineClassPlan(BaseModel):
    """What to do with one class of failure inside a cluster."""

    retry_class: RetryClass
    action: ActionType
    delay_hours: float = Field(
        ge=0, le=336, description="Hours after the failure to act. Never zero for a retry."
    )
    channel: Channel | None = Field(
        default=None, description="Required for contact actions, forbidden otherwise."
    )
    max_attempts: int = Field(ge=0, le=4, description="How many times to try this at most.")
    rationale: str = Field(max_length=300)


class RecoveryStrategy(BaseModel):
    reasoning: str = Field(default="", max_length=600)
    plans: list[DeclineClassPlan] = Field(min_length=1, max_length=5)


SYSTEM_PROMPT = """You are planning revenue recovery for one diagnosed incident
at a payment gateway.

You are given the root cause and the mix of failures inside the incident. For
each class of failure present, choose one action, when to take it, and how many
attempts at most.

The classes mean specific things and constrain what can work:

  transient             infrastructure blip; the instrument is fine. Retrying
                        works, but only once the fault has cleared.
  soft_retryable        funds or limits. May succeed later. Space attempts by
                        a day or more; salary timing is the biggest lever.
  user_action_required  cannot succeed without the customer. Retrying is
                        pointless. The only route is contacting them.
  hard_decline          the instrument is permanently dead. A new one is
                        needed, so contact is the only route.
  fraud_block           never retry, never contact. Escalate to risk.

Choose delays that respect the cause. Retrying into an outage that has not
finished manufactures failures. Contacting somebody about an issuer outage that
resolved itself annoys a customer whose payment would have worked anyway.

A separate deterministic policy engine will vet every action you propose and
refuse anything unsafe, so propose what you believe is correct rather than what
you think will be permitted. Do not propose an action you cannot justify from
the root cause you were given."""


# --------------------------------------------------------------------------
# The tail: deterministic, straight off the taxonomy
# --------------------------------------------------------------------------

#: What to do with an isolated failure that belongs to no incident. Retry
#: classes that cannot be fixed by re-presenting go to contact; fraud goes to
#: risk; the rest get one spaced retry. Conservative by construction -- the
#: policy engine will still trim it.
TAIL_PLAYBOOK: dict[RetryClass, tuple[ActionType, float, Channel | None, int]] = {
    RetryClass.TRANSIENT: (ActionType.RETRY_PAYMENT, 2.0, None, 2),
    RetryClass.SOFT_RETRYABLE: (ActionType.RETRY_PAYMENT, 26.0, None, 2),
    RetryClass.USER_ACTION_REQUIRED: (ActionType.SEND_DUNNING, 3.0, Channel.EMAIL, 2),
    RetryClass.HARD_DECLINE: (ActionType.SEND_DUNNING, 4.0, Channel.EMAIL, 1),
    RetryClass.FRAUD_BLOCK: (ActionType.ESCALATE_TO_RISK, 0.0, None, 1),
}


#: Mandate failures need their own verbs. A lapsed authorisation is not fixed
#: by a payment reminder -- the customer has to re-authorise, which is a
#: different ask with a different success rate. Keyed by code rather than by
#: retry class because the class cannot tell an expiry from an abandoned OTP.
MANDATE_PLAYBOOK: dict[DeclineCode, tuple[ActionType, float, Channel | None, int]] = {
    DeclineCode.MANDATE_EXPIRED: (
        ActionType.REQUEST_MANDATE_REREGISTRATION, 4.0, Channel.EMAIL, 2),
    DeclineCode.MANDATE_NOT_REGISTERED: (
        ActionType.REQUEST_MANDATE_REREGISTRATION, 6.0, Channel.EMAIL, 1),
    # Revoked is a decision, not a lapse. Recovery hands it to a person rather
    # than asking a customer to undo something they chose. The policy engine
    # refuses the alternative anyway; proposing it here would just log a veto.
    DeclineCode.MANDATE_REVOKED: (
        ActionType.ESCALATE_TO_HUMAN, 0.0, None, 1),
    # A pause usually ends on its own. Waiting costs nothing and asking a
    # customer to un-pause something they just paused rarely lands.
    DeclineCode.MANDATE_PAUSED: (
        ActionType.WAIT, 168.0, None, 1),
}


def tail_actions(order: Order) -> list[Action]:
    """Actions for a failure with no diagnosed incident behind it."""
    code = order.last_decline
    last = order.last_attempt
    if code is None or last is None:
        return []
    if code in MANDATE_PLAYBOOK:
        kind, delay, channel, attempts = MANDATE_PLAYBOOK[code]
    else:
        kind, delay, channel, attempts = TAIL_PLAYBOOK[code.retry_class]
    out: list[Action] = []
    for n in range(attempts):
        # Spacing widens with each attempt: a failure that survived one retry
        # is less likely to be fixed by an immediate second.
        at = last.at + timedelta(hours=delay * (n + 1) * (1.6**n))
        out.append(
            Action(
                type=kind,
                subject_id=order.id,
                scheduled_at=at,
                channel=channel,
                rationale=f"tail playbook for {code.value} ({code.retry_class.value})",
            )
        )
    return out


# --------------------------------------------------------------------------
# The recurring path
# --------------------------------------------------------------------------
#
# Lapsed mandates need a third path, because they are neither an incident nor
# a failed order. Nothing spiked and nothing was declined -- the authorisation
# is simply dead, and it will stay dead until somebody re-registers it. There
# is no evidence bundle to reason over and no judgement call to make, so like
# the long tail this is a lookup rather than a model call. What varies is not
# what to do but whether to do it at all, and that is a policy question.

#: What to propose for each mandate state, and how long to leave it first.
#:
#: The delays are the interesting column, and they are a genuine trade-off
#: rather than caution for its own sake. A paused mandate gets three days of
#: silence first, because a third of them resume unprompted and mailing those
#: customers spends money to be told something that was going to happen
#: anyway. But the delay is short, because customers go cold: waiting is only
#: worth what it saves, and past a few days it costs reach faster than it
#: saves postage. An expired mandate is chased promptly -- nothing about it
#: self-corrects -- and gets a second ask, because one email is easy to miss.
MANDATE_RECOVERY_PLAYBOOK: dict[MandateStatus, tuple[ActionType, float, Channel | None, int]] = {
    MandateStatus.EXPIRED: (
        ActionType.REQUEST_MANDATE_REREGISTRATION, 24.0, Channel.EMAIL, 2),
    MandateStatus.NOT_REGISTERED: (
        ActionType.REQUEST_MANDATE_REREGISTRATION, 48.0, Channel.EMAIL, 2),
    MandateStatus.PAUSED: (
        ActionType.REQUEST_MANDATE_REREGISTRATION, 72.0, Channel.EMAIL, 1),
    # Not a lapse. A person decides whether to ask somebody who cancelled on
    # purpose, and the policy engine refuses to let automation make that call
    # anyway -- so proposing anything else here would be proposing a veto.
    MandateStatus.REVOKED: (
        ActionType.ESCALATE_TO_HUMAN, 0.0, None, 1),
}

#: Days between the two re-registration attempts where there are two.
#: Recurring recovery is a patient problem: the customer has no deadline, so
#: neither should the sequence.
REREGISTRATION_SPACING_DAYS = 9.0


def mandate_actions(sub: Subscription, now: datetime) -> list[Action]:
    """Actions for one mandate that cannot currently collect.

    `now` is when the book was scanned. Unlike a failed payment there is no
    failure moment to schedule from -- an expired mandate has no event, only a
    state -- so everything is relative to the scan.
    """
    if sub.cancelled_at is not None or sub.is_chargeable:
        return []
    plan = MANDATE_RECOVERY_PLAYBOOK.get(sub.mandate_status)
    if plan is None:
        return []
    kind, delay_hours, channel, attempts = plan
    out: list[Action] = []
    for n in range(attempts):
        at = now + timedelta(hours=delay_hours) + timedelta(days=REREGISTRATION_SPACING_DAYS * n)
        out.append(
            Action(
                type=kind,
                subject_id=sub.id,
                scheduled_at=at,
                channel=channel,
                rationale=(
                    f"mandate playbook for {sub.mandate_status.value}: "
                    f"{sub.annual_value} per year cannot be collected"
                ),
            )
        )
    return out


# --------------------------------------------------------------------------
# The receivables path
# --------------------------------------------------------------------------
#
# A fourth path, because an overdue invoice fails in a fourth way. A payment
# fails at a moment. A mandate fails into a state. A receivable fails by
# getting older -- nothing breaks, nothing flips, the money simply does not
# arrive and every week it is slightly less likely to. So what selects the
# response here is not a cause or a status but an *age*, and the playbook is
# an escalation ladder rather than a lookup on what went wrong.


#: The collections ladder, by aging bracket.
#:
#: Three judgements are encoded here and each one gives something up.
#:
#: The recent bracket gets **one** reminder and no more. Most invoices that
#: are a fortnight late are not a collections problem at all -- they are an
#: accounts-payable cycle that has not turned over yet -- and the money
#: arrives whether or not anybody writes. A second and third chase into that
#: bracket buys almost nothing and spends a buyer relationship to get it.
#:
#: The middle brackets pair a reminder with a **part-payment offer** a few
#: days later, because by sixty days the buyers who have not paid are
#: increasingly ones who cannot pay in one piece. Dunning somebody harder for
#: a balance they do not have is the collections equivalent of retrying a
#: stolen card: more pressure applied to a blocker that pressure cannot move.
#: The offer costs a contact and recovers a fraction rather than the whole,
#: and it is still worth more than a fourth email.
#:
#: The oldest bracket is handed to a **person**, and this is the one that
#: costs real money. Ninety-day paper is where collections, settlement and
#: write-off decisions live, and those are not automation's to make. An agent
#: that decides on its own to keep chasing a buyer who has stopped answering
#: for three months is one that will eventually chase somebody into a
#: complaint, a legal letter, or a relationship a sales team spent years on.
#:
#: A ladder is a schedule, not a decision tree, and that costs something too:
#: where a reminder lands first, the offer behind it is spent on an invoice
#: already collected. The executor reports those as wasted and charges for
#: them rather than hiding them, because the honest alternative -- re-planning
#: after each outcome -- is a different architecture, and pretending a fixed
#: plan has the reach of a conditional one would flatter this one.
RECEIVABLES_LADDER: dict[AgingBucket, list[tuple[ActionType, float, Channel | None]]] = {
    AgingBucket.DAYS_1_30: [
        (ActionType.SEND_DUNNING, 24.0, Channel.EMAIL),
    ],
    AgingBucket.DAYS_31_60: [
        (ActionType.SEND_DUNNING, 12.0, Channel.EMAIL),
        (ActionType.OFFER_PART_PAYMENT, 12.0 + 24 * 7, Channel.EMAIL),
    ],
    AgingBucket.DAYS_61_90: [
        (ActionType.SEND_DUNNING, 12.0, Channel.EMAIL),
        (ActionType.OFFER_PART_PAYMENT, 12.0 + 24 * 4, Channel.EMAIL),
    ],
    AgingBucket.DAYS_90_PLUS: [
        (ActionType.ESCALATE_TO_HUMAN, 0.0, None),
    ],
}

#: What share of the outstanding balance a part-payment offer proposes as a
#: first instalment. Deliberately not the whole: an offer to split that asks
#: for the full amount is not an offer, and one that asks for a token is a
#: discount the merchant did not agree to.
PART_PAYMENT_OFFER_SHARE = 0.5


def receivable_actions(inv: Invoice, now: datetime) -> list[Action]:
    """Actions for one invoice, chosen by how overdue it is.

    Two invoices get an inert action rather than nothing, and the difference
    matters for the audit trail. A disputed invoice and one under a live
    promise are both *seen* and both deliberately not chased; recording an
    escalation or a wait says so, where proposing nothing would look identical
    to never having scanned them.

    `now` is when the ledger was aged. An invoice has no failure event to
    schedule from -- only an issue date and a due date -- so everything is
    relative to the scan.
    """
    if inv.is_settled or not inv.is_overdue(now):
        return []

    if inv.disputed_at is not None:
        # A dispute is a disagreement about whether the money is owed, and no
        # amount of dunning settles one. `dispute_freeze` stands behind this,
        # but the planner should not be proposing work the engine has to catch.
        return [
            Action(
                type=ActionType.ESCALATE_TO_HUMAN,
                subject_id=inv.id,
                scheduled_at=now,
                rationale=f"invoice disputed since {inv.disputed_at:%Y-%m-%d}; a person owns it",
            )
        ]

    if inv.promise is not None and inv.promise.is_live(now):
        return [
            Action(
                type=ActionType.WAIT,
                subject_id=inv.id,
                scheduled_at=now,
                rationale=(
                    f"buyer committed to pay by {inv.promise.promised_for:%Y-%m-%d}; "
                    "chasing inside a promise is how a cooperative buyer stops being one"
                ),
            )
        ]

    ladder = RECEIVABLES_LADDER.get(bucket_for(inv.days_overdue(now)))
    if not ladder:
        return []

    days = inv.days_overdue(now)
    out: list[Action] = []
    for kind, delay_hours, channel in ladder:
        out.append(
            Action(
                type=kind,
                subject_id=inv.id,
                scheduled_at=now + timedelta(hours=delay_hours),
                channel=channel,
                amount_paise=(
                    round(inv.outstanding.paise * PART_PAYMENT_OFFER_SHARE)
                    if kind is ActionType.OFFER_PART_PAYMENT
                    else None
                ),
                rationale=(
                    f"collections ladder at {days}d overdue: "
                    f"{inv.outstanding} outstanding"
                ),
            )
        )
    return out


# --------------------------------------------------------------------------
# The incident path
# --------------------------------------------------------------------------


def orders_in_cluster(cluster: RiskCluster, orders: list[Order]) -> list[Order]:
    """Failed orders whose last attempt sits inside this cluster's slice."""
    out = []
    for o in orders:
        if o.is_captured:
            continue
        last = o.last_attempt
        if last is None or not (cluster.starts_at <= last.at <= cluster.ends_at):
            continue
        if cluster.segment in segments_for(last):
            out.append(o)
    return out


def expand(
    strategy: RecoveryStrategy, orders: list[Order], *, outage_until: datetime | None = None
) -> list[Action]:
    """Turn a per-class strategy into concrete per-order actions."""
    by_class = {p.retry_class: p for p in strategy.plans}
    out: list[Action] = []
    for order in orders:
        code, last = order.last_decline, order.last_attempt
        if code is None or last is None:
            continue
        plan = by_class.get(code.retry_class)
        if plan is None or plan.max_attempts == 0:
            continue
        for n in range(plan.max_attempts):
            at = last.at + timedelta(hours=plan.delay_hours * (n + 1))
            if outage_until is not None and at < outage_until:
                at = outage_until
            out.append(
                Action(
                    type=plan.action,
                    subject_id=order.id,
                    scheduled_at=at,
                    channel=plan.channel,
                    rationale=plan.rationale[:400],
                )
            )
    return out


@dataclass
class StrategyResult:
    cluster_id: str
    strategy: RecoveryStrategy | None
    repaired: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.strategy is not None


class Planner:
    """Asks the model for a recovery shape, once per diagnosed cluster."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def plan(
        self, cluster: RiskCluster, diagnosis: Diagnosis, orders: list[Order]
    ) -> StrategyResult:
        """Propose a strategy, or record that none came back.

        A failure here means the cluster falls through to the tail playbook,
        which is the safe default: a deterministic response derived from the
        decline taxonomy rather than an improvised one.
        """
        mix: dict[RetryClass, int] = {}
        for o in orders:
            if o.last_decline:
                k = o.last_decline.retry_class
                mix[k] = mix.get(k, 0) + 1
        if not mix:
            return StrategyResult(cluster.id, None, error="no failed orders in cluster")

        lines = "\n".join(f"  {k.value:<22} {n:>5} orders" for k, n in sorted(mix.items()))
        user = (
            f"ROOT CAUSE: {diagnosis.root_cause.value} (confidence {diagnosis.confidence:.2f})\n"
            f"LOCUS: {diagnosis.locus}\n"
            f"EVIDENCE:\n" + "\n".join(f"  - {e}" for e in diagnosis.key_evidence) + "\n\n"
            f"WINDOW: {cluster.starts_at:%Y-%m-%d %H:%M} to {cluster.ends_at:%H:%M} UTC\n"
            f"MONEY AT RISK: {cluster.money_at_risk}\n\n"
            f"FAILURES BY CLASS:\n{lines}\n\n"
            "Plan one action per class present above."
        )
        try:
            result = self.client.structured(
                system=SYSTEM_PROMPT, user=user, schema=RecoveryStrategy
            )
        except LLMError as err:
            return StrategyResult(cluster.id, None, error=str(err))
        return StrategyResult(cluster.id, result.value, repaired=result.repaired)


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def do_nothing(orders: list[Order]) -> list[Action]:
    """The floor. Whatever this recovers, recovery did not cause."""
    return []


def naive_retry(orders: list[Order], *, retries: int = 3, spacing_hours: float = 1.0) -> list[Action]:
    """What gets built without a policy engine: retry everything, mail everyone.

    This is not a strawman. It is the obvious implementation, it is what most
    recovery scripts actually do, and it recovers real money -- which is the
    point of measuring it. The interesting question is not whether it works but
    what it costs in contacts, wasted attempts and rules broken.
    """
    out: list[Action] = []
    for order in orders:
        last = order.last_attempt
        if last is None:
            continue
        for n in range(retries):
            out.append(
                Action(
                    type=ActionType.RETRY_PAYMENT,
                    subject_id=order.id,
                    scheduled_at=last.at + timedelta(hours=spacing_hours * (n + 1)),
                    rationale="naive baseline: retry every failure on a fixed schedule",
                )
            )
        out.append(
            Action(
                type=ActionType.SEND_DUNNING,
                subject_id=order.id,
                scheduled_at=last.at + timedelta(hours=spacing_hours * (retries + 1)),
                channel=Channel.EMAIL,
                rationale="naive baseline: mail every customer who failed",
            )
        )
    return out


def naive_mandate_chase(
    subscriptions: list[Subscription], now: datetime, *, contacts: int = 3
) -> list[Action]:
    """Chase every lapsed mandate, hard, on the channel that gets answered.

    Again not a strawman: "we have 1,300 dead mandates, mail them all and text
    the ones who don't reply" is what a growth team ships on a Friday. It does
    recover money. It also cannot tell a customer who let an authorisation
    lapse from one who deliberately switched it off, and it re-asks both.
    """
    out: list[Action] = []
    for sub in subscriptions:
        if sub.is_chargeable:
            continue
        for n in range(contacts):
            out.append(
                Action(
                    type=ActionType.REQUEST_MANDATE_REREGISTRATION,
                    subject_id=sub.id,
                    scheduled_at=now + timedelta(hours=6 * (n + 1)),
                    channel=Channel.EMAIL if n == 0 else Channel.SMS,
                    rationale="naive baseline: re-ask every mandate that stopped collecting",
                )
            )
    return out


def naive_invoice_chase(
    invoices: list[Invoice], now: datetime, *, contacts: int = 3
) -> list[Action]:
    """Chase every overdue invoice three times, escalating channel.

    Not a strawman either. "Pull the aged debtors report and dun everything on
    it" is the default behaviour of most collections tooling and of every
    finance team that has just been asked about DSO. It does collect money.

    What it cannot do is read the two suppressions. It chases invoices the
    buyer is actively disputing, which turns a billing disagreement into a
    grievance, and it chases buyers who already committed to a date, which is
    how a cooperative payer learns that committing to a date buys nothing. It
    also cannot see that a buyer with four overdue invoices is one person: it
    chases the invoices, so that person gets twelve messages.
    """
    out: list[Action] = []
    for inv in invoices:
        if inv.is_settled or not inv.is_overdue(now):
            continue
        for n in range(contacts):
            out.append(
                Action(
                    type=ActionType.SEND_DUNNING,
                    subject_id=inv.id,
                    scheduled_at=now + timedelta(hours=6 * (n + 1)),
                    channel=Channel.EMAIL if n == 0 else Channel.SMS,
                    rationale="naive baseline: dun everything on the aged debtors report",
                )
            )
    return out


#: Causes where the right move is to wait for the fault to pass rather than
#: retry into it. Used to set an outage hold when expanding a strategy.
SELF_HEALING = {
    RootCause.ISSUER_OUTAGE,
    RootCause.PSP_OR_RAIL_OUTAGE,
    RootCause.GATEWAY_ROUTING_DEGRADATION,
}
