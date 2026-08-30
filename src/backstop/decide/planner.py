"""Turning a diagnosis into proposed actions.

Two paths, because revenue recovery has two shapes and only one of them needs
a model.

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
from backstop.diagnose.diagnoser import Diagnosis
from backstop.domain.actions import Action, ActionType
from backstop.domain.declines import DeclineCode, RetryClass, RootCause
from backstop.domain.entities import Channel, Order
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


#: Causes where the right move is to wait for the fault to pass rather than
#: retry into it. Used to set an outage hold when expanding a strategy.
SELF_HEALING = {
    RootCause.ISSUER_OUTAGE,
    RootCause.PSP_OR_RAIL_OUTAGE,
    RootCause.GATEWAY_ROUTING_DEGRADATION,
}
