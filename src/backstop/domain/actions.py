"""The closed set of things recovery is allowed to attempt.

Pydantic rather than a dataclass, deliberately: this type sits *on* the model
boundary. The planner emits these, so the schema is the contract that stops a
model inventing an action the executor has no code path for. "Call the customer
and offer a 40% discount" is not representable here, which is the point.

Every action names one subject and one moment. Nothing is open-ended, nothing
is free text that an executor would have to interpret, and `rationale` is
explicitly *not* load-bearing -- it is for the audit trail, never for deciding
what happens.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from backstop.domain.entities import Channel


class ActionType(StrEnum):
    RETRY_PAYMENT = "retry_payment"
    """Re-present the same instrument. Only for failures that time can fix."""

    SWITCH_ROUTE = "switch_route"
    """Re-present through a different acquirer. For routing degradation."""

    SEND_DUNNING = "send_dunning"
    """Contact the customer to act. The only route for user-action failures."""

    REGENERATE_PAYMENT_LINK = "regenerate_payment_link"
    """Issue a fresh link for an expired or abandoned checkout."""

    REQUEST_MANDATE_REREGISTRATION = "request_mandate_reregistration"
    """Ask the customer to re-authorise a lapsed mandate."""

    OFFER_PART_PAYMENT = "offer_part_payment"
    """Split a receivable the buyer cannot clear in one go."""

    ESCALATE_TO_HUMAN = "escalate_to_human"
    """Hand to a person. Not a failure -- often the correct outcome."""

    ESCALATE_TO_RISK = "escalate_to_risk"
    """Hand to the risk team. Recovery must not fight the risk engine."""

    WAIT = "wait"
    """Deliberately hold, e.g. until an issuer outage clears or payday lands."""

    DO_NOTHING = "do_nothing"
    """Chasing costs more than it recovers, or nothing is actually broken."""


#: Actions that put money through the rails again. These are the ones that can
#: damage a merchant's standing with an issuer if fired at the wrong failure.
CHARGING_ACTIONS = {ActionType.RETRY_PAYMENT, ActionType.SWITCH_ROUTE}

#: Actions that contact a person. These carry consent and frequency duties.
CONTACT_ACTIONS = {
    ActionType.SEND_DUNNING,
    ActionType.REGENERATE_PAYMENT_LINK,
    ActionType.REQUEST_MANDATE_REREGISTRATION,
    ActionType.OFFER_PART_PAYMENT,
}

#: Actions that are always safe: they move no money and contact nobody.
INERT_ACTIONS = {ActionType.WAIT, ActionType.DO_NOTHING, ActionType.ESCALATE_TO_HUMAN,
                 ActionType.ESCALATE_TO_RISK}


class Action(BaseModel):
    """One proposed step. Proposed -- nothing here has been permitted yet."""

    type: ActionType
    subject_id: str = Field(description="Order, invoice or subscription this acts on.")
    scheduled_at: datetime = Field(description="When to execute, UTC. Never 'as soon as possible'.")
    channel: Channel | None = Field(
        default=None, description="Required for contact actions, forbidden otherwise."
    )
    route: str | None = Field(default=None, description="Acquirer, for switch_route only.")
    amount_paise: int | None = Field(
        default=None, ge=0, description="For part payment offers. Integer paise."
    )
    rationale: str = Field(
        max_length=400,
        description="Why, for the audit trail. Never used to decide whether this is allowed.",
    )

    @property
    def is_charging(self) -> bool:
        return self.type in CHARGING_ACTIONS

    @property
    def is_contact(self) -> bool:
        return self.type in CONTACT_ACTIONS

    @property
    def is_inert(self) -> bool:
        return self.type in INERT_ACTIONS

    def describe(self) -> str:
        bits = [self.type.value, self.subject_id]
        if self.channel:
            bits.append(f"via {self.channel.value}")
        if self.route:
            bits.append(f"via {self.route}")
        return " ".join(bits) + f" at {self.scheduled_at:%m-%d %H:%M}"


class RecoveryPlan(BaseModel):
    """What the planner proposes for one detected problem."""

    actions: list[Action] = Field(default_factory=list, max_length=12)
    reasoning: str = Field(default="", max_length=800)
