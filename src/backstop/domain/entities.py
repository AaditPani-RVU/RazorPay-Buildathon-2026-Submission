"""The objects recovery operates on.

Deliberately plain dataclasses. Pydantic lives at the LLM boundary, where
schema validation earns its keep; the domain core should not depend on the
reasoning layer's serialisation library, because the simulator, detector and
ledger all round-trip these and none of them talk to a model.

Time is UTC-aware everywhere. Quiet-hours policy converts to IST at the point
of decision -- a contact cap that silently uses server-local time is the kind
of bug that only shows up as complaints.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from backstop.domain.declines import DeclineCode, Rail
from backstop.domain.money import Money


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:14]}"


def utc(ts: datetime) -> datetime:
    """Normalise to aware UTC. Naive datetimes are a correctness hazard here."""
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


# --------------------------------------------------------------------------
# Customers and consent
# --------------------------------------------------------------------------


class Channel(StrEnum):
    EMAIL = "email"
    SMS = "sms"
    WHATSAPP = "whatsapp"
    VOICE = "voice"


class ContactOutcome(StrEnum):
    DELIVERED = "delivered"
    BOUNCED = "bounced"
    OPENED = "opened"
    CLICKED = "clicked"
    REPLIED = "replied"
    OPTED_OUT = "opted_out"


@dataclass
class Customer:
    id: str
    email: str | None = None
    phone: str | None = None
    consented_channels: set[Channel] = field(default_factory=set)
    """Explicit opt-in. Absence of consent is not permission."""
    dnd_registered: bool = False
    """On the national Do Not Disturb registry: no promotional SMS or voice."""
    opted_out_at: datetime | None = None
    """A hard stop across every channel, permanently."""
    timezone: str = "Asia/Kolkata"

    def reachable_on(self, channel: Channel) -> bool:
        """Whether contact is permissible at all, ignoring frequency caps."""
        if self.opted_out_at is not None:
            return False
        if channel not in self.consented_channels:
            return False
        if self.dnd_registered and channel in (Channel.SMS, Channel.VOICE):
            return False
        if channel is Channel.EMAIL and not self.email:
            return False
        if channel in (Channel.SMS, Channel.WHATSAPP, Channel.VOICE) and not self.phone:
            return False
        return True


@dataclass
class ContactRecord:
    """One outbound touch. The basis of every frequency and fatigue rule."""

    id: str
    customer_id: str
    channel: Channel
    at: datetime
    subject_ref: str
    """What this contact was about -- an order, invoice or subscription id."""
    outcome: ContactOutcome = ContactOutcome.DELIVERED
    template: str = ""


# --------------------------------------------------------------------------
# Payments
# --------------------------------------------------------------------------


class AttemptStatus(StrEnum):
    CAPTURED = "captured"
    FAILED = "failed"
    PENDING = "pending"


@dataclass
class PaymentAttempt:
    """A single authorisation attempt. Orders accumulate these."""

    id: str
    order_id: str
    customer_id: str
    amount: Money
    rail: Rail
    at: datetime
    status: AttemptStatus
    issuer: str | None = None
    """Bank or PSP handling authorisation, e.g. 'HDFC'. Segments detection."""
    bin: str | None = None
    """First six of the card. Distinguishes a BIN issue from an issuer outage."""
    acquirer: str | None = None
    """Route the attempt took. Distinguishes routing degradation from issuer."""
    decline_code: DeclineCode | None = None
    attempt_no: int = 1
    is_recovery_attempt: bool = False
    """True when the agent caused this attempt. Keeps measurement honest: money
    recovered must be attributable to recovery, not to the customer retrying."""

    def __post_init__(self) -> None:
        self.at = utc(self.at)
        if self.status is AttemptStatus.FAILED and self.decline_code is None:
            raise ValueError(f"failed attempt {self.id} has no decline code")
        if self.status is AttemptStatus.CAPTURED and self.decline_code is not None:
            raise ValueError(f"captured attempt {self.id} carries a decline code")

    @property
    def succeeded(self) -> bool:
        return self.status is AttemptStatus.CAPTURED


@dataclass
class Order:
    """A payment intent and every attempt made against it."""

    id: str
    customer_id: str
    amount: Money
    created_at: datetime
    attempts: list[PaymentAttempt] = field(default_factory=list)
    abandoned_at_checkout: bool = False
    """Customer never reached an attempt. Recoverable, but not by retrying."""

    def __post_init__(self) -> None:
        self.created_at = utc(self.created_at)

    @property
    def is_captured(self) -> bool:
        return any(a.succeeded for a in self.attempts)

    @property
    def last_attempt(self) -> PaymentAttempt | None:
        return max(self.attempts, key=lambda a: a.at) if self.attempts else None

    @property
    def last_decline(self) -> DeclineCode | None:
        last = self.last_attempt
        return last.decline_code if last and not last.succeeded else None

    @property
    def recovery_attempts(self) -> int:
        return sum(1 for a in self.attempts if a.is_recovery_attempt)

    @property
    def amount_at_risk(self) -> Money:
        """Unrecovered value. Zero once captured -- there is nothing to chase."""
        return Money.zero() if self.is_captured else self.amount

    @property
    def recovered_amount(self) -> Money:
        """Value captured *because of* recovery, not by the customer unaided."""
        for a in self.attempts:
            if a.succeeded and a.is_recovery_attempt:
                return a.amount
        return Money.zero()


# --------------------------------------------------------------------------
# Subscriptions
# --------------------------------------------------------------------------


class MandateStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    EXPIRED = "expired"
    REVOKED = "revoked"
    NOT_REGISTERED = "not_registered"


@dataclass
class Subscription:
    id: str
    customer_id: str
    amount: Money
    rail: Rail
    mandate_status: MandateStatus
    next_charge_at: datetime
    consecutive_failures: int = 0
    orders: list[str] = field(default_factory=list)
    cancelled_at: datetime | None = None

    def __post_init__(self) -> None:
        self.next_charge_at = utc(self.next_charge_at)

    @property
    def is_chargeable(self) -> bool:
        return self.cancelled_at is None and self.mandate_status is MandateStatus.ACTIVE

    @property
    def needs_re_registration(self) -> bool:
        return self.mandate_status in (MandateStatus.EXPIRED, MandateStatus.NOT_REGISTERED)


# --------------------------------------------------------------------------
# Receivables
# --------------------------------------------------------------------------


class InvoiceStatus(StrEnum):
    OPEN = "open"
    PAID = "paid"
    PART_PAID = "part_paid"
    WRITTEN_OFF = "written_off"


@dataclass
class PromiseToPay:
    """A commitment from the buyer. Suppresses chasing until it lapses."""

    promised_at: datetime
    promised_for: datetime
    amount: Money
    kept: bool | None = None

    def is_live(self, now: datetime) -> bool:
        return self.kept is None and utc(now) <= utc(self.promised_for)


@dataclass
class Invoice:
    """A B2B receivable. Same engine as payments, different failure surface."""

    id: str
    buyer_id: str
    amount: Money
    issued_at: datetime
    due_at: datetime
    status: InvoiceStatus = InvoiceStatus.OPEN
    amount_paid: Money = field(default_factory=Money.zero)
    disputed_at: datetime | None = None
    """A disputed invoice must never be auto-chased; it goes to a human."""
    promise: PromiseToPay | None = None
    contacts: list[ContactRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.issued_at = utc(self.issued_at)
        self.due_at = utc(self.due_at)

    @property
    def outstanding(self) -> Money:
        return self.amount - self.amount_paid

    @property
    def is_settled(self) -> bool:
        return self.status in (InvoiceStatus.PAID, InvoiceStatus.WRITTEN_OFF)

    def days_overdue(self, now: datetime) -> int:
        delta = utc(now) - self.due_at
        return max(0, delta.days)

    def is_overdue(self, now: datetime) -> bool:
        return not self.is_settled and utc(now) > self.due_at

    def is_chaseable(self, now: datetime) -> bool:
        """Disputes and live promises both suspend collection."""
        if self.is_settled or self.disputed_at is not None:
            return False
        if self.promise is not None and self.promise.is_live(now):
            return False
        return self.is_overdue(now)


__all__ = [
    "AttemptStatus", "Channel", "ContactOutcome", "ContactRecord", "Customer",
    "Invoice", "InvoiceStatus", "MandateStatus", "Order", "PaymentAttempt",
    "PromiseToPay", "Subscription", "new_id", "utc", "replace", "timedelta",
]
