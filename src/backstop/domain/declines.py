"""Failure taxonomy: what went wrong, and what you are allowed to do about it.

This is the foundation the policy engine stands on. "Never retry a stolen card"
has to be a lookup on a frozen table, not a sentence in a prompt that a model
may or may not honour.

Two enums matter downstream:

*   `DeclineCode` -- the normalised reason a specific attempt failed, carrying a
    `RetryClass` that determines whether recovery may touch it at all.
*   `RootCause` -- the *systemic* explanation the diagnosis stage must choose
    from. It is a closed set precisely because a weaker model asked for a root
    cause as free text returns prose ("HDFC issuer system outage or
    connectivity failure") that no rule can switch on. A closed enum turns that
    drift into a validation error and a repair attempt.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Rail(StrEnum):
    CARD = "card"
    UPI = "upi"
    NETBANKING = "netbanking"
    WALLET = "wallet"
    EMANDATE_NACH = "emandate_nach"
    UPI_AUTOPAY = "upi_autopay"


class RetryClass(StrEnum):
    """What recovery is permitted to do with a failure.

    Ordered loosely by how much freedom the agent has, most to least.
    """

    TRANSIENT = "transient"
    """Infrastructure blip. Retry quickly; the instrument is fine."""

    SOFT_RETRYABLE = "soft_retryable"
    """May succeed later (funds, limits). Retry on a schedule, not immediately."""

    USER_ACTION_REQUIRED = "user_action_required"
    """Cannot succeed without the customer. Re-engage; never silently retry."""

    HARD_DECLINE = "hard_decline"
    """Instrument is permanently unusable. Never retry; collect a new one."""

    FRAUD_BLOCK = "fraud_block"
    """Never retry, never dun, escalate to risk. Retrying is itself harmful."""


@dataclass(frozen=True)
class DeclineSpec:
    code: "DeclineCode"
    rail: Rail
    retry_class: RetryClass
    description: str
    max_retries: int
    """Hard ceiling on automated attempts. Zero means no automated retry, ever."""
    min_retry_delay_s: int
    """Floor on spacing between attempts. Retrying an NSF decline in 30 seconds
    just burns an issuer's patience and the merchant's decline ratio."""
    dunning_allowed: bool
    """Whether the customer may be contacted about this failure at all."""

    @property
    def is_retryable(self) -> bool:
        return self.max_retries > 0


class DeclineCode(StrEnum):
    # --- Card: funds and limits (recoverable with time) ---
    INSUFFICIENT_FUNDS = "insufficient_funds"
    EXCEEDS_LIMIT = "exceeds_limit"
    DO_NOT_HONOR = "do_not_honor"

    # --- Card: instrument is unusable ---
    CARD_EXPIRED = "card_expired"
    INVALID_CVV = "invalid_cvv"
    INVALID_CARD_NUMBER = "invalid_card_number"
    RESTRICTED_CARD = "restricted_card"
    TRANSACTION_NOT_PERMITTED = "transaction_not_permitted"

    # --- Card: fraud (retrying is actively harmful) ---
    STOLEN_OR_LOST_CARD = "stolen_or_lost_card"
    RISK_DECLINED_BY_GATEWAY = "risk_declined_by_gateway"

    # --- Card: authentication drop-off ---
    AUTH_3DS_FAILED = "auth_3ds_failed"
    OTP_ABANDONED = "otp_abandoned"

    # --- Infrastructure, any rail ---
    ISSUER_UNAVAILABLE = "issuer_unavailable"
    GATEWAY_TIMEOUT = "gateway_timeout"
    NETWORK_ERROR = "network_error"

    # --- UPI ---
    UPI_COLLECT_EXPIRED = "upi_collect_expired"
    UPI_DECLINED_BY_USER = "upi_declined_by_user"
    UPI_LIMIT_EXCEEDED = "upi_limit_exceeded"
    INVALID_VPA = "invalid_vpa"
    PSP_UNAVAILABLE = "psp_unavailable"

    # --- Netbanking ---
    BANK_UNAVAILABLE = "bank_unavailable"
    NB_SESSION_EXPIRED = "nb_session_expired"

    # --- Mandates: eNACH and UPI Autopay ---
    MANDATE_INSUFFICIENT_FUNDS = "mandate_insufficient_funds"
    MANDATE_PAUSED = "mandate_paused"
    MANDATE_EXPIRED = "mandate_expired"
    MANDATE_REVOKED = "mandate_revoked"
    MANDATE_NOT_REGISTERED = "mandate_not_registered"

    @property
    def spec(self) -> DeclineSpec:
        return DECLINE_SPECS[self]

    @property
    def retry_class(self) -> RetryClass:
        return DECLINE_SPECS[self].retry_class


_HOUR = 3600
_DAY = 24 * _HOUR


def _spec(code, rail, klass, desc, max_retries, delay, dunning) -> DeclineSpec:
    return DeclineSpec(code, rail, klass, desc, max_retries, delay, dunning)


# Retry ceilings on mandate rails are deliberately conservative. NACH
# re-presentation is capped by the sponsor-bank agreement and scheme rules
# rather than by anything we can infer, so these are floors that a merchant
# tightens in policy config -- never loosens by accident.
DECLINE_SPECS: dict[DeclineCode, DeclineSpec] = {
    d.code: d
    for d in [
        # Funds and limits: worth retrying, spaced out. Salary-credit timing is
        # the single biggest lever on NSF recovery, so the delay floor is long.
        _spec(DeclineCode.INSUFFICIENT_FUNDS, Rail.CARD, RetryClass.SOFT_RETRYABLE,
              "Insufficient funds in the account", 3, 24 * _HOUR, True),
        _spec(DeclineCode.EXCEEDS_LIMIT, Rail.CARD, RetryClass.SOFT_RETRYABLE,
              "Transaction exceeds the card limit", 2, 24 * _HOUR, True),
        _spec(DeclineCode.DO_NOT_HONOR, Rail.CARD, RetryClass.SOFT_RETRYABLE,
              "Generic issuer decline; cause not disclosed", 2, 12 * _HOUR, True),

        # Instrument unusable: retrying cannot help, the customer must act.
        _spec(DeclineCode.CARD_EXPIRED, Rail.CARD, RetryClass.USER_ACTION_REQUIRED,
              "Card has expired", 0, 0, True),
        _spec(DeclineCode.INVALID_CVV, Rail.CARD, RetryClass.USER_ACTION_REQUIRED,
              "CVV did not match", 0, 0, True),
        _spec(DeclineCode.INVALID_CARD_NUMBER, Rail.CARD, RetryClass.HARD_DECLINE,
              "Card number is invalid", 0, 0, True),
        _spec(DeclineCode.RESTRICTED_CARD, Rail.CARD, RetryClass.HARD_DECLINE,
              "Card is restricted by the issuer", 0, 0, True),
        _spec(DeclineCode.TRANSACTION_NOT_PERMITTED, Rail.CARD, RetryClass.HARD_DECLINE,
              "Transaction type not permitted on this card", 0, 0, True),

        # Fraud: no retry, no contact. Dunning a stolen-card victim is a harm,
        # not a recovery, so dunning_allowed is False on both.
        _spec(DeclineCode.STOLEN_OR_LOST_CARD, Rail.CARD, RetryClass.FRAUD_BLOCK,
              "Card reported stolen or lost", 0, 0, False),
        _spec(DeclineCode.RISK_DECLINED_BY_GATEWAY, Rail.CARD, RetryClass.FRAUD_BLOCK,
              "Declined by gateway risk engine", 0, 0, False),

        # Authentication drop-off: the money is still winnable, but only by
        # bringing the customer back, never by a silent retry.
        _spec(DeclineCode.AUTH_3DS_FAILED, Rail.CARD, RetryClass.USER_ACTION_REQUIRED,
              "3-D Secure authentication failed", 0, 0, True),
        _spec(DeclineCode.OTP_ABANDONED, Rail.CARD, RetryClass.USER_ACTION_REQUIRED,
              "Customer abandoned the OTP step", 0, 0, True),

        # Infrastructure: retry fast, the instrument is fine.
        _spec(DeclineCode.ISSUER_UNAVAILABLE, Rail.CARD, RetryClass.TRANSIENT,
              "Issuer host unreachable", 4, 5 * 60, False),
        _spec(DeclineCode.GATEWAY_TIMEOUT, Rail.CARD, RetryClass.TRANSIENT,
              "Gateway timed out awaiting the issuer", 4, 2 * 60, False),
        _spec(DeclineCode.NETWORK_ERROR, Rail.CARD, RetryClass.TRANSIENT,
              "Network error in the authorisation path", 4, 2 * 60, False),

        # UPI
        _spec(DeclineCode.UPI_COLLECT_EXPIRED, Rail.UPI, RetryClass.USER_ACTION_REQUIRED,
              "Collect request expired unactioned", 0, 0, True),
        _spec(DeclineCode.UPI_DECLINED_BY_USER, Rail.UPI, RetryClass.USER_ACTION_REQUIRED,
              "Customer declined the collect request", 0, 0, True),
        _spec(DeclineCode.UPI_LIMIT_EXCEEDED, Rail.UPI, RetryClass.SOFT_RETRYABLE,
              "Per-transaction or daily UPI limit exceeded", 2, 24 * _HOUR, True),
        _spec(DeclineCode.INVALID_VPA, Rail.UPI, RetryClass.HARD_DECLINE,
              "VPA does not exist", 0, 0, True),
        _spec(DeclineCode.PSP_UNAVAILABLE, Rail.UPI, RetryClass.TRANSIENT,
              "Payment service provider unavailable", 4, 5 * 60, False),

        # Netbanking
        _spec(DeclineCode.BANK_UNAVAILABLE, Rail.NETBANKING, RetryClass.TRANSIENT,
              "Bank netbanking endpoint unavailable", 4, 10 * 60, False),
        _spec(DeclineCode.NB_SESSION_EXPIRED, Rail.NETBANKING, RetryClass.USER_ACTION_REQUIRED,
              "Netbanking session expired before completion", 0, 0, True),

        # Mandates
        _spec(DeclineCode.MANDATE_INSUFFICIENT_FUNDS, Rail.EMANDATE_NACH,
              RetryClass.SOFT_RETRYABLE, "Insufficient funds on mandate presentation",
              2, 3 * _DAY, True),
        _spec(DeclineCode.MANDATE_PAUSED, Rail.EMANDATE_NACH, RetryClass.SOFT_RETRYABLE,
              "Mandate temporarily paused by the customer", 1, 7 * _DAY, True),
        _spec(DeclineCode.MANDATE_EXPIRED, Rail.EMANDATE_NACH, RetryClass.USER_ACTION_REQUIRED,
              "Mandate validity has lapsed; re-registration needed", 0, 0, True),
        _spec(DeclineCode.MANDATE_REVOKED, Rail.EMANDATE_NACH, RetryClass.HARD_DECLINE,
              "Mandate revoked by the customer", 0, 0, True),
        _spec(DeclineCode.MANDATE_NOT_REGISTERED, Rail.UPI_AUTOPAY, RetryClass.HARD_DECLINE,
              "No active mandate registered", 0, 0, True),
    ]
}


class RootCause(StrEnum):
    """Systemic explanations the diagnosis stage may return. Closed by design."""

    ISSUER_OUTAGE = "issuer_outage"
    """A specific issuer's authorisation host is degraded or down."""

    PSP_OR_RAIL_OUTAGE = "psp_or_rail_outage"
    """A UPI PSP, netbanking endpoint or whole rail is degraded."""

    GATEWAY_ROUTING_DEGRADATION = "gateway_routing_degradation"
    """One acquirer or route is underperforming; other routes are healthy."""

    BIN_SPECIFIC_DECLINE = "bin_specific_decline"
    """Failures concentrated in a BIN range rather than a whole issuer."""

    CHECKOUT_REGRESSION = "checkout_regression"
    """A merchant-side change broke the flow; failures start at a deploy."""

    AUTHENTICATION_DROPOFF = "authentication_dropoff"
    """Customers reaching 3DS/OTP and abandoning; auth step is the leak."""

    INSUFFICIENT_FUNDS_CLUSTER = "insufficient_funds_cluster"
    """Genuine NSF concentration, e.g. pre-payday billing runs."""

    MANDATE_LIFECYCLE_FAILURE = "mandate_lifecycle_failure"
    """Subscriptions failing on expired, paused or revoked mandates."""

    BUYER_CASHFLOW_DELAY = "buyer_cashflow_delay"
    """A receivable is late because the buyer cannot pay yet."""

    INVOICE_DISPUTE = "invoice_dispute"
    """A receivable is late because the buyer contests it. Never auto-chase."""

    FRAUD_PRESSURE = "fraud_pressure"
    """Failures are risk declines; recovery must not fight the risk engine."""

    NO_SYSTEMIC_CAUSE = "no_systemic_cause"
    """Baseline noise. The correct action is frequently to do nothing."""


def classify(codes) -> dict[RetryClass, int]:
    """Count failures by retry class. The first thing any triage view needs."""
    counts: dict[RetryClass, int] = {k: 0 for k in RetryClass}
    for c in codes:
        counts[c.retry_class] += 1
    return counts
