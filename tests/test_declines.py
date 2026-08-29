"""Invariants the policy engine relies on. If these break, recovery becomes unsafe."""

import pytest

from backstop.domain.declines import (
    DECLINE_SPECS,
    DeclineCode,
    RetryClass,
    RootCause,
    classify,
)

NEVER_AUTO_RETRY = {
    RetryClass.HARD_DECLINE,
    RetryClass.FRAUD_BLOCK,
    RetryClass.USER_ACTION_REQUIRED,
}


def test_every_decline_code_has_a_spec():
    assert set(DECLINE_SPECS) == set(DeclineCode)


@pytest.mark.parametrize("code", list(DeclineCode))
def test_spec_is_self_consistent(code):
    spec = DECLINE_SPECS[code]
    assert spec.code is code
    assert spec.max_retries >= 0
    assert spec.min_retry_delay_s >= 0
    if spec.is_retryable:
        assert spec.min_retry_delay_s > 0, "a retryable code needs a spacing floor"


@pytest.mark.parametrize("code", list(DeclineCode))
def test_unrecoverable_classes_are_never_auto_retryable(code):
    spec = DECLINE_SPECS[code]
    if spec.retry_class in NEVER_AUTO_RETRY:
        assert spec.max_retries == 0


def test_fraud_blocks_forbid_customer_contact():
    """Dunning a stolen-card victim is a harm, not a recovery."""
    for spec in DECLINE_SPECS.values():
        if spec.retry_class is RetryClass.FRAUD_BLOCK:
            assert not spec.dunning_allowed


def test_stolen_card_is_never_retryable():
    spec = DeclineCode.STOLEN_OR_LOST_CARD.spec
    assert spec.retry_class is RetryClass.FRAUD_BLOCK
    assert not spec.is_retryable and not spec.dunning_allowed


def test_transient_failures_retry_sooner_than_funds_failures():
    """Spacing must reflect the cause: infra recovers in minutes, wallets don't."""
    assert (
        DeclineCode.GATEWAY_TIMEOUT.spec.min_retry_delay_s
        < DeclineCode.INSUFFICIENT_FUNDS.spec.min_retry_delay_s
    )


def test_classify_counts_every_class():
    counts = classify([DeclineCode.STOLEN_OR_LOST_CARD, DeclineCode.INSUFFICIENT_FUNDS])
    assert set(counts) == set(RetryClass)
    assert counts[RetryClass.FRAUD_BLOCK] == 1
    assert counts[RetryClass.SOFT_RETRYABLE] == 1
    assert counts[RetryClass.HARD_DECLINE] == 0


def test_root_cause_is_a_closed_set():
    """Diagnosis returns these tokens and nothing else, so rules can switch on them."""
    assert RootCause("issuer_outage") is RootCause.ISSUER_OUTAGE
    with pytest.raises(ValueError):
        RootCause("HDFC issuer system outage or connectivity failure")
