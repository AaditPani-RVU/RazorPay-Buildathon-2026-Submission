"""The eval is only as trustworthy as the batch it runs on."""

from collections import Counter
from datetime import UTC, datetime

import pytest

from backstop.domain.declines import Rail
from backstop.domain.entities import AttemptStatus
from backstop.simulate.generator import SimConfig, _diurnal_weight, generate

START = datetime(2026, 8, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def scenario():
    return generate(SimConfig(seed=7, days=2, orders_per_day=20000, start=START))


def test_same_seed_reproduces_the_batch_exactly():
    """Eval arms must be comparable on genuinely identical data."""
    a = generate(SimConfig(seed=42, days=1, orders_per_day=500, start=START))
    b = generate(SimConfig(seed=42, days=1, orders_per_day=500, start=START))
    assert [o.id for o in a.orders] == [o.id for o in b.orders]
    assert a.total_at_risk == b.total_at_risk
    assert [i.money_at_risk for i in a.incidents] == [i.money_at_risk for i in b.incidents]


def test_different_seeds_produce_different_batches():
    a = generate(SimConfig(seed=1, days=1, orders_per_day=500, start=START))
    b = generate(SimConfig(seed=2, days=1, orders_per_day=500, start=START))
    assert a.total_at_risk != b.total_at_risk


def test_incident_attribution_excludes_baseline_failures(scenario):
    """Money at risk must be excess loss, never a window sum.

    If this regresses, every recovery number the project reports is inflated.
    """
    for inc in scenario.incidents:
        assert inc.baseline_failures_in_window > 0, "expect some unrelated noise in-window"
        assert len(inc.affected_order_ids) > 0
        by_id = {o.id: o for o in scenario.orders}
        attributed = sum(by_id[oid].amount.paise for oid in inc.affected_order_ids)
        assert inc.money_at_risk.paise == attributed


def test_attributed_orders_all_actually_failed(scenario):
    by_id = {o.id: o for o in scenario.orders}
    for inc in scenario.incidents:
        for oid in inc.affected_order_ids:
            assert not by_id[oid].is_captured


def test_incidents_are_large_enough_to_be_detectable(scenario):
    for inc in scenario.incidents:
        assert len(inc.affected_order_ids) >= 10, f"{inc.root_cause} too small to detect"


def test_traffic_follows_a_diurnal_curve():
    """Flat arrivals would let a detector cheat by thresholding raw counts."""
    weights = [_diurnal_weight(datetime(2026, 8, 1, h, tzinfo=UTC)) for h in range(24)]
    assert max(weights) / min(weights) > 3.0


def test_failed_attempts_always_carry_a_decline_code(scenario):
    for order in scenario.orders:
        for attempt in order.attempts:
            if attempt.status is AttemptStatus.FAILED:
                assert attempt.decline_code is not None


def test_checkout_abandonment_produces_orders_with_no_attempts(scenario):
    abandoned = [o for o in scenario.orders if o.abandoned_at_checkout]
    assert abandoned, "abandonment is a distinct recovery surface"
    assert all(not o.attempts for o in abandoned)
    assert all(o.amount_at_risk.paise > 0 for o in abandoned)


def test_disputed_invoices_are_never_chaseable(scenario):
    now = scenario.ends_at
    disputed = [i for i in scenario.invoices if i.disputed_at is not None]
    assert disputed, "the batch must contain contested receivables"
    assert all(not i.is_chaseable(now) for i in disputed)


def test_some_invoices_are_settled(scenario):
    """A ledger where nothing is ever paid makes the at-risk total meaningless."""
    settled = [i for i in scenario.invoices if i.is_settled]
    assert 0.4 < len(settled) / len(scenario.invoices) < 0.9


def test_rail_mix_roughly_matches_configuration(scenario):
    # One-off payment rails only. Mandate presentations share the order pool
    # but are governed by MANDATE_RAILS, so including them would measure the
    # subscription population rather than the configured checkout mix.
    payment_rails = {Rail.UPI, Rail.CARD, Rail.NETBANKING}
    counts = Counter(
        a.rail for o in scenario.orders for a in o.attempts if a.rail in payment_rails
    )
    total = sum(counts.values())
    assert 0.50 < counts[Rail.UPI] / total < 0.60
    assert 0.25 < counts[Rail.CARD] / total < 0.35


def test_incidents_can_be_disabled():
    s = generate(SimConfig(seed=3, days=1, orders_per_day=300, start=START, inject_incidents=False))
    assert s.incidents == []


def test_determinism_survives_a_fresh_process():
    """Guards a bug class in-process tests cannot see.

    Python salts string hashing per process (PYTHONHASHSEED), and uuid4 ignores
    seeding entirely. Both reproduce fine within one run and silently diverge
    across runs, which would make eval arms incomparable without ever failing a
    normal test.
    """
    import os
    import subprocess
    import sys

    script = (
        "import sys, hashlib; sys.path.insert(0, 'src');"
        "from datetime import datetime, UTC;"
        "from backstop.simulate.generator import generate, SimConfig;"
        "s = generate(SimConfig(seed=11, days=1, orders_per_day=800,"
        " start=datetime(2026, 8, 1, tzinfo=UTC)));"
        "blob = '|'.join(f'{o.id}:{o.amount.paise}:{o.is_captured}'"
        " + (o.attempts[0].bin or '-') for o in s.orders if o.attempts);"
        "print(hashlib.sha256(blob.encode()).hexdigest())"
    )
    digests = set()
    for salt in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": salt}
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True
        )
        digests.add(out.stdout.strip())
    assert len(digests) == 1, "generation must not depend on process hash salt"
