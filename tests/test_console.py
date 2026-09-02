"""The operator console: a viewer, and the four things it must not become.

The console is the only place a person can push a button and make this system
act, so what it is tested for is mostly what it refuses:

*   It must not decide anything. Every verdict on the page comes from the
    engine, and a route that could permit an action the engine denied would be
    a hole in the boundary exactly the width of an HTTP handler.
*   It must not dispatch what a person has not approved. The backtest executes
    `REQUIRE_APPROVAL` on a stated assumption; a live path may not make that
    assumption on somebody's behalf.
*   It must not fire anything early, and its clock must not be able to fire
    something twice.
*   It must not open on a batch it has already given up on. The batch is
    historical, so a console that read `now` as the clock would start by
    dropping the whole plan as stale.

Offline: a small generated batch, no model, no network. The live Razorpay
routes are exercised only for their refusals, which need no key.
"""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from backstop.console import session as session_module
from backstop.console.app import SESSION, app
from backstop.domain.entities import utc
from backstop.ledger.ledger import Surface
from backstop.policy.engine import Disposition
from backstop.schedule import SchedulerState

SMALL = {"seed": 1, "days": 3, "orders_per_day": 4000, "sample": 40}


@pytest.fixture(scope="module")
def live_session():
    s = session_module.ConsoleSession()
    s.build(seed=SMALL["seed"], days=SMALL["days"],
            orders_per_day=SMALL["orders_per_day"], sample=SMALL["sample"])
    return s


@pytest.fixture
def fresh():
    s = session_module.ConsoleSession()
    s.build(seed=SMALL["seed"], days=SMALL["days"],
            orders_per_day=SMALL["orders_per_day"], sample=SMALL["sample"])
    return s


# -- what it shows ---------------------------------------------------------


def test_both_plans_are_ruled_by_the_same_engine(live_session):
    sources = {row.source for row in live_session.rows}
    assert sources == {"backstop", "naive"}
    assert live_session.rows, "no actions were proposed at all"


def test_only_the_operated_plan_is_routed_anywhere(live_session):
    """The naive plan is ruled beside Backstop's and dispatched nowhere. A
    console that ran an unpoliced arm at a person would not be a console."""
    assert all(row.fate == "shown" for row in live_session.rows
               if row.source == "naive")


def test_a_denied_action_reaches_neither_the_queue_nor_the_scheduler(live_session):
    denied = [
        row for row in live_session.rows
        if row.source == "backstop" and row.ruling.disposition is Disposition.DENY
    ]
    for row in denied:
        assert row.fate == "refused"
    ids = set(live_session.scheduler.entries) | set(live_session.queue.requests)
    assert not {row.id for row in denied} & ids


def test_what_needs_a_person_goes_to_the_queue_and_not_to_the_scheduler(live_session):
    for request in live_session.queue.requests.values():
        assert request.ruling.disposition is Disposition.REQUIRE_APPROVAL
        assert request.id not in live_session.scheduler.entries


def test_the_naive_plan_puts_more_rules_under_load(live_session):
    """The argument the second stream exists to make: a good planner rarely
    proposes anything the rules must refuse, so a page showing only its own
    plan shows a boundary holding back nothing."""
    spoke = {
        source: {
            v.rule_id for row in live_session.rows if row.source == source
            for v in row.ruling.verdicts
        }
        for source in ("backstop", "naive")
    }
    assert len(spoke["naive"]) > len(spoke["backstop"])


def test_rule_pressure_lists_every_rule_including_the_silent_ones(live_session):
    listed = {row["rule"] for row in live_session.rule_pressure()}
    assert {rule.id for rule in live_session.engine.rules} <= listed


# -- the clock -------------------------------------------------------------


def test_the_clock_opens_on_the_earliest_moment_the_plan_asks_for(live_session):
    """A batch is historical. Read the clock as `now` and the scheduler is
    right to drop the entire plan as stale before anybody has done anything."""
    assert live_session.clock <= live_session.require().ends_at
    assert live_session.scheduler.counts()[SchedulerState.STALE] == 0


def test_nothing_fires_before_its_moment(fresh):
    due = fresh.scheduler.next_due
    assert due is not None
    fresh.clock = due - timedelta(minutes=1)
    fired = fresh.advance(hours=0.0001)
    assert fired["fired"] == 0


def test_advancing_to_the_next_due_moment_fires_something(fresh):
    moved = fresh.advance(to_next=True)
    assert moved["fired"] >= 1


def test_an_action_is_only_ever_executed_once(fresh):
    for _ in range(20):
        fresh.advance(hours=12)
    executed = [entry.action.describe() for entry in fresh.ledger.executed]
    assert len(executed) == len(set(executed))


def test_an_unanswered_request_expires_rather_than_dispatching(fresh):
    assert fresh.queue.pending(fresh.clock)
    for _ in range(20):
        fresh.advance(hours=12)
    counts = fresh.queue.counts()
    from backstop.approve import ApprovalState

    assert counts[ApprovalState.EXPIRED] > 0
    assert not fresh.queue.pending(fresh.clock)


# -- the desk --------------------------------------------------------------


def test_an_approval_must_name_a_person(fresh):
    request = fresh.queue.pending(fresh.clock)[0]
    with pytest.raises(ValueError, match="name the person"):
        fresh.decide(request.id, approve=True, by="")


def test_a_release_re_rules_and_can_still_refuse(fresh):
    """The property the queue exists to have. A yes discharges the
    `require_approval` verdict and nothing else."""
    for request in fresh.queue.pending(fresh.clock):
        fresh.decide(request.id, approve=True)
    fresh.advance(hours=18)
    releases = fresh.release()
    assert releases
    for release in releases:
        if release["outcome"] == "refused":
            assert release["rule"], "a refusal must name the rule that produced it"


def test_a_released_action_is_scheduled_rather_than_sent_now(fresh):
    request = fresh.queue.pending(fresh.clock)[0]
    fresh.decide(request.id, approve=True)
    before = len(fresh.scheduler.entries)
    releases = fresh.release()
    if any(r["outcome"] != "refused" for r in releases):
        assert len(fresh.scheduler.entries) > before


def test_one_approval_cannot_be_released_twice(fresh):
    for request in fresh.queue.pending(fresh.clock):
        fresh.decide(request.id, approve=True)
    first = fresh.release()
    second = fresh.release()
    assert first and not second


def test_a_rejected_request_is_never_released(fresh):
    request = fresh.queue.pending(fresh.clock)[0]
    fresh.decide(request.id, approve=False)
    assert all(r["id"] != request.id for r in fresh.release())


# -- measuring -------------------------------------------------------------


def test_the_measurement_reports_each_surface_apart(fresh):
    """Four arms on one batch. The surfaces are tabled apart and never added:
    a recovered payment and a restored year of billing are not the same unit."""
    fresh.measure(offline=True)
    report = fresh.measurement()
    assert set(report["surfaces"]) == {s.value for s in Surface}
    assert "total" not in report["surfaces"]
    for rows in report["surfaces"].values():
        assert [r["arm"] for r in rows] == [
            "do-nothing", "naive-retry", "planner-unpoliced", "backstop"
        ]


def test_the_policed_arm_breaks_no_rules_on_any_surface(fresh):
    """The claim the whole project rests on, checked through the console's own
    reading of the ledger rather than through the backtest's printout."""
    fresh.measure(offline=True)
    for rows in fresh.measurement()["surfaces"].values():
        backstop = next(r for r in rows if r["arm"] == "backstop")
        assert backstop["violations"] == 0
        assert not backstop["illegal"]


def test_an_unrun_measurement_says_so_rather_than_reporting_zeroes(fresh):
    assert fresh.measurement() == {"backend": "", "surfaces": {}}


# -- reaching outside ------------------------------------------------------


def test_dispatching_needs_a_bound_adapter(fresh):
    entry = next(iter(fresh.scheduler.entries))
    with pytest.raises(RuntimeError, match="not bound"):
        fresh.dispatch(entry)


def test_the_receiver_refuses_to_exist_without_a_secret(fresh):
    with pytest.raises(RuntimeError, match="secret"):
        fresh.webhook(b"{}", "signature")


# -- the HTTP surface ------------------------------------------------------


@pytest.fixture(scope="module")
def client(live_session):
    SESSION.__dict__.update(
        {k: v for k, v in live_session.__dict__.items() if k != "lock"}
    )
    return TestClient(app)


def test_bootstrap_reports_every_rule_with_its_reason(client):
    body = client.get("/api/bootstrap").json()
    assert len(body["rules"]) == 16
    assert all(rule["why"] for rule in body["rules"])


def test_state_reports_each_surface_apart_and_never_summed(client):
    body = client.get("/api/state").json()
    assert set(body["ledger_by_surface"]) == {s.value for s in Surface}
    assert "total" not in body["ledger_by_surface"]


def test_rows_can_be_filtered_by_the_rule_that_spoke(client):
    body = client.get("/api/rows?rule=contact_consent&limit=5").json()
    for row in body["rows"]:
        assert any(v["rule"] == "contact_consent" for v in row["verdicts"])


def test_every_row_carries_the_rules_that_produced_its_ruling(client):
    body = client.get("/api/rows?disposition=deny&limit=20").json()
    assert body["rows"]
    for row in body["rows"]:
        assert row["blocking_rule"]


def test_the_bench_route_reports_matches_against_expectation(client):
    body = client.get("/api/probes").json()
    assert body["total"] >= 10
    assert body["matched"] == body["total"]


def test_an_approval_without_a_reviewer_is_refused_over_http(client):
    body = client.get("/api/approvals?state=pending").json()
    if not body["requests"]:
        pytest.skip("no pending requests on this batch")
    res = client.post(
        "/api/approvals/decide",
        json={"ids": [body["requests"][0]["id"]], "approve": True, "by": "  "},
    )
    assert res.status_code == 400


def test_the_webhook_route_refuses_when_no_secret_is_configured(client):
    res = client.post("/webhooks/razorpay", content=b"{}")
    assert res.status_code == 503


def test_a_cold_console_answers_409_rather_than_pretending(client):
    cold = session_module.ConsoleSession()
    saved = dict(SESSION.__dict__)
    try:
        SESSION.__dict__.update(
            {k: v for k, v in cold.__dict__.items() if k != "lock"}
        )
        assert client.get("/api/state").status_code == 409
    finally:
        SESSION.__dict__.update(saved)


def test_times_are_reported_with_a_zone_on_them(client):
    """A naive timestamp on the wire is a timestamp the page will read in
    whatever zone the reader's browser happens to be in."""
    body = client.get("/api/state").json()
    for iso in (body["clock"], body["batch_ends_at"], body["started_at"]):
        assert datetime.fromisoformat(iso).tzinfo is not None
        assert utc(datetime.fromisoformat(iso)).tzinfo is UTC
