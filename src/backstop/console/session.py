"""One operator's view of a running batch, and the controls over it.

State lives here rather than in the HTTP layer, for the ordinary reason: the
things this holds -- an approval queue, a scheduler, a clock -- are the same
objects a deployment would hold, and a request handler that owned them would
make them a property of the transport. `app.py` is a translation of these
methods into routes and nothing else.

Three things the console does that the walkthrough cannot.

**It rules on two plans at once.** The engine's argument is comparative: the
same rules, the same contexts, the same moments, and the only difference is
whether the rulings are obeyed. A stream drawn from Backstop's own playbook
shows that argument badly, because a good playbook rarely proposes anything
the rules must refuse -- on a sample of 450 subjects it draws nine denials.
Ruling the naive plan beside it, in the same engine, lights eleven of the
sixteen rules and shows what the boundary is actually holding back.

**It moves a clock.** `quiet_hours`, `retry_spacing` and `outage_hold` exist
to move an action rather than refuse it, and an approval can go stale while it
waits. Neither is visible in a process that runs to completion in one second.
Advancing the console's clock is the same loop the scheduler would run against
real time, with the waiting taken out, and it is the only way to see a
reviewer's approval overruled by a rule that became true after they gave it.

**It reads, it does not decide.** Nothing on this page recomputes a verdict,
a recovered amount or a violation count. Every figure is read off the ledger,
the queue or the scheduler, so a console that disagreed with the measurement
would be a bug in one of them rather than a second opinion.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from backstop.approve import ApprovalQueue, ApprovalState, ReleaseOutcome
from backstop.config import Settings
from backstop.decide.planner import (
    mandate_actions,
    naive_invoice_chase,
    naive_mandate_chase,
    naive_retry,
    receivable_actions,
    tail_actions,
)
from backstop.detect import mandates as mandate_scan
from backstop.detect import receivables as receivables_scan
from backstop.detect.correlate import RiskCluster, correlate
from backstop.detect.multires import MultiResolutionDetector
from backstop.diagnose.diagnoser import Diagnoser, DiagnosisResult
from backstop.diagnose.evidence import EvidenceBuilder
from backstop.domain.actions import Action
from backstop.domain.entities import utc
from backstop.domain.money import Money
from backstop.evaluation import detection_score
from backstop.execute.executor import ExecutionResult, SimulatedExecutor
from backstop.ledger.ledger import LedgerEntry, RecoveryLedger, Surface
from backstop.llm import GroqProvider, LLMClient
from backstop.policy.engine import Disposition, PolicyContext, PolicyEngine
from backstop.policy.probes import bench
from backstop.policy.subjects import ContactBook, SubjectIndex
from backstop.schedule import Fate, Scheduler
from backstop.simulate.generator import SimConfig, generate
from backstop.simulate.scenario import Scenario

#: How many subjects per surface the operate view rules on. Small enough that
#: a page can hold the whole stream and a person can read any row in it, large
#: enough that the caps and the fatigue rule have something to bite on. The
#: measurement runs on the whole batch; this is a window onto the live path.
DEFAULT_SAMPLE = 150

#: Who the console's decisions are recorded as. There is no auto-approve, so
#: something has to name the person at the desk, and a console that recorded
#: "system" would be a rubber stamp with a nicer font.
DEFAULT_REVIEWER = "console:operator"


@dataclass
class Row:
    """One proposed action and what the engine said about it."""

    id: str
    source: str
    """`backstop` or `naive`. Which plan proposed it, not which engine ruled."""
    surface: Surface
    action: Action
    ruling: object
    """A `Ruling`. Untyped here only to keep the dataclass import surface flat."""
    fate: str = "ruled"
    """Where it went afterwards: queued, scheduled, refused, or nowhere."""


@dataclass
class Event:
    """Something the operator did, or something that happened because of it."""

    at: datetime
    kind: str
    text: str
    rule: str = ""


@dataclass
class ConsoleSession:
    """Everything one browser tab is looking at.

    Single-writer by construction and guarded by a lock, because the clock and
    the queue are mutable and two overlapping requests advancing time would
    interleave firings. That is the same limitation the journal has and it is
    stated for the same reason: it wants a database, not a comment.
    """

    seed: int = 1
    days: int = 7
    orders_per_day: int = 20000
    sample: int = DEFAULT_SAMPLE

    scenario: Scenario | None = None
    clusters: list[RiskCluster] = field(default_factory=list)
    diagnoses: list[tuple[RiskCluster, DiagnosisResult]] = field(default_factory=list)
    diagnosis_backend: str = ""

    engine: PolicyEngine = field(default_factory=PolicyEngine)
    index: SubjectIndex | None = None
    rows: list[Row] = field(default_factory=list)
    queue: ApprovalQueue = field(default_factory=ApprovalQueue)
    scheduler: Scheduler = field(default_factory=Scheduler)
    ledger: RecoveryLedger = field(default_factory=lambda: RecoveryLedger(arm="console"))
    book: ContactBook = field(default_factory=ContactBook)
    executor: SimulatedExecutor | None = None
    events: list[Event] = field(default_factory=list)
    clock: datetime | None = None
    started_at: datetime | None = None
    dispatches: list[ExecutionResult] = field(default_factory=list)
    """Live Razorpay results, kept apart from the ledger on purpose: a dispatch
    is not a recovery and must not be counted as one."""

    arms: list = field(default_factory=list)
    """The four-arm measurement, once somebody runs it. Kept beside the live
    path rather than merged into it: the console operates one plan and measures
    four, and the measurement needs a counterfactual reality does not offer."""
    measurement_backend: str = ""

    live: object | None = None
    """A `RazorpayExecutor` bound to this batch, once somebody asks for one.

    Off until asked. Constructing it is what turns a page that describes
    dispatching into a page that dispatches, and that is a decision a person
    makes rather than a side effect of opening a tab.
    """
    receiver: object | None = None
    """The webhook receiver, sharing the live executor. Present only with a
    secret configured: a receiver that treats "no secret" as "everything is
    authentic" is worse than one with no check at all."""
    journal: object | None = None

    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # -- building ----------------------------------------------------------

    def build(self, *, seed: int | None = None, days: int | None = None,
              orders_per_day: int | None = None, sample: int | None = None) -> None:
        """Generate a batch and run the three detection scans over it.

        No model anywhere in here, which is the same claim the pipeline makes:
        finding the money is statistics, and a scan that took a second round
        trip to a language model to notice a success rate had halved would be
        slower and worse at it.
        """
        self.seed = seed if seed is not None else self.seed
        self.days = days if days is not None else self.days
        self.orders_per_day = orders_per_day if orders_per_day is not None else self.orders_per_day
        self.sample = sample if sample is not None else self.sample

        self.scenario = generate(
            SimConfig(seed=self.seed, days=self.days, orders_per_day=self.orders_per_day)
        )
        attempts = [a for o in self.scenario.orders for a in o.attempts]
        self.clusters = correlate(MultiResolutionDetector().run(attempts))
        self.index = SubjectIndex.of(self.scenario)
        self.diagnoses = []
        self.diagnosis_backend = ""
        self.reset_operations()

    @property
    def ready(self) -> bool:
        return self.scenario is not None

    def require(self) -> Scenario:
        if self.scenario is None:
            raise RuntimeError("no batch has been generated yet")
        return self.scenario

    # -- diagnosis ---------------------------------------------------------

    def diagnose(self, *, model: str | None = None) -> None:
        """Ask the model why each detected cluster is failing, and score it.

        The one place on this page where a model decides anything. Its answer
        is graded against the incident that actually produced the cluster,
        printed beside it, because a diagnosis nobody checks is a sentence
        rather than a measurement.
        """
        scenario = self.require()
        settings = Settings.load()
        if not settings.has_groq:
            raise RuntimeError("GROQ_API_KEY is not set; diagnosis needs a model")
        chosen = model or settings.reasoning_model
        client = LLMClient(
            provider=GroqProvider(api_key=settings.groq_api_key), model=chosen
        )
        diagnoser = Diagnoser(client)
        builder = EvidenceBuilder([a for o in scenario.orders for a in o.attempts])
        self.diagnoses = [
            (cluster, diagnoser.diagnose(builder.build(cluster)))
            for cluster in self.clusters
        ]
        self.diagnosis_backend = chosen

    # -- the live path -----------------------------------------------------

    def reset_operations(self) -> None:
        """Rule both plans over the same sample and route what survives.

        Two contact books, one per plan, for the same reason the backtest gives
        each arm its own: frequency and fatigue are counted against what that
        plan has already sent, and sharing a book would let one plan's restraint
        pay for the other's volume.
        """
        scenario = self.require()
        index = self.index or SubjectIndex.of(scenario)
        self.index = index
        now = scenario.ends_at
        self.clock = now
        self.started_at = now
        self.queue = ApprovalQueue()
        self.scheduler = Scheduler()
        self.ledger = RecoveryLedger(arm="console")
        self.book = ContactBook()
        self.events = []
        self.dispatches = []
        self.executor = SimulatedExecutor(
            orders=index.orders,
            recoverability=scenario.recoverability,
            subscriptions=index.subscriptions,
            mandate_recovery=scenario.mandate_recovery,
            invoices=index.invoices,
            invoice_recovery=scenario.invoice_recovery,
        )

        orders = scenario.failed_orders[: self.sample]
        subs = scenario.lapsed_subscriptions[: self.sample]
        invoices = scenario.overdue_invoices[: self.sample]

        planned: list[Action] = []
        for order in orders:
            planned.extend(tail_actions(order))
        for sub in subs:
            planned.extend(mandate_actions(sub, now))
        for invoice in invoices:
            planned.extend(receivable_actions(invoice, now))

        naive: list[Action] = (
            naive_retry(orders)
            + naive_mandate_chase(subs, now)
            + naive_invoice_chase(invoices, now)
        )

        # The clock starts at the earliest moment the plan asks for, not at the
        # end of the batch. The batch is historical, so a plan drawn against it
        # is drawn for moments that have already passed -- a retry timed for
        # the hour after a failure is timed for last Tuesday. Read the clock as
        # `now` and the scheduler is correct to drop the lot as stale, and the
        # console would open on 239 actions it had already given up on. The
        # walkthrough starts its clock the same way and for the same reason.
        earliest = min((utc(a.scheduled_at) for a in planned), default=now)
        self.clock = min(earliest, now)
        self.started_at = self.clock

        self.rows = []
        self.rows.extend(self._rule_stream(planned, source="backstop", route=True))
        self.rows.extend(
            self._rule_stream(naive, source="naive", route=False, book=ContactBook())
        )
        self._log("run", f"{len(self.rows)} actions ruled across two plans")

    def _rule_stream(
        self,
        actions: Iterable[Action],
        *,
        source: str,
        route: bool,
        book: ContactBook | None = None,
    ) -> list[Row]:
        """Evaluate one plan in schedule order, and optionally route the result.

        `route` is what separates the plan being *operated* from the plan being
        *shown*. Backstop's actions go on to the queue and the scheduler, which
        is where they would go in a deployment. Naive's are ruled and displayed
        and nothing else: the console is not a place to run an unpoliced arm at
        a person, and the argument only needs the verdicts.
        """
        index = self.index
        assert index is not None
        book = book if book is not None else self.book
        out: list[Row] = []
        for action in sorted(actions, key=lambda a: utc(a.scheduled_at)):
            subject = index.resolve(action.subject_id)
            ctx = index.context(action, contacts=book)
            ruling = self.engine.evaluate(action, ctx)
            row = Row(
                id=f"{source}:{len(out)}",
                source=source,
                surface=subject.surface,
                action=action,
                ruling=ruling,
            )
            if not route:
                # The unpoliced counterpart still writes to its own book: its
                # later actions have to be judged against the contacts it would
                # actually have sent, or the fatigue rule never fires on it.
                book.record(
                    action,
                    customer_id=subject.customer.id if subject.customer else "",
                    at=utc(action.scheduled_at),
                )
                row.fate = "shown"
                out.append(row)
                continue

            if ruling.disposition is Disposition.DENY:
                row.fate = "refused"
                self.ledger.record(
                    LedgerEntry(action=action, surface=subject.surface,
                                ruling=ruling, execution=None)
                )
            elif action.is_inert:
                row.fate = "inert"
            elif ruling.disposition is Disposition.REQUIRE_APPROVAL:
                self.queue.submit(action, ruling, surface=subject.surface, at=self.clock)
                row.fate = "queued"
            else:
                entry = self.scheduler.submit(
                    ruling.final or action, surface=subject.surface, at=self.clock
                )
                row.id = entry.id
                row.fate = "scheduled"
            out.append(row)
        return out

    def _context_for(self, action: Action) -> PolicyContext | None:
        """The context an action is re-judged in, at the console's clock.

        Returns None for a subject that is not in the batch, which both the
        queue and the scheduler read as a refusal. That is the right answer:
        an action whose subject cannot be found is one the rules have nothing
        to check, and dispatching it would be dispatching blind.
        """
        index = self.index
        if index is None:
            return None
        if not index.resolve(action.subject_id).known:
            return None
        return index.context(action, now=self.clock, contacts=self.book)

    # -- the desk ----------------------------------------------------------

    def decide(self, request_id: str, *, approve: bool, by: str = DEFAULT_REVIEWER,
               note: str = "") -> None:
        with self.lock:
            at = self.clock or utc(datetime.now(UTC))
            call = self.queue.approve if approve else self.queue.reject
            request = call(request_id, by=by, at=at, note=note)
            self._log(
                "approval",
                f"{by} {'approved' if approve else 'rejected'} "
                f"{request.action.describe()}",
            )
            if request.state is ApprovalState.EXPIRED:
                self._log("approval", "answered after it had already expired")

    def release(self) -> list[dict]:
        """Re-rule everything a person approved, and report what survives.

        The interesting outcome is `refused`: a reviewer said yes and a rule
        said no anyway, because the world moved while the request sat. That is
        the property the queue exists to have, and it is the one thing about
        an approval queue a screenshot cannot assert.
        """
        with self.lock:
            at = self.clock
            assert at is not None
            releases = self.queue.release(self.engine, self._context_for, at=at)
            out: list[dict] = []
            for release in releases:
                if release.outcome is ReleaseOutcome.REFUSED:
                    self._log(
                        "release",
                        f"refused after approval: {release.request.action.describe()}",
                        rule=release.blocking_rule or "subject left the batch",
                    )
                else:
                    entry = self.scheduler.submit(
                        release.action, surface=release.request.surface, at=at
                    )
                    self._log(
                        "release",
                        f"{release.outcome.value} to {entry.due_at:%m-%d %H:%M}: "
                        f"{release.action.describe()}",
                    )
                out.append({
                    "id": release.request.id,
                    "outcome": release.outcome.value,
                    "rule": release.blocking_rule or "",
                    "action": release.request.action.describe(),
                    "reason": next(
                        (v.reason for v in release.ruling.verdicts
                         if v.disposition is Disposition.DENY),
                        "",
                    ),
                })
            return out

    # -- the clock ---------------------------------------------------------

    def advance(self, *, hours: float | None = None, to_next: bool = False) -> dict:
        """Move the clock, then let everything that came due happen.

        Expiry runs before firing, on purpose. An approval whose shelf life ran
        out during the jump is expired at the moment it expired rather than
        being answered by a jump that went past it.
        """
        with self.lock:
            assert self.clock is not None
            before = self.clock
            if to_next:
                due = self.scheduler.next_due
                self.clock = max(due, before) if due else before
            else:
                self.clock = before + timedelta(hours=hours or 1)
            moved = self.clock - before

            expired = self.queue.expire_due(self.clock)
            for request in expired:
                self._log(
                    "expiry",
                    f"nobody answered in time: {request.action.describe()}",
                    rule=request.asking_rule,
                )

            firings = self.scheduler.run_due(
                self.executor, self.engine, self._context_for, at=self.clock
            )
            for firing in firings:
                self._record_firing(firing)

            return {
                "moved_hours": moved.total_seconds() / 3600,
                "expired": len(expired),
                "fired": sum(1 for f in firings if f.fate is Fate.DISPATCHED),
                "refused": sum(1 for f in firings if f.fate is Fate.REFUSED),
                "deferred": sum(1 for f in firings if f.fate is Fate.DEFERRED),
                "abandoned": sum(1 for f in firings if f.fate is Fate.ABANDONED),
                "stale": sum(1 for f in firings if f.fate is Fate.STALE),
            }

    def _record_firing(self, firing) -> None:
        entry = firing.scheduled
        action = entry.action
        if firing.fate is Fate.DISPATCHED and firing.result is not None:
            subject = self.index.resolve(action.subject_id) if self.index else None
            self.ledger.record(
                LedgerEntry(action=action, surface=entry.surface,
                            ruling=firing.ruling, execution=firing.result)
            )
            self.book.record(
                action,
                customer_id=subject.customer.id if subject and subject.customer else "",
                at=self.clock,
            )
            self._log(
                "fire",
                f"{action.describe()} -> {firing.result.outcome.value}"
                + (f" {firing.result.recovered}" if firing.result.is_recovery else ""),
            )
        else:
            self._log(
                firing.fate.value,
                f"{action.describe()}: {firing.detail or firing.fate.value}",
                rule=firing.blocking_rule or "",
            )

    def _log(self, kind: str, text: str, rule: str = "") -> None:
        self.events.append(
            Event(at=self.clock or utc(datetime.now(UTC)), kind=kind, text=text, rule=rule)
        )

    # -- the bench ---------------------------------------------------------

    def probes(self) -> list[dict]:
        """Run the fixed bench and report each ruling against its expectation."""
        scenario = self.require()
        out: list[dict] = []
        for probe in bench(scenario):
            ruling = self.engine.evaluate(probe.action, probe.context)
            moved = (
                ruling.final is not None
                and utc(ruling.final.scheduled_at) != utc(probe.action.scheduled_at)
            )
            out.append({
                "label": probe.label,
                "subject": probe.action.subject_id,
                "proposed": probe.action.describe(),
                "expected": probe.expected,
                "got": ruling.disposition.value,
                "matches": ruling.disposition is probe.expect,
                "moved_to": (
                    f"{ruling.final.scheduled_at:%Y-%m-%d %H:%M} UTC"
                    if moved and ruling.final else ""
                ),
                "verdicts": [
                    {"rule": v.rule_id, "disposition": v.disposition.value,
                     "reason": v.reason}
                    for v in ruling.verdicts
                ],
            })
        return out

    # -- reading -----------------------------------------------------------

    def rule_pressure(self) -> list[dict]:
        """How often each rule spoke, per plan. The load-bearing rules first."""
        tally: dict[str, dict] = {
            rule.id: {"rule": rule.id, "backstop": 0, "naive": 0,
                      "dispositions": {}, "example": ""}
            for rule in self.engine.rules
        }
        for row in self.rows:
            for verdict in row.ruling.verdicts:
                slot = tally.setdefault(
                    verdict.rule_id,
                    {"rule": verdict.rule_id, "backstop": 0, "naive": 0,
                     "dispositions": {}, "example": ""},
                )
                slot[row.source] = slot.get(row.source, 0) + 1
                key = verdict.disposition.value
                slot["dispositions"][key] = slot["dispositions"].get(key, 0) + 1
                if not slot["example"]:
                    slot["example"] = verdict.reason
        return sorted(
            tally.values(),
            key=lambda s: (s["backstop"] + s["naive"]),
            reverse=True,
        )

    def surfaces(self) -> dict:
        """The three scans, side by side, with what each one is denominated in.

        Reported apart and never summed. A recovered payment is one amount that
        landed; a re-registered mandate is a year of billing restored; a
        collected invoice is a balance already owed. A console that added them
        into one headline would be inventing a number nobody could reconcile,
        and the tables in the measurement go to some trouble not to.
        """
        scenario = self.require()
        mandates = mandate_scan.scan(scenario.subscriptions)
        aging = receivables_scan.scan(scenario.invoices, scenario.ends_at)
        at_risk = sum((o.amount_at_risk for o in scenario.orders), start=Money.zero())
        return {
            "payment": {
                "label": "Payments",
                "subtitle": "one-off checkout failures",
                "denominated": "money that did not land",
                "subjects": len(scenario.failed_orders),
                "at_risk": str(at_risk),
                "detail": f"{len(self.clusters)} correlated clusters",
            },
            "recurring": {
                "label": "Recurring",
                "subtitle": "mandates that stopped collecting",
                "denominated": "a year of billing per mandate",
                "subjects": len(scenario.lapsed_subscriptions),
                "at_risk": str(mandates.total_annual_value) + " / yr",
                "detail": f"expected {mandates.expected_recovery} if all are chased",
            },
            "receivable": {
                "label": "Receivables",
                "subtitle": "invoices that were never paid",
                "denominated": "a balance already owed",
                "subjects": len(scenario.overdue_invoices),
                "at_risk": str(aging.total_outstanding),
                "detail": (
                    f"{aging.chaseable_value} chaseable, "
                    f"{aging.weighted_days_overdue:.0f}d weighted age"
                ),
            },
        }

    def detection(self) -> dict:
        """What the payments scan found, scored against the incidents behind it.

        The score is the point. A detector reporting seven clusters is not
        evidence of anything until somebody says how many incidents there were
        and whether these are them.
        """
        scenario = self.require()
        report = detection_score.score(scenario, [c.primary for c in self.clusters])
        return {
            "clusters": [
                {
                    "id": c.id,
                    "segment": c.segment.describe(),
                    "starts_at": c.starts_at.isoformat(),
                    "ends_at": c.ends_at.isoformat(),
                    "minutes": round(c.primary.duration_minutes),
                    "at_risk": str(c.money_at_risk),
                    "decline": (
                        c.primary.dominant_decline.value
                        if c.primary.dominant_decline else ""
                    ),
                    "share": round(c.primary.dominant_share, 3),
                    "breadth": c.breadth,
                    "truth": (
                        t.root_cause.value
                        if (t := detection_score.match(c.primary, scenario.incidents))
                        else ""
                    ),
                }
                for c in self.clusters
            ],
            "incidents": len(scenario.incidents),
            "found": len(report.detected),
            "missed": len(report.missed),
            "false_positives": len(report.false_positives),
            "redundant": len(report.redundant),
            "recall": round(report.recall, 3),
            "precision": round(report.precision, 3),
            "latency_minutes": round(report.mean_latency_minutes),
            "money_found": str(report.money_found),
            "money_missed": str(report.money_missed),
        }

    # -- reaching outside --------------------------------------------------

    def go_live(self) -> dict:
        """Bind a Razorpay test-mode adapter to this batch, and say what it can reach.

        Deliberately explicit. Everything else on this page is a claim about
        what recovery *would* do; this is the one component that reaches
        outside the process and does it, and it is constructed when somebody
        asks rather than when the server starts.

        Live keys are refused here as they are everywhere else. The adapter
        will not construct against one, and the console does not offer an
        override, because the difference between dispatching a payment link at
        a test customer and at a real one is a single token in a `.env` file.
        """
        from backstop.execute.razorpay import HttpTransport, RazorpayExecutor, capabilities
        from backstop.execute.webhook import WebhookReceiver
        from backstop.store import Journal

        scenario = self.require()
        index = self.index or SubjectIndex.of(scenario)
        settings = Settings.load()
        if not settings.has_razorpay:
            raise RuntimeError("RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET are not set")
        if not settings.razorpay_is_test_mode:
            raise RuntimeError(
                "the configured key is not a test key, and this adapter dispatches "
                "payment links at people"
            )
        transport = HttpTransport.from_settings(settings)
        self.journal = Journal(settings.journal_path)
        self.live = RazorpayExecutor(
            transport=transport,
            orders=index.orders,
            subscriptions=index.subscriptions,
            invoices=index.invoices,
            customers=index.customers,
            subscription_by_order=index.subscription_by_order,
            notify=False,
            journal=self.journal,
        )
        self.receiver = (
            WebhookReceiver(executor=self.live, secret=settings.razorpay_webhook_secret)
            if settings.has_webhook_secret else None
        )
        self._log("live", f"bound to {settings.razorpay_key_id} in test mode")
        return {
            "key": settings.razorpay_key_id,
            "notify": False,
            "webhook": self.receiver is not None,
            "journal": str(settings.journal_path),
            "known_dispatches": len(self.live.dispatched),
            "already_credited": len(self.live.reconciled),
            "capabilities": [
                {"name": name, "ok": ok, "why": why, "error": err}
                for name, ok, why, err in capabilities(transport)
            ],
        }

    def dispatch(self, entry_id: str, *, notify: bool = False) -> dict:
        """Send one scheduled action onto the real rails, after ruling on it again.

        The re-rule is not ceremony. The action was permitted for a moment,
        and the moment a person clicks a button is a different one; dispatching
        on the strength of the earlier verdict would be the same hole the
        approval queue exists to close. A denial here sends nothing and names
        the rule.
        """
        if self.live is None:
            raise RuntimeError("the live adapter is not bound; enable it first")
        entry = self.scheduler.entries.get(entry_id)
        if entry is None:
            raise KeyError(f"nothing scheduled under {entry_id}")
        ctx = self._context_for(entry.action)
        if ctx is None:
            raise RuntimeError("this action's subject is not in the batch")
        ruling = self.engine.evaluate(entry.action, ctx)
        if ruling.disposition is Disposition.DENY:
            self._log("live", f"refused before dispatch: {entry.action.describe()}",
                      rule=ruling.blocking_rule or "")
            return {
                "dispatched": False,
                "disposition": ruling.disposition.value,
                "rule": ruling.blocking_rule or "",
                "reason": next(
                    (v.reason for v in ruling.verdicts
                     if v.disposition is Disposition.DENY), ""
                ),
            }
        if ruling.disposition is Disposition.REQUIRE_APPROVAL:
            # The backtest executes these on its stated assumption that a
            # merchant staffs the desk. A live adapter may not make that
            # assumption on somebody's behalf: a gate that sends while it waits
            # is not a gate.
            return {
                "dispatched": False,
                "disposition": ruling.disposition.value,
                "rule": next(
                    (v.rule_id for v in ruling.verdicts
                     if v.disposition is Disposition.REQUIRE_APPROVAL), ""
                ),
                "reason": "a person has not approved this yet",
            }
        self.live.notify = notify
        result = self.live.execute(ruling.final or entry.action, self.clock)
        self.dispatches.append(result)
        self._log(
            "live",
            f"{result.outcome.value}: {result.detail}"
            + (f"  {result.external.describe()}" if result.external else ""),
        )
        return {
            "dispatched": result.is_pending or result.is_recovery,
            "disposition": ruling.disposition.value,
            "outcome": result.outcome.value,
            "detail": result.detail,
            "external": (
                {"entity": result.external.entity, "id": result.external.id,
                 "url": result.external.url}
                if result.external else None
            ),
        }

    def webhook(self, body: bytes, signature: str) -> object:
        """Hand a pushed delivery to the receiver, verbatim.

        Raw bytes, never a parsed body. Re-serialising a parsed payload to
        check the signature breaks on key order and quietly tempts somebody to
        skip the check, so the transport's only job here is not to touch it.
        """
        if self.receiver is None:
            raise RuntimeError(
                "no webhook receiver: RAZORPAY_WEBHOOK_SECRET is not set, and a "
                "receiver with no secret would treat every delivery as authentic"
            )
        return self.receiver.receive(body, signature, at=self.clock)

    # -- measuring ---------------------------------------------------------

    def measure(self, *, offline: bool = True, model: str | None = None) -> None:
        """Replay this batch through four arms and keep the result.

        The same call the command-line measurement makes, on the same batch the
        rest of the page is looking at, so a reader can check that the console
        and the tables in the README are describing one run rather than two.

        The simulator is the backend here and the Razorpay adapter is not,
        which is a seam rather than a shortcut: the comparison needs to know
        what would have happened had nobody acted, and reality does not offer a
        counterfactual. Swapping a live adapter in would not make this more
        real, it would make it unmeasurable.
        """
        from backstop.evaluation import backtest

        scenario = self.require()
        arms, proposal = backtest.run(scenario, offline=offline, model=model)
        self.arms = arms
        self.measurement_backend = proposal.backend

    def measurement(self) -> dict:
        """The three surface tables, apart and never summed."""
        if not self.arms:
            return {"backend": "", "surfaces": {}}
        scenario = self.require()
        out: dict[str, Any] = {}
        for surface in Surface:
            rows = []
            for arm in self.arms:
                led = arm.ledger.on(surface)
                illegal = led.recovered_in_violation
                rows.append({
                    "arm": arm.name,
                    "recovered": str(led.recovered),
                    "illegal": str(illegal) if illegal else "",
                    "keepable_net": str(led.compliant_net),
                    "subjects": led.orders_recovered,
                    "charges": led.charges_attempted,
                    "contacts": led.contacts_sent,
                    "burst": led.worst_contact_burst,
                    "violations": sum(1 for v in arm.violations if v.surface is surface),
                })
            out[surface.value] = rows
        return {
            "backend": self.measurement_backend,
            "surfaces": out,
            "at_risk": {
                "payment": str(
                    sum((o.amount_at_risk for o in scenario.orders), start=Money.zero())
                ),
                "recurring": str(scenario.recurring_at_risk) + " / yr",
                "receivable": str(scenario.receivables_at_risk),
            },
        }
