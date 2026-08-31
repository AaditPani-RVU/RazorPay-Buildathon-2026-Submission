"""Where an action waits for a person, and what happens when it stops waiting.

`REQUIRE_APPROVAL` was the one disposition with nowhere to go. The engine
produced it, the ledger counted it, the backtest executed it on the stated
assumption that a merchant staffs the queue -- and the live walkthrough, quite
correctly, held those actions back and then dropped them on the floor. A gate
with nothing behind it is not a gate, it is a deletion.

This is what is behind it.

Four properties it exists to give, each of which costs the recovery number
something:

**An approval is permission from a person, not an exemption from the rules.**
A release re-evaluates the action against the policy engine in the context
that holds *at release time*, not the one that held when it was queued. Time
passes while a request sits: the customer's fortnight contact budget may have
filled, the hour may now be inside quiet hours, the mandate may have been
revoked, the invoice may have gone into dispute. A queue that dispatched
whatever a reviewer approved on Tuesday would be a hole in the policy engine
exactly the width of the approval queue, and every guarantee the engine makes
would carry a silent asterisk. So `DENY` still denies after a yes, and the
reviewer is told which rule overtook their decision.

What the approval *does* discharge is the `require_approval` verdict itself.
That is the whole content of the human decision, and re-raising it on release
would make every approved action immortal in the queue.

**An unanswered request is not a yes.** Requests expire. This is the property
that makes the backtest's assumption falsifiable rather than convenient: a
merchant who does not staff the queue does not thereby get the revenue, they
get a pile of expired requests, and the money is reported as given up rather
than quietly booked. Expiry is recorded as an outcome, never as a silent drop,
because revenue foregone by a *staffing* decision should be as visible as
revenue foregone by a rule.

**A shelf life, not a queue depth.** The expiry is per request and measured
from when it was raised, because a recovery action goes stale on its own
schedule -- chasing a three-day-old failure is a different act from chasing
the same failure three weeks later, whatever the queue's backlog looks like.

**Every decision names a human being.** `decided_by` is not optional and there
is no auto-approve on this class. An approval with no reviewer on it is not an
audit trail, it is a rubber stamp with extra steps. The backtest's assumption
that somebody is at the desk is expressed as `StandingApproval` -- a named
reviewer, written down, so that reading the backtest tells you an assumption
was made and by whom.

Identity comes from `reference_for`, the same fingerprint that gives the
Razorpay adapter its idempotency: an action's verb, subject, moment, channel
and amount, and deliberately not its `rationale`. So re-running a plan finds
the request a reviewer has already seen instead of asking them a second time,
and a model rewording its own justification cannot manufacture a fresh ask.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from backstop.domain.actions import Action
from backstop.domain.entities import utc
from backstop.execute.razorpay import reference_for
from backstop.ledger.ledger import Surface
from backstop.policy.engine import (
    Disposition,
    PolicyContext,
    PolicyEngine,
    Ruling,
)

#: How long a request stays answerable. Two working days: long enough that a
#: reviewer who is not at their desk on a Friday afternoon still gets there,
#: short enough that nothing dispatches into a week-old world.
DEFAULT_TTL = timedelta(days=2)


class ApprovalState(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    """A person said yes. Not yet dispatched -- the rules get a second look."""
    REJECTED = "rejected"
    EXPIRED = "expired"
    """Nobody answered in time. Revenue given up by a staffing decision."""


class ReleaseOutcome(StrEnum):
    RELEASED = "released"
    """Approved by a person and still permitted by the rules. Dispatchable."""
    REFUSED = "refused"
    """Approved by a person and refused by the rules anyway. Deny wins."""
    RESCHEDULED = "rescheduled"
    """Permitted, but not at the moment it was queued for."""


@dataclass
class ApprovalRequest:
    """One action waiting on a human decision, and the trail it leaves."""

    id: str
    """`reference_for(action)`. Two proposals of the same action are one ask."""
    action: Action
    """As proposed. Never mutated -- a release produces a new action."""
    ruling: Ruling
    """The ruling that sent it here, carrying the rule that asked for a human."""
    surface: Surface
    submitted_at: datetime
    expires_at: datetime
    state: ApprovalState = ApprovalState.PENDING
    decided_at: datetime | None = None
    decided_by: str = ""
    note: str = ""

    @property
    def asking_rule(self) -> str:
        """Which rule wanted a person. What the reviewer is being asked about."""
        for v in self.ruling.verdicts:
            if v.disposition is Disposition.REQUIRE_APPROVAL:
                return v.rule_id
        return "unknown"

    @property
    def reason(self) -> str:
        for v in self.ruling.verdicts:
            if v.disposition is Disposition.REQUIRE_APPROVAL:
                return v.reason
        return ""

    def is_open(self, at: datetime) -> bool:
        return self.state is ApprovalState.PENDING and utc(at) < utc(self.expires_at)

    def describe(self) -> str:
        # No square brackets: this string is printed through rich in the
        # walkthrough, which would read them as markup and silently drop the
        # half of the line that says why a person is being asked.
        return f"{self.action.describe()}  <- {self.asking_rule}: {self.reason}"


@dataclass
class Release:
    """What became of an approved action when the rules looked again."""

    request: ApprovalRequest
    outcome: ReleaseOutcome
    ruling: Ruling
    """The *re*-evaluation, at release time. Not the one that queued it."""
    action: Action | None = None
    """The action as permitted now. None when the rules refused it."""

    @property
    def dispatchable(self) -> bool:
        return self.action is not None

    @property
    def blocking_rule(self) -> str | None:
        return self.ruling.blocking_rule


@dataclass
class StandingApproval:
    """The backtest's assumption, written down and given a name.

    The four-arm measurement executes `REQUIRE_APPROVAL` actions because its
    question is what recovery is worth *if* a merchant staffs the queue. That
    is a defensible assumption and it stays -- but an assumption stated as a
    reviewer is one a reader can see, argue with, and vary. `latency` is how
    long the desk takes to answer, and it is not zero, because a queue that
    answers instantly is not a queue.

    `rate` below 1.0 models a desk that does not get to everything. What it
    misses expires, which is the honest outcome and the reason this class
    exists rather than a boolean.
    """

    by: str = "backtest:assumed-reviewer"
    latency: timedelta = timedelta(hours=4)
    rate: float = 1.0
    note: str = "standing approval assumed by the backtest, not a real decision"

    def decides_at(self, request: ApprovalRequest) -> datetime | None:
        """When this desk answers, or None if it never gets to it."""
        if self.rate >= 1.0:
            answered = True
        else:
            # Deterministic from the request id: the same batch must produce
            # the same staffing outcome on every arm and every re-run, or the
            # comparison would be measuring the sampler.
            answered = (int(self.id_hash(request.id), 16) % 1000) / 1000.0 < self.rate
        if not answered:
            return None
        when = utc(request.submitted_at) + self.latency
        return when if when < utc(request.expires_at) else None

    @staticmethod
    def id_hash(request_id: str) -> str:
        import hashlib

        return hashlib.sha1(request_id.encode()).hexdigest()[:8]


@dataclass
class ApprovalQueue:
    """Holds actions the engine will not let automation decide alone.

    Deliberately not a general work queue: it can only ever hold something the
    policy engine sent here, and it can only ever release something a named
    person approved *and* the rules re-permitted.
    """

    requests: dict[str, ApprovalRequest] = field(default_factory=dict)
    ttl: timedelta = DEFAULT_TTL
    released: set[str] = field(default_factory=set, init=False)
    """Requests already dispatched. A queue that can release twice is a queue
    that can contact somebody twice for one approval."""

    # -- in ----------------------------------------------------------------

    def submit(
        self,
        action: Action,
        ruling: Ruling,
        *,
        surface: Surface = Surface.PAYMENT,
        at: datetime,
        ttl: timedelta | None = None,
    ) -> ApprovalRequest:
        """Queue an action for a person. Re-submitting one is not a second ask.

        Raises if the ruling is not actually an approval request: everything
        else has a disposition that means something already, and quietly
        accepting a DENY here would turn the queue into a way around it.
        """
        if ruling.disposition is not Disposition.REQUIRE_APPROVAL:
            raise ValueError(
                f"only require_approval rulings belong in the queue, not "
                f"{ruling.disposition.value}"
            )
        rid = reference_for(action)
        existing = self.requests.get(rid)
        if existing is not None:
            return existing
        at = utc(at)
        request = ApprovalRequest(
            id=rid,
            action=action,
            ruling=ruling,
            surface=surface,
            submitted_at=at,
            expires_at=at + (ttl or self.ttl),
        )
        self.requests[rid] = request
        return request

    # -- the desk ----------------------------------------------------------

    def approve(
        self, request_id: str, *, by: str, at: datetime, note: str = ""
    ) -> ApprovalRequest:
        return self._decide(request_id, ApprovalState.APPROVED, by=by, at=at, note=note)

    def reject(
        self, request_id: str, *, by: str, at: datetime, note: str = ""
    ) -> ApprovalRequest:
        return self._decide(request_id, ApprovalState.REJECTED, by=by, at=at, note=note)

    def _decide(
        self,
        request_id: str,
        state: ApprovalState,
        *,
        by: str,
        at: datetime,
        note: str,
    ) -> ApprovalRequest:
        if not by:
            raise ValueError("an approval must name the person who gave it")
        request = self.requests.get(request_id)
        if request is None:
            raise KeyError(f"no approval request {request_id}")
        at = utc(at)
        if request.state is not ApprovalState.PENDING:
            raise ValueError(
                f"{request_id} was already {request.state.value}; a decision is "
                "made once"
            )
        if at >= utc(request.expires_at):
            # Answered after the shelf life ran out. The honest record is that
            # nobody got to it in time, not that somebody did.
            request.state = ApprovalState.EXPIRED
            request.decided_at = at
            return request
        request.state = state
        request.decided_at = at
        request.decided_by = by
        request.note = note
        return request

    def staff(self, reviewer: StandingApproval, *, at: datetime) -> list[ApprovalRequest]:
        """Run a standing approval over everything open. The backtest's desk.

        Separate from `approve` on purpose: this is an assumption being applied
        in bulk, and it should not be reachable by the same call a real
        reviewer's decision comes through.
        """
        at = utc(at)
        decided: list[ApprovalRequest] = []
        for request in list(self.requests.values()):
            if request.state is not ApprovalState.PENDING:
                continue
            when = reviewer.decides_at(request)
            if when is None or when > at:
                continue
            decided.append(
                self._decide(
                    request.id, ApprovalState.APPROVED,
                    by=reviewer.by, at=when, note=reviewer.note,
                )
            )
        return decided

    # -- out ---------------------------------------------------------------

    def expire_due(self, at: datetime) -> list[ApprovalRequest]:
        """Mark everything nobody answered in time. Call before reading state.

        Expiry is a transition rather than a computed property so that it lands
        in the record once, with a time on it.
        """
        at = utc(at)
        out: list[ApprovalRequest] = []
        for request in self.requests.values():
            if request.state is ApprovalState.PENDING and at >= utc(request.expires_at):
                request.state = ApprovalState.EXPIRED
                request.decided_at = utc(request.expires_at)
                out.append(request)
        return out

    def release(
        self,
        engine: PolicyEngine,
        context_for: Callable[[Action], PolicyContext | None],
        *,
        at: datetime,
    ) -> list[Release]:
        """Re-run the rules over everything approved, and report what survives.

        The second evaluation is the point of the whole class. A reviewer's yes
        discharges the `require_approval` verdict and nothing else; if the
        world moved while the request sat -- the contact budget filled, the
        hour became quiet, the mandate was revoked -- the rules refuse it now
        and say which one did.

        A subject that has left the batch entirely returns no context, and is
        refused rather than dispatched blind.
        """
        at = utc(at)
        out: list[Release] = []
        for request in self.requests.values():
            if request.state is not ApprovalState.APPROVED:
                continue
            if request.id in self.released:
                continue
            ctx = context_for(request.action)
            if ctx is None:
                out.append(
                    Release(request, ReleaseOutcome.REFUSED, request.ruling, None)
                )
                self.released.add(request.id)
                continue
            ruling = engine.evaluate(request.action, ctx)
            self.released.add(request.id)
            if ruling.disposition is Disposition.DENY:
                out.append(Release(request, ReleaseOutcome.REFUSED, ruling, None))
                continue
            final = ruling.final or request.action
            # Read the move off the *time*, not the disposition label. An
            # action that is both high-value and inside quiet hours composes
            # to REQUIRE_APPROVAL, because the engine ranks a pending human
            # decision above a schedule change -- while `final` still carries
            # the new hour. A reviewer who approved something for Tuesday
            # 22:30 and had it leave at Wednesday 09:00 should see that on the
            # record rather than the word "released".
            moved = utc(final.scheduled_at) != utc(request.action.scheduled_at)
            outcome = ReleaseOutcome.RESCHEDULED if moved else ReleaseOutcome.RELEASED
            out.append(Release(request, outcome, ruling, final))
        return out

    # -- reading -----------------------------------------------------------

    def pending(self, at: datetime) -> list[ApprovalRequest]:
        """Open requests, oldest first. What a reviewer would see."""
        at = utc(at)
        return sorted(
            (r for r in self.requests.values() if r.is_open(at)),
            key=lambda r: utc(r.submitted_at),
        )

    def in_state(self, state: ApprovalState) -> list[ApprovalRequest]:
        return [r for r in self.requests.values() if r.state is state]

    @property
    def expired(self) -> list[ApprovalRequest]:
        return self.in_state(ApprovalState.EXPIRED)

    def counts(self) -> dict[ApprovalState, int]:
        out = dict.fromkeys(ApprovalState, 0)
        for r in self.requests.values():
            out[r.state] += 1
        return out

    def on(self, surface: Surface) -> ApprovalQueue:
        view = ApprovalQueue(
            requests={k: v for k, v in self.requests.items() if v.surface is surface},
            ttl=self.ttl,
        )
        view.released.update(self.released & view.requests.keys())
        return view
