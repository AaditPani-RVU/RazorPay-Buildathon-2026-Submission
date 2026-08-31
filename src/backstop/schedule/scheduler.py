"""Holding each action until its moment, which is the moment the rules chose.

Every action carries a `scheduled_at`, and three rules exist mainly to move it:
`quiet_hours` pushes an SMS out of the night, `retry_spacing` refuses to
re-present a card thirty seconds after it declined, and `outage_hold` defers a
charge until a degraded rail recovers. The backtest honours those times --
it sorts by `scheduled_at`, judges each action at its own moment, and executes
it there. The live path did not. It dispatched immediately and threw the
schedule away.

That made `RESCHEDULE` decorative in exactly the place it mattered most. An
engine that moves a 22:30 SMS to 09:00, followed by an adapter that sends it
at 22:30, has not protected anybody; it has produced a log entry saying it
did. This is the component that makes the disposition mean something outside
the simulator.

Four properties, and the first is the whole point.

**Nothing fires before it is due.** That is the guarantee, and it is the only
one a reader has to take on trust rather than infer -- everything else here
exists to keep it honest under the ways real deployments go wrong.

**The rules run again at fire time.** The same principle the approval queue
runs on, for the same reason: time passed, and the world the action was judged
in is not the world it will land in. A contact budget fills, a mandate gets
revoked, an invoice goes into dispute. So a due action is re-ruled before it
is dispatched, and a denial still denies. The re-rule is also what makes
lateness *safe* rather than merely tolerated -- an action that comes due at an
awkward hour is pushed again by `quiet_hours` at that point, rather than
squeaking through on a ruling made yesterday.

**An action can be re-held, but not forever.** If the re-rule moves it again,
it goes back into the queue with the new time rather than firing early or
firing anyway. Something rescheduled repeatedly is not being carefully timed,
it is being chased by a blocker that is not clearing, so there is a bound --
after `max_deferrals` the action is dropped and recorded. A recovery agent
with no stopping condition is a harassment engine, and that is as true of a
retry loop that never fires as of one that fires too often.

**A missed window is not a licence to fire late.** A process that was down for
a day comes back holding actions whose moment has passed, and firing them now
is not the same act as firing them then -- a retry timed for the hour after a
failure is a different thing three days later, and a reminder about an invoice
can arrive after the invoice was paid. Past `max_lateness` an action is
dropped as stale and said to be dropped. This is a relevance bound, not a
safety one; safety is the re-rule's job.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from backstop.domain.actions import Action
from backstop.domain.entities import utc
from backstop.execute.executor import ExecutionResult, Executor
from backstop.execute.razorpay import reference_for
from backstop.ledger.ledger import Surface
from backstop.policy.engine import (
    Disposition,
    PolicyContext,
    PolicyEngine,
    Ruling,
)
from backstop.store.codec import dump_action, dump_time, load_action, load_time
from backstop.store.journal import SCHEDULED, Journal

#: How late an action may fire and still be the action that was planned. A
#: recovery step timed for the hour after a failure is a different act a day
#: later, and a reminder can arrive after the invoice was settled.
DEFAULT_MAX_LATENESS = timedelta(hours=24)

#: How many times the rules may push one action before it is abandoned. Three
#: is enough to compose a quiet-hours move with an outage hold and a spacing
#: delay; past that the blocker is not clearing and the moment has gone.
DEFAULT_MAX_DEFERRALS = 3


class SchedulerState(StrEnum):
    WAITING = "waiting"
    FIRED = "fired"
    """Dispatched to an executor. Says nothing about whether money moved."""
    REFUSED = "refused"
    """Came due, and the rules denied it at that later moment."""
    STALE = "stale"
    """Its moment passed while nobody was running. Dropped, not fired late."""
    ABANDONED = "abandoned"
    """Deferred past the limit. The blocker never cleared."""


class Fate(StrEnum):
    DISPATCHED = "dispatched"
    REFUSED = "refused"
    DEFERRED = "deferred"
    """Re-held with a later time. Still waiting, and not an outcome yet."""
    ABANDONED = "abandoned"
    STALE = "stale"


@dataclass
class ScheduledAction:
    """One action waiting for its moment."""

    id: str
    """`reference_for(action)`. Scheduling the same action twice is one entry."""
    action: Action
    surface: Surface
    due_at: datetime
    submitted_at: datetime
    state: SchedulerState = SchedulerState.WAITING
    deferrals: int = 0
    """How many times the rules have pushed this action since it was queued."""
    history: list[datetime] = field(default_factory=list)
    """Every time it has been due for, oldest first. The audit trail of moves."""
    settled_at: datetime | None = None
    note: str = ""

    def is_due(self, at: datetime) -> bool:
        return self.state is SchedulerState.WAITING and utc(at) >= utc(self.due_at)

    def is_stale(self, at: datetime, max_lateness: timedelta) -> bool:
        return (
            self.state is SchedulerState.WAITING
            and utc(at) > utc(self.due_at) + max_lateness
        )

    def describe(self) -> str:
        moved = f"  moved {self.deferrals}x" if self.deferrals else ""
        return f"{self.due_at:%m-%d %H:%M}  {self.action.describe()}{moved}"

    # -- durability --------------------------------------------------------

    def to_record(self) -> dict:
        """A whole snapshot of this entry, not a description of what changed.

        `deferrals` and `history` travel with it because they are bounds, not
        statistics: an entry restored with its counter at zero would be an
        entry that can be pushed three more times, and the limit that stops a
        never-clearing blocker would reset on every restart.
        """
        return {
            "id": self.id,
            "action": dump_action(self.action),
            "surface": self.surface.value,
            "due_at": dump_time(self.due_at),
            "submitted_at": dump_time(self.submitted_at),
            "state": self.state.value,
            "deferrals": self.deferrals,
            "history": [dump_time(h) for h in self.history],
            "settled_at": dump_time(self.settled_at),
            "note": self.note,
        }

    @classmethod
    def from_record(cls, body: dict) -> ScheduledAction:
        return cls(
            id=str(body["id"]),
            action=load_action(body["action"]),
            surface=Surface(body["surface"]),
            due_at=load_time(body["due_at"]),
            submitted_at=load_time(body["submitted_at"]),
            state=SchedulerState(body["state"]),
            deferrals=int(body.get("deferrals", 0)),
            history=[load_time(h) for h in body.get("history", []) if h],
            settled_at=load_time(body.get("settled_at")),
            note=str(body.get("note") or ""),
        )


@dataclass
class Firing:
    """What happened when one scheduled action came due."""

    scheduled: ScheduledAction
    fate: Fate
    ruling: Ruling | None = None
    """The re-evaluation at fire time. None when the action never got that far."""
    result: ExecutionResult | None = None
    detail: str = ""

    @property
    def dispatched(self) -> bool:
        return self.fate is Fate.DISPATCHED

    @property
    def blocking_rule(self) -> str | None:
        return self.ruling.blocking_rule if self.ruling else None


@dataclass
class Scheduler:
    """Holds permitted actions until the moment the rules chose for them.

    Deliberately not a general job queue. It takes actions that have already
    been ruled on, it re-rules them before firing, and it cannot be made to
    fire one early -- there is no `flush()` and no `force` argument, because
    the single guarantee this class makes is the one such a method would take
    away.
    """

    entries: dict[str, ScheduledAction] = field(default_factory=dict)
    max_lateness: timedelta = DEFAULT_MAX_LATENESS
    max_deferrals: int = DEFAULT_MAX_DEFERRALS
    journal: Journal | None = None
    """Where held actions are written down, if anywhere.

    Optional because the backtest has no use for one: it runs a batch to
    completion in a single process and its answer is a number, not a promise
    to somebody. A live deployment is the opposite -- what it holds is a
    promise to act at a stated time, and a promise that only exists in one
    process's memory is not one. Given a journal, the constructor restores
    from it, and every state change is written before it is returned.
    """

    def __post_init__(self) -> None:
        if self.journal is not None and not self.entries:
            self.entries = {
                body["id"]: ScheduledAction.from_record(body)
                for body in self.journal.replay().latest(SCHEDULED).values()
            }

    def _remember(self, entry: ScheduledAction) -> None:
        if self.journal is not None:
            self.journal.append(SCHEDULED, entry.id, entry.to_record())

    # -- in ----------------------------------------------------------------

    def submit(
        self,
        action: Action,
        *,
        surface: Surface = Surface.PAYMENT,
        at: datetime,
    ) -> ScheduledAction:
        """Queue a permitted action for its own `scheduled_at`.

        Re-submitting the same action is not a second dispatch: identity is the
        adapter's fingerprint, so a replayed plan finds the entry already
        waiting rather than queueing a duplicate that would fire twice.
        """
        rid = reference_for(action)
        existing = self.entries.get(rid)
        if existing is not None:
            return existing
        at = utc(at)
        entry = ScheduledAction(
            id=rid,
            action=action,
            surface=surface,
            due_at=utc(action.scheduled_at),
            submitted_at=at,
            history=[utc(action.scheduled_at)],
        )
        self.entries[rid] = entry
        self._remember(entry)
        return entry

    # -- out ---------------------------------------------------------------

    def run_due(
        self,
        executor: Executor,
        engine: PolicyEngine,
        context_for: Callable[[Action], PolicyContext | None],
        *,
        at: datetime,
    ) -> list[Firing]:
        """Fire everything whose moment has arrived, in the order it arrives.

        Due-time order rather than submission order, so a run behaves the way
        a deployment ticking through real time would: the action planned for
        Tuesday morning goes before the one planned for Tuesday afternoon,
        whatever order the planner emitted them in.

        Staleness is checked before dueness. An action a day past its moment is
        not fired and then apologised for; it is dropped, and the drop is the
        record.
        """
        at = utc(at)
        out: list[Firing] = []
        for entry in self._waiting_by_due():
            if entry.is_stale(at, self.max_lateness):
                entry.state = SchedulerState.STALE
                entry.settled_at = at
                entry.note = (
                    f"due {utc(at) - utc(entry.due_at)} ago; past the "
                    f"{self.max_lateness} horizon, so not fired late"
                )
                self._remember(entry)
                out.append(Firing(entry, Fate.STALE, detail=entry.note))
                continue
            if not entry.is_due(at):
                continue
            firing = self._fire(entry, executor, engine, context_for, at=at)
            # Written after every outcome, deferral included: a re-held action
            # that came back on its old time would fire inside the quiet hours
            # the deferral moved it out of.
            self._remember(entry)
            out.append(firing)
        return out

    def _fire(
        self,
        entry: ScheduledAction,
        executor: Executor,
        engine: PolicyEngine,
        context_for: Callable[[Action], PolicyContext | None],
        *,
        at: datetime,
    ) -> Firing:
        ctx = context_for(entry.action)
        if ctx is None:
            # The subject left the batch -- settled, cancelled, or gone. Firing
            # blind at an id nothing can describe is worse than dropping it.
            entry.state = SchedulerState.REFUSED
            entry.settled_at = at
            entry.note = "subject is no longer in the batch"
            return Firing(entry, Fate.REFUSED, detail=entry.note)

        ruling = engine.evaluate(entry.action, ctx)

        if ruling.disposition is Disposition.DENY:
            entry.state = SchedulerState.REFUSED
            entry.settled_at = at
            entry.note = f"{ruling.blocking_rule} refused it at its own moment"
            return Firing(entry, Fate.REFUSED, ruling=ruling, detail=entry.note)

        final = ruling.final or entry.action
        moved_to = utc(final.scheduled_at)
        if moved_to > at:
            # The rules want it later still. Re-hold rather than fire early,
            # and rather than firing anyway because it was "due enough".
            if entry.deferrals >= self.max_deferrals:
                entry.state = SchedulerState.ABANDONED
                entry.settled_at = at
                entry.note = (
                    f"deferred {entry.deferrals} times without clearing; "
                    "the blocker is not lifting and the moment has gone"
                )
                return Firing(entry, Fate.ABANDONED, ruling=ruling, detail=entry.note)
            entry.action = final
            entry.due_at = moved_to
            entry.deferrals += 1
            entry.history.append(moved_to)
            entry.note = (
                f"{ruling.moving_rule or 'a rule'} moved it to "
                f"{moved_to:%m-%d %H:%M}"
            )
            return Firing(entry, Fate.DEFERRED, ruling=ruling, detail=entry.note)

        result = executor.execute(final, at)
        entry.state = SchedulerState.FIRED
        entry.settled_at = at
        entry.action = final
        entry.note = result.detail
        return Firing(entry, Fate.DISPATCHED, ruling=ruling, result=result,
                      detail=result.detail)

    # -- reading -----------------------------------------------------------

    def _waiting_by_due(self) -> list[ScheduledAction]:
        return sorted(
            (e for e in self.entries.values() if e.state is SchedulerState.WAITING),
            key=lambda e: utc(e.due_at),
        )

    def waiting(self, *, at: datetime | None = None) -> list[ScheduledAction]:
        """What is still held, soonest first. What an operator would see."""
        entries = self._waiting_by_due()
        if at is None:
            return entries
        horizon = utc(at)
        return [e for e in entries if utc(e.due_at) >= horizon]

    @property
    def next_due(self) -> datetime | None:
        """When this scheduler next has something to do. None if it is idle."""
        pending = self._waiting_by_due()
        return utc(pending[0].due_at) if pending else None

    def in_state(self, state: SchedulerState) -> list[ScheduledAction]:
        return [e for e in self.entries.values() if e.state is state]

    def counts(self) -> dict[SchedulerState, int]:
        out = dict.fromkeys(SchedulerState, 0)
        for e in self.entries.values():
            out[e.state] += 1
        return out

    def on(self, surface: Surface) -> Scheduler:
        # No journal on a view. A surface view is a way of reading the queue,
        # and a reader that wrote to the store would put the same entry in it
        # under three different owners.
        return Scheduler(
            entries={k: v for k, v in self.entries.items() if v.surface is surface},
            max_lateness=self.max_lateness,
            max_deferrals=self.max_deferrals,
        )
