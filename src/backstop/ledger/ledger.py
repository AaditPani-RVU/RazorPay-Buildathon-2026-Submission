"""The audit trail. What was proposed, what was permitted, what actually ran.

"Zero policy violations" is only worth saying if it is a statement about a
record. This is that record: one entry per proposed action, carrying the
policy ruling that shaped it and the outcome if it ran.

The ledger also knows how to *audit* an arm that had no policy engine. Replay
every action an arm actually executed past the rules and count the ones that
would have been refused -- that is a violation count on identical terms for
every arm, rather than a claim that only the policed arm can make about itself.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum

from backstop.domain.actions import Action, ActionType
from backstop.domain.money import Money
from backstop.execute.executor import ExecutionResult, Outcome
from backstop.policy.engine import Disposition, PolicyContext, PolicyEngine, Ruling


class Surface(StrEnum):
    """Which revenue surface an action was working on.

    Recorded per entry rather than inferred from an id prefix, and reported
    separately rather than summed, because the surfaces are not denominated in
    the same thing. A recovered payment is one amount that landed; a
    re-registered mandate is a year of billing restored. Adding them would
    produce a headline number that no merchant could reconcile.
    """

    PAYMENT = "payment"
    RECURRING = "recurring"
    RECEIVABLE = "receivable"


@dataclass
class LedgerEntry:
    action: Action
    """The action as proposed, before any rule touched it."""
    surface: Surface = Surface.PAYMENT
    ruling: Ruling | None = None
    """None when the arm ran without a policy engine."""
    execution: ExecutionResult | None = None
    """None when the action was never executed."""

    @property
    def executed(self) -> bool:
        return self.execution is not None

    @property
    def recovered(self) -> Money:
        return self.execution.recovered if self.execution else Money.zero()

    @property
    def cost(self) -> Money:
        return self.execution.cost if self.execution else Money.zero()


@dataclass
class Violation:
    """An action that ran which the rules would have refused."""

    action: Action
    rule_id: str
    reason: str
    surface: Surface = Surface.PAYMENT


@dataclass
class RecoveryLedger:
    arm: str
    entries: list[LedgerEntry] = field(default_factory=list)

    def record(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)

    def on(self, surface: Surface) -> RecoveryLedger:
        """A view of one revenue surface, answering every question this ledger
        does. Reporting reads surfaces through this rather than through
        separate counters, so a per-surface number and a total can never be
        computed two different ways."""
        return RecoveryLedger(
            arm=self.arm, entries=[e for e in self.entries if e.surface is surface]
        )

    @property
    def surfaces(self) -> list[Surface]:
        """Surfaces this arm actually touched, in declaration order."""
        seen = {e.surface for e in self.entries}
        return [s for s in Surface if s in seen]

    # -- what happened -----------------------------------------------------

    @property
    def proposed(self) -> int:
        return len(self.entries)

    @property
    def executed(self) -> list[LedgerEntry]:
        return [e for e in self.entries if e.executed]

    @property
    def recovered(self) -> Money:
        total = Money.zero()
        for e in self.entries:
            total += e.recovered
        return total

    @property
    def cost(self) -> Money:
        total = Money.zero()
        for e in self.entries:
            total += e.cost
        return total

    @property
    def net(self) -> Money:
        return self.recovered - self.cost

    @property
    def recovered_in_violation(self) -> Money:
        """Money taken by actions the rules would have refused.

        Not a credit. An arm that recovers by texting people on the DND
        registry has not out-performed one that declines to -- it has taken
        revenue its merchant could not lawfully take, and counting it as a win
        would make the comparison reward exactly the behaviour the project
        exists to prevent.
        """
        total = Money.zero()
        for e in self.entries:
            if e.ruling and e.ruling.disposition is Disposition.DENY:
                total += e.recovered
        return total

    @property
    def compliant_recovered(self) -> Money:
        """Recovery a merchant could actually keep. The honest comparison."""
        return self.recovered - self.recovered_in_violation

    @property
    def compliant_net(self) -> Money:
        return self.compliant_recovered - self.cost

    @property
    def orders_recovered(self) -> int:
        return sum(
            1 for e in self.entries
            if e.execution and e.execution.outcome is Outcome.RECOVERED
        )

    @property
    def contacts_sent(self) -> int:
        return sum(1 for e in self.executed if e.action.is_contact)

    @property
    def charges_attempted(self) -> int:
        return sum(1 for e in self.executed if e.action.is_charging)

    @property
    def wasted_actions(self) -> int:
        """Ran, cost money, recovered nothing."""
        return sum(
            1 for e in self.executed
            if e.execution.outcome is Outcome.NO_EFFECT and e.cost
        )

    # -- what the rules did ------------------------------------------------

    @property
    def vetoed(self) -> list[LedgerEntry]:
        return [
            e for e in self.entries
            if e.ruling and e.ruling.disposition is Disposition.DENY
        ]

    @property
    def rescheduled(self) -> list[LedgerEntry]:
        return [
            e for e in self.entries
            if e.ruling and e.ruling.disposition is Disposition.RESCHEDULE
        ]

    @property
    def held_for_approval(self) -> list[LedgerEntry]:
        return [
            e for e in self.entries
            if e.ruling and e.ruling.disposition is Disposition.REQUIRE_APPROVAL
        ]

    def vetoes_by_rule(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for e in self.vetoed:
            counts[e.ruling.blocking_rule or "unknown"] += 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def contacts_per_subject(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for e in self.executed:
            if e.action.is_contact:
                counts[e.action.subject_id] += 1
        return dict(counts)

    @property
    def worst_contact_burst(self) -> int:
        """Most contacts any one subject received. The harassment ceiling."""
        per = self.contacts_per_subject()
        return max(per.values()) if per else 0

    # -- audit -------------------------------------------------------------

    def audit(
        self, engine: PolicyEngine, context_for
    ) -> list[Violation]:
        """Replay everything that ran past the rules and collect refusals.

        `context_for` maps an action to the PolicyContext it should be judged
        in. Applied identically to every arm, so an unpoliced arm is not being
        held to a standard the policed one escaped.
        """
        out: list[Violation] = []
        for entry in self.executed:
            ctx: PolicyContext | None = context_for(entry.action)
            if ctx is None:
                continue
            ruling = engine.evaluate(entry.action, ctx)
            if ruling.disposition is not Disposition.DENY:
                continue
            for v in ruling.verdicts:
                if v.disposition is Disposition.DENY:
                    out.append(Violation(entry.action, v.rule_id, v.reason))
                    break
        return out

    def action_mix(self) -> dict[ActionType, int]:
        counts: dict[ActionType, int] = defaultdict(int)
        for e in self.executed:
            counts[e.action.type] += 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
