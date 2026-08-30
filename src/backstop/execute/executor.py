"""Executing an approved action, and recording what actually happened.

Nothing reaches here that the policy engine did not permit. The executor's job
is narrow on purpose: take a permitted action, make it happen, and report the
outcome truthfully. It does not re-decide anything.

Two backends behind one interface:

*   `SimulatedExecutor` resolves an action against the batch's latent
    recoverability -- ground truth fixed at generation time, identical for
    every arm of the backtest. This is what makes arms comparable. It resolves
    two surfaces: failed orders, and the mandates behind recurring revenue.
*   A Razorpay test-mode adapter belongs here too and is not built yet; the
    protocol is shaped so it drops in without the ledger or backtest changing.

The executor is the only component permitted to read `Recoverability`. The
planner and the policy engine must never see it -- an agent that can look up
whether its own action will work is not being measured, it is cheating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from backstop.domain.actions import Action, ActionType
from backstop.domain.entities import Order, Subscription
from backstop.domain.money import Money
from backstop.simulate.recoverability import MandateRecovery, Recoverability


class Outcome(StrEnum):
    RECOVERED = "recovered"
    """Money landed as a direct result of this action."""

    NO_EFFECT = "no_effect"
    """The action ran and did not recover the money. A retry that declined
    again, or a contact the customer ignored."""

    NOT_APPLICABLE = "not_applicable"
    """Inert by nature -- a wait, a do-nothing, an escalation. Not a failure."""


@dataclass
class ExecutionResult:
    action: Action
    outcome: Outcome
    at: datetime
    recovered: Money = field(default_factory=Money.zero)
    cost: Money = field(default_factory=Money.zero)
    detail: str = ""

    @property
    def is_recovery(self) -> bool:
        return self.outcome is Outcome.RECOVERED


class Executor(Protocol):
    name: str

    def execute(self, action: Action, at: datetime) -> ExecutionResult: ...


@dataclass
class ExecutionCosts:
    """What an attempt costs the merchant. Same figures the policy engine uses
    to decide whether chasing is worth it, so the two cannot drift apart."""

    per_retry: Money = field(default_factory=lambda: Money.rupees(0.4))
    per_contact: Money = field(default_factory=lambda: Money.rupees(1.5))

    def for_action(self, action: Action) -> Money:
        if action.is_charging:
            return self.per_retry
        if action.is_contact:
            return self.per_contact
        return Money.zero()


@dataclass
class SimulatedExecutor:
    """Resolves actions against ground truth fixed before any arm ran.

    Once an order is recovered it stays recovered: later actions against the
    same subject report NO_EFFECT and still cost money. That is deliberate.
    An arm that keeps chasing an already-settled order should be charged for
    it, because a real merchant would be.
    """

    orders: dict[str, Order]
    recoverability: dict[str, Recoverability]
    subscriptions: dict[str, Subscription] = field(default_factory=dict)
    mandate_recovery: dict[str, MandateRecovery] = field(default_factory=dict)
    costs: ExecutionCosts = field(default_factory=ExecutionCosts)
    name: str = "simulated"
    settled: set[str] = field(default_factory=set, init=False)

    def execute(self, action: Action, at: datetime) -> ExecutionResult:
        cost = self.costs.for_action(action)

        if action.is_inert:
            return ExecutionResult(action, Outcome.NOT_APPLICABLE, at,
                                   detail=f"{action.type.value} moves no money")

        if action.subject_id in self.subscriptions:
            return self._execute_mandate(action, at, cost)

        order = self.orders.get(action.subject_id)
        rec = self.recoverability.get(action.subject_id)
        if order is None or rec is None:
            # Receivables have no latent model yet, so they are reported as
            # no-effect rather than silently counted as wins.
            return ExecutionResult(action, Outcome.NO_EFFECT, at, cost=cost,
                                   detail="no latent outcome model for this subject")

        if action.subject_id in self.settled:
            return ExecutionResult(action, Outcome.NO_EFFECT, at, cost=cost,
                                   detail="already recovered; this action was wasted")

        if action.type is ActionType.SWITCH_ROUTE:
            won = rec.route_switch_helps and rec.retry_would_succeed
            why = "another route was healthy" if won else "the route was not the problem"
        elif action.is_charging:
            won = rec.retry_succeeds_at(at)
            why = (
                "blocker had cleared" if won
                else "retried before the blocker cleared" if rec.retry_would_succeed
                else "this failure is not fixable by re-presenting"
            )
        else:
            won = rec.dunning_succeeds_at(at)
            why = "customer acted on the contact" if won else "customer did not act"

        if not won:
            return ExecutionResult(action, Outcome.NO_EFFECT, at, cost=cost, detail=why)

        self.settled.add(action.subject_id)
        return ExecutionResult(action, Outcome.RECOVERED, at,
                               recovered=order.amount, cost=cost, detail=why)

    def _execute_mandate(self, action: Action, at: datetime, cost: Money) -> ExecutionResult:
        """Resolve an action whose subject is a subscription, not an order.

        Two things are different from a payment and both are deliberate.

        A dead mandate cannot be *presented*, so a charging action against one
        recovers nothing however healthy the instrument behind it -- there is
        no authorisation to charge against. Only asking the customer to
        re-register can work.

        And what is credited is the year of billing that re-registration
        restores, not one charge. A mandate is a stream; pricing the recovery
        of a stream at one period would understate it by an order of
        magnitude. Recurring is reported on its own row for exactly this
        reason: the unit is not the same as a one-off payment's and adding the
        two together would produce a number that means nothing.
        """
        sub = self.subscriptions[action.subject_id]
        rec = self.mandate_recovery.get(action.subject_id)

        if action.is_charging:
            return ExecutionResult(action, Outcome.NO_EFFECT, at, cost=cost,
                                   detail="mandate is not active; there is nothing to present")
        if rec is None:
            return ExecutionResult(action, Outcome.NO_EFFECT, at, cost=cost,
                                   detail="mandate is active; nothing to recover")
        if action.subject_id in self.settled:
            return ExecutionResult(action, Outcome.NO_EFFECT, at, cost=cost,
                                   detail="already re-registered; this action was wasted")

        if not rec.reregisters_at(at):
            why = (
                "would have resumed unprompted; chasing recovered nothing extra"
                if rec.would_reregister and rec.resumes_unprompted
                else "asked after the customer had stopped paying attention"
                if rec.would_reregister
                else "customer would not re-authorise"
            )
            return ExecutionResult(action, Outcome.NO_EFFECT, at, cost=cost, detail=why)

        self.settled.add(action.subject_id)
        return ExecutionResult(action, Outcome.RECOVERED, at, recovered=sub.annual_value,
                               cost=cost, detail="customer re-authorised the mandate")
