"""Executing an approved action, and recording what actually happened.

Nothing reaches here that the policy engine did not permit. The executor's job
is narrow on purpose: take a permitted action, make it happen, and report the
outcome truthfully. It does not re-decide anything.

Two backends behind one interface:

*   `SimulatedExecutor` resolves an action against the batch's latent
    recoverability -- ground truth fixed at generation time, identical for
    every arm of the backtest. This is what makes arms comparable. It resolves
    all three surfaces: failed orders, the mandates behind recurring revenue,
    and overdue receivables.
*   `RazorpayExecutor` (in `razorpay.py`) satisfies the same protocol against
    the real test-mode API. It is not interchangeable with the simulator for
    *measurement* -- a live dispatch has no counterfactual and no synchronous
    outcome -- and `Outcome.DISPATCHED` exists to keep that distinction
    visible rather than papered over.

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
from backstop.domain.entities import Invoice, Order, Subscription
from backstop.domain.money import Money
from backstop.simulate.recoverability import (
    InvoiceRecovery,
    MandateRecovery,
    Recoverability,
)


class Outcome(StrEnum):
    RECOVERED = "recovered"
    """Money landed as a direct result of this action."""

    NO_EFFECT = "no_effect"
    """The action ran and did not recover the money. A retry that declined
    again, or a contact the customer ignored."""

    NOT_APPLICABLE = "not_applicable"
    """Inert by nature -- a wait, a do-nothing, an escalation. Not a failure."""

    DISPATCHED = "dispatched"
    """Handed to the real payment rails; the result is not knowable yet.

    Only a live adapter produces this. A simulated run resolves an action
    against latent truth the moment it fires, but a real link or order settles
    when a human being acts on it, hours or days later and out of band. It is
    deliberately neither RECOVERED nor NO_EFFECT: booking a dispatch as
    recovery would be the same error as crediting an invoice for arriving on
    the buyer's own cycle, and booking it as no-effect would write off money
    that has not had its chance yet. `RazorpayExecutor.reconcile` is what
    later turns one of these into an outcome.
    """


@dataclass(frozen=True)
class ExternalRef:
    """A handle on whatever a real backend created for an action.

    The audit trail is only as good as its ability to be checked, and "we sent
    a payment link" is checkable in a way "we contacted the customer" is not.
    A simulated run leaves this empty; a live one carries the id and the URL a
    reader can open.
    """

    entity: str
    """What kind of thing it is -- an order, a payment link, an auth link."""
    id: str
    url: str = ""

    def describe(self) -> str:
        return f"{self.entity} {self.id}" + (f"  {self.url}" if self.url else "")


@dataclass
class ExecutionResult:
    action: Action
    outcome: Outcome
    at: datetime
    recovered: Money = field(default_factory=Money.zero)
    cost: Money = field(default_factory=Money.zero)
    detail: str = ""
    external: ExternalRef | None = None
    """Set by a live backend. None for anything the simulator resolved."""

    @property
    def is_recovery(self) -> bool:
        return self.outcome is Outcome.RECOVERED

    @property
    def is_pending(self) -> bool:
        """Ran for real, and the outcome is not in yet."""
        return self.outcome is Outcome.DISPATCHED


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
    invoices: dict[str, Invoice] = field(default_factory=dict)
    invoice_recovery: dict[str, InvoiceRecovery] = field(default_factory=dict)
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

        if action.subject_id in self.invoices:
            return self._execute_receivable(action, at, cost)

        order = self.orders.get(action.subject_id)
        rec = self.recoverability.get(action.subject_id)
        if order is None or rec is None:
            # A subject with no latent model is reported as no-effect rather
            # than silently counted as a win.
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

    def _execute_receivable(self, action: Action, at: datetime, cost: Money) -> ExecutionResult:
        """Resolve an action whose subject is an invoice.

        Three things separate a receivable from a payment.

        There is nothing to *charge*. A B2B invoice has no instrument on file
        waiting to be re-presented -- collection happens when the buyer's
        accounts-payable department pays it -- so a charging action against one
        recovers nothing and says so rather than quietly reporting a decline.

        What is credited is only what would not have arrived anyway. Most
        overdue invoices are paid without anybody chasing, and an agent that
        mails those buyers and books the payment has measured the AP cycle.
        The latent model marks them non-incremental and they are worth nothing
        here however well the reminder was written.

        And a part-payment offer collects only the instalment it asked for.
        A buyer who cannot clear the balance in one go pays what they can,
        which is worth more than the nothing a fourth reminder would have
        collected. A buyer who *could* have paid in full pays the instalment
        and no more, so offering a split in the wrong place costs the merchant
        the remainder -- the offer is not free, and modelling it as strictly
        better than a reminder would make the playbook prefer it everywhere.
        """
        inv = self.invoices[action.subject_id]
        rec = self.invoice_recovery.get(action.subject_id)

        if action.is_charging:
            return ExecutionResult(action, Outcome.NO_EFFECT, at, cost=cost,
                                   detail="an invoice has no instrument to re-present")
        if rec is None:
            return ExecutionResult(action, Outcome.NO_EFFECT, at, cost=cost,
                                   detail="settled, not yet due, or disputed; no collections outcome")
        if action.subject_id in self.settled:
            return ExecutionResult(action, Outcome.NO_EFFECT, at, cost=cost,
                                   detail="already collected; this action was wasted")

        # An offer names the instalment it is asking for; a reminder asks for
        # the whole balance. The latent model needs to know which, because a
        # buyer who cannot pay in one piece answers only the first.
        offered: float | None = None
        if action.type is ActionType.OFFER_PART_PAYMENT and inv.outstanding.paise:
            offered = (action.amount_paise or 0) / inv.outstanding.paise
        part = offered is not None
        share = rec.collects_at(at, offered_share=offered)
        if not share:
            why = (
                "would have been paid on the buyer's own cycle; chasing collected nothing extra"
                if rec.pays_unprompted
                else "buyer cannot clear the balance in one payment; only an offer to split would land"
                if rec.needs_part_payment and rec.would_pay_if_chased
                else "chased after the invoice had gone cold"
                if rec.would_pay_if_chased
                else "buyer was never going to pay this"
            )
            return ExecutionResult(action, Outcome.NO_EFFECT, at, cost=cost, detail=why)

        self.settled.add(action.subject_id)
        collected = inv.outstanding * share
        detail = (
            f"buyer accepted a part payment of {share:.0%} of the balance"
            if part else "buyer paid the invoice"
        )
        return ExecutionResult(action, Outcome.RECOVERED, at, recovered=collected,
                               cost=cost, detail=detail)
