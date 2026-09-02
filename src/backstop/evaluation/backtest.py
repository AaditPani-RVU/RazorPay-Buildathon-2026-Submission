"""Replay one batch through three arms and report the difference.

This is the claim the whole project rests on, so the experiment is built to be
hard on itself:

*   **One batch, one world.** Every arm sees identical orders and identical
    latent recoverability, fixed before any arm ran. No arm can get a luckier
    draw than another.
*   **One engine, one set of contexts.** The policy engine evaluates every
    action in every arm, including the arms that ignore it. The only difference
    between a policed and an unpoliced arm is whether the ruling is *obeyed*.
    A violation count is therefore measured on identical terms rather than
    self-reported by the arm that would look best.
*   **Costs are charged.** An action that runs and recovers nothing still costs
    money. An arm that chases an already-settled order pays for the privilege.

The arms:

    do-nothing    the floor. Recovery causes nothing, so it recovers nothing.
    naive-retry   retry every failure on a fixed schedule, mail every customer.
                  Not a strawman -- it is the obvious implementation.
    backstop      detect, diagnose, plan, enforce, execute.

Read the result as a comparison under a stated world model, not as a
prediction of real recovery rates. The rates live in `simulate.recoverability`
and are configuration.

    python -m backstop.evaluation.backtest --seed 1
    python -m backstop.evaluation.backtest --offline --seeds 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Console
from rich.table import Table

from backstop.config import Settings
from backstop.decide.planner import (
    SELF_HEALING,
    Planner,
    RecoveryStrategy,
    expand,
    mandate_actions,
    naive_invoice_chase,
    naive_mandate_chase,
    naive_retry,
    orders_in_cluster,
    receivable_actions,
    tail_actions,
)
from backstop.detect.correlate import correlate
from backstop.detect.multires import MultiResolutionDetector
from backstop.diagnose.diagnoser import Diagnoser
from backstop.diagnose.evidence import EvidenceBuilder
from backstop.domain.actions import Action
from backstop.domain.declines import RootCause
from backstop.domain.entities import utc
from backstop.domain.money import Money
from backstop.execute.executor import SimulatedExecutor
from backstop.ledger.ledger import LedgerEntry, RecoveryLedger, Surface, Violation
from backstop.llm import GroqProvider, LLMClient, ScriptedProvider
from backstop.policy.engine import Disposition, PolicyEngine
from backstop.policy.subjects import ContactBook, SubjectIndex
from backstop.simulate.generator import SimConfig, generate
from backstop.simulate.scenario import Scenario

console = Console(highlight=False)


@dataclass
class Proposal:
    """Everything the Backstop pipeline wants to do, and where it came from."""

    actions: list[Action] = field(default_factory=list)
    from_model: int = 0
    """Actions expanded from an LLM strategy for a diagnosed incident."""
    from_tail: int = 0
    """Actions from the deterministic playbook, for failures with no incident."""
    from_mandates: int = 0
    """Actions from the mandate scan, for recurring revenue that stopped."""
    from_receivables: int = 0
    """Actions from the aged ledger, for invoices that were never paid."""
    diagnosis_by_order: dict[str, RootCause] = field(default_factory=dict)
    outage_by_order: dict[str, datetime] = field(default_factory=dict)
    llm_calls: int = 0
    backend: str = ""
    strategies: list[tuple[str, RootCause, RecoveryStrategy]] = field(default_factory=list)


@dataclass
class ArmResult:
    name: str
    ledger: RecoveryLedger
    violations: list[Violation] = field(default_factory=list)
    llm_calls: int = 0
    note: str = ""

    @property
    def recovered(self) -> Money:
        return self.ledger.recovered

    @property
    def net(self) -> Money:
        return self.ledger.net


def run_arm(
    name: str,
    actions: list[Action],
    scenario: Scenario,
    *,
    enforce: bool,
    diagnosis_by_order: dict[str, RootCause],
    outage_by_order: dict[str, datetime],
) -> ArmResult:
    """Execute one arm's actions in schedule order.

    The engine runs for every arm. `enforce` decides only whether its rulings
    are obeyed, which is what makes the violation counts comparable: an
    unpoliced arm is judged by the same rules, in the same contexts, at the
    same moments.
    """
    engine = PolicyEngine()
    index = SubjectIndex.of(scenario)
    executor = SimulatedExecutor(
        orders=index.orders,
        recoverability=scenario.recoverability,
        subscriptions=index.subscriptions,
        mandate_recovery=scenario.mandate_recovery,
        invoices=index.invoices,
        invoice_recovery=scenario.invoice_recovery,
    )
    ledger = RecoveryLedger(arm=name)
    violations: list[Violation] = []

    # Contact history accumulates as the arm runs, so frequency caps and
    # fatigue see what this arm has actually already sent.
    book = ContactBook()

    for action in sorted(actions, key=lambda a: utc(a.scheduled_at)):
        # An action's subject names its surface. On payments that is the
        # failed order; on recurring it is the subscription itself, because
        # re-registering a lapsed mandate acts on the authorisation rather
        # than on any one presentation of it; on receivables it is the invoice.
        subject = index.resolve(action.subject_id)
        surface = subject.surface
        customer = subject.customer
        ctx = index.context(
            action,
            contacts=book,
            diagnosis=diagnosis_by_order.get(action.subject_id),
            outage_until=outage_by_order.get(action.subject_id),
        )
        ruling = engine.evaluate(action, ctx)

        if enforce and not ruling.allowed:
            ledger.record(
                LedgerEntry(action=action, surface=surface, ruling=ruling, execution=None)
            )
            continue

        final = (ruling.final if enforce else None) or action
        at = utc(final.scheduled_at)
        result = executor.execute(final, at)
        ledger.record(
            LedgerEntry(action=final, surface=surface, ruling=ruling, execution=result)
        )

        if ruling.disposition is Disposition.DENY:
            for v in ruling.verdicts:
                if v.disposition is Disposition.DENY:
                    violations.append(Violation(final, v.rule_id, v.reason, surface))
                    break

        book.record(final, customer_id=customer.id if customer else "", at=at)

    return ArmResult(name=name, ledger=ledger, violations=violations)


def build_backstop_actions(
    scenario: Scenario, *, offline: bool, model: str | None
) -> Proposal:
    """Run detect -> diagnose -> plan and return everything the arm proposes."""
    attempts = [a for o in scenario.orders for a in o.attempts]
    clusters = correlate(MultiResolutionDetector().run(attempts))

    settings = Settings.load()
    use_model = not offline and settings.has_groq
    if use_model:
        provider = GroqProvider(api_key=settings.groq_api_key)
        backend = model or settings.reasoning_model
    else:
        provider = ScriptedProvider(responses=[])
        backend = "offline: tail playbook only, no model"

    proposal = Proposal(backend=backend)
    diagnosis_by_order = proposal.diagnosis_by_order
    outage_by_order = proposal.outage_by_order
    actions = proposal.actions
    planned_ids: set[str] = set()

    if use_model:
        client = LLMClient(provider=provider, model=model or settings.reasoning_model)
        diagnoser = Diagnoser(client)
        planner = Planner(client)
        builder = EvidenceBuilder(attempts)

        degraded = 0
        for cluster in clusters:
            members = orders_in_cluster(cluster, scenario.orders)
            if not members:
                continue
            diag = diagnoser.diagnose(builder.build(cluster))
            if not diag.ok:
                # No diagnosis means no strategy, and these orders fall through
                # to the tail playbook below. Recovery degrades to the
                # deterministic path rather than stopping, which is the whole
                # point of having one.
                degraded += 1
                continue
            cause = diag.diagnosis.root_cause
            hold = cluster.ends_at if cause in SELF_HEALING else None
            for o in members:
                diagnosis_by_order[o.id] = cause
                if hold is not None:
                    outage_by_order[o.id] = hold

            strategy = planner.plan(cluster, diag.diagnosis, members)
            if not strategy.ok:
                continue
            proposal.strategies.append((cluster.id, cause, strategy.strategy))
            actions.extend(expand(strategy.strategy, members, outage_until=hold))
            planned_ids.update(o.id for o in members)
        proposal.llm_calls = client.stats.calls
        proposal.from_model = len(actions)
        if degraded:
            proposal.backend += f"  ({degraded} cluster(s) fell back to the tail playbook)"

    # The long tail: everything that belonged to no diagnosed incident.
    for order in scenario.failed_orders:
        if order.id not in planned_ids:
            actions.extend(tail_actions(order))
    proposal.from_tail = len(actions) - proposal.from_model

    # The recurring surface. Not an incident and not a failed order: a scan of
    # the book for authorisations that stopped being able to collect. It runs
    # whether or not a model was reachable, because nothing here needs one.
    for sub in scenario.lapsed_subscriptions:
        actions.extend(mandate_actions(sub, scenario.ends_at))
    proposal.from_mandates = len(actions) - proposal.from_model - proposal.from_tail

    # The receivables surface. A third non-incident: the aged ledger, where
    # what selects the response is how old the money is rather than what went
    # wrong with it. Deterministic for the same reason as the other two.
    before = len(actions)
    for inv in scenario.overdue_invoices:
        actions.extend(receivable_actions(inv, scenario.ends_at))
    proposal.from_receivables = len(actions) - before

    return proposal


def run(scenario: Scenario, *, offline: bool, model: str | None) -> list[ArmResult]:
    failed = scenario.failed_orders

    arms: list[ArmResult] = [
        run_arm("do-nothing", [], scenario, enforce=False,
                diagnosis_by_order={}, outage_by_order={}),
    ]

    # The naive arm works all three surfaces, or the comparison would be
    # unfair in Backstop's favour: an arm that never touches a book of dead
    # mandates or an aged debtors ledger cannot be criticised for how it
    # touches them.
    naive_actions = (
        naive_retry(failed)
        + naive_mandate_chase(scenario.subscriptions, scenario.ends_at)
        + naive_invoice_chase(scenario.invoices, scenario.ends_at)
    )
    with console.status("naive-retry..."):
        arms.append(
            run_arm("naive-retry", naive_actions, scenario, enforce=False,
                    diagnosis_by_order={}, outage_by_order={})
        )

    with console.status("planning: detect, diagnose, strategy..."):
        prop = build_backstop_actions(scenario, offline=offline, model=model)

    # The controlled comparison. Both arms below execute the *same* proposed
    # actions from the *same* planner. The only difference is whether the
    # policy engine's rulings are obeyed, which is the whole thesis reduced to
    # one variable.
    with console.status("planner, unpoliced..."):
        unpoliced = run_arm(
            "planner-unpoliced", prop.actions, scenario, enforce=False,
            diagnosis_by_order=prop.diagnosis_by_order,
            outage_by_order=prop.outage_by_order,
        )
    unpoliced.note = "same actions as backstop, policy engine ignored"
    arms.append(unpoliced)

    with console.status("backstop: enforce and execute..."):
        result = run_arm(
            "backstop", prop.actions, scenario, enforce=True,
            diagnosis_by_order=prop.diagnosis_by_order,
            outage_by_order=prop.outage_by_order,
        )
    result.llm_calls = prop.llm_calls
    result.note = prop.backend
    arms.append(result)
    return arms, prop


def _arm_table(arms: list[ArmResult], surface: Surface, subject: str) -> Table:
    """One surface's results. Surfaces are tabled apart rather than summed --
    a recovered payment and a restored year of billing are not the same unit,
    and a single total mixing them would mean nothing."""
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("arm", no_wrap=True)
    for col in ("recovered", "of which illegal", "keepable net", subject,
                "charges", "contacts", "burst", "violations"):
        table.add_column(col, justify="right", no_wrap=True)

    for arm in arms:
        led = arm.ledger.on(surface)
        style = "green" if arm.name == "backstop" else ""
        viol = sum(1 for v in arm.violations if v.surface is surface)
        viol_txt = f"[green]{viol}[/green]" if viol == 0 else f"[red]{viol}[/red]"
        illegal = led.recovered_in_violation
        table.add_row(
            f"[{style}]{arm.name}[/{style}]" if style else arm.name,
            led.recovered.format(),
            f"[red]{illegal.format()}[/red]" if illegal else "-",
            led.compliant_net.format(),
            f"{led.orders_recovered:,}",
            f"{led.charges_attempted:,}",
            f"{led.contacts_sent:,}",
            str(led.worst_contact_burst),
            viol_txt,
        )
    return table


def render(scenario: Scenario, arms: list[ArmResult], prop: Proposal) -> None:
    console.print("[bold]payments[/bold]  one-off checkout failures\n")
    console.print(_arm_table(arms, Surface.PAYMENT, "orders"))

    console.print("\n[bold]recurring[/bold]  mandates that stopped collecting\n")
    console.print(_arm_table(arms, Surface.RECURRING, "mandates"))
    console.print(
        "  [dim]recurring recovery is a restored *year* of billing, not one charge, so it\n"
        "  is reported apart from payments rather than added to them. Credit is\n"
        "  incremental only: a paused mandate that would have resumed unprompted counts\n"
        f"  for nothing. {scenario.recurring_at_risk} per year was at risk across "
        f"{len(scenario.lapsed_subscriptions):,} lapsed mandates.[/dim]"
    )

    console.print("\n[bold]receivables[/bold]  invoices that were never paid\n")
    console.print(_arm_table(arms, Surface.RECEIVABLE, "invoices"))
    # Ground truth, read by the eval rather than by the agent. Printing the
    # counterfactual beside the result is the only way the receivables number
    # can be read correctly: without it, an arm that chased everything and
    # booked whatever arrived would look like the best collections team alive.
    outstanding = {i.id: i.outstanding for i in scenario.invoices}
    self_cure = Money.zero()
    incremental = Money.zero()
    for rec in scenario.invoice_recovery.values():
        amount = outstanding.get(rec.invoice_id, Money.zero())
        if rec.pays_unprompted:
            self_cure += amount
        elif rec.would_pay_if_chased:
            incremental += amount
    console.print(
        f"  [dim]{scenario.receivables_at_risk} outstanding across "
        f"{len(scenario.overdue_invoices):,} overdue invoices. Of that, "
        f"{self_cure} arrives on the\n"
        f"  buyer's own cycle whether anybody acts or not, and is credited to nobody; "
        f"{incremental}\n"
        "  is the most any collections effort could add. Credit is incremental here too,\n"
        "  and it bites hardest on this surface. A part-payment offer collects a share\n"
        "  of the balance, not the whole.[/dim]"
    )

    total_at_risk = Money.zero()
    for o in scenario.orders:
        total_at_risk += o.amount_at_risk
    console.print(f"\n  payment value at risk in this batch: {total_at_risk}")
    console.print(
        "  [dim]of which illegal = recovered by actions the rules refuse, so a merchant\n"
        "  could not keep it. keepable net = the rest, less cost. burst = most contacts\n"
        "  any one subject received.[/dim]"
    )

    naive = next((a for a in arms if a.name == "naive-retry"), None)
    back = next((a for a in arms if a.name == "backstop"), None)
    if naive and back:
        at_risk = total_at_risk.as_rupees or 1.0
        naive_pay = naive.ledger.on(Surface.PAYMENT).recovered
        back_pay = back.ledger.on(Surface.PAYMENT).recovered
        console.print(
            f"  share of it recovered: naive {naive_pay.as_rupees / at_risk:.1%}, "
            f"backstop {back_pay.as_rupees / at_risk:.1%}"
        )

    for arm in arms:
        if not arm.violations:
            continue
        console.print(f"\n[bold red]{arm.name} broke {len(arm.violations)} rules[/bold red]")
        by_rule: dict[str, int] = {}
        for v in arm.violations:
            by_rule[v.rule_id] = by_rule.get(v.rule_id, 0) + 1
        for rule_id, n in sorted(by_rule.items(), key=lambda kv: -kv[1]):
            example = next(v.reason for v in arm.violations if v.rule_id == rule_id)
            console.print(f"    {rule_id:<24} {n:>6}  [dim]{example}[/dim]")

    # The controlled comparison: identical actions, only obedience differs.
    loose = next((a for a in arms if a.name == "planner-unpoliced"), None)
    if loose and back:
        console.print("\n[bold]the leash, isolated[/bold]")
        console.print(
            "  [dim]Both arms below ran the same proposed actions from the same planner.\n"
            "  The only variable is whether the policy engine was obeyed.[/dim]\n"
        )
        for surface in back.ledger.surfaces:
            lo, hi = loose.ledger.on(surface), back.ledger.on(surface)
            console.print(
                f"  {surface.value:<11} keepable  unpoliced {lo.compliant_net}   "
                f"policed {hi.compliant_net}   delta {hi.compliant_net - lo.compliant_net}"
            )
        console.print(
            f"  contacts    unpoliced {loose.ledger.contacts_sent:,}   "
            f"policed {back.ledger.contacts_sent:,}"
        )
        console.print(
            f"  violations  unpoliced [red]{len(loose.violations):,}[/red]   "
            f"policed [green]{len(back.violations):,}[/green]"
        )

    if back:
        led = back.ledger
        console.print("\n[bold]what the policy engine did in the backstop arm[/bold]")
        console.print(
            f"  proposed          {led.proposed:,}   "
            f"[dim]({prop.from_model:,} from the model, {prop.from_tail:,} from the "
            f"tail playbook, {prop.from_mandates:,} from the mandate scan, "
            f"{prop.from_receivables:,} from the aged ledger)[/dim]"
        )
        console.print(f"  vetoed            {len(led.vetoed):,}")
        console.print(f"  rescheduled       {len(led.rescheduled):,}")
        console.print(f"  held for approval {len(led.held_for_approval):,}")
        console.print(f"  executed          {len(led.executed):,}")
        if led.vetoes_by_rule():
            console.print("\n  vetoes by rule:")
            for rule_id, n in led.vetoes_by_rule().items():
                console.print(f"    {rule_id:<24} {n:>6}")
        console.print(f"\n  reasoning backend: {back.note}  ({back.llm_calls} llm calls)")

    if prop.strategies:
        console.print("\n[bold]what the model proposed, per diagnosed incident[/bold]")
        for cluster_id, cause, strategy in prop.strategies:
            console.print(f"  [bold]{cluster_id}[/bold]  {cause.value}")
            for plan in strategy.plans:
                chan = f" via {plan.channel.value}" if plan.channel else ""
                console.print(
                    f"    {plan.retry_class.value:<22} {plan.action.value}{chan}  "
                    f"+{plan.delay_hours:g}h  x{plan.max_attempts}"
                )


def main() -> None:
    ap = argparse.ArgumentParser(description="Three-arm recovery backtest.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--orders-per-day", type=int, default=20000)
    ap.add_argument("--offline", action="store_true", help="no API calls; tail playbook only")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    console.print()
    console.print(
        f"[bold]BACKTEST[/bold]  seed {args.seed}, {args.days}d, "
        f"{args.orders_per_day:,} orders/day\n"
    )
    with console.status("generating batch..."):
        scenario = generate(
            SimConfig(seed=args.seed, days=args.days, orders_per_day=args.orders_per_day)
        )
    console.print(
        f"  {len(scenario.orders):,} orders, {len(scenario.failed_orders):,} failed, "
        f"{len(scenario.incidents)} incidents\n"
    )

    arms, prop = run(scenario, offline=args.offline, model=args.model)
    render(scenario, arms, prop)
    console.print()


if __name__ == "__main__":
    main()
