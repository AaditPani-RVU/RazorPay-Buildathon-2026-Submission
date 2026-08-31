"""Walk one batch through the pipeline, stage by stage, and show the work.

This exists so the system can be *watched* rather than described. Every number
printed is computed live from a seeded batch; nothing here is a recorded
transcript. Ground truth is printed alongside each prediction, so a reader can
check the pipeline rather than take its word.

    python -m backstop.demo                      # full walkthrough, live model
    python -m backstop.demo --offline            # no API calls; diagnosis is scripted
    python -m backstop.demo --stage policy       # just the safety boundary
    python -m backstop.demo --stage receivables  # just the aged ledger
    python -m backstop.demo --stage razorpay     # live dispatch, test-mode keys
    python -m backstop.demo --seed 3

The pipeline is Detect -> Diagnose -> Decide -> Enforce -> Execute -> Measure,
and the walkthrough runs all of it across the three revenue surfaces: payment
failures, lapsed mandates and overdue receivables. The Execute stage is the
only one that reaches outside the process: with test-mode keys configured it
dispatches three permitted actions onto Razorpay's real API, silently, and
skips itself entirely under --offline.

The Enforce stage is deliberately not driven by the planner. It puts
hand-written probes through the policy engine instead -- including ones that
must be refused -- because the guarantee worth watching is that the rules hold
against *any* proposed action, and a stage that only ever showed the planner's
own output could not demonstrate that. The probes are labelled as probes.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

from rich.console import Console
from rich.table import Table

from backstop.config import Settings
from backstop.detect.correlate import RiskCluster, correlate
from backstop.detect.multires import MultiResolutionDetector
from backstop.diagnose.diagnoser import Diagnoser, DiagnosisResult
from backstop.diagnose.evidence import EvidenceBuilder
from backstop.domain.actions import Action, ActionType
from backstop.domain.declines import DeclineCode, RootCause
from backstop.domain.entities import Channel, ContactRecord, Order, utc
from backstop.domain.money import Money
from backstop.evaluation import detection_score, diagnosis_score
from backstop.evaluation.bench import scope_is_exact
from backstop.llm import GroqProvider, LLMClient, ScriptedProvider
from backstop.policy.engine import IST, Disposition, PolicyContext, PolicyEngine
from backstop.simulate.generator import SimConfig, generate
from backstop.simulate.scenario import Scenario

console = Console(highlight=False)

DISPOSITION_STYLE = {
    Disposition.ALLOW: "green",
    Disposition.DENY: "red",
    Disposition.RESCHEDULE: "yellow",
    Disposition.REQUIRE_APPROVAL: "magenta",
}


def rule(title: str) -> None:
    console.rule(f"[bold]{title}", style="dim")
    console.print()


# --------------------------------------------------------------------------
# Stage 1 -- the world, and the truth about it
# --------------------------------------------------------------------------


def stage_scenario(cfg: SimConfig) -> Scenario:
    rule("1. SIMULATE  a seeded batch with labelled incidents")
    console.print(
        "[dim]No real transaction data exists for this, so the batch is generated. That is\n"
        "what makes the rest of the walkthrough checkable: every incident below was\n"
        "planted deliberately, so detection and diagnosis can be scored against truth\n"
        "rather than against a vibe. The detector never sees any of it.[/dim]\n"
    )
    with console.status("generating..."):
        scenario = generate(cfg)
    console.print(scenario.summary())
    console.print()
    return scenario


# --------------------------------------------------------------------------
# Stage 2 -- detection
# --------------------------------------------------------------------------


def stage_detect(scenario: Scenario) -> list[RiskCluster]:
    rule("2. DETECT  segmented anomaly detection, no LLM")
    console.print(
        "[dim]Statistics, not a model: this runs over every attempt in the batch and has to\n"
        "answer in seconds. Three resolutions run at once (15/60/240m) because a single\n"
        "BIN carries about four attempts per quarter hour and is unmeasurable at the\n"
        "fine grain. Overlapping signals are then rolled up to the scope the problem\n"
        "actually has.[/dim]\n"
    )
    attempts = [a for o in scenario.orders for a in o.attempts]
    with console.status("scanning..."):
        signals = MultiResolutionDetector().run(attempts)
        clusters = correlate(signals)

    console.print(
        f"  {len(attempts):,} attempts  ->  {len(signals)} raw signals  ->  "
        f"{len(clusters)} correlated clusters\n"
    )
    for c in clusters:
        console.print(f"  {c.describe()}")
    console.print()

    report = detection_score.score(scenario, [c.primary for c in clusters])
    exact = sum(scope_is_exact(i, report) for i in scenario.incidents)
    console.print("[bold]scored against ground truth[/bold]\n")
    console.print(report.render())
    console.print(
        f"\n  exact scope  {exact}/{len(scenario.incidents)} incidents reported at their own slice"
    )
    console.print()
    return clusters


# --------------------------------------------------------------------------
# Stage 3 -- diagnosis
# --------------------------------------------------------------------------


def _scripted_for(cluster: RiskCluster) -> str:
    """A canned answer keyed off the dominant decline code.

    Used only in --offline mode. This is a lookup table, not a model, and the
    walkthrough labels it that way -- otherwise the demo would be claiming
    reasoning it did not do.
    """
    code = cluster.primary.dominant_decline
    guess = {
        DeclineCode.ISSUER_UNAVAILABLE: RootCause.ISSUER_OUTAGE,
        DeclineCode.PSP_UNAVAILABLE: RootCause.PSP_OR_RAIL_OUTAGE,
        DeclineCode.GATEWAY_TIMEOUT: RootCause.GATEWAY_ROUTING_DEGRADATION,
        DeclineCode.DO_NOT_HONOR: RootCause.BIN_SPECIFIC_DECLINE,
        DeclineCode.OTP_ABANDONED: RootCause.AUTHENTICATION_DROPOFF,
        DeclineCode.AUTH_3DS_FAILED: RootCause.AUTHENTICATION_DROPOFF,
        DeclineCode.INSUFFICIENT_FUNDS: RootCause.INSUFFICIENT_FUNDS_CLUSTER,
    }.get(code, RootCause.NO_SYSTEMIC_CAUSE)
    seen = code.value if code else "none"
    return (
        f'{{"root_cause": "{guess.value}", "confidence": 0.5, '
        f'"locus": "{cluster.segment.describe()}", '
        f'"key_evidence": ["scripted: dominant decline is {seen}"], "ruled_out": []}}'
    )


def stage_diagnose(
    scenario: Scenario, clusters: list[RiskCluster], *, offline: bool, model: str | None
) -> list[tuple[RiskCluster, DiagnosisResult]]:
    rule("3. DIAGNOSE  root-cause attribution, LLM")
    console.print(
        "[dim]Detection says money is leaking and where. Diagnosis says why, which decides\n"
        "whether the answer is a retry, a re-engagement, a route switch or nothing. The\n"
        "evidence bundle always carries healthy peers: an issuer outage and a routing\n"
        "fault look identical from the failing slice alone. Output is a closed enum, so\n"
        "the policy engine downstream can switch on it.[/dim]\n"
    )

    settings = Settings.load()
    if offline:
        provider = ScriptedProvider(responses=[_scripted_for(c) for c in clusters])
        target = "scripted lookup table (NOT a model)"
    elif not settings.has_groq:
        console.print("[yellow]  GROQ_API_KEY not set -- falling back to --offline.[/yellow]\n")
        provider = ScriptedProvider(responses=[_scripted_for(c) for c in clusters])
        target = "scripted lookup table (NOT a model)"
    else:
        provider = GroqProvider(api_key=settings.groq_api_key)
        target = model or settings.reasoning_model

    client = LLMClient(provider=provider, model=model or settings.reasoning_model)
    diagnoser = Diagnoser(client)
    builder = EvidenceBuilder([a for o in scenario.orders for a in o.attempts])

    console.print(f"  reasoning backend: [bold]{target}[/bold]\n")

    pairs: list[tuple[RiskCluster, DiagnosisResult]] = []
    for cluster in clusters:
        bundle = builder.build(cluster)
        with console.status(f"diagnosing {cluster.id}..."):
            result = diagnoser.diagnose(bundle)
        pairs.append((cluster, result))

        truth = detection_score.match(cluster.primary, scenario.incidents)
        truth_name = truth.root_cause.value if truth else "(no matching incident)"

        console.print(f"[bold]{cluster.id}[/bold]  {cluster.segment.describe()}")
        if not result.ok:
            console.print(f"  [red]no usable diagnosis[/red]: {result.error}")
            console.print(f"  truth      {truth_name}\n")
            continue
        d = result.diagnosis
        hit = truth is not None and d.root_cause is truth.root_cause
        mark = "[green]correct[/green]" if hit else "[red]wrong[/red]"
        console.print(f"  predicted  {d.root_cause.value}  ({mark}, confidence {d.confidence:.2f})")
        console.print(f"  truth      {truth_name}")
        console.print(f"  locus      {d.locus}")
        for e in d.key_evidence:
            console.print(f"    [dim]- {e}[/dim]")
        if d.ruled_out:
            console.print(f"  [dim]ruled out  {', '.join(c.value for c in d.ruled_out)}[/dim]")
        if result.repaired:
            console.print("  [yellow]schema repair was needed[/yellow]")
        console.print()

    report = diagnosis_score.score(scenario, pairs)
    console.print("[bold]scored against ground truth[/bold]\n")
    console.print(report.render())
    s = client.stats
    console.print(
        f"\n  llm calls {s.calls}  repairs {s.repairs}  failures {s.failures}  "
        f"tokens {s.prompt_tokens}+{s.completion_tokens}"
    )
    console.print()
    return pairs


# --------------------------------------------------------------------------
# Stage 4 -- the policy engine
# --------------------------------------------------------------------------


def _ist(base: datetime, hour: int, minute: int = 0) -> datetime:
    """A UTC instant that lands at the given IST wall-clock time."""
    return base.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        hours=hour - 5, minutes=minute - 30
    )


def _pick(orders: list[Order], code: DeclineCode, **bounds) -> Order | None:
    lo, hi = bounds.get("above"), bounds.get("below")
    for o in orders:
        if o.last_decline is not code:
            continue
        if lo is not None and o.amount < lo:
            continue
        if hi is not None and o.amount >= hi:
            continue
        return o
    return None


def stage_policy(scenario: Scenario, pairs) -> None:
    rule("4. ENFORCE  the policy engine -- the safety boundary")
    console.print(
        "[dim]Everything above is advisory. The planner that will feed this stage is not\n"
        "built yet, so the actions below are hand-written probes: the things a planner\n"
        "might reasonably propose, plus the things it must never be allowed to do. Each\n"
        "one names a real order, invoice or customer from the batch above.\n\n"
        "Watch the rule ids. Every ruling names the rules that produced it, which is\n"
        "what makes 'zero policy violations' a claim about a log rather than a hope.[/dim]\n"
    )

    engine = PolicyEngine()
    failed = scenario.failed_orders
    now = scenario.ends_at
    day = now - timedelta(days=1)
    cust = scenario.customers

    probes: list[tuple[str, str, Action, PolicyContext]] = []

    def add(label, expect, action, ctx):
        probes.append((label, expect, action, ctx))

    # -- things that must never happen -----------------------------------
    stolen = _pick(failed, DeclineCode.STOLEN_OR_LOST_CARD)
    if stolen:
        add(
            "Retry a card reported stolen",
            "DENY",
            Action(type=ActionType.RETRY_PAYMENT, subject_id=stolen.id,
                   scheduled_at=now + timedelta(hours=1),
                   rationale="the amount is material and the order is unpaid"),
            PolicyContext(now=now, order=stolen, customer=cust.get(stolen.customer_id)),
        )
        add(
            "Dun the customer whose card was stolen",
            "DENY",
            Action(type=ActionType.SEND_DUNNING, subject_id=stolen.id,
                   scheduled_at=_ist(day, 11), channel=Channel.EMAIL,
                   rationale="ask them to complete payment with another method"),
            PolicyContext(now=now, order=stolen, customer=cust.get(stolen.customer_id)),
        )

    expired = _pick(failed, DeclineCode.CARD_EXPIRED)
    if expired:
        add(
            "Re-present an expired card",
            "DENY",
            Action(type=ActionType.RETRY_PAYMENT, subject_id=expired.id,
                   scheduled_at=now + timedelta(hours=6),
                   rationale="maybe it works on a second attempt"),
            PolicyContext(now=now, order=expired, customer=cust.get(expired.customer_id)),
        )

    disputed = next((i for i in scenario.invoices if i.disputed_at), None)
    if disputed:
        add(
            "Chase a disputed invoice",
            "DENY",
            Action(type=ActionType.SEND_DUNNING, subject_id=disputed.id,
                   scheduled_at=_ist(day, 11), channel=Channel.EMAIL,
                   rationale="it is overdue and unpaid"),
            PolicyContext(now=now, invoice=disputed),
        )

    dnd_order = next(
        (o for o in failed
         if (c := cust.get(o.customer_id)) and c.dnd_registered
         and Channel.SMS in c.consented_channels and o.amount >= Money.rupees(500)),
        None,
    )
    if dnd_order:
        add(
            "SMS a customer on the DND registry",
            "DENY",
            Action(type=ActionType.SEND_DUNNING, subject_id=dnd_order.id,
                   scheduled_at=_ist(day, 11), channel=Channel.SMS,
                   rationale="SMS gets read faster than email"),
            PolicyContext(now=now, order=dnd_order, customer=cust.get(dnd_order.customer_id)),
        )

    tiny = min(
        (o for o in failed
         if o.last_decline and o.last_decline.spec.max_retries == 0),
        key=lambda o: o.amount, default=None,
    )
    if tiny:
        add(
            f"Chase a balance of {tiny.amount}",
            "DENY",
            Action(type=ActionType.SEND_DUNNING, subject_id=tiny.id,
                   scheduled_at=_ist(day, 11), channel=Channel.EMAIL,
                   rationale="every rupee counts"),
            PolicyContext(now=now, order=tiny, customer=cust.get(tiny.customer_id)),
        )

    nsf = _pick(failed, DeclineCode.INSUFFICIENT_FUNDS, above=Money.rupees(500))
    if nsf:
        add(
            "Retry, when diagnosis says customers are abandoning at OTP",
            "DENY",
            Action(type=ActionType.RETRY_PAYMENT, subject_id=nsf.id,
                   scheduled_at=nsf.last_attempt.at + timedelta(days=2),
                   rationale="the decline code is technically retryable"),
            PolicyContext(now=now, order=nsf, customer=cust.get(nsf.customer_id),
                          diagnosis=RootCause.AUTHENTICATION_DROPOFF),
        )
        add(
            "A fourth contact on the same order inside 14 days",
            "DENY",
            Action(type=ActionType.SEND_DUNNING, subject_id=nsf.id,
                   scheduled_at=_ist(day, 11), channel=Channel.EMAIL,
                   rationale="they have not responded yet"),
            PolicyContext(
                now=now, order=nsf, customer=cust.get(nsf.customer_id),
                contacts=[
                    ContactRecord(id=f"c{n}", customer_id=nsf.customer_id,
                                  channel=Channel.EMAIL, at=now - timedelta(days=3 * n + 1),
                                  subject_ref=nsf.id)
                    for n in range(3)
                ],
            ),
        )

    # -- things that are permitted, but only on the system's terms --------
    if nsf:
        add(
            "Retry an NSF decline 30 seconds later",
            "RESCHEDULE to +24h",
            Action(type=ActionType.RETRY_PAYMENT, subject_id=nsf.id,
                   scheduled_at=nsf.last_attempt.at + timedelta(seconds=30),
                   rationale="try again immediately"),
            PolicyContext(now=now, order=nsf, customer=cust.get(nsf.customer_id)),
        )

    night = next(
        (o for o in failed
         if (c := cust.get(o.customer_id)) and not c.dnd_registered
         and Channel.SMS in c.consented_channels and o.amount >= Money.rupees(500)),
        None,
    )
    if night:
        add(
            "SMS at 03:00 IST",
            "RESCHEDULE to 09:00 IST",
            Action(type=ActionType.SEND_DUNNING, subject_id=night.id,
                   scheduled_at=_ist(day, 3), channel=Channel.SMS,
                   rationale="send the reminder now"),
            PolicyContext(now=now, order=night, customer=cust.get(night.customer_id)),
        )

    timeout = _pick(failed, DeclineCode.GATEWAY_TIMEOUT, above=Money.rupees(500))
    if timeout:
        add(
            "Retry into an outage that is still open",
            "RESCHEDULE past the outage",
            Action(type=ActionType.RETRY_PAYMENT, subject_id=timeout.id,
                   scheduled_at=timeout.last_attempt.at + timedelta(minutes=10),
                   rationale="the instrument is fine, it was a timeout"),
            PolicyContext(now=now, order=timeout, customer=cust.get(timeout.customer_id),
                          outage_until=timeout.last_attempt.at + timedelta(hours=2)),
        )
        add(
            "Retry a gateway timeout once the outage has cleared",
            "ALLOW",
            Action(type=ActionType.RETRY_PAYMENT, subject_id=timeout.id,
                   scheduled_at=timeout.last_attempt.at + timedelta(minutes=10),
                   rationale="transient infrastructure failure, instrument is healthy"),
            PolicyContext(now=now, order=timeout, customer=cust.get(timeout.customer_id)),
        )

    big = max(
        (i for i in scenario.invoices if i.is_chaseable(now)),
        key=lambda i: i.outstanding, default=None,
    )
    if big:
        add(
            f"Chase a {big.outstanding} receivable",
            "REQUIRE_APPROVAL",
            Action(type=ActionType.SEND_DUNNING, subject_id=big.id,
                   scheduled_at=_ist(day, 11), channel=Channel.EMAIL,
                   rationale="materially overdue, no dispute on record"),
            PolicyContext(now=now, invoice=big),
        )

    if stolen:
        add(
            "Escalate the stolen card to the risk team",
            "ALLOW",
            Action(type=ActionType.ESCALATE_TO_RISK, subject_id=stolen.id,
                   scheduled_at=now, rationale="fraud signal, recovery must not touch this"),
            PolicyContext(now=now, order=stolen, customer=cust.get(stolen.customer_id)),
        )

    # -- run them ---------------------------------------------------------
    counts: dict[Disposition, int] = {d: 0 for d in Disposition}
    surprises: list[str] = []

    for label, expect, action, ctx in probes:
        ruling = engine.evaluate(action, ctx)
        counts[ruling.disposition] += 1
        style = DISPOSITION_STYLE[ruling.disposition]
        got = ruling.disposition.value.upper()
        agrees = expect.split()[0] == got
        if not agrees:
            surprises.append(f"{label}: expected {expect}, got {got}")

        console.print(f"[bold]{label}[/bold]  [dim]({action.subject_id})[/dim]")
        console.print(f"  proposed   {action.describe()}")
        console.print(f"  ruling     [{style}]{got}[/{style}]  [dim]expected {expect}[/dim]")
        if ruling.final and utc(ruling.final.scheduled_at) != utc(action.scheduled_at):
            local = utc(ruling.final.scheduled_at).astimezone(IST)
            console.print(
                f"  moved to   {ruling.final.scheduled_at:%Y-%m-%d %H:%M} UTC "
                f"({local:%H:%M} IST)"
            )
        for v in ruling.verdicts:
            console.print(f"    [dim][{v.rule_id}] {v.disposition.value}: {v.reason}[/dim]")
        if not ruling.verdicts:
            console.print("    [dim]no rule objected[/dim]")
        console.print()

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("disposition")
    table.add_column("n", justify="right")
    for d in Disposition:
        table.add_row(f"[{DISPOSITION_STYLE[d]}]{d.value}[/{DISPOSITION_STYLE[d]}]", str(counts[d]))
    console.print(table)
    console.print(f"\n  {len(probes)} probes, {len(engine.rules)} rules, every rule consulted "
                  f"on every action")
    if surprises:
        console.print("\n[red]  rulings that did not match expectation:[/red]")
        for s in surprises:
            console.print(f"    [red]{s}[/red]")
    else:
        console.print("  [green]every probe was ruled on as expected[/green]")
    console.print()


# --------------------------------------------------------------------------


def stage_mandates(scenario) -> None:
    rule("4b. SCAN  recurring revenue at risk")
    console.print(
        "[dim]The anomaly detector answers 'did something just break'. That is the wrong\n"
        "question for most subscription revenue, because the usual failure is a state,\n"
        "not an event: mandates lapse quietly, one at a time, and nothing ever spikes.\n"
        "So this is a scan rather than a detector -- walk the book, price the dead\n"
        "authorisations, rank them. Both paths run; they find different problems.[/dim]\n"
    )
    from backstop.detect.mandates import scan

    report = scan(scenario.subscriptions)
    console.print(report.render())
    console.print("\n  [dim]worst five:[/dim]")
    for risk in report.at_risk[:5]:
        console.print(f"    {risk.describe()}")

    # What the scan produces is a bill, not a plan. Show the plan too, because
    # a number nobody acts on is a slide rather than a recovery system.
    console.print(
        "\n  [dim]what recovery proposes against that book, by mandate state:[/dim]"
    )
    from collections import Counter

    from backstop.decide.planner import mandate_actions

    proposed: Counter[tuple[str, str]] = Counter()
    for sub in scenario.lapsed_subscriptions:
        for action in mandate_actions(sub, scenario.ends_at):
            proposed[(sub.mandate_status.value, action.type.value)] += 1
    for (status, kind), n in sorted(proposed.items(), key=lambda kv: -kv[1]):
        note = (
            "  [dim]a decision, not a lapse -- automation does not re-ask[/dim]"
            if kind == "escalate_to_human" else ""
        )
        console.print(f"    {status:<16} {kind:<32} {n:>5}{note}")
    console.print(
        "\n  [dim]The measurement follows in stage 5, on its own row: restoring a mandate\n"
        "  recovers a year of billing, not one charge, so it is never added to the\n"
        "  payments number. Credit is incremental only -- a paused mandate that would\n"
        "  have resumed unprompted counts for nothing.[/dim]"
    )
    console.print()


def stage_receivables(scenario) -> None:
    rule("4c. AGE  receivables at risk")
    console.print(
        "[dim]The third surface, and a third shape of failure. A payment fails as an event.\n"
        "A mandate fails into a state. A receivable fails by *ageing* -- nothing breaks\n"
        "and nothing flips, the invoice simply gets older and every week the money is\n"
        "slightly less likely to arrive. So the output is the aging report a finance\n"
        "team already reads, and what selects the response is a duration, not a cause.[/dim]\n"
    )
    from backstop.detect.receivables import bucket_for, scan

    report = scan(scenario.invoices, scenario.ends_at)
    console.print(report.render())
    console.print("\n  [dim]largest five:[/dim]")
    for risk in report.at_risk[:5]:
        console.print(f"    {risk.describe()}")

    console.print(
        "\n  [dim]what collections proposes against that ledger, by aging bracket:[/dim]"
    )
    from collections import Counter

    from backstop.decide.planner import receivable_actions

    proposed: Counter[tuple[str, str]] = Counter()
    for inv in scenario.overdue_invoices:
        bracket = bucket_for(inv.days_overdue(scenario.ends_at)).value
        for action in receivable_actions(inv, scenario.ends_at):
            proposed[(bracket, action.type.value)] += 1
    notes = {
        "escalate_to_human": "  [dim]disputed, or old enough that a person decides[/dim]",
        "wait": "  [dim]the buyer committed to a date; chasing inside it is how they stop[/dim]",
        "offer_part_payment": "  [dim]the blocker is the balance, not the reminder[/dim]",
    }
    for (bracket, kind), n in sorted(proposed.items(), key=lambda kv: -kv[1]):
        console.print(f"    {bracket:<10} {kind:<28} {n:>5}{notes.get(kind, '')}")
    console.print(
        "\n  [dim]Measured in stage 5 on its own row. Credit is incremental here too and\n"
        "  it bites hardest on this surface: most overdue invoices are paid on the\n"
        "  buyer's own accounts-payable cycle whether or not anybody chases, and an\n"
        "  agent that mails them and books the payment has measured the AP cycle.[/dim]"
    )
    console.print()


def stage_backtest(scenario, *, offline: bool, model: str | None) -> None:
    rule("5. MEASURE  four arms, one batch")
    console.print(
        "[dim]Every arm below faces the same orders and the same latent recoverability,\n"
        "fixed before any of them ran. The policy engine evaluates every action in\n"
        "every arm -- the only difference is whether its rulings are obeyed, which is\n"
        "what makes the violation counts comparable rather than self-reported.[/dim]\n"
    )
    from backstop.evaluation.backtest import render, run

    arms, prop = run(scenario, offline=offline, model=model)
    render(scenario, arms, prop)
    console.print()


def stage_razorpay(scenario, *, notify: bool) -> None:
    """Dispatch a few permitted actions onto Razorpay's real test-mode rails.

    Everything before this stage is a claim about what recovery *would* do.
    This is the one place the system reaches outside itself, and it is
    deliberately placed after the policy engine rather than beside it: what
    hits the API is `ruling.final`, the action as the rules left it, and an
    action the rules refused makes no network call at all.

    Three actions, one per surface, and no notifications unless asked for.
    Test mode really delivers, and a walkthrough is not a reason to mail
    somebody.
    """
    rule("6. EXECUTE  the same actions, on Razorpay's test-mode rails")

    from dataclasses import replace

    from backstop.approve import ApprovalQueue, ReleaseOutcome
    from backstop.decide.planner import mandate_actions, receivable_actions, tail_actions
    from backstop.execute.razorpay import (
        HttpTransport,
        RazorpayExecutor,
        capabilities,
        reference_for,
    )
    from backstop.ledger.ledger import Surface
    from backstop.schedule import Scheduler

    settings = Settings.load()
    if not settings.has_razorpay:
        console.print(
            "  [yellow]RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET are not set in .env.[/yellow]\n"
            "  [dim]Everything above this line runs without them; only this stage needs a key.[/dim]\n"
        )
        return
    if not settings.razorpay_is_test_mode:
        console.print(
            "  [red]The configured key is not a test key.[/red]\n"
            "  [dim]The adapter refuses to construct against a live key, on purpose: it\n"
            "  dispatches payment links at people, and one token in a .env file is all\n"
            "  that separates a walkthrough from messaging real customers.[/dim]\n"
        )
        return

    console.print(
        "[dim]Nothing reaches this stage that the policy engine did not permit, and what\n"
        "is sent is the action as the rules left it -- rescheduled, downgraded or\n"
        "unchanged. A refused action makes no API call at all, which is the property\n"
        "worth having: the safety boundary is upstream of the network, not a filter\n"
        "applied to a log afterwards.[/dim]\n"
    )

    transport = HttpTransport.from_settings(settings)
    try:
        console.print("[bold]what this key can reach[/bold]  [dim]GET probes only[/dim]\n")
        for name, ok, why, err in capabilities(transport):
            mark = "[green]ok [/green]" if ok else "[yellow]no [/yellow]"
            note = f"  [dim]{why}[/dim]" if ok else f"  [dim]{why} -- {err}[/dim]"
            console.print(f"  {mark} {name:<15}{note}")
        console.print(
            "\n  [dim]Subscriptions is not enabled on this account and is not needed:\n"
            "  mandate re-registration goes through subscription_registration, which\n"
            "  works without it.[/dim]\n"
        )

        engine = PolicyEngine()
        now = scenario.ends_at
        executor = RazorpayExecutor(
            transport=transport,
            orders={o.id: o for o in scenario.orders},
            subscriptions={s.id: s for s in scenario.subscriptions},
            invoices={i.id: i for i in scenario.invoices},
            customers=scenario.customers,
            subscription_by_order=scenario.subscription_by_order,
            notify=notify,
        )

        # One candidate stream per surface. The subjects are chosen, the
        # rulings are not -- the first action the engine actually permits is
        # the one that gets dispatched, and if a surface has none, it says so.
        streams = [
            ("payment", (
                (a, PolicyContext(now=now, order=o, customer=scenario.customers.get(o.customer_id)))
                for o in scenario.failed_orders[:400] for a in tail_actions(o)
            )),
            ("recurring", (
                (a, PolicyContext(now=now, subscription=s,
                                  customer=scenario.customers.get(s.customer_id)))
                for s in scenario.lapsed_subscriptions[:400]
                for a in mandate_actions(s, now)
            )),
            ("receivable", (
                (a, PolicyContext(now=now, invoice=i,
                                  customer=scenario.customers.get(i.buyer_id)))
                for i in scenario.overdue_invoices[:400] for a in receivable_actions(i, now)
            )),
        ]

        console.print("[bold]three permitted actions, one per surface[/bold]\n")
        dispatched = 0
        queue = ApprovalQueue()
        scheduler = Scheduler()
        # The context each held or scheduled action was judged in, so it can be
        # re-judged against the same subject at a later clock. Both the queue's
        # release and the scheduler's fire-time re-rule read this.
        held_ctx: dict[str, PolicyContext] = {}
        for surface, candidates in streams:
            chosen = None
            for action, ctx in candidates:
                ruling = engine.evaluate(action, ctx)
                if action.is_inert:
                    continue
                # A live dispatch takes ALLOW and RESCHEDULE only. The backtest
                # executes REQUIRE_APPROVAL too, and is right to -- there the
                # question is what recovery is worth if a merchant staffs the
                # queue. Here there is a real person who has not said yes yet,
                # and sending anyway would make the approval gate decorative.
                #
                # The scan does not stop at the first dispatchable action: it
                # keeps going to collect what the engine sends to a person,
                # because the queue is a property of the whole candidate pool
                # and not of whatever happened to precede one dispatch.
                if ruling.disposition is Disposition.REQUIRE_APPROVAL:
                    request = queue.submit(
                        action, ruling, surface=Surface(surface), at=now
                    )
                    held_ctx[request.id] = ctx
                    continue
                if ruling.allowed and chosen is None:
                    # The context travels with the choice. The scan runs to
                    # exhaustion to fill the approval queue, so the loop
                    # variable no longer describes the chosen action by the
                    # time this loop ends.
                    chosen = (action, ruling, ctx)
            if chosen is None:
                console.print(f"  [yellow]{surface}[/yellow]  nothing permitted in the sample\n")
                continue

            action, ruling, chosen_ctx = chosen
            final = ruling.final or action
            style = DISPOSITION_STYLE[ruling.disposition]
            console.print(f"  [bold]{surface}[/bold]  {final.describe()}")
            console.print(
                f"    ruling     [{style}]{ruling.disposition.value.upper()}[/{style}]"
                f"  [dim]{len(engine.rules)} rules consulted[/dim]"
            )
            # Not dispatched here. A permitted action is a permitted action *at
            # its own moment*, and the moment is often not this one -- three
            # rules exist mainly to move it. Sending now would make the
            # reschedule a log entry rather than a protection.
            entry = scheduler.submit(final, surface=Surface(surface), at=now)
            held_ctx[entry.id] = chosen_ctx
            console.print(
                f"    scheduled  [dim]{entry.due_at:%m-%d %H:%M} UTC, "
                f"held until then[/dim]\n"
            )

        pending = queue.pending(now)
        if pending:
            console.print(
                f"  [magenta]{len(pending)} candidate{'s' if len(pending) > 1 else ''} "
                "held for approval, and queued rather than dropped[/magenta]\n"
                "  [dim]The backtest executes these, on the stated assumption that a\n"
                "  merchant staffs the queue. A live adapter may not make that\n"
                "  assumption on somebody's behalf -- an approval gate that sends while\n"
                "  it waits is not a gate -- so here they wait for a person.[/dim]\n"
            )
            for request in pending[:3]:
                console.print(f"    [magenta]{request.describe()}[/magenta]")
            console.print()

            # A reviewer answers one of them, four hours later. Everything else
            # is left alone on purpose, to show what an unstaffed desk costs.
            later = now + timedelta(hours=4)
            answered = pending[0]
            queue.approve(
                answered.id, by="ops@merchant.test", at=later,
                note="checked the buyer's history by hand",
            )
            console.print(
                f"  [bold]a person answers[/bold]  ops@merchant.test approves "
                f"{answered.action.type.value} on {answered.action.subject_id}"
            )

            def context_for(action: Action) -> PolicyContext | None:
                ctx = held_ctx.get(reference_for(action))
                return replace(ctx, now=later) if ctx else None

            for release in queue.release(engine, context_for, at=later):
                if release.outcome is ReleaseOutcome.REFUSED:
                    console.print(
                        f"  [red]refused anyway[/red]  {release.blocking_rule} "
                        "overtook the approval between the ask and the release"
                    )
                    continue
                word = (
                    "released" if release.outcome is ReleaseOutcome.RELEASED
                    else "released, rescheduled"
                )
                console.print(f"  [green]{word}[/green]  {release.action.describe()}")
                # An approved action still waits for its moment. A human yes
                # is not a reason to ignore the hour the rules chose.
                entry = scheduler.submit(
                    release.action, surface=release.request.surface, at=later
                )
                # The context the request was judged in, never a fallback:
                # firing against some other subject's context is exactly the
                # bug this line used to have.
                approved_ctx = held_ctx.get(release.request.id)
                if approved_ctx is not None:
                    held_ctx[entry.id] = approved_ctx
                console.print(
                    f"    scheduled  [dim]{entry.due_at:%m-%d %H:%M} UTC[/dim]"
                )

            expired = queue.expire_due(now + timedelta(days=3))
            console.print(
                f"\n  [yellow]{len(expired)} expired unanswered[/yellow]  "
                "[dim]one was answered above, on purpose[/dim]\n"
                "  [dim]Recorded as expired rather than dropped, and credited to\n"
                "  nobody. Silence is not consent, and revenue given up by a\n"
                "  staffing decision should be as visible as revenue given up by a\n"
                "  rule. An approval is permission from a person, not an exemption\n"
                "  from the rules -- the release above re-ran all "
                f"{len(engine.rules)} of them at the\n"
                "  later clock, and a DENY would still have won.[/dim]\n"
            )

        dispatched += run_schedule(scheduler, executor, engine, held_ctx)

        settled = executor.reconcile()
        console.print(
            f"  [bold]reconcile[/bold]  {len(executor.pending)} dispatched, "
            f"{len(settled)} settled so far"
        )
        console.print(
            "\n  [dim]Zero settled, and that is the correct answer rather than a\n"
            "  disappointing one. A payment link is paid when a human opens it, so a\n"
            "  live adapter cannot report recovery synchronously and does not pretend\n"
            "  to: it reports 'dispatched' and reconciles later. This is exactly why the\n"
            "  four-arm measurement above runs on the simulator -- reality has no\n"
            "  counterfactual, and an arm with no counterfactual cannot be scored.[/dim]"
        )

        show_webhook(executor)
        if dispatched and not notify:
            console.print(
                "\n  [dim]Nothing was sent to anybody: notifications are off unless\n"
                "  --notify is passed. The links exist in the dashboard, each tagged\n"
                "  with the action and subject that created it.[/dim]"
            )
    finally:
        transport.close()
    console.print()


def run_schedule(scheduler, executor, engine, held_ctx) -> int:
    """Advance a clock through the schedule and fire what comes due.

    A deployment would sleep between these moments. The walkthrough does not
    have that long, so it steps the clock to each scheduled moment in turn --
    which is the same loop with the waiting taken out, and worth saying out
    loud rather than implying that anything waited.

    Starting from the earliest scheduled moment rather than from `now` is not a
    convenience either. The batch is historical, so a plan drawn against it is
    drawn for moments that have already passed; replaying from the first of
    them is what the backtest does too, and reading the clock any other way
    would report every action as arriving a week late.
    """
    from dataclasses import replace

    from backstop.execute.razorpay import RazorpayError, reference_for
    from backstop.schedule import Fate

    waiting = scheduler.waiting()
    if not waiting:
        return 0

    console.print(
        f"[bold]the clock advances[/bold]  [dim]{len(waiting)} scheduled "
        "moment(s), stepped rather than slept through[/dim]\n"
    )

    fired = 0
    # Bounded: a fire-time reschedule adds a new moment, and a loop that
    # followed those indefinitely would be the unbounded retry the scheduler
    # exists to prevent.
    for _ in range(12):
        tick = scheduler.next_due
        if tick is None:
            break

        def context_for(action, *, tick=tick):
            ctx = held_ctx.get(reference_for(action))
            return replace(ctx, now=tick) if ctx else None

        try:
            firings = scheduler.run_due(executor, engine, context_for, at=tick)
        except RazorpayError as err:
            console.print(f"  [red]razorpay   {err}[/red]")
            break
        if not firings:
            break

        for firing in firings:
            when = f"{firing.scheduled.due_at:%m-%d %H:%M}"
            if firing.fate is Fate.DISPATCHED:
                result = firing.result
                console.print(
                    f"  [green]fired[/green]      {when}  "
                    f"{firing.scheduled.action.type.value} "
                    f"{firing.scheduled.action.subject_id}"
                )
                if result and result.external:
                    console.print(f"               created {result.external.describe()}")
                console.print(f"               [dim]{firing.detail}[/dim]")
                fired += int(bool(result) and result.outcome.value == "dispatched")
            elif firing.fate is Fate.DEFERRED:
                console.print(f"  [yellow]deferred[/yellow]   {when}  {firing.detail}")
            elif firing.fate is Fate.REFUSED:
                console.print(f"  [red]refused[/red]    {when}  {firing.detail}")
            elif firing.fate is Fate.STALE:
                console.print(f"  [dim]stale[/dim]      {when}  {firing.detail}")
            else:
                console.print(f"  [dim]abandoned[/dim]  {when}  {firing.detail}")

    console.print(
        "\n  [dim]Nothing above went out before the moment the rules chose for it,\n"
        "  and every action was ruled on again when it came due rather than\n"
        "  riding a verdict reached hours earlier. An engine that moves an SMS\n"
        "  out of the night, followed by an adapter that sends it at 22:30\n"
        "  anyway, has not protected anybody -- it has logged that it did.[/dim]\n"
    )
    return fired


def show_webhook(executor) -> None:
    """Close the loop the other way round: Razorpay tells us, we do not ask.

    An honest caveat first, because the rest of this project states its limits
    rather than burying them. Receiving a *real* webhook needs a public URL
    Razorpay can reach, and a walkthrough on somebody's laptop has none. So the
    delivery below is constructed locally and signed with a local secret. What
    is real is everything after the signature check: the same receiver, the
    same verification, the same matching against what was actually dispatched,
    and the same crediting path the poll uses. What is simulated is only the
    postman.
    """
    import hashlib
    import hmac
    import json

    from backstop.execute.webhook import WebhookReceiver

    pending = executor.pending
    if not pending:
        return

    console.print("\n[bold]a webhook arrives[/bold]  [dim]locally signed sample[/dim]\n")
    secret = "whsec_walkthrough_only"
    receiver = WebhookReceiver(executor=executor, secret=secret)
    dispatch = pending[0]

    # The event name has to match what the dispatch actually created: an auth
    # link is an invoice, dunning is a payment link, a re-presentment is an
    # order, and the receiver refuses a delivery whose kind disagrees with the
    # entity it names.
    kind = dispatch.ref.entity

    def deliver(entity_id: str, event_id: str, *, entity_kind: str = kind):
        body = json.dumps({
            "entity": "event",
            "event": f"{entity_kind}.paid",
            "contains": [entity_kind],
            "payload": {entity_kind: {"entity": {
                "id": entity_id, "status": "paid",
                "amount_paid": dispatch.amount.paise,
            }}},
            "created_at": int(datetime.now(UTC).timestamp()),
        }).encode()
        signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        return receiver.receive(body, signature, event_id=event_id)

    forged = receiver.receive(b'{"event":"payment_link.paid"}', "deadbeef")
    console.print(
        f"  [red]{forged.verdict.value:<10}[/red] a body signed with the wrong secret "
        f"-> HTTP {forged.status}"
    )

    stranger = deliver("plink_not_ours", "evt_stranger", entity_kind="payment_link")
    console.print(
        f"  [yellow]{stranger.verdict.value:<10}[/yellow] a link the merchant created "
        f"themselves -> HTTP {stranger.status}"
    )

    ours = deliver(dispatch.ref.id, "evt_ours")
    colour = "green" if ours.is_recovery else "yellow"
    console.print(
        f"  [{colour}]{ours.verdict.value:<10}[/{colour}] {dispatch.ref.id} "
        f"-> {ours.result.recovered if ours.result else 'nothing'}"
    )

    replay = deliver(dispatch.ref.id, "evt_ours")
    console.print(
        f"  [dim]{replay.verdict.value:<10}[/dim] the same delivery again "
        f"-> HTTP {replay.status}, credited nothing"
    )

    console.print(
        "\n  [dim]The middle line is the whole counterfactual argument arriving over\n"
        "  HTTP. A merchant's own payment links are paid all day; that is real\n"
        "  revenue and recovery caused none of it, and a receiver that credited\n"
        "  every payment_link.paid on the account would book the merchant's\n"
        "  ordinary business as its own work. Only an entity this process\n"
        "  dispatched is a recovery.[/dim]\n"
    )


def stage_gaps() -> None:
    rule("NOT BUILT YET")
    console.print(
        "  [yellow]scheduling[/yellow]  nothing holds a plan across real time. Every\n"
        "              action carries a scheduled_at the rules can move, and the\n"
        "              walkthrough dispatches immediately regardless. A deployment\n"
        "              would wait.\n"
        "  [yellow]a real URL[/yellow]  the webhook receiver is exercised against a\n"
        "              locally signed delivery, because receiving one from Razorpay\n"
        "              needs a public endpoint this walkthrough does not have. The\n"
        "              verification, matching and crediting are the real ones.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk a batch through the Backstop pipeline.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--orders-per-day", type=int, default=20000)
    ap.add_argument("--offline", action="store_true", help="no API calls; diagnosis is scripted")
    ap.add_argument("--model", default=None, help="override the reasoning model")
    ap.add_argument(
        "--notify", action="store_true",
        help="let the razorpay stage actually send. Test mode really delivers.",
    )
    ap.add_argument(
        "--stage", default="all",
        choices=["all", "detect", "diagnose", "policy", "mandates", "receivables",
                 "enforce", "backtest", "razorpay"],
        help="stop after this stage",
    )
    args = ap.parse_args()

    console.print()
    console.print("[bold]BACKSTOP[/bold]  bounded revenue recovery  "
                  f"[dim]seed {args.seed}, {args.days}d, "
                  f"{args.orders_per_day:,} orders/day[/dim]")
    console.print()

    scenario = stage_scenario(
        SimConfig(seed=args.seed, days=args.days, orders_per_day=args.orders_per_day)
    )

    if args.stage == "policy":
        stage_policy(scenario, [])
        return
    if args.stage == "mandates":
        stage_mandates(scenario)
        return
    if args.stage == "receivables":
        stage_receivables(scenario)
        return
    if args.stage == "razorpay":
        stage_razorpay(scenario, notify=args.notify)
        return

    clusters = stage_detect(scenario)
    if args.stage == "detect":
        return

    pairs = stage_diagnose(scenario, clusters, offline=args.offline, model=args.model)
    if args.stage == "diagnose":
        return

    stage_policy(scenario, pairs)
    stage_mandates(scenario)
    stage_receivables(scenario)
    if args.stage == "enforce":
        return

    stage_backtest(scenario, offline=args.offline, model=args.model)
    if not args.offline:
        stage_razorpay(scenario, notify=args.notify)
    stage_gaps()


if __name__ == "__main__":
    main()
