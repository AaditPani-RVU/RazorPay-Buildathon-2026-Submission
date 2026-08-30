"""Walk one batch through the pipeline, stage by stage, and show the work.

This exists so the system can be *watched* rather than described. Every number
printed is computed live from a seeded batch; nothing here is a recorded
transcript. Ground truth is printed alongside each prediction, so a reader can
check the pipeline rather than take its word.

    python -m backstop.demo                 # full walkthrough, live model
    python -m backstop.demo --offline       # no API calls; diagnosis is scripted
    python -m backstop.demo --stage policy  # just the safety boundary
    python -m backstop.demo --seed 3

The pipeline is Detect -> Diagnose -> Decide -> Enforce -> Execute -> Measure.
`decide`, `execute` and `measure` are not built yet, so the walkthrough covers
Detect, Diagnose and Enforce, and says so at the end rather than papering over
the gap. The Enforce stage is driven by hand-written probes standing in for the
planner -- clearly labelled as such, because a demo that quietly simulates a
missing component is worse than no demo.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

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


def stage_gaps() -> None:
    rule("NOT BUILT YET")
    console.print(
        "  [yellow]mandates[/yellow]  subscriptions are generated but nothing charges or detects\n"
        "            them. Mandate decline codes and a root cause exist; the pipeline\n"
        "            does not reach them.\n"
        "  [yellow]receivables[/yellow]  invoices have policy rules but no detection, and buyers\n"
        "            have no Customer record, so ConsentRule denies every contact.\n"
        "  [yellow]razorpay[/yellow]  execute/ has one simulated backend. A test-mode adapter\n"
        "            drops in behind the same protocol.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk a batch through the Backstop pipeline.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--orders-per-day", type=int, default=20000)
    ap.add_argument("--offline", action="store_true", help="no API calls; diagnosis is scripted")
    ap.add_argument("--model", default=None, help="override the reasoning model")
    ap.add_argument(
        "--stage", default="all",
        choices=["all", "detect", "diagnose", "policy", "enforce", "backtest"],
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

    clusters = stage_detect(scenario)
    if args.stage == "detect":
        return

    pairs = stage_diagnose(scenario, clusters, offline=args.offline, model=args.model)
    if args.stage == "diagnose":
        return

    stage_policy(scenario, pairs)
    if args.stage == "enforce":
        return

    stage_backtest(scenario, offline=args.offline, model=args.model)
    stage_gaps()


if __name__ == "__main__":
    main()
