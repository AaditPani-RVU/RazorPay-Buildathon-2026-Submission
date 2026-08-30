"""Deterministic scenario generation with labelled incidents.

Traffic follows an IST diurnal curve with lunchtime and evening peaks. That
matters: with a flat arrival rate a detector can find incidents by thresholding
raw failure *counts*, which is not a technique that survives contact with real
traffic. Making volume swing by 4x forces rate-based, segment-aware detection.

Everything is seeded. The same seed yields byte-identical scenarios, so eval
arms can be compared on genuinely identical batches.
"""

from __future__ import annotations

import itertools
import math
import random
import zlib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from backstop.domain.declines import DeclineCode, Rail, RootCause
from backstop.domain.entities import (
    AttemptStatus,
    Channel,
    Customer,
    Invoice,
    InvoiceStatus,
    MandateStatus,
    Order,
    PaymentAttempt,
    Subscription,
)
from backstop.domain.money import Money
from backstop.simulate import recoverability as recov
from backstop.simulate.scenario import (
    DEFAULT_ACQUIRERS,
    DEFAULT_ISSUERS,
    DEFAULT_RAILS,
    Incident,
    RailProfile,
    Scenario,
    Segment,
)


@dataclass
class SimConfig:
    seed: int = 7
    days: int = 7
    orders_per_day: int = 20000
    start: datetime | None = None
    n_customers: int = 25000
    n_subscriptions: int = 600
    n_invoices: int = 220
    rails: list[RailProfile] = field(default_factory=lambda: list(DEFAULT_RAILS))
    issuers: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ISSUERS))
    acquirers: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ACQUIRERS))
    inject_incidents: bool = True

    def resolved_start(self) -> datetime:
        if self.start is not None:
            return self.start
        base = datetime.now(UTC) - timedelta(days=self.days)
        return base.replace(hour=0, minute=0, second=0, microsecond=0)


def _diurnal_weight(at: datetime) -> float:
    """Relative arrival rate at a given instant, on IST wall-clock.

    Two peaks (lunch, evening) over a low overnight floor -- roughly a 4x swing
    between trough and peak, which is what makes count-based detection fail.
    """
    ist_hour = ((at.hour + at.minute / 60) + 5.5) % 24
    lunch = math.exp(-(((ist_hour - 13.0) / 2.4) ** 2))
    evening = math.exp(-(((ist_hour - 21.0) / 2.6) ** 2))
    return 0.25 + 1.35 * evening + 0.95 * lunch


def _weighted(rng: random.Random, weights: dict):
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def _issuer_bins(issuer: str) -> list[str]:
    """Stable per-issuer BIN pool, so BIN-level incidents are expressible.

    Derived with crc32 rather than the builtin hash: string hashing is salted
    per process via PYTHONHASHSEED, which would silently break reproducibility
    across runs while looking fine inside any single one.
    """
    h = zlib.crc32(issuer.encode())
    return [f"{4 if i % 2 else 5}{h % 100:02d}{i:03d}" for i in range(4)]


class ScenarioGenerator:
    def __init__(self, config: SimConfig | None = None) -> None:
        self.cfg = config or SimConfig()
        self.rng = random.Random(self.cfg.seed)
        self._seq = itertools.count()
        self._bins: dict[str, list[str]] = {}

    def _id(self, prefix: str) -> str:
        """Seeded, collision-free ids. uuid4 would ignore the seed and make
        two runs of the same configuration incomparable."""
        return f"{prefix}_{self.cfg.seed:04x}{next(self._seq):09x}"

    def _bin_for(self, issuer: str) -> str:
        pool = self._bins.setdefault(issuer, _issuer_bins(issuer))
        return self.rng.choice(pool)

    # -- incidents ---------------------------------------------------------

    def plan_incidents(self, start: datetime, end: datetime) -> list[Incident]:
        """Place a fixed, varied set of incidents in the window.

        The mix is intentional: a long obvious outage, a subtle BIN issue, a
        routing degradation that looks like an issuer problem unless you segment
        by acquirer, an auth drop-off with no decline code at all, and a payday
        NSF cluster that is *not* an incident to act on. A detector that fires
        on the last one is producing false interventions.
        """
        if not self.cfg.inject_incidents:
            return []
        rng = self.rng
        span = (end - start).total_seconds()

        def at(frac: float) -> datetime:
            return start + timedelta(seconds=span * frac)

        issuer_a = _weighted(rng, self.cfg.issuers)
        issuer_b = _weighted(rng, {k: v for k, v in self.cfg.issuers.items() if k != issuer_a})

        specs = [
            # Loud and unambiguous: a whole issuer's auth host drops.
            (RootCause.ISSUER_OUTAGE, Segment(rail=Rail.CARD, issuer=issuer_a),
             0.18, 95, 0.28, DeclineCode.ISSUER_UNAVAILABLE, 0.88),
            # Quiet: one BIN range, easy to miss inside issuer-level averages.
            (RootCause.BIN_SPECIFIC_DECLINE,
             Segment(rail=Rail.CARD, issuer=issuer_b, bin=self._bin_for(issuer_b)),
             0.38, 400, 0.42, DeclineCode.DO_NOT_HONOR, 0.79),
            # Looks like an issuer problem until you segment by acquirer.
            (RootCause.GATEWAY_ROUTING_DEGRADATION, Segment(acquirer="acq_beta"),
             0.55, 150, 0.55, DeclineCode.GATEWAY_TIMEOUT, 0.72),
            # A rail outage on the dominant rail: highest money at risk.
            (RootCause.PSP_OR_RAIL_OUTAGE, Segment(rail=Rail.UPI),
             0.71, 55, 0.35, DeclineCode.PSP_UNAVAILABLE, 0.91),
            # Customers reach 3DS and leave. Recoverable, but never by retrying.
            (RootCause.AUTHENTICATION_DROPOFF, Segment(rail=Rail.CARD),
             0.84, 180, 0.62, DeclineCode.OTP_ABANDONED, 0.83),
            # A genuine funds crunch, not a system fault. Detection *should*
            # fire -- the money is really at risk -- but diagnosis must not call
            # it an outage and recovery must wait for payday rather than
            # hammering the issuer. Discrimination, not just alarm.
            (RootCause.INSUFFICIENT_FUNDS_CLUSTER, Segment(),
             0.29, 240, 0.74, DeclineCode.INSUFFICIENT_FUNDS, 0.68),
        ]
        incidents = []
        for cause, seg, frac, minutes, mult, code, share in specs:
            s = at(frac)
            incidents.append(
                Incident(
                    id=self._id("inc"), root_cause=cause, segment=seg, starts_at=s,
                    ends_at=s + timedelta(minutes=minutes),
                    success_rate_multiplier=mult, dominant_decline=code,
                    dominant_share=share,
                )
            )
        return incidents

    # -- population --------------------------------------------------------

    def _customers(self) -> dict[str, Customer]:
        rng = self.rng
        out: dict[str, Customer] = {}
        for _ in range(self.cfg.n_customers):
            cid = self._id("cust")
            channels = {Channel.EMAIL}
            if rng.random() < 0.82:
                channels.add(Channel.SMS)
            if rng.random() < 0.64:
                channels.add(Channel.WHATSAPP)
            if rng.random() < 0.11:
                channels.add(Channel.VOICE)
            out[cid] = Customer(
                id=cid,
                email=f"{cid}@example.test",
                phone=f"+9198{rng.randint(10**7, 10**8 - 1)}",
                consented_channels=channels,
                dnd_registered=rng.random() < 0.23,
                opted_out_at=None,
            )
        return out

    def _amount(self) -> Money:
        """Log-normal order values: many small, a long tail of large ones."""
        return Money.rupees(round(min(60000, math.exp(self.rng.gauss(6.4, 0.95))), 2))

    # -- attempts ----------------------------------------------------------

    def _decline_for(
        self, profile: RailProfile, active: list[Incident], seg_key: dict
    ) -> DeclineCode:
        rng = self.rng
        for inc in active:
            if inc.segment.matches(**seg_key) and rng.random() < inc.dominant_share:
                return inc.dominant_decline
        return _weighted(rng, profile.decline_mix)

    def generate(self) -> Scenario:
        cfg, rng = self.cfg, self.rng
        start = cfg.resolved_start()
        end = start + timedelta(days=cfg.days)
        customers = self._customers()
        cust_ids = list(customers)
        incidents = self.plan_incidents(start, end)

        rail_weights = {p.rail: p.volume_share for p in cfg.rails}
        profiles = {p.rail: p for p in cfg.rails}

        # Arrival times shaped by the diurnal curve via rejection sampling.
        total_orders = cfg.orders_per_day * cfg.days
        times: list[datetime] = []
        span_s = (end - start).total_seconds()
        while len(times) < total_orders:
            t = start + timedelta(seconds=rng.random() * span_s)
            if rng.random() < _diurnal_weight(t) / 1.85:
                times.append(t)
        times.sort()

        orders: list[Order] = []
        for t in times:
            cid = rng.choice(cust_ids)
            rail = _weighted(rng, rail_weights)
            profile = profiles[rail]
            issuer = _weighted(rng, cfg.issuers)
            acquirer = _weighted(rng, cfg.acquirers)
            bin_ = self._bin_for(issuer) if rail is Rail.CARD else None
            seg_key = dict(rail=rail, issuer=issuer, bin=bin_, acquirer=acquirer)

            active = [i for i in incidents if i.is_active(t) and i.segment.matches(**seg_key)]
            multiplier = 1.0
            for inc in active:
                multiplier *= inc.success_rate_multiplier

            amount = self._amount()
            order = Order(id=self._id("order"), customer_id=cid, amount=amount, created_at=t)

            # A slice never reaches an attempt at all: checkout abandonment.
            if rng.random() < 0.06:
                order.abandoned_at_checkout = True
                orders.append(order)
                continue

            # One uniform draw, compared against two thresholds. The order
            # fails iff u >= base*multiplier, and *would have succeeded absent
            # the incident* iff u < base. That makes incident attribution an
            # exact counterfactual rather than an estimate: failures in the
            # window that would have failed anyway are never charged to the
            # incident, so "money at risk" is not inflated by baseline noise.
            u = rng.random()
            base = profile.base_success_rate
            success = u < base * multiplier
            counterfactually_ok = u < base
            if success:
                status, code = AttemptStatus.CAPTURED, None
            else:
                status = AttemptStatus.FAILED
                code = self._decline_for(profile, active, seg_key)

            order.attempts.append(
                PaymentAttempt(
                    id=self._id("pay"), order_id=order.id, customer_id=cid, amount=amount,
                    rail=rail, at=t, status=status, issuer=issuer, bin=bin_,
                    acquirer=acquirer, decline_code=code, attempt_no=1,
                )
            )
            orders.append(order)
            if not success and counterfactually_ok:
                for inc in active:
                    inc.affected_order_ids.append(order.id)
                    inc.money_at_risk += amount
            elif not success:
                for inc in active:
                    inc.baseline_failures_in_window += 1

        subs = self._subscriptions(start, end, cust_ids)
        invoices = self._invoices(start, end)

        # Latent recoverability, drawn once so every backtest arm faces the
        # same world. An incident-caused failure heals when its incident ends,
        # which is what makes "wait for the outage to pass" a strategy the
        # backtest can actually reward rather than merely permit.
        heals: dict[str, datetime] = {}
        routing: set[str] = set()
        for inc in incidents:
            for oid in inc.affected_order_ids:
                heals[oid] = inc.ends_at
                if inc.root_cause is RootCause.GATEWAY_ROUTING_DEGRADATION:
                    routing.add(oid)

        return Scenario(
            customers=customers, orders=orders, subscriptions=subs, invoices=invoices,
            incidents=incidents, starts_at=start, ends_at=end, seed=cfg.seed,
            recoverability=recov.build(orders, heals, routing, cfg.seed),
        )

    def _subscriptions(self, start, end, cust_ids) -> list[Subscription]:
        rng = self.rng
        statuses = [
            (MandateStatus.ACTIVE, 0.83), (MandateStatus.PAUSED, 0.05),
            (MandateStatus.EXPIRED, 0.07), (MandateStatus.REVOKED, 0.05),
        ]
        out = []
        for _ in range(self.cfg.n_subscriptions):
            status = rng.choices([s for s, _ in statuses], [w for _, w in statuses])[0]
            out.append(
                Subscription(
                    id=self._id("sub"), customer_id=rng.choice(cust_ids),
                    amount=Money.rupees(rng.choice([199, 299, 499, 799, 1499, 2999])),
                    rail=rng.choice([Rail.EMANDATE_NACH, Rail.UPI_AUTOPAY]),
                    mandate_status=status,
                    next_charge_at=start + timedelta(seconds=rng.random() * (end - start).total_seconds()),
                    consecutive_failures=0 if status is MandateStatus.ACTIVE else rng.randint(1, 3),
                )
            )
        return out

    def _invoices(self, start, end) -> list[Invoice]:
        """B2B receivables, aged around the window so some are already overdue."""
        rng = self.rng
        out = []
        for _ in range(self.cfg.n_invoices):
            issued = start - timedelta(days=rng.randint(5, 120))
            terms = rng.choice([15, 30, 45, 60])
            amount = Money.rupees(round(math.exp(rng.gauss(11.2, 1.0)), 2))
            inv = Invoice(
                id=self._id("inv"), buyer_id=self._id("buyer"), amount=amount,
                issued_at=issued, due_at=issued + timedelta(days=terms),
            )
            # Most receivables are collected without help. Only the residue is
            # a recovery problem, and a ledger where nothing is ever paid would
            # make the at-risk total meaningless.
            roll = rng.random()
            if roll < 0.62:
                inv.status = InvoiceStatus.PAID
                inv.amount_paid = amount
            elif roll < 0.70:
                inv.status = InvoiceStatus.PART_PAID
                inv.amount_paid = amount * rng.uniform(0.2, 0.7)
            # A minority are contested. These must never be auto-chased.
            if not inv.is_settled and rng.random() < 0.09:
                inv.disputed_at = inv.due_at + timedelta(days=rng.randint(1, 10))
            out.append(inv)
        return out


def generate(config: SimConfig | None = None) -> Scenario:
    return ScenarioGenerator(config).generate()
