"""Segment-aware detection of revenue at risk. No LLM anywhere in this file.

Detection is a statistics problem and should be solved as one: it must run over
every attempt, respond in seconds, and produce a number the rest of the system
can act on. A model here would be slower, costlier and worse.

Three things make this harder than thresholding a failure count:

*   Volume swings ~4x across the day, so counts are meaningless and only rates
    carry signal.
*   The interesting failures hide in slices. An issuer-level view averages away
    a single bad BIN, and a routing problem looks like an issuer problem until
    you cut by acquirer.
*   Scanning many slices invites false positives, so a signal must clear a
    statistical bar, a minimum effect size, a volume floor and a sustain
    requirement before it is worth anyone's attention.

Money at risk is reported as *excess* loss -- observed failures beyond what the
segment's own baseline predicts -- to stay consistent with how the simulator
attributes ground truth. A window sum would overstate every signal.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from backstop.domain.declines import DeclineCode
from backstop.domain.entities import PaymentAttempt
from backstop.domain.money import Money


@dataclass(frozen=True, order=True)
class SegmentKey:
    """A slice of traffic. `dimension` names the cut, `value` identifies it."""

    dimension: str
    value: str

    def describe(self) -> str:
        return f"{self.dimension}={self.value}"


def segments_for(attempt: PaymentAttempt) -> list[SegmentKey]:
    """Every slice an attempt belongs to.

    Includes the rail-by-issuer cross because a card problem at one bank is
    invisible in both the rail view and the issuer view when that bank is
    healthy on UPI.
    """
    # "global" exists so an incident that affects everything has a parent to
    # roll up to. Without it, an all-traffic fault breaches every issuer and
    # every rail as siblings and correlation cannot collapse them.
    keys = [SegmentKey("global", "all"), SegmentKey("rail", str(attempt.rail))]
    if attempt.issuer:
        keys.append(SegmentKey("issuer", attempt.issuer))
        keys.append(SegmentKey("rail_issuer", f"{attempt.rail}:{attempt.issuer}"))
        if attempt.bin:
            # Qualified by issuer so the hierarchy is explicit: correlation
            # needs to know a BIN sits under an issuer to roll up to it.
            keys.append(SegmentKey("issuer_bin", f"{attempt.issuer}:{attempt.bin}"))
    if attempt.acquirer:
        keys.append(SegmentKey("acquirer", attempt.acquirer))
    return keys


@dataclass
class Bucket:
    attempts: int = 0
    successes: int = 0
    failed_amount: Money = field(default_factory=Money.zero)
    total_amount: Money = field(default_factory=Money.zero)
    declines: dict[DeclineCode, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0

    @property
    def mean_amount(self) -> Money:
        return Money(self.total_amount.paise // self.attempts) if self.attempts else Money.zero()


@dataclass
class RiskSignal:
    """Detected revenue at risk in one segment over one contiguous window."""

    id: str
    segment: SegmentKey
    starts_at: datetime
    ends_at: datetime
    baseline_rate: float
    observed_rate: float
    attempts: int
    failures: int
    z_score: float
    excess_failures: float
    """Failures beyond what the baseline predicts. The honest count."""
    money_at_risk: Money
    """Excess failures priced at the segment's mean order value."""
    decline_mix: dict[DeclineCode, int] = field(default_factory=dict)

    @property
    def dominant_decline(self) -> DeclineCode | None:
        return max(self.decline_mix, key=self.decline_mix.get) if self.decline_mix else None

    @property
    def dominant_share(self) -> float:
        if not self.decline_mix:
            return 0.0
        return max(self.decline_mix.values()) / sum(self.decline_mix.values())

    @property
    def rate_drop(self) -> float:
        return self.baseline_rate - self.observed_rate

    @property
    def duration_minutes(self) -> float:
        return (self.ends_at - self.starts_at).total_seconds() / 60

    def describe(self) -> str:
        d = self.dominant_decline
        return (
            f"{self.segment.describe():<30} {self.observed_rate:.0%} vs {self.baseline_rate:.0%} "
            f"baseline  z={self.z_score:>6.1f}  {self.duration_minutes:>4.0f}m  "
            f"{self.money_at_risk}  {d.value if d else '-'} {self.dominant_share:.0%}"
        )


@dataclass
class DetectorConfig:
    bucket_minutes: int = 15
    min_attempts: int = 30
    """Volume floor per bucket. Below this a rate is noise, not a measurement."""
    z_threshold: float = 4.0
    """Strict, because many segments are scanned every bucket. See note below."""
    min_absolute_drop: float = 0.08
    """Statistical significance is not the same as mattering. A 2% drop on huge
    volume is significant and usually not worth waking anyone for."""
    sustained_buckets: int = 2
    """Consecutive breaching buckets before a signal is emitted. Kills flapping."""
    warmup_buckets: int = 8
    """Buckets used to learn a segment's baseline before it can alert."""
    ewma_alpha: float = 0.15
    min_money_at_risk: Money = field(default_factory=lambda: Money.rupees(2000))


@dataclass
class _SegmentState:
    baseline: float | None = None
    seen: int = 0
    breach_run: list[tuple[datetime, Bucket, float]] = field(default_factory=list)


class Detector:
    """Streaming EWMA baseline per segment with a binomial z-test per bucket."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.cfg = config or DetectorConfig()
        self._state: dict[SegmentKey, _SegmentState] = defaultdict(_SegmentState)
        self._seq = 0

    def _bucket_start(self, at: datetime, origin: datetime) -> datetime:
        minutes = self.cfg.bucket_minutes
        delta = int((at - origin).total_seconds() // (minutes * 60))
        return origin + timedelta(minutes=minutes * delta)

    def run(self, attempts: list[PaymentAttempt]) -> list[RiskSignal]:
        """Detect over a batch of attempts, replaying them in time order."""
        if not attempts:
            return []
        attempts = sorted(attempts, key=lambda a: a.at)
        origin = attempts[0].at

        # Bucket every attempt into each of its segments.
        grid: dict[SegmentKey, dict[datetime, Bucket]] = defaultdict(dict)
        for a in attempts:
            bstart = self._bucket_start(a.at, origin)
            for key in segments_for(a):
                b = grid[key].setdefault(bstart, Bucket())
                b.attempts += 1
                b.total_amount += a.amount
                if a.succeeded:
                    b.successes += 1
                else:
                    b.failed_amount += a.amount
                    if a.decline_code:
                        b.declines[a.decline_code] += 1

        signals: list[RiskSignal] = []
        for key, buckets in grid.items():
            signals.extend(self._scan_segment(key, buckets))
        return sorted(signals, key=lambda s: s.starts_at)

    def _scan_segment(self, key: SegmentKey, buckets: dict[datetime, Bucket]) -> list[RiskSignal]:
        cfg, st = self.cfg, self._state[key]
        out: list[RiskSignal] = []

        for bstart in sorted(buckets):
            b = buckets[bstart]
            if b.attempts < cfg.min_attempts:
                continue  # too thin to say anything responsible about

            st.seen += 1
            if st.baseline is None:
                st.baseline = b.rate
                continue
            if st.seen <= cfg.warmup_buckets:
                st.baseline += cfg.ewma_alpha * (b.rate - st.baseline)
                continue

            z = _binomial_z(b.rate, st.baseline, b.attempts)
            drop = st.baseline - b.rate
            breaching = z <= -cfg.z_threshold and drop >= cfg.min_absolute_drop

            if breaching:
                st.breach_run.append((bstart, b, z))
                # Baseline is frozen while breaching. Updating it here would let
                # a long incident normalise itself and silence the alert.
                continue

            if st.breach_run:
                sig = self._emit(key, st)
                if sig:
                    out.append(sig)
                st.breach_run = []
            st.baseline += cfg.ewma_alpha * (b.rate - st.baseline)

        if st.breach_run:
            sig = self._emit(key, st)
            if sig:
                out.append(sig)
            st.breach_run = []
        return out

    def _emit(self, key: SegmentKey, st: _SegmentState) -> RiskSignal | None:
        cfg = self.cfg
        run = st.breach_run
        if len(run) < cfg.sustained_buckets:
            return None

        attempts = sum(b.attempts for _, b, _ in run)
        successes = sum(b.successes for _, b, _ in run)
        failures = attempts - successes
        baseline = st.baseline or 0.0
        observed = successes / attempts if attempts else 0.0

        # Excess, not total: failures the baseline already predicted are not
        # this incident's doing, and pricing them here would inflate the signal.
        excess = max(0.0, (baseline - observed) * attempts)
        total_amount = Money.zero()
        for _, b, _ in run:
            total_amount += b.total_amount
        mean_amount = Money(total_amount.paise // attempts) if attempts else Money.zero()
        money = mean_amount * excess
        if money < cfg.min_money_at_risk:
            return None

        mix: dict[DeclineCode, int] = defaultdict(int)
        for _, b, _ in run:
            for code, n in b.declines.items():
                mix[code] += n

        self._seq += 1
        worst = min(z for _, _, z in run)
        return RiskSignal(
            id=f"sig_{self._seq:05d}",
            segment=key,
            starts_at=run[0][0],
            ends_at=run[-1][0] + timedelta(minutes=cfg.bucket_minutes),
            baseline_rate=baseline,
            observed_rate=observed,
            attempts=attempts,
            failures=failures,
            z_score=worst,
            excess_failures=excess,
            money_at_risk=money,
            decline_mix=dict(mix),
        )


def _binomial_z(observed: float, baseline: float, n: int) -> float:
    """Standardised deviation of an observed rate from its baseline.

    Normal approximation to the binomial. The volume floor in DetectorConfig is
    what keeps that approximation honest; below it, rates are not measured.
    """
    if n <= 0 or baseline <= 0.0 or baseline >= 1.0:
        return 0.0
    se = math.sqrt(baseline * (1.0 - baseline) / n)
    return (observed - baseline) / se if se > 0 else 0.0
