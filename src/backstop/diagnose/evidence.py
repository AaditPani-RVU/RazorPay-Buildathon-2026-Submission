"""Build the evidence a root cause can actually be argued from.

The quality ceiling on diagnosis is set here, not in the prompt. An issuer
outage and a routing fault look identical when you only look at the failing
slice: both are a rate collapse dominated by a timeout-ish decline code. What
separates them is whether the *siblings* are healthy. One bad acquirer beside
two clean ones is a routing problem; one bad issuer beside seven clean ones is
an issuer problem; everything degraded at once is neither.

So a bundle always carries peers, including the healthy ones. Their absence is
what makes models guess.

Reference rates come from the same segment outside the incident window, which
keeps the comparison within-segment: rails and issuers have genuinely different
baseline success rates, and comparing across them would invent effects.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from backstop.detect.correlate import RiskCluster
from backstop.detect.detector import SegmentKey, segments_for
from backstop.domain.declines import DeclineCode
from backstop.domain.entities import PaymentAttempt
from backstop.domain.money import Money


@dataclass
class SegmentObservation:
    segment: SegmentKey
    attempts: int
    successes: int
    reference_rate: float
    """This segment's own rate outside the window. Never another segment's."""
    declines: dict[DeclineCode, int] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0

    @property
    def delta_pp(self) -> float:
        """Signed change against reference. Negative is worse, as a reader
        expects; reporting a collapse as a positive number invites exactly the
        misreading the evidence exists to prevent."""
        return (self.rate - self.reference_rate) * 100

    @property
    def drop_pp(self) -> float:
        """Magnitude of degradation, never negative. For ranking only."""
        return max(0.0, -self.delta_pp)

    @property
    def top_decline(self) -> tuple[DeclineCode, float] | None:
        if not self.declines:
            return None
        total = sum(self.declines.values())
        code = max(self.declines, key=self.declines.get)
        return code, self.declines[code] / total

    def line(self) -> str:
        top = self.top_decline
        tail = f"  {top[0].value} {top[1]:.0%}" if top else ""
        flag = "  <-- FAILING" if self.drop_pp >= 8 else ""
        return (
            f"    {self.segment.value:<22} n={self.attempts:>5}  "
            f"{self.rate:>6.1%} vs {self.reference_rate:>6.1%} ref  "
            f"({self.delta_pp:>+6.1f}pp){tail}{flag}"
        )


@dataclass
class EvidenceBundle:
    cluster_id: str
    starts_at: datetime
    ends_at: datetime
    primary: SegmentObservation
    peers_by_dimension: dict[str, list[SegmentObservation]]
    money_at_risk: Money
    total_failures: int
    duration_minutes: float

    def render(self) -> str:
        """The text a model reasons over. Compact, factual, no leading."""
        lines = [
            f"INCIDENT WINDOW: {self.starts_at:%Y-%m-%d %H:%M} to {self.ends_at:%H:%M} UTC "
            f"({self.duration_minutes:.0f} minutes)",
            f"DETECTED SEGMENT: {self.primary.segment.describe()}",
            f"  success rate {self.primary.rate:.1%} against {self.primary.reference_rate:.1%} "
            f"reference ({self.primary.delta_pp:+.1f}pp)",
            f"  {self.total_failures} failed attempts, {self.money_at_risk} excess loss",
            "",
            "DECLINE CODES IN THE FAILING SEGMENT:",
        ]
        total = sum(self.primary.declines.values()) or 1
        for code, n in sorted(self.primary.declines.items(), key=lambda kv: -kv[1])[:6]:
            lines.append(f"    {code.value:<28} {n:>5}  {n / total:>5.1%}")

        lines.append("")
        lines.append("PEER SEGMENTS IN THE SAME WINDOW (healthy peers narrow the cause):")
        for dim, obs in self.peers_by_dimension.items():
            if not obs:
                continue
            lines.append(f"  by {dim}:")
            for o in sorted(obs, key=lambda o: -o.drop_pp)[:10]:
                lines.append(o.line())
        return "\n".join(lines)


class EvidenceBuilder:
    """Indexes a batch once, then answers window queries for any cluster."""

    DIMENSIONS = ("rail", "issuer", "acquirer")

    def __init__(self, attempts: list[PaymentAttempt]) -> None:
        self._attempts = sorted(attempts, key=lambda a: a.at)
        self._by_segment: dict[SegmentKey, list[PaymentAttempt]] = defaultdict(list)
        for a in self._attempts:
            for key in segments_for(a):
                self._by_segment[key].append(a)

    def _observe(self, key: SegmentKey, start: datetime, end: datetime) -> SegmentObservation:
        inside, outside_ok, outside_n = [], 0, 0
        for a in self._by_segment.get(key, ()):
            if start <= a.at <= end:
                inside.append(a)
            else:
                outside_n += 1
                outside_ok += a.succeeded
        declines: dict[DeclineCode, int] = defaultdict(int)
        successes = 0
        for a in inside:
            if a.succeeded:
                successes += 1
            elif a.decline_code:
                declines[a.decline_code] += 1
        return SegmentObservation(
            segment=key,
            attempts=len(inside),
            successes=successes,
            reference_rate=(outside_ok / outside_n) if outside_n else 0.0,
            declines=dict(declines),
        )

    def build(self, cluster: RiskCluster) -> EvidenceBundle:
        start, end = cluster.starts_at, cluster.ends_at
        primary = self._observe(cluster.segment, start, end)

        peers: dict[str, list[SegmentObservation]] = {}
        for dim in self.DIMENSIONS:
            values = {k.value for k in self._by_segment if k.dimension == dim}
            obs = [self._observe(SegmentKey(dim, v), start, end) for v in sorted(values)]
            # Thin slices produce meaningless rates and crowd out the signal.
            peers[dim] = [o for o in obs if o.attempts >= 20]

        # BIN detail only when the detector already pointed inside an issuer:
        # every BIN of every issuer would bury the comparison that matters.
        if cluster.segment.dimension in ("issuer", "rail_issuer", "issuer_bin"):
            issuer = cluster.segment.value.split(":")[-2 if ":" in cluster.segment.value else 0]
            if cluster.segment.dimension == "issuer_bin":
                issuer = cluster.segment.value.split(":")[0]
            elif cluster.segment.dimension == "rail_issuer":
                issuer = cluster.segment.value.split(":")[1]
            bins = {
                k.value
                for k in self._by_segment
                if k.dimension == "issuer_bin" and k.value.startswith(f"{issuer}:")
            }
            obs = [self._observe(SegmentKey("issuer_bin", v), start, end) for v in sorted(bins)]
            peers["issuer_bin"] = [o for o in obs if o.attempts >= 15]

        return EvidenceBundle(
            cluster_id=cluster.id,
            starts_at=start,
            ends_at=end,
            primary=primary,
            peers_by_dimension=peers,
            money_at_risk=cluster.money_at_risk,
            total_failures=primary.attempts - primary.successes,
            duration_minutes=(end - start).total_seconds() / 60,
        )
