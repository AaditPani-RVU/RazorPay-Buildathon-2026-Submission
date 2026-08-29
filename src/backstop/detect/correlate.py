"""Roll co-occurring signals up to the scope the problem actually has.

Scanning many slices means one incident lights up many of them. A card-wide
authentication fault breaches card:HDFC, card:ICICI, card:AXIS and rail=card
all at once. Reporting those as five findings is wrong twice over: it buries an
operator in alerts, and it hands diagnosis the wrong scope -- "ICICI is broken"
sends someone to call a bank about a problem that is in the checkout flow.

Correlation groups signals that overlap in time and picks the segment that
*covers* the others as the primary, keeping the rest as supporting evidence.
Breadth alone is not enough to win: a parent is only chosen when enough of its
children are actually breaching, otherwise a genuinely BIN-specific problem
would be laundered into a bank-wide one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from backstop.detect.detector import RiskSignal, SegmentKey
from backstop.domain.money import Money


def covers(parent: SegmentKey, child: SegmentKey) -> bool:
    """Whether every attempt in `child` is also in `parent`."""
    if parent == child:
        return False
    p, c = parent.dimension, child.dimension
    if p == "global":
        return True
    if p == "rail" and c == "rail_issuer":
        return child.value.startswith(f"{parent.value}:")
    if p == "issuer":
        if c == "rail_issuer":
            return child.value.endswith(f":{parent.value}")
        if c == "issuer_bin":
            return child.value.startswith(f"{parent.value}:")
    if p == "rail_issuer" and c == "issuer_bin":
        _, _, issuer = parent.value.partition(":")
        return child.value.startswith(f"{issuer}:")
    return False


@dataclass
class RiskCluster:
    """One problem, at the scope it actually has, with its corroboration."""

    id: str
    primary: RiskSignal
    supporting: list[RiskSignal] = field(default_factory=list)

    @property
    def segment(self) -> SegmentKey:
        return self.primary.segment

    @property
    def starts_at(self) -> datetime:
        return min([self.primary.starts_at] + [s.starts_at for s in self.supporting])

    @property
    def ends_at(self) -> datetime:
        return max([self.primary.ends_at] + [s.ends_at for s in self.supporting])

    @property
    def money_at_risk(self) -> Money:
        """The primary's figure. Summing children would double-count the same
        failed attempts, which appear in every slice they belong to."""
        return self.primary.money_at_risk

    @property
    def breadth(self) -> int:
        return 1 + len(self.supporting)

    def describe(self) -> str:
        extra = f"  (+{len(self.supporting)} corroborating)" if self.supporting else ""
        return self.primary.describe() + extra


def _overlap(a: RiskSignal, b: RiskSignal) -> bool:
    return a.starts_at <= b.ends_at and b.starts_at <= a.ends_at


def correlate(signals: list[RiskSignal], *, margin: float = 1.3) -> list[RiskCluster]:
    """Group overlapping signals and elect the right scope for each group.

    `margin` is how much sharper a narrow signal must be than the broadest one
    in its group before it is treated as the true locus rather than a slice
    that happened to look worst.
    """
    if not signals:
        return []

    # Group by time overlap *and* dominant decline code. Time alone is not
    # enough: at coarse resolutions almost everything overlaps everything, so
    # a routing fault and a funds crunch running the same afternoon would be
    # merged into one finding. The decline code is what says they are two
    # different problems.
    ordered = sorted(signals, key=lambda s: s.starts_at)
    groups: list[list[RiskSignal]] = []
    for sig in ordered:
        for g in groups:
            if g[0].dominant_decline == sig.dominant_decline and any(
                _overlap(sig, other) for other in g
            ):
                g.append(sig)
                break
        else:
            groups.append([sig])

    clusters: list[RiskCluster] = []
    for n, group in enumerate(groups, start=1):
        clusters.extend(_elect(group, n, margin))
    return sorted(clusters, key=lambda c: c.starts_at)


SPECIFICITY = {"global": 0, "rail": 1, "issuer": 1, "acquirer": 1, "rail_issuer": 2, "issuer_bin": 3}


def _elect(group: list[RiskSignal], n: int, margin: float) -> list[RiskCluster]:
    """Choose the scope a group of co-occurring signals actually has.

    Breadth cannot decide this: the global segment covers every other one, so
    counting children always elects it. Dilution can. A problem confined to one
    acquirer shows a large drop in that slice and a small one globally, because
    the healthy traffic averages it away. A genuinely global problem shows
    comparable drops at every level.

    So the narrowest strong signal only wins if it beats the broadest by
    `margin`. Otherwise the broad reading stands, and a real all-traffic
    incident is not mistaken for whichever slice happened to look worst.
    """
    broadest = min(group, key=lambda s: (SPECIFICITY.get(s.segment.dimension, 1), -s.rate_drop))
    sharpest = max(group, key=lambda s: (s.rate_drop, SPECIFICITY.get(s.segment.dimension, 1)))

    primary = sharpest if sharpest.rate_drop >= margin * broadest.rate_drop else broadest
    supporting = [s for s in group if s is not primary]
    return [RiskCluster(id=f"cluster_{n:03d}", primary=primary, supporting=supporting)]
