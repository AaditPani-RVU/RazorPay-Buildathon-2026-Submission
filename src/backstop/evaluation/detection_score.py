"""Score detection against the simulator's labels.

Matching is deliberately strict on cause. Time overlap and segment
compatibility alone would let a signal be credited to whichever incident
happened to be running, so a signal must also agree on the dominant decline
code. That is what separates "found the UPI outage through the acquirer view"
from "fired during an unrelated incident and got lucky".

Signals are sorted into three outcomes rather than two. A second signal for an
already-detected incident is *redundant*, not a false positive -- it costs an
operator attention but it is not a wrong claim, and collapsing the two
categories would hide which problem the detector actually has.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backstop.detect.detector import RiskSignal, SegmentKey
from backstop.domain.money import Money
from backstop.simulate.scenario import Incident, Scenario


def _segment_compatible(key: SegmentKey, inc: Incident) -> bool:
    """Whether a signal's slice could be a view of this incident's slice."""
    seg = inc.segment
    if key.dimension == "global":
        return True
    if key.dimension == "rail":
        return seg.rail is None or str(seg.rail) == key.value
    if key.dimension == "issuer":
        return seg.issuer is None or seg.issuer == key.value
    if key.dimension == "acquirer":
        return seg.acquirer is None or seg.acquirer == key.value
    if key.dimension == "issuer_bin":
        issuer, _, bin_ = key.value.partition(":")
        return (seg.issuer is None or seg.issuer == issuer) and (
            seg.bin is None or seg.bin == bin_
        )
    if key.dimension == "rail_issuer":
        rail, _, issuer = key.value.partition(":")
        return (seg.rail is None or str(seg.rail) == rail) and (
            seg.issuer is None or seg.issuer == issuer
        )
    return False


def _overlaps(sig: RiskSignal, inc: Incident) -> bool:
    return sig.starts_at <= inc.ends_at and inc.starts_at <= sig.ends_at


def match(sig: RiskSignal, incidents: list[Incident]) -> Incident | None:
    """Best incident this signal can be credited to, if any."""
    candidates = [
        inc
        for inc in incidents
        if _overlaps(sig, inc)
        and _segment_compatible(sig.segment, inc)
        and sig.dominant_decline == inc.dominant_decline
    ]
    if not candidates:
        return None
    # Prefer the most specific segment: a BIN incident beats an all-traffic one.
    return max(candidates, key=lambda i: sum(v is not None for v in vars(i.segment).values()))


@dataclass
class DetectionReport:
    detected: dict[str, RiskSignal] = field(default_factory=dict)
    """incident id -> earliest signal that found it."""
    missed: list[Incident] = field(default_factory=list)
    false_positives: list[RiskSignal] = field(default_factory=list)
    redundant: list[RiskSignal] = field(default_factory=list)
    latency_minutes: dict[str, float] = field(default_factory=dict)
    incidents: list[Incident] = field(default_factory=list)

    @property
    def recall(self) -> float:
        n = len(self.incidents)
        return len(self.detected) / n if n else 0.0

    @property
    def precision(self) -> float:
        """Redundant signals count as correct: they name a real incident."""
        useful = len(self.detected) + len(self.redundant)
        total = useful + len(self.false_positives)
        return useful / total if total else 0.0

    @property
    def money_found(self) -> Money:
        total = Money.zero()
        for inc in self.incidents:
            if inc.id in self.detected:
                total += inc.money_at_risk
        return total

    @property
    def money_missed(self) -> Money:
        total = Money.zero()
        for inc in self.incidents:
            if inc.id not in self.detected:
                total += inc.money_at_risk
        return total

    @property
    def mean_latency_minutes(self) -> float:
        v = list(self.latency_minutes.values())
        return sum(v) / len(v) if v else 0.0

    def render(self) -> str:
        lines = [
            f"recall     {self.recall:>6.0%}  ({len(self.detected)}/{len(self.incidents)} incidents)",
            f"precision  {self.precision:>6.0%}  ({len(self.false_positives)} false, "
            f"{len(self.redundant)} redundant)",
            f"latency    {self.mean_latency_minutes:>6.0f}m mean to first signal",
            f"money      {self.money_found} found / {self.money_missed} missed",
            "",
        ]
        for inc in self.incidents:
            sig = self.detected.get(inc.id)
            if sig:
                lines.append(
                    f"  FOUND   {inc.root_cause.value:<28} {inc.segment.describe():<30}"
                    f" via {sig.segment.describe():<22} +{self.latency_minutes[inc.id]:>4.0f}m"
                )
            else:
                lines.append(
                    f"  MISSED  {inc.root_cause.value:<28} {inc.segment.describe():<30}"
                    f" {len(inc.affected_order_ids)} orders  {inc.money_at_risk}"
                )
        for sig in self.false_positives:
            lines.append(f"  FALSE   {sig.describe()}")
        return "\n".join(lines)


def score(scenario: Scenario, signals: list[RiskSignal]) -> DetectionReport:
    report = DetectionReport(incidents=list(scenario.incidents))
    for sig in sorted(signals, key=lambda s: s.starts_at):
        inc = match(sig, scenario.incidents)
        if inc is None:
            report.false_positives.append(sig)
        elif inc.id in report.detected:
            report.redundant.append(sig)
        else:
            report.detected[inc.id] = sig
            delay = (sig.starts_at - inc.starts_at).total_seconds() / 60
            report.latency_minutes[inc.id] = max(0.0, delay)
    report.missed = [i for i in scenario.incidents if i.id not in report.detected]
    return report
