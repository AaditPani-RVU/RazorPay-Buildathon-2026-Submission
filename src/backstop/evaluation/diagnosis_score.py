"""Score root-cause attribution against the simulator's labels.

Accuracy alone would hide the failure that matters. Confusing an issuer outage
for a rail outage is an operational annoyance; confusing an authentication
drop-off for either is a *recovery* error, because it turns "bring the customer
back" into "retry the charge", which cannot work and burns issuer goodwill on
every attempt. So the confusion matrix is reported, not just the score.

Confidence is scored for calibration as well as accuracy. A diagnoser that is
0.95 confident when wrong is more dangerous than one that is 0.6 confident when
right, because the planner downstream is entitled to trust it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from backstop.detect.correlate import RiskCluster
from backstop.diagnose.diagnoser import DiagnosisResult
from backstop.domain.declines import RootCause
from backstop.evaluation.detection_score import match
from backstop.simulate.scenario import Scenario

# Mistaking any of these for one another sends recovery down a path that cannot
# work, rather than merely mislabelling a real problem.
RETRY_SAFE = {
    RootCause.ISSUER_OUTAGE,
    RootCause.PSP_OR_RAIL_OUTAGE,
    RootCause.GATEWAY_ROUTING_DEGRADATION,
    RootCause.BIN_SPECIFIC_DECLINE,
}
NEVER_RETRY = {
    RootCause.AUTHENTICATION_DROPOFF,
    RootCause.INVOICE_DISPUTE,
    RootCause.FRAUD_PRESSURE,
    RootCause.MANDATE_LIFECYCLE_FAILURE,
}


@dataclass
class DiagnosisReport:
    correct: int = 0
    wrong: int = 0
    unparseable: int = 0
    confusions: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    confidence_when_right: list[float] = field(default_factory=list)
    confidence_when_wrong: list[float] = field(default_factory=list)
    dangerous: list[tuple[str, str]] = field(default_factory=list)
    repairs: int = 0

    @property
    def total(self) -> int:
        return self.correct + self.wrong + self.unparseable

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def mean_confidence_right(self) -> float:
        v = self.confidence_when_right
        return sum(v) / len(v) if v else 0.0

    @property
    def mean_confidence_wrong(self) -> float:
        v = self.confidence_when_wrong
        return sum(v) / len(v) if v else 0.0

    @property
    def calibration_gap(self) -> float:
        """How much more confident the model is when right than when wrong.
        Near zero means confidence carries no information and the planner
        should not be allowed to gate on it."""
        if not self.confidence_when_wrong:
            return float("nan")
        return self.mean_confidence_right - self.mean_confidence_wrong

    def render(self) -> str:
        lines = [
            f"accuracy      {self.accuracy:.0%}  ({self.correct}/{self.total})",
            f"unparseable   {self.unparseable}   repairs {self.repairs}",
            (f"confidence    {self.mean_confidence_right:.2f} when right, "
            f"{self.mean_confidence_wrong:.2f} when wrong "
            f"(gap {self.calibration_gap:+.2f})"),
            f"dangerous     {len(self.dangerous)} retry-class errors",
        ]
        if self.confusions:
            lines.append("confusions:")
            for (truth, got), n in sorted(self.confusions.items(), key=lambda kv: -kv[1]):
                mark = "  <-- changes the recovery action" if (truth, got) in self.dangerous else ""
                lines.append(f"    {truth:<28} -> {got:<28} x{n}{mark}")
        return "\n".join(lines)


def is_dangerous(truth: RootCause, got: RootCause) -> bool:
    """Whether the mistake flips the class of intervention, not just the label."""
    return (truth in NEVER_RETRY and got in RETRY_SAFE) or (
        truth in RETRY_SAFE and got in NEVER_RETRY
    )


def score(
    scenario: Scenario,
    pairs: list[tuple[RiskCluster, DiagnosisResult]],
    report: DiagnosisReport | None = None,
) -> DiagnosisReport:
    rep = report or DiagnosisReport()
    for cluster, result in pairs:
        incident = match(cluster.primary, scenario.incidents)
        if incident is None:
            continue  # a false-positive cluster has no cause to be right about
        if result.repaired:
            rep.repairs += 1
        if not result.ok:
            rep.unparseable += 1
            continue
        truth, got = incident.root_cause, result.diagnosis.root_cause
        if truth == got:
            rep.correct += 1
            rep.confidence_when_right.append(result.diagnosis.confidence)
        else:
            rep.wrong += 1
            rep.confidence_when_wrong.append(result.diagnosis.confidence)
            rep.confusions[(truth.value, got.value)] += 1
            if is_dangerous(truth, got):
                rep.dangerous.append((truth.value, got.value))
    return rep
