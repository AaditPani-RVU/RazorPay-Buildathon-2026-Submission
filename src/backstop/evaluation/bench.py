"""Reproducible detection benchmark across seeds.

Single-seed numbers are how you fool yourself: thresholds tuned against one
scenario look excellent on it and mediocre everywhere else. This sweeps seeds
so the reported figure is a distribution with a worst case, not a best case.

    python -m backstop.evaluation.bench --seeds 10
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

from backstop.detect.correlate import correlate
from backstop.detect.multires import MultiResolutionDetector
from backstop.evaluation.detection_score import DetectionReport, score
from backstop.simulate.generator import SimConfig, generate
from backstop.simulate.scenario import Incident, Scenario

_DIMENSION_FOR_SEGMENT = {
    "global": lambda s: not any([s.rail, s.issuer, s.acquirer, s.bin]),
    "rail": lambda s: s.rail is not None and not any([s.issuer, s.bin, s.acquirer]),
    "issuer": lambda s: s.issuer is not None and not any([s.rail, s.bin]),
    "acquirer": lambda s: s.acquirer is not None,
    "rail_issuer": lambda s: s.rail is not None and s.issuer is not None and s.bin is None,
    "issuer_bin": lambda s: s.bin is not None,
}


def scope_is_exact(incident: Incident, report: DetectionReport) -> bool:
    """Whether detection named the incident's own slice, not merely a
    compatible view of it. Diagnosis consumes this scope, so a card-wide fault
    reported as one bank's problem sends the next stage somewhere wrong."""
    sig = report.detected.get(incident.id)
    if sig is None:
        return False
    test = _DIMENSION_FOR_SEGMENT.get(sig.segment.dimension)
    return bool(test and test(incident.segment))


@dataclass
class BenchRow:
    seed: int
    clusters: int
    recall: float
    precision: float
    exact_scope: float
    latency_minutes: float


def run_one(seed: int, days: int, orders_per_day: int) -> tuple[BenchRow, Scenario]:
    scenario = generate(SimConfig(seed=seed, days=days, orders_per_day=orders_per_day))
    attempts = [a for o in scenario.orders for a in o.attempts]
    clusters = correlate(MultiResolutionDetector().run(attempts))
    report = score(scenario, [c.primary for c in clusters])
    exact = sum(scope_is_exact(i, report) for i in scenario.incidents)
    return (
        BenchRow(
            seed=seed,
            clusters=len(clusters),
            recall=report.recall,
            precision=report.precision,
            exact_scope=exact / len(scenario.incidents) if scenario.incidents else 0.0,
            latency_minutes=report.mean_latency_minutes,
        ),
        scenario,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--orders-per-day", type=int, default=20000)
    args = ap.parse_args()

    rows = [run_one(s, args.days, args.orders_per_day)[0] for s in range(1, args.seeds + 1)]

    head = f"{'seed':>5} {'clusters':>9} {'recall':>7} {'precision':>10} {'scope':>7} {'latency':>8}"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r.seed:>5} {r.clusters:>9} {r.recall:>6.0%} {r.precision:>9.0%} "
            f"{r.exact_scope:>6.0%} {r.latency_minutes:>7.0f}m"
        )
    print("-" * len(head))

    def line(name: str, values: list[float], fmt: str = ".0%") -> str:
        return (
            f"  {name:<12} mean {format(statistics.mean(values), fmt)}"
            f"   worst {format(min(values), fmt)}"
        )

    print(line("recall", [r.recall for r in rows]))
    print(line("precision", [r.precision for r in rows]))
    print(line("exact scope", [r.exact_scope for r in rows]))
    lat = [r.latency_minutes for r in rows]
    print(f"  {'latency':<12} mean {statistics.mean(lat):.0f}m   worst {max(lat):.0f}m")


if __name__ == "__main__":
    main()
