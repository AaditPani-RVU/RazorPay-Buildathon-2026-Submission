"""Multi-resolution detection.

A single bucket size cannot work. The volume floor that keeps a rate estimate
honest also makes the detector blind wherever traffic is thin -- and traffic is
thin in exactly the two places incidents like to hide:

*   Narrow slices. A single BIN carries ~4 attempts per 15 minutes here, so a
    BIN-level problem is unmeasurable at that resolution however large it is.
*   Off-peak hours. Traffic swings ~4x across the day, so an incident starting
    at 02:00 IST is measured on a quarter of the volume of one at 21:00. An
    incident that would be obvious at lunchtime is invisible overnight.

So detection runs at several resolutions at once. Coarse windows see thin
slices and quiet hours; fine windows see fast, loud incidents sooner. The cost
is latency, and it is paid only where it is needed: a signal is reported at the
finest resolution that could see it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from backstop.detect.detector import Detector, DetectorConfig, RiskSignal


@dataclass(frozen=True)
class Resolution:
    bucket_minutes: int
    min_attempts: int
    sustained_buckets: int
    label: str

    def config(self, base: DetectorConfig) -> DetectorConfig:
        return replace(
            base,
            bucket_minutes=self.bucket_minutes,
            min_attempts=self.min_attempts,
            sustained_buckets=self.sustained_buckets,
        )


# Coarser windows aggregate more evidence per test, so a single breaching
# bucket is already strong; requiring two would double an already slow
# detection for no statistical gain.
DEFAULT_RESOLUTIONS: list[Resolution] = [
    Resolution(15, 30, 2, "fast"),
    Resolution(60, 40, 2, "medium"),
    Resolution(240, 50, 1, "slow"),
]


@dataclass
class MultiResolutionDetector:
    base: DetectorConfig = field(default_factory=DetectorConfig)
    resolutions: list[Resolution] = field(default_factory=lambda: list(DEFAULT_RESOLUTIONS))

    def run(self, attempts) -> list[RiskSignal]:
        found: list[tuple[int, RiskSignal]] = []
        for idx, res in enumerate(self.resolutions):
            for sig in Detector(res.config(self.base)).run(attempts):
                found.append((idx, sig))
        return self._dedupe(found)

    def _dedupe(self, found: list[tuple[int, RiskSignal]]) -> list[RiskSignal]:
        """Collapse the same incident seen at several resolutions.

        Finest resolution wins, because it found the problem soonest. Signals
        in *different* segments are kept: a routing fault genuinely visible in
        both the acquirer and rail views is two pieces of evidence, and
        diagnosis wants both.
        """
        kept: list[tuple[int, RiskSignal]] = []
        for idx, sig in sorted(found, key=lambda p: (p[0], p[1].starts_at)):
            duplicate = any(
                k.segment == sig.segment
                and k.starts_at <= sig.ends_at
                and sig.starts_at <= k.ends_at
                for _, k in kept
            )
            if not duplicate:
                kept.append((idx, sig))
        return sorted((s for _, s in kept), key=lambda s: s.starts_at)
