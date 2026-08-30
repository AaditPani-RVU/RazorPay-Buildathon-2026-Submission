"""Root-cause attribution. The first stage where a model does real work.

Detection says *that* money is leaking and where. Diagnosis says *why*, which
is what determines whether the right response is a retry, a re-engagement, a
routing switch or nothing at all. Getting this wrong is expensive in both
directions: retrying an authentication drop-off burns issuer goodwill and
recovers nothing, and re-engaging customers during an issuer outage annoys
people whose payments would have worked on their own.

The output is a closed enum, so the planner can switch on it and the policy
engine can reason about it. `ruled_out` is required rather than decorative: an
explanation that cannot say what it is *not* is usually pattern-matching on the
loudest decline code, and forcing the contrast is what makes the routing versus
issuer distinction stick.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from backstop.diagnose.evidence import EvidenceBundle
from backstop.domain.declines import RootCause
from backstop.llm import LLMClient, LLMError

TAXONOMY: dict[RootCause, str] = {
    RootCause.ISSUER_OUTAGE:
        "One issuer's authorisation host is degraded. That issuer fails badly "
        "while other issuers stay healthy, usually with timeout or "
        "issuer-unavailable declines.",
    RootCause.PSP_OR_RAIL_OUTAGE:
        "A whole rail or PSP is degraded. Every issuer on that rail suffers "
        "together while other rails are fine.",
    RootCause.GATEWAY_ROUTING_DEGRADATION:
        "One acquirer or route is underperforming. The failing acquirer is far "
        "worse than its peers, and issuers look mildly degraded only because "
        "that acquirer carries traffic for all of them.",
    RootCause.BIN_SPECIFIC_DECLINE:
        "Failures concentrate in one BIN range rather than a whole issuer. The "
        "issuer's other BINs are healthy.",
    RootCause.CHECKOUT_REGRESSION:
        "A merchant-side change broke the flow. Failures start abruptly and cut "
        "across issuers, rails and acquirers alike.",
    RootCause.AUTHENTICATION_DROPOFF:
        "Customers reach the 3-D Secure or OTP step and abandon. Dominated by "
        "otp_abandoned or auth_3ds_failed. The instruments are fine; the "
        "authentication step is the leak.",
    RootCause.INSUFFICIENT_FUNDS_CLUSTER:
        "A genuine funds crunch, not a system fault. Dominated by "
        "insufficient_funds across many segments at once, typically before "
        "payday. Nothing is broken.",
    RootCause.MANDATE_LIFECYCLE_FAILURE:
        "Subscription charges failing on expired, paused or revoked mandates.",
    RootCause.BUYER_CASHFLOW_DELAY:
        "A receivable is late because the buyer cannot pay yet.",
    RootCause.INVOICE_DISPUTE:
        "A receivable is late because the buyer contests it.",
    RootCause.FRAUD_PRESSURE:
        "Failures are risk declines. Recovery must not fight the risk engine.",
    RootCause.NO_SYSTEMIC_CAUSE:
        "Ordinary baseline noise. The correct action is frequently to do nothing.",
}

SYSTEM_PROMPT = """You are a payments reliability analyst at a payment gateway.
You are given one detected degradation and the state of every comparable
segment in the same window. Attribute the systemic root cause.

Work in two steps, in this order. They answer different questions and the
first one takes precedence.

STEP 1 -- WHAT FAILED. Read the dominant decline code. It is direct evidence of
the failure mechanism and it constrains which causes are possible at all:

  otp_abandoned, auth_3ds_failed        -> authentication_dropoff
  psp_unavailable                       -> psp_or_rail_outage
  issuer_unavailable                    -> issuer_outage
  gateway_timeout                       -> routing degradation, or a rail fault
  do_not_honor                          -> bin_specific_decline or issuer_outage
  insufficient_funds                    -> insufficient_funds_cluster
  risk_declined_by_gateway, stolen_card -> fraud_pressure
  mandate_*                             -> mandate_lifecycle_failure

Infrastructure faults produce infrastructure declines. Customers abandoning an
OTP screen is a behavioural failure, not an outage, however the traffic happens
to be sliced -- so a segment dominated by otp_abandoned is an authentication
drop-off even when exactly one rail is affected and the others look healthy.

STEP 2 -- WHERE IT ORIGINATES. Now read the PEER SEGMENTS to place the locus
among the causes step 1 left open. A failing slice looks the same for many
causes; only the peers separate them.

- One acquirer far worse than its peers => routing degradation, even though
  every issuer will also look mildly degraded, because that acquirer carries
  traffic for all of them.
- One issuer far worse than its peers, across acquirers => issuer outage.
- Every issuer on one rail degraded while other rails are healthy => rail or
  PSP outage -- but only if step 1 pointed at an infrastructure decline.
- One BIN far worse while the issuer's other BINs are healthy => BIN-specific.
- Everything degraded roughly equally, dominated by insufficient_funds => a
  funds cluster, not a fault. Nothing is broken.

Judge by the size of the gap between the worst segment and its peers, not by
how many segments carry a warning marker. Contamination from a shared component
is expected and is not itself evidence of a second cause.

Cite only figures that appear in the evidence. Do not invent deploys, incidents
or history you were not given. Set confidence below 0.6 when steps 1 and 2
disagree or the peers do not separate the candidates."""


class Diagnosis(BaseModel):
    root_cause: RootCause
    confidence: float = Field(ge=0.0, le=1.0)
    locus: str = Field(
        description="The single segment that is the true origin, e.g. 'acquirer=acq_beta'."
    )
    key_evidence: list[str] = Field(
        min_length=1, max_length=4,
        description="Specific figures from the evidence that support the conclusion.",
    )
    ruled_out: list[RootCause] = Field(
        default_factory=list, max_length=3,
        description="Causes this evidence excludes, and which were plausible before it.",
    )


@dataclass
class DiagnosisResult:
    cluster_id: str
    diagnosis: Diagnosis | None
    repaired: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.diagnosis is not None


class Diagnoser:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def diagnose(self, bundle: EvidenceBundle) -> DiagnosisResult:
        """Attribute a cause, or record that no usable answer came back.

        A failure here is not fatal. An undiagnosed cluster simply carries no
        recommended intervention, which is the safe default: the pipeline does
        nothing rather than acting on a guess.
        """
        catalogue = "\n".join(f"- {c.value}: {d}" for c, d in TAXONOMY.items())
        user = f"{bundle.render()}\n\nCAUSES YOU MAY CHOOSE FROM:\n{catalogue}"
        try:
            result = self.client.structured(
                system=SYSTEM_PROMPT, user=user, schema=Diagnosis
            )
        except LLMError as err:
            return DiagnosisResult(bundle.cluster_id, None, error=str(err))
        return DiagnosisResult(bundle.cluster_id, result.value, repaired=result.repaired)
