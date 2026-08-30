# Backstop

A bounded revenue-recovery agent for the Razorpay AI Buildathon 2026 — **AI Revenue Recovery** track.

Backstop detects revenue at risk, diagnoses the root cause, and executes a
**policy-constrained** recovery workflow across payment failures, failed
subscription mandates, and overdue receivables — then measures the money it
actually recovered against baselines on an identical batch.

## The design in one line

**The model proposes, the policy engine disposes.**

Every recovery action an LLM suggests is validated against a deterministic
policy engine before it can execute. Retry budgets, decline-code rules, contact
caps, quiet hours, value thresholds and stopping rules are code, not prompts.
A vetoed action is logged with the rule that fired, so "zero policy violations"
is a provable claim rather than an assertion.

## Pipeline

```
Detect  ->  Diagnose  ->  Decide  ->  Enforce  ->  Execute  ->  Measure
(stats)     (LLM)         (LLM)       (rules)      (adapter)   (backtest)
```

- **Detect** — segmented success-rate anomaly detection, receivables aging,
  mandate-failure clustering. Deliberately no LLM: statistics are cheaper,
  faster and more accurate for this, and the money at risk is a number.
- **Diagnose** — structured root-cause attribution over an evidence bundle.
  Messy multi-signal attribution is where a model genuinely earns its place.
- **Decide** — a plan drawn from a typed action catalog, never free text.
- **Enforce** — the deterministic policy engine. The heart of the system.
- **Execute** — simulated adapters and Razorpay test-mode APIs behind one interface.
- **Measure** — replay one labelled batch through three arms and report the delta.

## Status

**Built:** the domain model (money, decline taxonomy, entities, typed action
catalog), a provider-agnostic LLM layer with schema validation and bounded
repair, a seeded scenario generator with labelled incidents, multi-resolution
detection with scope correlation, LLM root-cause diagnosis, the policy engine,
and scoring for detection and diagnosis.

**Not built:** `decide/` (the planner), `execute/` (rail adapters), `ledger/`,
and the three-arm backtest that produces the money-recovered figure.

## Seeing it run

```bash
.venv/bin/python -m backstop.demo              # full walkthrough, live model
.venv/bin/python -m backstop.demo --offline    # no API calls
.venv/bin/python -m backstop.demo --stage policy   # just the safety boundary
.venv/bin/python -m backstop.evaluation.bench --seeds 10
```

The walkthrough generates a labelled batch, detects and scores against truth,
diagnoses each cluster and scores that, then puts fourteen probe actions
through the policy engine -- including ones that must be refused. Ground truth
is printed beside every prediction.

## Setup

```bash
uv venv --python 3.13 .venv
uv pip install -e ".[dev]"
cp .env.example .env    # add GROQ_API_KEY; RAZORPAY_* keys are optional
.venv/bin/python -m pytest -q
```

The test suite runs fully offline against `ScriptedProvider` — no API key needed.
