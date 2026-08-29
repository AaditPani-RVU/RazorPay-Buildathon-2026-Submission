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

Foundations in place: money type, provider-agnostic LLM layer with schema
validation and bounded repair, Groq backend, scripted offline backend.

## Setup

```bash
uv venv --python 3.13 .venv
uv pip install -e ".[dev]"
cp .env.example .env    # add GROQ_API_KEY; RAZORPAY_* keys are optional
.venv/bin/python -m pytest -q
```

The test suite runs fully offline against `ScriptedProvider` — no API key needed.
