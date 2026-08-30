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

## What it measures

Four arms, one batch, identical latent recoverability fixed before any arm
runs. The policy engine evaluates every action in every arm; the only
difference is whether its rulings are obeyed. Seed 1, 140k orders, 14,725
failures, `openai/gpt-oss-120b`:

| arm | recovered | orders | charges | contacts | wasted | violations |
|---|---|---|---|---|---|---|
| do-nothing | ₹0 | 0 | 0 | 0 | 0 | 0 |
| naive-retry | ₹43,01,596 | 4,576 | 44,175 | 14,725 | 54,324 | **21,734** |
| planner-unpoliced | ₹53,26,140 | 5,563 | 15,001 | 13,728 | 23,094 | 88 |
| **backstop** | **₹53,24,544** | 5,562 | 14,995 | 13,644 | 23,077 | **0** |

Three things worth reading off that table.

**Against the realistic alternative**, Backstop recovers ₹10.2L more than naive
retry while making 66% fewer charge attempts -- and naive breaks 21,734 rules
doing it, including 600 retries of cards reported stolen.

**The leash is nearly free.** The policed and unpoliced arms run identical
proposed actions, so their gap is the cost of compliance alone: ₹1,596 out of
₹53.2L, or 0.03% of recovered revenue, to go from 88 violations to zero.

**It holds when the model gets worse.** Swapping in `openai/gpt-oss-20b` drops
recovery 0.5% (₹53,00,397) and changes the guarantee not at all -- still zero
violations. The weak model proposes *worse* actions, including 10 SMS to
customers on the national DND registry and 49 badly-timed attempts, and the
engine refuses every one. That is the whole argument: safety that does not
depend on model quality.

Rates in `simulate/recoverability.py` are stated configuration, not measured
from Razorpay traffic. Read the table as a comparison under a declared world
model, not as a forecast.

## Status

**Built:** the full payments pipeline, end to end -- domain model, LLM layer,
seeded generator with labelled incidents, multi-resolution detection with scope
correlation, LLM diagnosis, the policy engine, the planner, execution, the
ledger, and the four-arm backtest. 192 tests.

**Not built:** subscriptions are generated but nothing charges or detects
mandates; receivables have policy rules but no detection, and invoice buyers
have no `Customer` record so every contact is denied by `ConsentRule`; and
`execute/` has only the simulated backend, no Razorpay test-mode adapter.

## Seeing it run

```bash
.venv/bin/python -m backstop.demo                    # full walkthrough, live model
.venv/bin/python -m backstop.demo --offline          # no API calls
.venv/bin/python -m backstop.demo --stage policy     # just the safety boundary
.venv/bin/python -m backstop.evaluation.backtest     # the four-arm measurement
.venv/bin/python -m backstop.evaluation.backtest --model openai/gpt-oss-20b
.venv/bin/python -m backstop.evaluation.bench --seeds 10   # detection across seeds
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
