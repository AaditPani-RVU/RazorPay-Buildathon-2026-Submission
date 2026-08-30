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

| arm | recovered | of which illegal | keepable net | charges | contacts | violations |
|---|---|---|---|---|---|---|
| do-nothing | ₹0 | – | ₹0 | 0 | 0 | 0 |
| naive-retry | ₹49,16,423 | ₹49,821 | ₹48,20,689 | 51,015 | 17,005 | **25,312** |
| planner-unpoliced | ₹58,34,191 | ₹1,565 | ₹58,03,296 | 16,634 | 15,118 | 100 |
| **backstop** | **₹58,32,626** | **–** | **₹58,03,689** | 16,027 | 15,018 | **0** |

*"Of which illegal" is revenue taken by actions the rules refuse — a merchant
cannot keep it, so crediting it would score the baseline for exactly the
behaviour the policy engine exists to stop.*

Three things worth reading off that table.

**Against the realistic alternative**, Backstop keeps ₹9.8L more than naive
retry while making 69% fewer charge attempts -- and naive breaks 25,312 rules
doing it, including 600 retries of cards reported stolen and 1,524 attempts to
chase customers who explicitly revoked their mandate.

**The leash pays for itself.** The policed and unpoliced arms run identical
proposed actions, so their gap is the cost of compliance alone. It is negative:
the unpoliced arm recovers ₹1,565 it cannot keep and spends more to do it, so
policing ends up ahead on keepable revenue as well as on safety.

**It holds when the model gets worse.** Swapping in `openai/gpt-oss-20b` drops
recovery 0.5% and changes the guarantee not at all -- still zero violations.
The weak model proposes *worse* actions, including 10 SMS to customers on the
national DND registry and 49 badly-timed attempts, and the engine refuses every
one. That is the whole argument: safety that does not depend on model quality.

**It holds when the model is gone.** A provider outage or a rate limit
degrades recovery to the deterministic playbook rather than stopping it -- on a
live 429 the pipeline still proposed 7,858 actions. Nothing about the guarantee
depends on the model being reachable either.

Rates in `simulate/recoverability.py` are stated configuration, not measured
from Razorpay traffic. Read the table as a comparison under a declared world
model, not as a forecast.

### Recurring revenue

Subscriptions fail differently: the usual failure is a *state*, not an event.
Mandates lapse one at a time, nothing spikes, and an anomaly detector correctly
reports all clear. So a second path scans the book and prices the dead
authorisations:

```
active mandates    6,669
lapsed mandates    1,331
recurring at risk  ₹1,70,30,028.00 per year
expected recovery  ₹60,96,557.76 if all are chased

  expired   542  ₹70,48,296/yr  ~42% recoverable  lapsed; re-registration restores it
  revoked   381  ₹50,07,828/yr  ~8% recoverable   a decision, not a lapse
  paused    408  ₹49,73,904/yr  ~55% recoverable  may resume on its own
```

The revoked row is the interesting one. Recovery never auto-chases it: somebody
who cancelled a mandate mostly meant to, and an agent that responds by
re-requesting authorisation has not recovered revenue, it has ignored a
cancellation. `revoked_mandate` refuses that and routes it to a person. Naive
retry attempts it 1,524 times.

## Status

**Built:** payments and subscriptions, end to end -- domain model, LLM layer,
seeded generator with labelled incidents, multi-resolution detection with scope
correlation, mandate lifecycle scanning, LLM diagnosis, a 15-rule policy
engine, the planner, execution, the ledger, and the four-arm backtest. 222
tests, and the repo is lint clean.

**Not built:** receivables are the last of the three revenue surfaces -- they
have policy rules but no detection, and invoice buyers have no `Customer`
record so every contact is denied by `ConsentRule`. `execute/` also has only
the simulated backend, no Razorpay test-mode adapter.

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
