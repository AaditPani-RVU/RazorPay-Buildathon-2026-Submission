# Backstop

A bounded revenue-recovery agent for the Razorpay AI Buildathon 2026 — **AI Revenue Recovery** track.

Backstop detects revenue at risk, diagnoses the root cause, and executes a
**policy-constrained** recovery workflow across payment failures, failed
subscription mandates, and overdue receivables — then measures the money it
actually recovered against baselines on an identical batch.

## The design in one line

**The model proposes, the policy engine disposes.**

Every recovery action an LLM suggests is validated against a deterministic
policy engine before it can execute. Retry budgets, decline-code rules, per-
subject and per-person contact caps, quiet hours, value thresholds and stopping
rules are code, not prompts. A vetoed action is logged with the rule that
fired, so "zero policy violations" is a provable claim rather than an
assertion.

## Pipeline

```
Detect -> Diagnose -> Decide -> Enforce -> Schedule -> Execute -> Measure
(stats)   (LLM)       (LLM)     (rules)    (a clock)   (adapter)  (backtest)
                                   |                      ^
                                Approve                Reconcile
                              (a person)               (webhooks)
```

- **Detect** — three scans for three shapes of failure: segmented success-rate
  anomaly detection for payments, a lifecycle scan of the mandate book for
  recurring revenue, and an aging scan of the ledger for receivables.
  Deliberately no LLM anywhere in here: statistics are cheaper, faster and more
  accurate for this, and the money at risk is a number.
- **Diagnose** — structured root-cause attribution over an evidence bundle.
  Messy multi-signal attribution is where a model genuinely earns its place.
- **Decide** — a plan drawn from a typed action catalog, never free text.
- **Enforce** — the deterministic policy engine. The heart of the system.
- **Approve** — what the rules will not let automation decide alone waits for a
  person. An approval discharges the `require_approval` verdict and nothing
  else; the rules run again at release, and an unanswered request expires
  rather than firing.
- **Schedule** — each action is held until the moment the rules chose for it,
  and ruled on again when it gets there. Nothing fires early; nothing fires a
  day late.
- **Execute** — two backends behind one protocol: the simulator the measurement
  runs on, and a Razorpay test-mode adapter that really dispatches.
- **Reconcile** — a dispatch is not a recovery. A signature-verifying webhook
  receiver credits what actually settled, and only for entities recovery itself
  dispatched.
- **Measure** — replay one labelled batch through four arms and report the delta,
  on each surface separately and never summed.

## What it measures

Four arms, one batch, identical latent recoverability fixed before any arm
runs. The policy engine evaluates every action in every arm; the only
difference is whether its rulings are obeyed. Seed 1, 148k orders, 17,005
failures, 1,331 lapsed mandates, 377 overdue invoices, `openai/gpt-oss-20b`:

**Payments** — one-off checkout failures.

| arm | recovered | of which illegal | keepable net | charges | contacts | violations |
|---|---|---|---|---|---|---|
| do-nothing | ₹0 | – | ₹0 | 0 | 0 | 0 |
| naive-retry | ₹49,16,423 | ₹49,821 | ₹48,20,689 | 51,015 | 17,005 | **25,312** |
| planner-unpoliced | ₹58,12,106 | ₹13,540 | ₹57,69,575 | 16,668 | 14,883 | 203 |
| **backstop** | **₹57,96,298** | **–** | **₹57,67,780** | 15,979 | 14,751 | **0** |

**Recurring** — mandates that stopped collecting. ₹1,70,30,028 per year was at
risk across 1,331 lapsed authorisations.

| arm | recovered | of which illegal | keepable net | mandates | contacts | violations |
|---|---|---|---|---|---|---|
| do-nothing | ₹0 | – | ₹0 | 0 | 0 | 0 |
| naive-retry | ₹44,05,644 | ₹1,85,796 | ₹42,13,859 | 363 | 3,993 | **1,899** |
| planner-unpoliced | ₹42,19,848 | ₹49,140 | ₹41,68,470 | 346 | 1,492 | 20 |
| **backstop** | **₹41,81,472** | **–** | **₹41,79,263** | 344 | 1,473 | **0** |

**Receivables** — invoices that were never paid. ₹4,01,69,841 was outstanding
across 377 overdue invoices, of which only ₹69,06,052 was ever incrementally
collectable; the rest either arrives on the buyer's own cycle or never arrives.

| arm | recovered | of which illegal | keepable net | invoices | contacts | violations |
|---|---|---|---|---|---|---|
| do-nothing | ₹0 | – | ₹0 | 0 | 0 | 0 |
| naive-retry | ₹44,40,758 | ₹82,497 | ₹43,56,565 | 51 | 1,131 | **678** |
| planner-unpoliced | ₹47,27,686 | ₹36,612 | ₹46,90,405 | 57 | 446 | 18 |
| **backstop** | **₹46,91,074** | **–** | **₹46,90,432** | 55 | 428 | **0** |

*"Of which illegal" is revenue taken by actions the rules refuse — a merchant
cannot keep it, so crediting it would score the baseline for exactly the
behaviour the policy engine exists to stop.*

*The three surfaces are tabled apart and never added. A recovered payment is
one amount that landed; a re-registered mandate is a year of billing restored;
a collected invoice is a balance that was already owed. Summing them would
produce a headline nobody could reconcile.*

Five things worth reading off those tables.

**Against the realistic alternative**, Backstop keeps ₹9.5L more than naive
retry on payments while making 69% fewer charge attempts — and naive breaks
27,889 rules across the three surfaces doing it, including 600 retries of cards
reported stolen, 2,667 attempts to chase customers who explicitly revoked their
mandate, 93 chases of invoices the buyer is actively disputing and 84 sent to
buyers who had already committed to a payment date.

**Nobody hears from Backstop more than six times a fortnight.** Not per order,
not per invoice, not per mandate — per person, across all three surfaces at
once. That is the guarantee `contact_fatigue` exists to make, and it is
measured end to end rather than asserted: over six seeds the policed arm's
worst 14-day burst against any one human being is exactly 6, with nobody above
it, where naive reaches 15 to 18. A customer with a failed order, a lapsed
mandate and two overdue invoices is one person, and every per-subject cap can
be scrupulously observed while they are written to a dozen times.

**The leash is close to free, and where it is not, the price is stated.** The
policed and unpoliced arms run identical proposed actions, so their gap is the
cost of compliance alone. On receivables it is positive — the unpoliced arm
takes ₹36,612 it cannot keep and spends more doing it. On recurring it is
positive too, ₹10,792. On payments it is *negative*: policing costs ₹1,795 out
of ₹57.7L, or 0.03%. Against naive rather than against itself, the largest
single cost is the ninety-day bracket Backstop refuses to let automation work
at all, described under Receivables below.

**It holds when the model gets worse.** The table above *is* the weak model.
`openai/gpt-oss-20b` proposes visibly worse actions — 73 re-presentations of
mandates whose diagnosis had already established that re-presenting cannot
work, 12 re-asks of customers who revoked their authorisation, and SMS dunning
where email was the safe channel — and the engine refuses every one of them.
241 rule breaks in the unpoliced arm, 0 in the policed one. That is the whole
argument: safety that does not depend on model quality.

**It holds when the model is gone.** On 2026-08-30 the account's Groq quota
(200,000 tokens/day) was exhausted mid-run and every cluster returned a 429.
Recovery degraded to the deterministic playbook rather than stopping: the
pipeline still proposed 35,158 actions, still recovered ₹58,30,774 on payments,
and still broke zero rules. It is worth saying plainly that the fallback
*out-recovered the weak model* on that batch by about ₹34,000 — the tail
playbook is a real playbook, not a stub, and on failures with no incident to
reason about a lookup table beats a small model. The model earns its place on
diagnosis and cluster-level strategy, and nowhere else.

Rates in `simulate/recoverability.py` are stated configuration, not measured
from Razorpay traffic. Read the tables as a comparison under a declared world
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
cancellation. The planner routes all 381 to a person, and `revoked_mandate`
stands behind it as the guarantee. Naive retry re-asks them 2,667 times.

Three decisions make that recurring table honest, and each one costs the
number something.

**A restored mandate is credited a year, not a charge.** Re-registering an
authorisation does not recover one ₹499 billing period, it restores the
stream. Everything that reasons about a mandate prices it the same way — the
risk scan, the policy engine's value thresholds, and the ledger — so the
engine cannot be deciding about one number while the ledger reports another.

**Recovery is a delta, not a total.** A third of paused mandates resume with
no prompting at all. An agent that mails those customers and books the
resumption has measured its own postage, so the ground truth marks them
non-incremental and they are credited nothing however well the contact is
timed. This is the single largest downward pressure on the recurring figure
and it is deliberate.

**The world does not run on the agent's estimate.** The ~42%/~55%/~8%
re-registration odds above are what Backstop *believes* and reports with.
`simulate/recoverability.py` holds separate, deliberately different rates for
what actually happens. Grading the agent against its own assumption would make
the recurring result a tautology.

What comes out of it is a sharper comparison than a bigger number would have
been. Naive chasing recovers ₹44,05,644, of which ₹1,85,796 was taken by
actions the rules refuse; what survives is ₹42,19,848. Backstop recovers
₹41,81,472 — within 0.9% of it, for 63% of the contacts.

That near-equality is not a coincidence. Re-registration is drawn once per
customer, so asking three times does not buy three chances; both arms reach
essentially every mandate they are allowed to reach, inside the window where
the customer is still listening, and naive's extra 2,520 contacts buy almost
nothing.

The gap that remains is the per-person contact cap, and it is worth naming as
a price rather than rounding away. The cap does not care that a customer's
lapsed mandate and their failed orders are different subjects; past six
messages in a fortnight it stops writing to the person, and occasionally the
message it stops would have landed. Across six seeds that costs this surface
between nothing and 1.1% of what naive keeps. Six messages a fortnight to one
human being, or another 1% of recurring revenue — that is the trade, it is
stated so a reader can disagree with it, and a test fails if the price starts
growing.

### Receivables

Invoices fail a third way. A payment fails as an *event* — one authorisation,
one decline code, one moment a detector can find. A mandate fails into a
*state* and stays there. A receivable fails by **ageing**: nothing breaks,
nothing flips, the invoice simply gets older and every week the money is a
little less likely to arrive. So the output is the aging report a finance team
already reads, and what selects the response is a duration rather than a cause.

```
settled invoices   909
current (not due)  114  ₹1,32,10,138.37
overdue invoices   377
receivables at risk ₹4,01,69,841.00
  of which chaseable ₹3,43,06,256.07  [61 suppressed: ₹27,23,166.89 disputed]
weighted age       44 days overdue, money-weighted

  1-30   135  ₹1,40,49,331  ~88% collectable  recently late; usually an AP cycle
  31-60  126  ₹1,55,93,578  ~71% collectable  late enough to be deliberate or forgotten
  61-90   73    ₹67,27,169  ~52% collectable  the buyer probably cannot clear it in one payment
  90+     43    ₹37,99,761  ~24% collectable  collections or write-off; a person decides
```

Two suppressions are built into the scan rather than left to policy, because
they change what is *chaseable* and not merely what is permitted. A **disputed**
invoice is a disagreement about whether the money is owed, and no amount of
dunning settles one. A **live promise to pay** is a commitment with a date on
it. Both stay in the at-risk total — the money is genuinely at risk — and
neither is counted as collectable. The engine refuses them anyway;
counting them here too is deliberate redundancy, because a number that says
"this is what we could collect" should not include money nobody may chase.

The counterfactual bites harder here than anywhere else in the system.

**Most overdue invoices are paid whether or not anybody chases them.** An
accounts-payable department runs on a cycle; an invoice twelve days late is
usually not a collections problem, it is a Tuesday. Of the ₹4.02 crore
outstanding, **₹1.92 crore arrives on the buyer's own cycle with no action at
all** and is credited to nobody, and only **₹69,06,052** is incremental — the
ceiling on what any collections effort could add. A tool that dunned the whole
aged debtors report and booked everything that subsequently arrived would show
a spectacular recovery rate having caused almost none of it.

That shape has a consequence worth stating: the incremental value of chasing is
thin among recent invoices, which mostly pay themselves, thin again among
ancient ones, which mostly never pay, and thickest in the middle. So the
playbook sends the 1–30 bracket exactly **one** reminder and no more.

**The blocker is sometimes the balance, not the attention.** By sixty days the
buyers who have not paid are increasingly ones who *cannot* pay in one piece,
and dunning them harder is the collections equivalent of retrying a stolen
card: more pressure applied to a blocker pressure cannot move. The middle
brackets pair a reminder with a part-payment offer, and the offer is modelled
with its real cost — it collects the instalment it asked for, so offering a
split to a buyer who could have paid in full loses the remainder. It is not a
free action and the playbook cannot treat it as one.

**The oldest bracket is revenue the policy engine gives up.** Ninety-day paper
is where settlement and write-off decisions live, and those are not
automation's to make, so Backstop hands the bracket to a person and books
nothing from it. Naive dunns it and collects. Across six seeds that bracket is
worth ₹83,267–₹5,16,803 to the naive arm, and it is the single largest thing
policing costs on this surface — enough that on one seed in six naive comes out
ahead overall. That is a real price for a real choice, and it is stated rather
than buried in a net figure.

### Executing for real

Everything above is a claim about what recovery *would* do. `execute/razorpay.py`
is the one component that reaches outside the process and does it, against the
Razorpay test API with the keys in `.env`. It satisfies the same `Executor`
protocol the simulator does, so the ledger and the policy engine did not change
to accommodate it.

Each verb in the action catalog maps to one API call, or to none:

```
retry_payment / switch_route      POST /orders
send_dunning                      POST /payment_links  (+ notify_by)
regenerate_payment_link           POST /payment_links
offer_part_payment                POST /payment_links, accept_partial
request_mandate_reregistration    POST /subscription_registration/auth_links
wait / do_nothing / escalate_*    no call at all
```

The inert verbs make no network call on purpose. An escalation is a handoff
inside the merchant, not a message to a customer, and an adapter that pinged an
API to record one would be inventing traffic.

**A dispatch is not a recovery, and the type system says so.** A payment link is
paid when a human being opens it, hours or days later and out of band. So a live
dispatch reports a fourth outcome — `dispatched` — which is neither recovered nor
no-effect, and `reconcile()` re-reads the entity afterwards to find out which it
became. Booking a dispatch as recovery would be the same error as crediting an
invoice for arriving on the buyer's own accounts-payable cycle.

**Which is exactly why this adapter cannot run the backtest.** The four-arm
comparison needs a counterfactual — what would have happened had nobody acted —
and reality does not offer one. Swapping the real adapter into the measurement
would not make it more real, it would make it unmeasurable. The simulator stays
the measurement backend; this is the execution backend, and the seam between
them is drawn in the code rather than left to good faith.

**A re-presentment is the one action test mode cannot honestly perform.**
Charging a customer again without asking needs a saved token or a live mandate,
and a test account has neither. `POST /orders` records the intent a merchant's
own checkout would then satisfy. It is the honest floor rather than a charge,
and the result string says so instead of implying an attempt was made.

Endpoint behaviour was probed against the live API rather than taken from the
docs, the same way model availability was, because the docs and a given
account's enabled products are not the same thing. `GET /plans` returns 401 —
Subscriptions is not enabled on this account — and it turns out not to matter:
mandate re-registration wants `subscription_registration/auth_links`, which
works without it. E-mandate authorises at zero; UPI and card reject a
zero-amount authorisation and need the ₹1 one.

**A mandate-rail payment fails as an order.** So the playbook's answer — re-
authorise — arrives naming the failed order, while the thing that needs
re-registering is the subscription behind it. The adapter carries the same
order-to-mandate map the backtest does and resolves through it; refusing that
action would have been the adapter rejecting a good plan for naming the
subject the planner naturally names. It surfaced on the first live run against
a seed where that decline code came up, which is roughly the argument for
having built the thing rather than describing it.

**Idempotency was found by re-running, not by reading.** Payment links enforce a
unique `reference_id` and auth links enforce a unique `receipt`, in different
words and with no distinct error code, so a duplicate dispatch is answered
rather than duplicated — the adapter treats both as "already sent" and reports a
no-op. `POST /orders` enforces nothing, and a repeat silently creates a second
order, which for a *charging* action is the worst thing here could do. So
charging actions look themselves up by receipt before creating anything and pay
one extra round trip for idempotency that survives a restart rather than only
surviving a loop. The reference is derived from the action's verb, subject and
scheduled minute, and deliberately not from its `rationale`: a model rewording
its own justification must not be able to buy an extra message to a customer.

Two guardrails, both stated as refusals rather than as options.

**Live keys are refused outright.** This code dispatches payment links and
charges at people, and the difference between `rzp_test_` and `rzp_live_` is one
token in a `.env` file. There is no override flag.

**Notifications are off unless asked for.** Test mode really delivers — a probe
of `notify_by/email` returned `{"success": true}` — so links are created
silently and `--notify` is a decision somebody makes on purpose.

One thing the walkthrough does that the backtest does not: it dispatches only
`ALLOW` and `RESCHEDULE` rulings, and holds `REQUIRE_APPROVAL` back. The
backtest executes those, on its stated assumption that a merchant staffs the
queue. A live adapter may not make that assumption on somebody's behalf — an
approval gate that sends while it waits is not a gate.

### The queue behind the gate

Which raises the question of where a held action actually goes.
`REQUIRE_APPROVAL` was the one disposition with nowhere to be: the engine
produced it, the ledger counted it, the backtest executed it on that stated
assumption, and the live walkthrough correctly refused to send it and then
dropped it. A gate with nothing behind it is a deletion. `approve/queue.py` is
what is behind it, and four of its properties each cost the recovery number
something.

**An approval is permission from a person, not an exemption from the rules.**
A release re-evaluates the action against all sixteen rules in the world that
holds *at release time*, not the one that held when it was queued. Time passes
while a request sits: a customer's fortnight contact budget fills, an hour
becomes a quiet hour, a mandate gets revoked, an invoice goes into dispute. A
queue that dispatched whatever a reviewer approved on Tuesday would be a hole
in the policy engine exactly the width of the queue, and every guarantee above
would carry a silent asterisk. So `DENY` still denies after a yes, and the
reviewer is told which rule overtook their decision. What the approval *does*
discharge is the `require_approval` verdict itself — that is the entire content
of the human decision, and re-raising it would make every approved action
immortal in the queue.

**Silence is not consent.** Requests expire, and expiry is recorded as an
outcome rather than a silent drop. This is what makes the backtest's
assumption falsifiable instead of merely convenient: a merchant who does not
staff the desk does not thereby get the revenue, they get a pile of expired
requests, and the money is reported as given up. On seed 1's live sample, 430
actions are held, one is answered, and 429 expire — revenue foregone by a
*staffing* decision, as visible as revenue foregone by a rule.

**Every decision names a human being.** `decided_by` is not optional and there
is no auto-approve. The backtest's assumption that somebody is at the desk is
expressed as a `StandingApproval` — a named reviewer with a stated latency and
a staffing rate, deterministic from the request id so that every arm and every
re-run sees the same desk. An assumption written as a reviewer is one a reader
can see and argue with.

**Asking twice is not allowed.** A request's identity is `reference_for` — the
same fingerprint that gives the adapter its idempotency, derived from the
action's verb, subject, moment, channel and amount and deliberately not from
its `rationale`. Re-running a plan finds the request a reviewer has already
seen; a model rewording its own justification cannot manufacture a fresh ask.

### Webhooks, and what a receiver must refuse

`reconcile()` polls, and polling is the wrong shape for the job — a dispatch
settles when a human opens a link, so a poll either runs constantly and mostly
learns nothing or runs rarely and reports recovery long after it happened.
`execute/webhook.py` takes the push instead. It is bytes in, verdict out: no
server and no framework, so a deployment can front it with anything and the
test suite can exercise it with no network at all.

A receiver is the one place in this system where an *outsider* proposes a
change to the revenue number, so it is written as a list of refusals.

**It verifies before it parses**, against the raw bytes, always. Parsing first
would hand unauthenticated JSON to the parser; re-serialising a parsed body to
check the signature breaks on key order and quietly tempts somebody to skip
the check. `receive` takes `bytes` and there is no overload that takes a dict.
An absent secret fails closed — a receiver that treats "no secret configured"
as "everything is authentic" is worse than one with no check at all, because
it looks like it has one.

**It credits only what recovery dispatched.** A merchant's own dashboard-created
payment links are paid all day; that is real revenue and recovery caused none
of it. A receiver that credited every `payment_link.paid` on the account would
make exactly the error the receivables surface exists to warn about — measuring
the world's ordinary behaviour and booking it as the agent's work. This is the
counterfactual argument arriving over HTTP, and it is one line of the
walkthrough's output.

**It credits once, in two layers.** Razorpay redelivers on any non-2xx, and
`order.paid` and `payment_link.paid` can both describe one settlement — so
deduplication on the event id is only an optimisation, and the load-bearing
layer is on the *dispatch*, which is the thing that can be recovered once. An
unmatched or duplicate event answers 200, because it was handled correctly and
redelivering it forever would not improve the answer; only a signature failure
is a 400.

**It does not decide what money is worth.** Settlement value goes through
`RazorpayExecutor.credit`, the same call the poll uses, so a re-registered
mandate is a year of billing whichever way the news arrives. Two paths with
their own opinions would make the recurring number a function of network
timing.

### Holding a plan across real time

Three rules exist mainly to move an action's `scheduled_at`: `quiet_hours`
pushes an SMS out of the night, `retry_spacing` refuses to re-present a card
thirty seconds after it declined, and `outage_hold` defers a charge until a
degraded rail recovers. The backtest honours those times — it sorts by
`scheduled_at`, judges each action at its own moment and executes it there.
The live path did not. It dispatched immediately and threw the schedule away.

That made `RESCHEDULE` decorative in the one place it mattered. An engine that
moves a 22:30 SMS to 09:00, followed by an adapter that sends it at 22:30, has
not protected anybody — it has produced a log entry saying it did.
`schedule/scheduler.py` is what makes the disposition mean something outside
the simulator, and it holds four properties.

**Nothing fires before it is due.** That is the guarantee. There is no
`flush()` and no `force` argument, because a method that skipped the wait would
remove the only thing this component promises.

**The rules run again at fire time**, on the same reasoning as the approval
queue: the world an action was judged in is not the world it lands in. A
denial still denies at the later clock. The re-rule is also what makes lateness
*safe* rather than merely tolerated — an action coming due at an awkward hour is
pushed again by `quiet_hours` then, rather than squeaking through on yesterday's
verdict.

**An action can be re-held, but not forever.** If the re-rule moves it again it
goes back into the queue with the new time rather than firing early or firing
anyway. Something rescheduled repeatedly is not being carefully timed, it is
being chased by a blocker that is not clearing — so after three deferrals it is
abandoned and recorded. An unbounded retry loop is the same failure as an
unbounded contact loop.

**A missed window is not a licence to fire late.** A process down for a day
comes back holding actions whose moment has passed, and firing them now is not
the same act: a retry timed for the hour after a failure is a different thing
three days later, and a reminder can arrive after the invoice was paid. Past
twenty-four hours an action is dropped as stale and said to be dropped. That is
a relevance bound; safety is the re-rule's job.

The walkthrough steps a clock through each scheduled moment rather than
sleeping between them, and says so — it is the same loop with the waiting taken
out. It starts from the earliest scheduled moment rather than from `now`
because the batch is historical, so a plan drawn against it is drawn for
moments that have already passed; reading the clock any other way would report
every action as arriving a week late.

### The restart hole, found by the demo

Building the receiver surfaced a bug in the adapter that had been there since
it was written. `dispatched` lives in memory, and a reference Razorpay already
holds was reported as "not sent twice" and otherwise forgotten — so a process
that sent a link and then died had lost its only handle on the thing that went
out. A payment landing on that link could never be credited: not by a poll,
which had nothing to poll, and not by a webhook, which would find no matching
dispatch and correctly refuse to credit a stranger.

So an already-sent action now looks its reference back up and re-registers the
dispatch. Probed rather than assumed, on 2026-08-31: `GET /payment_links` takes
a `reference_id` filter and returns the one link, and `GET /orders` takes
`receipt` — the lookup the charging path already pays for on the way in. Auth
links are the honest gap: they are invoices of type `link`, they do not come
back from `GET /invoices`, and there is no filter for the receipt they were
created with, so a re-registration dispatched by a process that has since died
cannot be recovered. Guessing at one risks reconciling against somebody else's
invoice, so the adapter declines and says so.

## Status

**Built:** all three revenue surfaces, end to end and all three *measured* --
domain model, LLM layer, seeded generator with labelled incidents,
multi-resolution detection with scope correlation, mandate lifecycle scanning,
receivables aging, a recovery playbook for each surface, LLM diagnosis, a
16-rule policy engine, the planner, the ledger, the four-arm backtest reported
per surface, two execution backends behind one protocol -- the simulator that
the measurement runs on, and a Razorpay test-mode adapter that really
dispatches -- an approval queue that holds what the engine will not let
automation decide alone, a scheduler that holds every action until its own
moment and re-rules it there, and a signature-verifying webhook receiver that
closes the loop without polling. 415 tests, and the repo is lint clean.

**Not built:** the webhook receiver is exercised against a locally signed
delivery, because receiving one from Razorpay needs a public endpoint a
walkthrough on a laptop does not have; the verification, matching and crediting
are the real ones, and only the postman is simulated. And the scheduler steps a
clock rather than sleeping -- a deployment would run the same loop on a timer,
with the actions persisted rather than held in memory.

## Seeing it run

```bash
.venv/bin/python -m backstop.demo                    # full walkthrough, live model
.venv/bin/python -m backstop.demo --offline          # no API calls
.venv/bin/python -m backstop.demo --stage policy     # just the safety boundary
.venv/bin/python -m backstop.demo --stage receivables  # just the aged ledger
.venv/bin/python -m backstop.demo --stage razorpay   # live dispatch, test-mode keys
.venv/bin/python -m backstop.demo --stage razorpay --notify   # and actually send
.venv/bin/python -m backstop.execute.razorpay        # what the keys can reach
.venv/bin/python -m backstop.evaluation.backtest     # the four-arm measurement
.venv/bin/python -m backstop.evaluation.backtest --model openai/gpt-oss-20b
.venv/bin/python -m backstop.evaluation.bench --seeds 10   # detection across seeds
```

The walkthrough generates a labelled batch, detects and scores against truth,
diagnoses each cluster and scores that, then puts fourteen probe actions
through the policy engine -- including ones that must be refused. Ground truth
is printed beside every prediction. With test-mode keys configured it finishes
by dispatching three permitted actions, one per surface, onto the real
Razorpay API -- silently, unless `--notify` says otherwise. `--offline` skips
that stage along with the model calls.

## Setup

```bash
uv venv --python 3.13 .venv
uv pip install -e ".[dev]"
cp .env.example .env    # add GROQ_API_KEY; RAZORPAY_* keys drive the live stage
                        # RAZORPAY_WEBHOOK_SECRET verifies pushed deliveries
.venv/bin/python -m pytest -q
```

The test suite runs fully offline against `ScriptedProvider` and
`RecordedTransport` — no API key needed, and nothing in it touches a network.
