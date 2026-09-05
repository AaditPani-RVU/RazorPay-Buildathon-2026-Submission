# Backstop — video demo script

**Target: 4:50 against a 5:00 cap.** If you run long, trim in this order: the third
paragraph of the Measurement beat, then Detect, then the Clock beat. Never the Approvals
beat — it is the only fifty seconds nobody else will have.

## Setup before you hit record

The console runs on the Linux box; you record from the Mac. Same wifi, no tunnel, no
deploy:

```bash
# on the Linux box
.venv/bin/python -m backstop.console --host 0.0.0.0   # every interface, not just loopback
ip -4 addr show scope global | grep inet              # confirm the address first
```

Then open **http://192.168.0.3:8000** on the Mac. That address comes from DHCP, so check
it before each session rather than trusting this line. Both machines are on Tailscale, so
if they end up on different networks **http://100.122.110.15:8000** reaches the same
console from anywhere.

- Seed `1`, sample `150`. Every number below depends on both. Reset if you touch either.
- **Press Diagnose before you start recording.** It is six live model calls and takes
  about a hundred seconds — you want that tab already full when you reach it at 1:40.
- **Do not press Measure yet.** You press it on camera at 3:45 and it lands in about
  fifteen seconds, which is exactly one paragraph of talking.
- Browser at 1600×1100, zoom 100%, dark mode. Close the terminal; the console is the
  whole show.

---

## 0:00 — Cold open

> *[Rulings tab already on screen. Don't talk over the first two seconds. Let them read the money.]*

**"Two crore forty-one lakh. One crore seventy a year. Four crore one.**

**Three different ways a merchant's money doesn't arrive. Payments that failed, mandates
that quietly stopped collecting, invoices nobody ever paid.**

**Most recovery demos add those together. I'm not going to, and by the end you'll see why
that would be lying."**

> *[Point at the three cards. Never let them merge into one number — that's the whole thesis.]*

---

## 0:20 — The thesis

> *[Scroll to the two bars. This is the most important image in the video.]*

**"Here's the same sixteen rules ruling two different plans.**

**On the left, Backstop's playbook: five hundred and ten allowed, nine denied. Looks great.
Looks, honestly, like a policy engine that isn't doing anything.**

**On the right — the naive plan. The one that just chases every rupee it can see. Seven
hundred and thirty-three denials."**

> *[Beat. This is the line.]*

**"That's the point. A good planner rarely proposes something the rules have to refuse. So
if I only showed you my own plan, I'd be showing you a safety boundary holding back
nothing. The naive plan lights up eleven of the sixteen rules. That's the proof the fence
is real — I had to build something that runs into it."**

---

## 0:55 — The rules have names

> *[Filter dropdown → pick `quiet_hours`. Then `contact_consent`. Then `promise_to_pay`.]*

**"Every ruling names the rule that fired. Not 'blocked by policy' — `quiet_hours`.
`contact_consent`, that's the DND registry. `promise_to_pay`, that's a customer who already
told you a date.**

**Sixteen rules. This isn't a prompt asking a model to be careful. The model never gets a
vote on any of this."**

---

## 1:15 — Finding it without a model

> *[Detect tab.]*

**"Six of seven incidents found. Eighty-six percent recall, a hundred percent precision,
one minute mean latency, and no LLM anywhere in it.**

**HDFC card issuer down for two hours. A UPI-Kotak insufficient-funds cluster. This runs
over every attempt in the batch and has to answer in seconds — that's not a language
model's job, and the money at risk is a number, not an opinion."**

---

## 1:40 — Where the model actually is

> *[Diagnoses tab. Already populated — you ran it before recording.]*

**"So where is the AI? Here. One job: why is this cluster failing.**

**Six clusters, six root causes, and each one is graded against the incident that actually
produced it. Six for six, confidence around nine-tenths. Issuer outage. Gateway routing
degradation. Authentication drop-off — which is a different problem with a different fix
from the outage two rows up.**

**That's the one place messy multi-signal attribution beats a statistic. And notice it is
graded on screen. A diagnosis nobody checks is a sentence, not a measurement."**

---

## 2:10 — The fifty seconds ⭐

> *[Approvals tab. **This is the beat. Do not rush it.** 161 pending.]*

**"A hundred and sixty-one actions need a human. Big receivables, disputes, things
automation shouldn't decide alone.**

**Watch. I'm the reviewer. I approve all of them."**

> *[Type your name. Approve visible. Advance the clock eighteen hours. Then Release approved.]*

**"Now — an approval is permission from a person. It is *not* an exemption from the rules.**

**Releasing re-runs all sixteen against the world that holds *now*.**

**A hundred and fifty-seven went out. Four were refused."**

> *[Zoom in on a refusal. Read the date off the screen, don't recite one from here.]*

**"'Promise to pay is live until' — that date.**

**While that request sat in the queue waiting for me, the customer committed to a payment
date. So the yes I gave got overruled — by the rules, not by me.**

**Most approval queues in the world would have sent that dunning message. Mine tells the
reviewer which rule overtook them."**

---

## 3:00 — A clock, because some of this only happens in time

> *[Clock control. Advance twelve hours, then forty-eight.]*

**"Nothing here fired when it was decided. Every action was held until the moment the rules
chose for it, and ruled on again when it got there.**

**Twelve hours on — thirty-four fire, and you can watch the recoveries land in the log.**

**Jump two days and forty-three go stale. They don't fire late. A retry timed for Tuesday
morning is not a retry you send on Thursday, and a system that catches up on its backlog
after an outage is a system that texts somebody nine times."**

---

## 3:25 — Sixteen tests it can't quietly fail

> *[Bench tab. 16/16.]*

**"Sixteen probes the engine must rule the same way every single time. Retry a stolen card
— deny. SMS someone at three in the morning IST — reschedule to nine. Chase a customer who
already promised to pay — deny.**

**Half of these are things recovery must never be allowed to do. If a rule quietly stops
denying something, this goes red in CI. Zero policy violations is a claim about a record,
not a sentence in a README."**

---

## 3:45 — Now measure it

> *[Measure tab. Press it. It takes about fifteen seconds — the next paragraph covers it.]*

**"Everything so far is a claim about behaviour. This is the money.**

**One batch, a hundred and forty-eight thousand orders, replayed through four arms — do
nothing, naive retry, the planner with the rules switched off, and Backstop. Same latent
recoverability fixed before any arm runs. The only variable is whether the rulings are
obeyed."**

> *[Table lands. Three tables, not one.]*

**"Naive retry recovers forty-nine lakh on payments and breaks twenty-seven thousand eight
hundred and eighty-nine rules doing it. Six hundred retries of cards reported stolen.
Two thousand six hundred chases of customers who revoked their mandate. Backstop recovers
fifty-eight lakh — nine lakh more — with sixty-eight percent fewer charge attempts, and
breaks zero.**

**And three tables, never one. A recovered payment is money that landed. A re-registered
mandate is a year of billing restored. A collected invoice is a balance you were already
owed. Adding those gives you a headline nobody can reconcile."**

---

## 4:25 — It actually reaches outside

> *[Live tab. Bind the test-mode adapter. Dispatch.]*

**"One thing left. All of that ran on a simulator, because a counterfactual is the only way
to measure recovery — reality doesn't offer you the batch where nobody acted.**

**This one really goes to Razorpay."**

> *[Click. Wait for the id. Open the payment link in a new tab — let them see a real
> Razorpay checkout page.]*

**"Real order. Real payment link. I can open it right now.**

**Same executor protocol as the simulator, so the ledger and the rules didn't change to
accommodate it. Nothing reaches it that the engine didn't permit — and it's ruled *again*
at the moment I press the button. Live keys are refused outright. There's no override."**

---

## 4:50 — Close

> *[Back to the three cards.]*

**"Three surfaces, measured apart, never added. Sixteen rules a person cannot approve their
way around. Four hundred and ninety-nine tests.**

**The model proposes. The policy engine disposes. That's Backstop."**

---

## Numbers, verified 2026-09-05 — do not improvise these

| Where | Number |
|---|---|
| At risk | ₹2,41,51,784.17 payments · ₹1,70,30,028.00/yr recurring · ₹4,01,69,841.00 receivables |
| Backstop plan | 510 allow · 14 reschedule · 161 approval · 9 deny |
| Naive plan | 349 allow · 237 reschedule · 181 approval · **733 deny** |
| Rules | 16, of which the naive plan lights 11 |
| Diagnoses | 6 clusters · **6 of 6 correct** · confidence 0.90–0.93 · `openai/gpt-oss-120b` |
| Console opens holding | 405 scheduled actions · 161 approval requests |
| Approvals beat | 161 approved → **157 released, 4 refused** by `promise_to_pay` |
| Clock | +18h fires 143 · +12h more fires 34 · +48h more drops **43 stale** |
| Bench | 16/16 as expected |
| Detect | 6 of 7 found · 86% recall · 100% precision · 1m latency |
| Measurement | naive **27,889** rule breaks · backstop **0** · payments ₹49,16,423 vs ₹58,30,774 |
| Tests | 499 passing, lint clean |

Seed `1`, sample `150`. Change either and every number above changes — reset before you
record.

**The one number that moves on its own** is the approvals split. Promises to pay are dated
relative to now, so a batch generated next week carries different live ones: expect three,
four or five refusals rather than exactly four. That is why the script has you read the
date off the screen instead of saying it. The count is honest either way.

## If it goes wrong on camera

- **The page won't load from the Mac** — the Linux box's DHCP address moved, or the console
  is bound to loopback. `ip -4 addr show scope global`, and make sure you started it with
  `--host 0.0.0.0`. Tailscale is the fallback that doesn't move.
- **Diagnoses tab is empty or erroring** — the Groq daily quota is 200k tokens and it has
  run out mid-run before. Don't debug on camera: say the model is one component and the
  fallback playbook is measured too — which is true, and the offline arm in the Measure tab
  proves it.
- **Live dispatch fails** — you're offline or the test keys rotated. Say "that's the one
  component that needs the internet" and go to the close.
- **Release refuses a different number** — you advanced a different number of hours, or the
  calendar moved. It's eighteen hours. Read what's on screen.
- **Numbers look wrong everywhere** — you're not on seed 1 / sample 150. Reset, regenerate,
  start again.
