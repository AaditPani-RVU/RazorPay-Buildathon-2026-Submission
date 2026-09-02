# Backstop — video demo script

**Target: 2 minutes 40.** Trim the Bench beat first if you run long, the Live beat last —
never the Approvals beat, which is the only fifteen seconds nobody else will have.

**Setup before you hit record**

```bash
.venv/bin/python -m backstop.console      # http://127.0.0.1:8000
```

Fresh session, seed `1`, sample `150`. Browser at 1600×1100, zoom 100%, dark mode looks
better on camera than light. Close the terminal — the console is the whole show. Have
`https://rzp.io/rzp/…` ready to paste nowhere; you're going to make a fresh one on air.

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

## 0:50 — The rules have names

> *[Filter dropdown → pick `quiet_hours`. Then `contact_consent`. Then `promise_to_pay`.]*

**"Every ruling names the rule that fired. Not 'blocked by policy' — `quiet_hours`.
`contact_consent`, that's the DND registry. `promise_to_pay`, that's a customer who already
told you a date.**

**Sixteen rules. This isn't a prompt asking a model to be careful. The model never gets a
vote on any of this."**

---

## 1:10 — The fifteen seconds ⭐

> *[Approvals tab. **This is the beat. Do not rush it.** 161 pending.]*

**"A hundred and sixty-one actions need a human. Big receivables, disputes, things
automation shouldn't decide alone.**

**Watch. I'm the reviewer. I approve all of them."**

> *[Type your name. Click Approve visible. Then Release approved.]*

**"Now — an approval is permission from a person. It is *not* an exemption from the rules.**

**Releasing re-runs all sixteen against the world that holds *now*.**

**A hundred and fifty-seven went out. Four were refused."**

> *[Zoom in on a refusal. Read it out loud, slowly.]*

**"'Promise to pay is live until September first.'**

**While that request sat in the queue waiting for me, the customer committed to a payment
date. So the yes I gave on Tuesday got overruled on Wednesday — by the rules, not by me.**

**Most approval queues in the world would have sent that dunning message. Mine tells the
reviewer which rule overtook them."**

---

## 1:45 — Sixteen tests it can't quietly fail

> *[Bench tab. 16/16.]*

**"Sixteen probes the engine must rule the same way every single time. Retry a stolen card
— deny. SMS someone at three in the morning IST — reschedule to nine. Chase a customer who
already promised to pay — deny.**

**Half of these are things recovery must never be allowed to do. If a rule quietly stops
denying something, this goes red in CI. Zero policy violations is a claim about a record,
not a sentence in a README."**

---

## 2:05 — Finding it without a model

> *[Detect tab.]*

**"Six of seven incidents found. Eighty-six percent recall, a hundred percent precision,
one minute mean latency, and no LLM anywhere in it.**

**HDFC card issuer down for two hours. UPI-Kotak insufficient funds cluster. This runs over
every attempt in the batch and has to answer in seconds — that's not a language model's
job, and the money at risk is a number, not an opinion."**

---

## 2:20 — It actually reaches outside

> *[Live tab. Bind the test-mode adapter. Dispatch.]*

**"Everything you've seen is a claim about what recovery *would* do.**

**This one really goes to Razorpay."**

> *[Click. Wait for the id. Open the payment link in a new tab — let them see a real
> Razorpay checkout page.]*

**"Real order. Real payment link. I can open it right now.**

**Same executor protocol as the simulator, so the ledger and the rules didn't change to
accommodate it. Nothing reaches it that the engine didn't permit — and it's ruled *again*
at the moment I press the button. Live keys are refused outright. There's no override."**

---

## 2:35 — Close

> *[Back to the three cards.]*

**"Three surfaces, measured apart, never added.**

**Sixteen rules that a person cannot approve their way around.**

**Four hundred and ninety-nine tests.**

**The model proposes. The policy engine disposes. That's Backstop."**

---

## Numbers, verified 2026-09-02 — do not improvise these

| Where | Number |
|---|---|
| At risk | ₹2,41,51,784.17 payments · ₹1,70,30,028.00/yr recurring · ₹4,01,69,841.00 receivables |
| Backstop plan | 510 allow · 14 reschedule · 161 approval · 9 deny |
| Naive plan | 349 allow · 237 reschedule · 181 approval · **733 deny** |
| Rules | 16, of which the naive plan lights 11 |
| Approvals beat | 161 approved → **157 released, 4 refused** by `promise_to_pay` |
| Bench | 16/16 as expected |
| Detect | 6 of 7 found · 86% recall · 100% precision · 1m latency |
| Tests | 499 passing, lint clean |

Seed `1`, sample `150`. Change either and every number above changes — reset before you record.

## If it goes wrong on camera

- **Live dispatch fails** — you're offline or the test keys rotated. Say "that's the one
  component that needs the internet" and move to the close. Don't debug on camera.
- **Release refuses more or fewer than 4** — you advanced a different number of hours. It's
  18. The count is honest either way; just read what's on screen instead of the script.
- **Numbers look wrong** — you're not on seed 1 / sample 150. Reset and re-generate.
