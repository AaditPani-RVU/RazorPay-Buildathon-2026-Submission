"""The console's HTTP surface. A translation layer and nothing more.

Every route here calls a method on `ConsoleSession` and serialises the result.
No route decides anything: there is no rule, no threshold and no verdict in
this file, on purpose. A safety boundary that could be reached around by
adding an HTTP handler would not be one, and the easiest way to guarantee that
is for the transport to have no opinions to reach around with.

Two things it does own, because they belong to the transport rather than to
the domain.

**Long work runs off the request.** Generating a batch takes seconds and
diagnosing one takes a model round trip per cluster. Both are started as jobs
and polled, so a page never sits on an open socket waiting for a language
model, and a browser that gives up does not cancel the work.

**One session, one lock.** The clock, the queue and the scheduler are mutable
and shared, so two overlapping requests advancing time would interleave
firings. The lock is held for the whole of a mutating call. That is the same
single-writer assumption the journal makes, and it is stated rather than
hidden for the same reason: it wants a database before it wants a second user.

    .venv/bin/python -m backstop.console
"""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backstop.config import Settings
from backstop.console.session import DEFAULT_REVIEWER, ConsoleSession
from backstop.domain.entities import utc
from backstop.execute.webhook import SIGNATURE_HEADER
from backstop.ledger.ledger import Surface
from backstop.policy.engine import Disposition

STATIC = Path(__file__).parent / "static"


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------


@dataclass
class Job:
    """Work too slow to hold a request open for."""

    id: str
    label: str
    state: str = "running"
    detail: str = ""
    error: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    def as_dict(self) -> dict:
        seconds = ((self.finished_at or datetime.now(UTC)) - self.started_at).total_seconds()
        return {
            "id": self.id, "label": self.label, "state": self.state,
            "detail": self.detail, "error": self.error, "seconds": round(seconds, 1),
        }


JOBS: dict[str, Job] = {}
SESSION = ConsoleSession()


def start(label: str, work) -> Job:
    job = Job(id=uuid.uuid4().hex[:8], label=label)
    JOBS[job.id] = job

    def run() -> None:
        try:
            job.detail = work() or ""
            job.state = "done"
        except Exception as exc:  # noqa: BLE001 -- a job runner is the top of
            # its own stack. Anything it does not catch dies in a thread nobody
            # is watching and the page waits forever on a job that will never
            # finish, which is a worse failure than a wide except with the
            # error put on the screen.
            job.state = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            job.finished_at = datetime.now(UTC)

    threading.Thread(target=run, daemon=True).start()
    return job


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def dump_verdicts(ruling) -> list[dict]:
    return [
        {"rule": v.rule_id, "disposition": v.disposition.value, "reason": v.reason}
        for v in ruling.verdicts
    ]


def dump_row(row) -> dict:
    action = row.action
    final = row.ruling.final or action
    return {
        "id": row.id,
        "source": row.source,
        "surface": row.surface.value,
        "type": action.type.value,
        "subject": action.subject_id,
        "channel": action.channel.value if action.channel else "",
        "amount": str(action.amount_paise) if action.amount_paise else "",
        "scheduled_at": utc(action.scheduled_at).isoformat(),
        "final_at": utc(final.scheduled_at).isoformat(),
        "moved": utc(final.scheduled_at) != utc(action.scheduled_at),
        "disposition": row.ruling.disposition.value,
        "fate": row.fate,
        "blocking_rule": row.ruling.blocking_rule or "",
        "moving_rule": row.ruling.moving_rule or "",
        "rationale": action.rationale,
        "verdicts": dump_verdicts(row.ruling),
    }


def dump_request(request, *, now: datetime) -> dict:
    return {
        "id": request.id,
        "surface": request.surface.value,
        "state": request.state.value,
        "action": request.action.describe(),
        "type": request.action.type.value,
        "subject": request.action.subject_id,
        "asking_rule": request.asking_rule,
        "reason": request.reason,
        "submitted_at": utc(request.submitted_at).isoformat(),
        "expires_at": utc(request.expires_at).isoformat(),
        "hours_left": round(
            (utc(request.expires_at) - utc(now)).total_seconds() / 3600, 1
        ),
        "decided_by": request.decided_by,
        "note": request.note,
    }


def dump_entry(entry) -> dict:
    return {
        "id": entry.id,
        "surface": entry.surface.value,
        "state": entry.state.value,
        "action": entry.action.describe(),
        "type": entry.action.type.value,
        "subject": entry.action.subject_id,
        "due_at": utc(entry.due_at).isoformat(),
        "deferrals": entry.deferrals,
        "history": [utc(h).isoformat() for h in entry.history],
        "note": entry.note,
    }


def dump_ledger(ledger) -> dict:
    return {
        "proposed": ledger.proposed,
        "executed": len(ledger.executed),
        "recovered": str(ledger.recovered),
        "cost": str(ledger.cost),
        "net": str(ledger.net),
        "illegal": str(ledger.recovered_in_violation),
        "keepable_net": str(ledger.compliant_net),
        "subjects_recovered": ledger.orders_recovered,
        "charges": ledger.charges_attempted,
        "contacts": ledger.contacts_sent,
        "burst": ledger.worst_contact_burst,
        "vetoed": len(ledger.vetoed),
        "vetoes_by_rule": ledger.vetoes_by_rule(),
    }


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

app = FastAPI(title="Backstop operator console", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
def index() -> HTMLResponse:
    """The page shell, with its assets stamped by their modification time.

    A browser holding yesterday's stylesheet is a nuisance in development and a
    disaster on camera, and neither `ETag` nor a reload the presenter forgets to
    make hard is a guarantee. Stamping the query string means the URL itself
    changes whenever the file does, so there is nothing to invalidate.
    """
    html = (STATIC / "index.html").read_text()
    for asset in ("console.css", "console.js"):
        stamp = int((STATIC / asset).stat().st_mtime)
        html = html.replace(f"/static/{asset}", f"/static/{asset}?v={stamp}")
    return HTMLResponse(html, headers={"cache-control": "no-store"})


@app.get("/api/bootstrap")
def bootstrap() -> dict:
    settings = Settings.load()
    return {
        "rules": [
            {
                "id": rule.id,
                "why": (type(rule).__doc__ or "").strip().split("\n\n")[0].replace("\n", " "),
            }
            for rule in SESSION.engine.rules
        ],
        "dispositions": [d.value for d in Disposition],
        "surfaces": [s.value for s in Surface],
        "settings": {
            "groq": settings.has_groq,
            "model": settings.reasoning_model,
            "razorpay": settings.has_razorpay,
            "razorpay_test_mode": settings.razorpay_is_test_mode,
            "webhook_secret": settings.has_webhook_secret,
        },
        "defaults": {
            "seed": SESSION.seed, "days": SESSION.days,
            "orders_per_day": SESSION.orders_per_day, "sample": SESSION.sample,
            "reviewer": DEFAULT_REVIEWER,
        },
        "ready": SESSION.ready,
    }


class BuildRequest(BaseModel):
    seed: int = 1
    days: int = 7
    orders_per_day: int = 20000
    sample: int = 150


@app.post("/api/build")
def build(body: BuildRequest) -> dict:
    def work() -> str:
        with SESSION.lock:
            SESSION.build(
                seed=body.seed, days=body.days,
                orders_per_day=body.orders_per_day, sample=body.sample,
            )
            return f"{len(SESSION.rows)} actions ruled"

    return start("generate and rule", work).as_dict()


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict:
    found = JOBS.get(job_id)
    if found is None:
        raise HTTPException(404, "no such job")
    return found.as_dict()


def _require_ready() -> ConsoleSession:
    if not SESSION.ready:
        raise HTTPException(409, "no batch has been generated yet")
    return SESSION


@app.get("/api/state")
def state() -> dict:
    s = _require_ready()
    with s.lock:
        counts: dict[str, int] = {}
        for row in s.rows:
            key = f"{row.source}:{row.ruling.disposition.value}"
            counts[key] = counts.get(key, 0) + 1
        return {
            "seed": s.seed,
            "sample": s.sample,
            "clock": utc(s.clock).isoformat() if s.clock else "",
            "started_at": utc(s.started_at).isoformat() if s.started_at else "",
            "batch_ends_at": utc(s.require().ends_at).isoformat(),
            "surfaces": s.surfaces(),
            "detection": s.detection(),
            "counts": counts,
            "queue": {k.value: v for k, v in s.queue.counts().items()},
            "pending": len(s.queue.pending(s.clock)),
            "scheduler": {k.value: v for k, v in s.scheduler.counts().items()},
            "next_due": (
                utc(s.scheduler.next_due).isoformat() if s.scheduler.next_due else ""
            ),
            "ledger": dump_ledger(s.ledger),
            "ledger_by_surface": {
                surface.value: dump_ledger(s.ledger.on(surface)) for surface in Surface
            },
            "diagnosis_backend": s.diagnosis_backend,
            "events": len(s.events),
        }


@app.get("/api/rows")
def rows(
    source: str = "", surface: str = "", disposition: str = "",
    rule: str = "", q: str = "", limit: int = 200, offset: int = 0,
) -> dict:
    s = _require_ready()
    with s.lock:
        selected = []
        for row in s.rows:
            if source and row.source != source:
                continue
            if surface and row.surface.value != surface:
                continue
            if disposition and row.ruling.disposition.value != disposition:
                continue
            if rule and not any(v.rule_id == rule for v in row.ruling.verdicts):
                continue
            if q and q.lower() not in (
                row.action.subject_id + row.action.type.value + row.action.rationale
            ).lower():
                continue
            selected.append(row)
        return {
            "total": len(selected),
            "rows": [dump_row(r) for r in selected[offset: offset + limit]],
        }


@app.get("/api/rules")
def rules() -> dict:
    s = _require_ready()
    with s.lock:
        return {"rules": s.rule_pressure()}


@app.get("/api/probes")
def probes() -> dict:
    s = _require_ready()
    with s.lock:
        found = s.probes()
        return {
            "probes": found,
            "matched": sum(1 for p in found if p["matches"]),
            "total": len(found),
        }


@app.get("/api/approvals")
def approvals(state: str = "", limit: int = 200) -> dict:
    s = _require_ready()
    with s.lock:
        now = s.clock
        requests = list(s.queue.requests.values())
        if state == "pending":
            requests = s.queue.pending(now)
        elif state:
            requests = [r for r in requests if r.state.value == state]
        requests.sort(key=lambda r: utc(r.submitted_at))
        return {
            "total": len(requests),
            "counts": {k.value: v for k, v in s.queue.counts().items()},
            "released": len(s.queue.released),
            "requests": [dump_request(r, now=now) for r in requests[:limit]],
        }


class DecideRequest(BaseModel):
    ids: list[str]
    approve: bool = True
    by: str = DEFAULT_REVIEWER
    note: str = ""


@app.post("/api/approvals/decide")
def decide(body: DecideRequest) -> dict:
    s = _require_ready()
    if not body.by.strip():
        raise HTTPException(400, "an approval must name the person who gave it")
    decided, refused = 0, []
    for request_id in body.ids:
        try:
            s.decide(request_id, approve=body.approve, by=body.by, note=body.note)
            decided += 1
        except (KeyError, ValueError) as exc:
            refused.append(f"{request_id}: {exc}")
    return {"decided": decided, "refused": refused}


@app.post("/api/approvals/release")
def release() -> dict:
    s = _require_ready()
    releases = s.release()
    return {
        "released": releases,
        "counts": {
            outcome: sum(1 for r in releases if r["outcome"] == outcome)
            for outcome in ("released", "rescheduled", "refused")
        },
    }


@app.get("/api/schedule")
def schedule(state: str = "", limit: int = 200) -> dict:
    s = _require_ready()
    with s.lock:
        entries = list(s.scheduler.entries.values())
        if state:
            entries = [e for e in entries if e.state.value == state]
        entries.sort(key=lambda e: utc(e.due_at))
        return {
            "total": len(entries),
            "counts": {k.value: v for k, v in s.scheduler.counts().items()},
            "next_due": (
                utc(s.scheduler.next_due).isoformat() if s.scheduler.next_due else ""
            ),
            "entries": [dump_entry(e) for e in entries[:limit]],
        }


class ClockRequest(BaseModel):
    hours: float | None = None
    to_next: bool = False


@app.post("/api/clock")
def clock(body: ClockRequest) -> dict:
    s = _require_ready()
    moved = s.advance(hours=body.hours, to_next=body.to_next)
    moved["clock"] = utc(s.clock).isoformat()
    return moved


@app.get("/api/events")
def events(limit: int = 200, kind: str = "") -> dict:
    s = _require_ready()
    with s.lock:
        selected = [e for e in s.events if not kind or e.kind == kind]
        return {
            "total": len(selected),
            "events": [
                {"at": utc(e.at).isoformat(), "kind": e.kind,
                 "text": e.text, "rule": e.rule}
                for e in selected[-limit:][::-1]
            ],
        }


class DiagnoseRequest(BaseModel):
    model: str | None = None


@app.post("/api/diagnose")
def diagnose(body: DiagnoseRequest) -> dict:
    _require_ready()

    def work() -> str:
        SESSION.diagnose(model=body.model)
        return f"{len(SESSION.diagnoses)} clusters diagnosed"

    return start("diagnose clusters", work).as_dict()


@app.get("/api/diagnoses")
def diagnoses() -> dict:
    s = _require_ready()
    from backstop.evaluation import detection_score

    scenario = s.require()
    out: list[dict] = []
    for cluster, result in s.diagnoses:
        truth = detection_score.match(cluster.primary, scenario.incidents)
        row: dict[str, Any] = {
            "cluster": cluster.id,
            "segment": cluster.segment.describe(),
            "at_risk": str(cluster.money_at_risk),
            "truth": truth.root_cause.value if truth else "",
            "ok": result.ok,
            "error": result.error,
            "repaired": result.repaired,
        }
        if result.ok:
            d = result.diagnosis
            row |= {
                "root_cause": d.root_cause.value,
                "confidence": d.confidence,
                "locus": d.locus,
                "evidence": list(d.key_evidence),
                "ruled_out": [c.value for c in d.ruled_out],
                "correct": bool(truth and d.root_cause is truth.root_cause),
            }
        out.append(row)
    return {"backend": s.diagnosis_backend, "diagnoses": out}


class MeasureRequest(BaseModel):
    offline: bool = True
    model: str | None = None


@app.post("/api/measure")
def measure(body: MeasureRequest) -> dict:
    _require_ready()

    def work() -> str:
        SESSION.measure(offline=body.offline, model=body.model)
        return f"{len(SESSION.arms)} arms replayed"

    return start("replay four arms", work).as_dict()


@app.get("/api/measurement")
def measurement() -> dict:
    return _require_ready().measurement()


# --------------------------------------------------------------------------
# Reaching outside
# --------------------------------------------------------------------------


@app.post("/api/live")
def go_live() -> dict:
    s = _require_ready()
    try:
        return s.go_live()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


class DispatchRequest(BaseModel):
    id: str
    notify: bool = False
    """Test mode really delivers, so sending is a decision somebody makes."""


@app.post("/api/dispatch")
def dispatch(body: DispatchRequest) -> dict:
    s = _require_ready()
    try:
        return s.dispatch(body.id, notify=body.notify)
    except (RuntimeError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/dispatches")
def dispatches() -> dict:
    s = _require_ready()
    return {
        "live": s.live is not None,
        "results": [
            {
                "action": r.action.describe(),
                "outcome": r.outcome.value,
                "detail": r.detail,
                "external": (
                    {"entity": r.external.entity, "id": r.external.id, "url": r.external.url}
                    if r.external else None
                ),
            }
            for r in s.dispatches
        ],
    }


@app.post("/webhooks/razorpay")
async def webhook(request: Request) -> Response:
    """The receiver, with a server in front of it at last.

    Everything that decides anything is in `execute/webhook.py`; this reads the
    raw body, passes the signature header through untouched, and returns the
    status the receipt asks for. Razorpay redelivers anything that is not 2xx,
    so an unmatched or duplicate event answers 200 -- it was handled correctly
    and redelivering it forever would not improve the answer. Only a signature
    failure is a 400.
    """
    s = _require_ready()
    raw = await request.body()
    signature = request.headers.get(SIGNATURE_HEADER, "")
    try:
        receipt = s.webhook(raw, signature)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return JSONResponse(
        status_code=receipt.status,
        content={
            "verdict": receipt.verdict.value,
            "detail": receipt.detail,
            "recovered": str(receipt.result.recovered) if receipt.result else "",
        },
    )


def main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="Serve the Backstop operator console.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument(
        "--cold", action="store_true",
        help="do not generate a batch at startup; the page asks for one",
    )
    args = ap.parse_args()

    if not args.cold:
        print(f"generating seed {args.seed}...", flush=True)
        SESSION.build(seed=args.seed, sample=args.sample)
        print(f"  {len(SESSION.rows)} actions ruled", flush=True)
    print(f"console on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
