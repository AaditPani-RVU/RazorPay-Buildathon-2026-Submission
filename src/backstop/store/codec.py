"""Turning the types the live path holds into JSON, and back without loss.

Small and boring on purpose. Everything a component needs to persist is
already a value type -- an `Action` is a pydantic model, `Money` is an integer,
a `Verdict` is four fields -- so the codecs here are direct rather than
generic. There is no reflection, no registry and no schema version, because a
clever encoder is a place for a restart to restore something subtly different
from what was saved, and the entire point of the store is that it does not.

Two rules the encodings follow.

**Times go out as ISO strings and come back through `utc`.** A naive datetime
read back from disk would compare against an aware one and raise, or worse,
compare wrongly -- and the scheduler decides whether to fire on exactly that
comparison.

**Money goes out as integer paise.** The same reason it is integer paise
everywhere else: a float in the file would round-trip a recovered amount
through binary fraction and report a number no merchant could reconcile.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from backstop.domain.actions import Action
from backstop.domain.entities import utc
from backstop.domain.money import Money
from backstop.execute.executor import ExternalRef
from backstop.policy.engine import Disposition, Ruling, Verdict


def dump_time(value: datetime | None) -> str | None:
    return utc(value).isoformat() if value is not None else None


def load_time(value: Any) -> datetime | None:
    if not value:
        return None
    return utc(datetime.fromisoformat(str(value)))


def dump_action(action: Action) -> dict[str, Any]:
    return action.model_dump(mode="json")


def load_action(body: dict[str, Any]) -> Action:
    return Action.model_validate(body)


def dump_money(amount: Money | None) -> int | None:
    return amount.paise if amount is not None else None


def load_money(value: Any) -> Money | None:
    return Money(int(value)) if value is not None else None


def dump_ref(ref: ExternalRef | None) -> dict[str, Any] | None:
    if ref is None:
        return None
    return {"entity": ref.entity, "id": ref.id, "url": ref.url}


def load_ref(body: Any) -> ExternalRef | None:
    if not isinstance(body, dict):
        return None
    return ExternalRef(
        entity=str(body.get("entity") or ""),
        id=str(body.get("id") or ""),
        url=str(body.get("url") or ""),
    )


def dump_verdict(verdict: Verdict) -> dict[str, Any]:
    return {
        "rule_id": verdict.rule_id,
        "disposition": verdict.disposition.value,
        "reason": verdict.reason,
        "reschedule_to": dump_time(verdict.reschedule_to),
    }


def load_verdict(body: dict[str, Any]) -> Verdict:
    return Verdict(
        rule_id=str(body.get("rule_id") or ""),
        disposition=Disposition(body["disposition"]),
        reason=str(body.get("reason") or ""),
        reschedule_to=load_time(body.get("reschedule_to")),
    )


def dump_ruling(ruling: Ruling) -> dict[str, Any]:
    """The ruling that sent an action somewhere, kept whole.

    Persisted rather than recomputed because it is the *record of what was
    decided*, and a rule set that has been edited since would produce a
    different answer on restore. The re-evaluation that actually governs
    dispatch happens at release or fire time against the live engine -- this
    is the trail, not the authority.
    """
    return {
        "proposed": dump_action(ruling.proposed),
        "disposition": ruling.disposition.value,
        "verdicts": [dump_verdict(v) for v in ruling.verdicts],
        "final": dump_action(ruling.final) if ruling.final is not None else None,
    }


def load_ruling(body: dict[str, Any]) -> Ruling:
    final = body.get("final")
    return Ruling(
        proposed=load_action(body["proposed"]),
        disposition=Disposition(body["disposition"]),
        verdicts=[load_verdict(v) for v in body.get("verdicts", [])],
        final=load_action(final) if final else None,
    )
