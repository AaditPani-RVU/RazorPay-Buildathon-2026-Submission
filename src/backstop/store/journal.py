"""What has to outlive the process, and why that is a safety property.

Three things in the live path are held in memory, and losing each one costs
something different.

The scheduler holds actions that have not fired yet. Its guarantee is that
nothing fires before its moment -- which a process that forgets everything on
restart satisfies trivially and uselessly, by never firing anything at all.
The other half of holding an action is that its moment eventually arrives, or
that the miss is recorded. Neither can happen if the queue dies with the
process.

The approval queue holds `released`, and the adapter holds `dispatched` and
`reconciled`. These are not work-in-progress, they are the sets that make
"once" mean once: released is what stops a reviewer's single yes from
dispatching twice, dispatched is the only handle on a link that already went
out, and reconciled is what stops a settlement being credited to the ledger
again on the next webhook. A restart that empties them does not merely lose
work -- it removes a guarantee. Somebody gets contacted a second time for one
approval, and one payment is booked as recovery twice. So durability here is
a policy concern rather than an operational nicety, and it belongs in the repo
next to the rules it protects.

This is that store, and it is deliberately the smallest thing that does the
job.

**One file, appended to, never edited in place.** Every state change writes a
complete snapshot of the entity it changed. Replay reads forward and the last
snapshot of an id wins, so restoring is one pass with no merge logic to get
wrong.

**Snapshots rather than deltas, and the reason is failure.** A journal of
changes ("deferred", "approved") is smaller and reconstructs state only if
every record survives. A torn write in a delta log silently produces an
entity in the wrong state -- an approval whose release record was lost is an
approval that can be released again, which is exactly the failure this file
exists to prevent. With whole snapshots a damaged record costs one update and
the entity falls back to its previous known state, which is stale but never
invented.

**A torn tail is dropped and counted, not guessed at.** A process killed
mid-append leaves a partial line. Replay skips anything it cannot parse and
reports how many it skipped, so a caller can say so out loud rather than
quietly restoring less than it thinks.

**Every append is flushed and fsynced.** A durability layer that can lose the
last write in exactly the crash it exists for is decoration. The volume here
is a handful of writes per action, so the cost is not worth trading the
property for.

What this is not: a database, a work queue, or a lock. Two processes writing
one journal would interleave snapshots and the last writer would win, which is
wrong in a way this file does not try to solve -- a deployment that wants two
schedulers wants Postgres, and the components take a journal by interface so
that swap is a constructor argument rather than a rewrite.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: What a component calls itself in the journal. Kinds share one file, and
#: replay filters by kind, so a single path holds the whole live path's state
#: and a restart restores it in one read.
SCHEDULED = "scheduled"
APPROVAL = "approval"
DISPATCH = "dispatch"


@dataclass(frozen=True)
class Record:
    """One snapshot of one entity, as it stood when it was written."""

    kind: str
    id: str
    at: datetime
    """When the snapshot was taken -- wall clock, not the entity's own time."""
    body: dict[str, Any]


@dataclass(frozen=True)
class Replay:
    """Everything a journal held, and what could not be read.

    `damaged` is carried rather than logged and forgotten: a restart that
    silently restored less than the file contains is the failure mode this
    whole module is trying to avoid, so the number travels to the caller.
    """

    records: list[Record]
    damaged: int = 0

    def latest(self, kind: str) -> dict[str, dict[str, Any]]:
        """The last snapshot of each id of one kind, in first-seen order."""
        out: dict[str, dict[str, Any]] = {}
        for record in self.records:
            if record.kind == kind:
                out[record.id] = record.body
        return out

    def history(self, kind: str, entity_id: str) -> list[Record]:
        """Every snapshot of one entity, oldest first. The audit trail."""
        return [r for r in self.records if r.kind == kind and r.id == entity_id]


@dataclass
class Journal:
    """An append-only file of entity snapshots, and the replay that reads it.

    Cheap to construct and safe to point at a path that does not exist yet:
    the file and its parent are created on the first append, and replaying a
    missing file is an empty replay rather than an error, because "nothing has
    happened yet" is the ordinary state of a fresh deployment.
    """

    path: Path
    writes: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    # -- writing -----------------------------------------------------------

    def append(self, kind: str, entity_id: str, body: dict[str, Any]) -> Record:
        """Write one snapshot, durably, before returning.

        Synchronous on purpose. The caller has just changed state that another
        process will act on -- a dispatch that went out, an approval that was
        released -- and returning before that is on disk would leave a window
        in which the action happened and the record of it did not.
        """
        record = Record(kind=kind, id=entity_id, at=datetime.now(UTC), body=body)
        line = json.dumps(
            {
                "kind": record.kind,
                "id": record.id,
                "at": record.at.isoformat(),
                "body": record.body,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.writes += 1
        return record

    # -- reading -----------------------------------------------------------

    def replay(self) -> Replay:
        """Read the whole journal forward, skipping what cannot be read.

        A line is skipped when it is not JSON, when it is JSON that is not an
        object, or when it is missing the fields that make it a record. All
        three mean the same thing in practice -- a write that did not finish --
        and all three are counted rather than raised, because a single bad line
        must not stop a deployment from recovering the other several thousand.
        """
        if not self.path.exists():
            return Replay(records=[], damaged=0)

        records: list[Record] = []
        damaged = 0
        with open(self.path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = _parse(line)
                if record is None:
                    damaged += 1
                    continue
                records.append(record)
        return Replay(records=records, damaged=damaged)

    # -- keeping it a reasonable size --------------------------------------

    def compact(self) -> int:
        """Rewrite the file as one snapshot per entity. Returns records dropped.

        An append-only file of snapshots grows with every state change, and the
        superseded ones are only worth keeping while somebody wants the audit
        trail. Compaction is therefore explicit rather than automatic: it is a
        deliberate trade of history for size, and a component should not make
        that trade on an operator's behalf in the middle of a run.

        The rewrite goes to a temporary file and is renamed over the original,
        so a crash during compaction leaves the old journal intact. There is no
        window in which the file is half a journal.
        """
        replay = self.replay()
        if not replay.records:
            return 0

        latest: dict[tuple[str, str], Record] = {}
        for record in replay.records:
            latest[(record.kind, record.id)] = record

        dropped = len(replay.records) - len(latest)
        tmp = self.path.with_name(self.path.name + ".compacting")
        with open(tmp, "w", encoding="utf-8") as fh:
            for record in latest.values():
                fh.write(
                    json.dumps(
                        {
                            "kind": record.kind,
                            "id": record.id,
                            "at": record.at.isoformat(),
                            "body": record.body,
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)
        _fsync_dir(self.path.parent)
        return dropped

    # -- reading without a component ---------------------------------------

    def describe(self) -> str:
        replay = self.replay()
        kinds = sorted({r.kind for r in replay.records})
        parts = [f"{len(replay.records)} record(s)"]
        for kind in kinds:
            parts.append(f"{len(replay.latest(kind))} {kind}")
        if replay.damaged:
            parts.append(f"{replay.damaged} unreadable")
        return f"{self.path.name}: " + ", ".join(parts)


def _parse(line: str) -> Record | None:
    try:
        raw = json.loads(line)
    except ValueError:
        return None
    if not isinstance(raw, dict):
        return None
    kind, entity_id, at, body = (
        raw.get("kind"), raw.get("id"), raw.get("at"), raw.get("body")
    )
    if not isinstance(kind, str) or not isinstance(entity_id, str):
        return None
    if not isinstance(body, dict) or not isinstance(at, str):
        return None
    try:
        when = datetime.fromisoformat(at)
    except ValueError:
        return None
    return Record(kind=kind, id=entity_id, at=when, body=body)


def _fsync_dir(directory: Path) -> None:
    """Make a rename durable, not merely visible.

    `os.replace` is atomic, which is a statement about what other readers can
    see rather than about what survives power loss. Without this the rename
    can be the thing that is lost.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def latest_bodies(journal: Journal | None, kind: str) -> Iterable[dict[str, Any]]:
    """The restore-time read every component does. None means a fresh start."""
    if journal is None:
        return []
    return journal.replay().latest(kind).values()
