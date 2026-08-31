"""Configuration and .env loading.

Deliberately dependency-free: a fifteen-line parser beats another package for
reading four keys, and it keeps `.env` handling explicit and auditable.
Secrets are never logged -- `describe()` reports presence and length only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from backstop.llm.groq_provider import FAST_MODEL, REASONING_MODEL

#: Where the live path's journal lives unless `BACKSTOP_JOURNAL` says
#: otherwise. `runs/` is untracked, which is the point: the file names real
#: payment links and the customers they went to.
DEFAULT_JOURNAL = "runs/live.jsonl"


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE pairs into os.environ. Shell values win unless override."""
    p = Path(path)
    loaded: dict[str, str] = {}
    if not p.exists():
        return loaded
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if not value:
            continue
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    razorpay_key_id: str | None
    razorpay_key_secret: str | None
    razorpay_webhook_secret: str | None
    reasoning_model: str
    fast_model: str
    journal_path: Path
    """Where the live path writes what has to outlive the process.

    Under `runs/` by default, which is already untracked: a journal holds real
    dispatch references and the ids of people who were contacted, and that is
    not something to commit by accident.
    """

    @classmethod
    def load(cls, path: str | Path = ".env") -> Settings:
        load_dotenv(path)
        return cls(
            groq_api_key=os.environ.get("GROQ_API_KEY") or None,
            razorpay_key_id=os.environ.get("RAZORPAY_KEY_ID") or None,
            razorpay_key_secret=os.environ.get("RAZORPAY_KEY_SECRET") or None,
            razorpay_webhook_secret=os.environ.get("RAZORPAY_WEBHOOK_SECRET") or None,
            reasoning_model=os.environ.get("BACKSTOP_LLM_MODEL") or REASONING_MODEL,
            fast_model=os.environ.get("BACKSTOP_FAST_MODEL") or FAST_MODEL,
            journal_path=Path(os.environ.get("BACKSTOP_JOURNAL") or DEFAULT_JOURNAL),
        )

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_razorpay(self) -> bool:
        return bool(self.razorpay_key_id and self.razorpay_key_secret)

    @property
    def has_webhook_secret(self) -> bool:
        """Without it the receiver refuses every delivery, which is correct:
        an unverifiable webhook is not evidence that money arrived."""
        return bool(self.razorpay_webhook_secret)

    @property
    def razorpay_is_test_mode(self) -> bool:
        """Live keys in a recovery agent that executes actions would move real money."""
        return bool(self.razorpay_key_id and self.razorpay_key_id.startswith("rzp_test_"))

    def describe(self) -> str:
        def mark(name: str, value: str | None) -> str:
            return f"  {name}: {'set (' + str(len(value)) + ' chars)' if value else 'MISSING'}"

        return "\n".join(
            [
                mark("GROQ_API_KEY", self.groq_api_key),
                mark("RAZORPAY_KEY_ID", self.razorpay_key_id),
                mark("RAZORPAY_KEY_SECRET", self.razorpay_key_secret),
                mark("RAZORPAY_WEBHOOK_SECRET", self.razorpay_webhook_secret),
                f"  reasoning model: {self.reasoning_model}",
                f"  fast model:      {self.fast_model}",
                f"  razorpay mode:   {'TEST' if self.razorpay_is_test_mode else 'NOT TEST MODE'}",
                f"  journal:         {self.journal_path}",
            ]
        )
