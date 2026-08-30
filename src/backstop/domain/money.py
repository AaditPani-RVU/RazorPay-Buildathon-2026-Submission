"""Money is integer paise. Never floats.

Rounding drift in a system that reports "money recovered" would undermine the
one number the whole project is judged on, so amounts are integer minor units
end to end and only become rupees at the display boundary.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Money:
    paise: int

    def __post_init__(self) -> None:
        if not isinstance(self.paise, int):
            raise TypeError(f"Money must be integer paise, got {type(self.paise).__name__}")

    @classmethod
    def rupees(cls, amount: float) -> Money:
        """Build from rupees. Rounds half-up to the nearest paisa at the boundary."""
        return cls(round(amount * 100))

    @classmethod
    def zero(cls) -> Money:
        return cls(0)

    def __add__(self, other: Money) -> Money:
        return Money(self.paise + other.paise)

    def __sub__(self, other: Money) -> Money:
        return Money(self.paise - other.paise)

    def __mul__(self, factor: float) -> Money:
        return Money(round(self.paise * factor))

    def __bool__(self) -> bool:
        return self.paise != 0

    @property
    def as_rupees(self) -> float:
        return self.paise / 100

    def format(self) -> str:
        """Indian digit grouping: 12,34,567.89 rather than 1,234,567.89."""
        neg = self.paise < 0
        whole, frac = divmod(abs(self.paise), 100)
        s = str(whole)
        if len(s) > 3:
            head, tail = s[:-3], s[-3:]
            groups = []
            while len(head) > 2:
                groups.insert(0, head[-2:])
                head = head[:-2]
            if head:
                groups.insert(0, head)
            s = ",".join(groups + [tail])
        return f"{'-' if neg else ''}₹{s}.{frac:02d}"

    def __str__(self) -> str:
        return self.format()


def total(amounts: object) -> Money:
    """Sum an iterable of Money, empty-safe."""
    acc = 0
    for m in amounts:  # type: ignore[attr-defined]
        acc += m.paise
    return Money(acc)
