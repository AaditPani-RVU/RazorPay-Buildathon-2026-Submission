"""The operator console: the live path with a face on it.

Everything this serves already existed. The pipeline, the rules, the queue,
the scheduler and the adapter are the same objects the walkthrough and the
backtest use, and nothing here re-implements a decision. What was missing was
a way to *watch* it: a terminal transcript can show that an action was denied
and name the rule, but it cannot let somebody filter thirty thousand rulings
by the rule that produced them, approve a held action and see the engine
overrule the approval, or push a clock forward and watch a deferral land.

The console is therefore a viewer and a set of controls, not a second
implementation. Every number on the screen is read off the same ledger the
measurement reports, and every button calls the method the walkthrough calls.
"""

from backstop.console.session import ConsoleSession

__all__ = ["ConsoleSession"]
