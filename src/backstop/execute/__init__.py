"""Execution backends.

`razorpay` is deliberately not re-exported here. It carries a CLI, and a
package `__init__` that imports its own submodule makes `python -m
backstop.execute.razorpay` import the module twice. Import it by path, the way
`evaluation.backtest` is imported.
"""

from backstop.execute.executor import (
    ExecutionCosts,
    ExecutionResult,
    Executor,
    ExternalRef,
    Outcome,
    SimulatedExecutor,
)

__all__ = [
    "ExecutionCosts",
    "ExecutionResult",
    "Executor",
    "ExternalRef",
    "Outcome",
    "SimulatedExecutor",
]
