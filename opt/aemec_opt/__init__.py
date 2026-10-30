"""AEMEC membrane-thickness optimization package."""

__all__ = ("OptimizationError",)


def __getattr__(name):
    # CLI protocol flags are also used by ELA's sample/analyze commands, which
    # do not need Optuna. Load the engine only when its public error is needed.
    if name == "OptimizationError":
        from .engine import OptimizationError
        return OptimizationError
    raise AttributeError(name)
