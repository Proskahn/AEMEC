"""Allow ``python -m crossover_experiments`` to run the workflow."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())

