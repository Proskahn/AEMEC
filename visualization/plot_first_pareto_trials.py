#!/usr/bin/env python3
"""Plot only the first N Optuna trials from an AEMEC Pareto CSV."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Trial:
    number: int
    membrane_thickness_um: float
    cell_voltage_v: float
    crossover_rate_mol_s: float


def clean_csv_value(value: str | None) -> str:
    """Accept both an ordinary CSV and the Markdown-decorated pasted table."""
    return "" if value is None else value.strip().strip("*").replace("\\", "")


def load_trials(path: Path) -> list[Trial]:
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        records: list[Trial] = []
        for raw_row in reader:
            row = {
                clean_csv_value(name): clean_csv_value(value)
                for name, value in raw_row.items()
                if name is not None
            }
            records.append(
                Trial(
                    number=int(row["trial"]),
                    membrane_thickness_um=float(row["membrane_thickness_um"]),
                    cell_voltage_v=float(row["cell_voltage_v"]),
                    crossover_rate_mol_s=float(row["crossover_rate_mol_s"]),
                )
            )
    return records


def select_first_trials(trials: list[Trial], count: int = 20) -> list[Trial]:
    """Select Optuna trial numbers 0 through count - 1, not the first CSV rows."""
    if count <= 0:
        raise ValueError("trial count must be positive")
    return sorted(
        (trial for trial in trials if 0 <= trial.number < count),
        key=lambda trial: trial.cell_voltage_v,
    )


def plot_trials(trials: list[Trial], output: Path, count: int) -> None:
    if not trials:
        raise ValueError(f"no trials with trial number below {count} were found")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    scatter = ax.scatter(
        [trial.cell_voltage_v for trial in trials],
        [trial.crossover_rate_mol_s for trial in trials],
        c=[trial.membrane_thickness_um for trial in trials],
        cmap="viridis",
        s=58,
        edgecolors="black",
        linewidths=0.35,
        zorder=2,
    )
    ax.plot(
        [trial.cell_voltage_v for trial in trials],
        [trial.crossover_rate_mol_s for trial in trials],
        color="tab:red",
        linewidth=1.4,
        alpha=0.8,
        zorder=1,
    )
    for trial in trials:
        ax.annotate(
            str(trial.number),
            (trial.cell_voltage_v, trial.crossover_rate_mol_s),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Membrane thickness (um)")
    ax.set_xlabel("Cell voltage (V)")
    ax.set_ylabel("Crossover rate (mol/s)")
    ax.set_title(f"Pareto trials 0-{count - 1}")
    ax.grid(True, alpha=0.25)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.savefig(output, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to pareto_front.csv or optimization_results.csv.",
    )
    parser.add_argument(
        "--trial-count",
        type=int,
        default=20,
        help="Select trial numbers from 0 through N-1 (default: 20).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path (default: pareto_first_N_trials.png beside the CSV).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or args.csv_path.with_name(
        f"pareto_first_{args.trial_count}_trials.png"
    )
    selected = select_first_trials(load_trials(args.csv_path), args.trial_count)
    plot_trials(selected, output, args.trial_count)
    print(f"Plotted {len(selected)} trials to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
