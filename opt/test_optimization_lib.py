from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from optimization_lib import (
    ObjectiveResult,
    OptimizationError,
    SearchConfig,
    create_or_load_study,
    pareto_mask,
    run_study,
)


class OptimizationLibraryTests(unittest.TestCase):
    def test_pareto_mask_handles_tradeoffs_dominance_and_duplicates(self) -> None:
        values = [(1.0, 5.0), (2.0, 4.0), (3.0, 6.0), (1.0, 5.0)]
        self.assertEqual(pareto_mask(values, ("minimize", "minimize")), [True, True, False, True])

    def test_completed_budget_ignores_a_failed_evaluation_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = SearchConfig("x", 0.0, 1.0, 3, 2, 7, 3)
            study = create_or_load_study(
                storage_path=Path(directory) / "study.db",
                study_name="generic",
                config=config,
                application_settings={"adapter": "fake"},
            )
            calls = 0

            def evaluate(_number: int, value: float) -> ObjectiveResult:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise OptimizationError("deliberate failure")
                return ObjectiveResult((value, 1.0 - value), {"value": value})

            run_study(study=study, config=config, evaluate=evaluate, progress=lambda _: None)
            complete = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
            failed = [trial for trial in study.trials if trial.state.name == "FAIL"]
            self.assertEqual(len(complete), 3)
            self.assertEqual(len(failed), 1)


if __name__ == "__main__":
    unittest.main()
