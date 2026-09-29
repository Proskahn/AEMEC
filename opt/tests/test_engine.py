from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from aemec_opt.engine import (
    ObjectiveResult,
    OptimizationError,
    SearchConfig,
    SearchParameter,
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
            config = SearchConfig((SearchParameter("x", 0.0, 1.0),), 3, 2, 7, 3)
            study = create_or_load_study(
                storage_path=Path(directory) / "study.db",
                study_name="generic",
                config=config,
                application_settings={"adapter": "fake"},
            )
            calls = 0

            def evaluate(_number: int, parameters: dict[str, float]) -> ObjectiveResult:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise OptimizationError("deliberate failure")
                value = parameters["x"]
                return ObjectiveResult((value, 1.0 - value), {"value": value})

            run_study(study=study, config=config, evaluate=evaluate, progress=lambda _: None)
            complete = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
            failed = [trial for trial in study.trials if trial.state.name == "FAIL"]
            self.assertEqual(len(complete), 3)
            self.assertEqual(len(failed), 1)

    def test_two_parameters_are_sampled_independently_and_resumed(self) -> None:
        config = SearchConfig(
            (SearchParameter("thickness", 10, 100), SearchParameter("temperature", 298.15, 353.15)),
            7, 5, 42, 3,
        )
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(storage_path=Path(directory) / "study.db", study_name="two-parameters", application_settings={})
            study = create_or_load_study(config=config, **kwargs)
            startup = [trial.system_attrs["fixed_params"] for trial in study.trials]
            self.assertEqual(len({point["temperature"] for point in startup}), 5)
            self.assertEqual(min(point["temperature"] for point in startup), 298.15)
            self.assertEqual(max(point["temperature"] for point in startup), 353.15)
            ordered = sorted(startup, key=lambda point: point["thickness"])
            temperatures = [point["temperature"] for point in ordered]
            self.assertNotEqual(temperatures, sorted(temperatures))
            self.assertNotEqual(temperatures, sorted(temperatures, reverse=True))

            def evaluate(_number, parameters):
                self.assertEqual(set(parameters), {"thickness", "temperature"})
                self.assertTrue(10 <= parameters["thickness"] <= 100)
                self.assertTrue(298.15 <= parameters["temperature"] <= 353.15)
                return ObjectiveResult((parameters["thickness"], 1 / parameters["temperature"]), {})

            run_study(study=study, config=config, evaluate=evaluate, progress=lambda _: None)
            resumed = create_or_load_study(config=config, **kwargs)
            run_study(study=resumed, config=config, evaluate=lambda *_: self.fail("completed trial rerun"))
            self.assertEqual(len(resumed.trials), 7)
            with self.assertRaisesRegex(OptimizationError, "different engine"):
                create_or_load_study(config=replace(config, parameters=(config.parameters[0], SearchParameter("temperature", 298.15, 343.15))), **kwargs)

    def test_search_rejects_invalid_bounds_and_duplicate_names(self) -> None:
        for parameters in (
            (),
            (SearchParameter("T", float("nan"), 350),),
            (SearchParameter("T", 350, 300),),
            (SearchParameter("T", 300, 350), SearchParameter("T", 310, 360)),
        ):
            with self.subTest(parameters=parameters), self.assertRaises(OptimizationError):
                SearchConfig(parameters, 5, 2, 42, 3).validate()


if __name__ == "__main__":
    unittest.main()
