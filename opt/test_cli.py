from __future__ import annotations

import unittest

from cli import build_parser
from run_optimization import OptimizationError, main


class CliTests(unittest.TestCase):
    def test_default_cli_keeps_legacy_options_and_fifty_evaluations(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.iterations, 50)
        self.assertEqual(args.run_mode, "fast")
        self.assertEqual(args.solver_iterations, 250)
        self.assertEqual(args.stability_samples, 5)

    def test_compatibility_entrypoint_exports_shared_error(self) -> None:
        self.assertTrue(issubclass(OptimizationError, RuntimeError))
        self.assertTrue(callable(main))


if __name__ == "__main__":
    unittest.main()
