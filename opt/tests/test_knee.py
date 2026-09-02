from __future__ import annotations

import unittest

from aemec_opt.knee import (
    find_bend_angle_knee,
    find_chebyshev_knee,
    normalize_objectives,
)


class KneePointTests(unittest.TestCase):
    def setUp(self) -> None:
        # A strictly non-dominated front whose two definitions select
        # different interior points after normalization.
        self.points = (
            (0.0, 10.0),
            (2.0, 9.0),
            (4.0, 6.0),
            (5.5, 5.5),
            (6.5, 2.0),
            (10.0, 0.0),
        )
        self.directions = ("minimize", "minimize")

    def test_chebyshev_knee_minimizes_normalized_distance_to_ideal(self) -> None:
        knee = find_chebyshev_knee(self.points, self.directions)
        self.assertIsNotNone(knee)
        assert knee is not None
        self.assertEqual(knee.index, 3)
        self.assertAlmostEqual(knee.distance, 0.55)
        self.assertEqual(knee.normalized_objectives, (0.55, 0.55))

    def test_bend_angle_uses_extreme_points_and_threshold(self) -> None:
        knee = find_bend_angle_knee(self.points, self.directions)
        self.assertIsNotNone(knee)
        assert knee is not None
        self.assertEqual(knee.index, 4)
        self.assertEqual((knee.left_index, knee.right_index), (0, 5))
        self.assertAlmostEqual(knee.angle_degrees, 21.1612598168)
        self.assertIsNone(
            find_bend_angle_knee(
                self.points,
                self.directions,
                threshold_degrees=22.0,
            )
        )

    def test_straight_front_has_no_positive_bend_angle(self) -> None:
        points = ((0.0, 10.0), (5.0, 5.0), (10.0, 0.0))
        self.assertIsNone(find_bend_angle_knee(points, self.directions))

    def test_normalization_accounts_for_maximize_directions(self) -> None:
        normalized = normalize_objectives(
            ((1.0, 0.0), (2.0, 5.0), (3.0, 10.0)),
            ("minimize", "maximize"),
        )
        self.assertEqual(normalized, [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
