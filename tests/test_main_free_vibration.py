"""Checks for main result validation before plotting."""

import numpy as np

from src.main import validate_results_for_plotting


def _build_results(initial_value, final_value):
    """Create a minimal results dict matching main.py expectations."""
    initial = np.full((4, 6), initial_value, dtype=float)
    final = np.full((4, 6), final_value, dtype=float)
    return {
        "snapshots": [initial, final],
        "center_history": np.array([initial_value, final_value], dtype=float),
    }


def test_validate_results_for_plotting_accepts_bounded_finite_values():
    results = _build_results(1.0e-3, 2.0e-3)
    validate_results_for_plotting(results)


def test_validate_results_for_plotting_rejects_nan_inf():
    results = _build_results(1.0e-3, 2.0e-3)
    results["center_history"][1] = np.nan
    with np.testing.assert_raises(ValueError):
        validate_results_for_plotting(results)


def test_validate_results_for_plotting_rejects_large_growth():
    results = _build_results(1.0e-3, 20.0)
    with np.testing.assert_raises(ValueError):
        validate_results_for_plotting(results)


if __name__ == "__main__":
    test_validate_results_for_plotting_accepts_bounded_finite_values()
    test_validate_results_for_plotting_rejects_nan_inf()
    test_validate_results_for_plotting_rejects_large_growth()
    print("test_main_free_vibration: all checks passed")
