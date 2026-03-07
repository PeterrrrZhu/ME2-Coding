"""Smoke checks for plotting functions."""

from pathlib import Path

import numpy as np

from src.discretization import create_polar_grid
from src.plotting import ensure_output_dir, plot_center_history, plot_contour_snapshot, plot_surface_snapshot


def test_plotting_outputs_are_created():
    r, theta, _, _ = create_polar_grid(0.1, 20, 36)
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    u = np.sin(np.pi * rr / 0.1) * np.cos(tt)

    out = ensure_output_dir(Path("d:/OneDrive - Imperial College London/Documents/GitHub/ME2-Coding/outputs/plot_test"))
    contour_path = out / "contour_test.png"
    surface_path = out / "surface_test.png"
    line_path = out / "line_test.png"

    plot_contour_snapshot(r, theta, u, contour_path, "contour test")
    plot_surface_snapshot(r, theta, u, surface_path, "surface test")
    plot_center_history(np.linspace(0.0, 0.01, 100), np.sin(np.linspace(0.0, 2.0 * np.pi, 100)), line_path, "line test")

    assert contour_path.exists()
    assert surface_path.exists()
    assert line_path.exists()


if __name__ == "__main__":
    test_plotting_outputs_are_created()
    print("test_plotting: all checks passed")
