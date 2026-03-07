"""Small verification tests for free-vibration discretization helpers."""

import numpy as np

from src.discretization_free_vibration import compute_laplacian_polar, create_polar_grid, create_u_history


def test_create_polar_grid_shapes():
    r, theta, dr, dtheta = create_polar_grid(1.0, 10, 16)
    assert len(r) == 10
    assert len(theta) == 16
    assert np.isclose(r[-1], 1.0)
    assert dr > 0.0
    assert dtheta > 0.0


def test_laplacian_zero_field():
    r, theta, dr, dtheta = create_polar_grid(1.0, 12, 24)
    u = np.zeros((len(r), len(theta)))
    lap = compute_laplacian_polar(u, r, dr, dtheta)
    assert np.allclose(lap, 0.0)


def test_laplacian_manufactured_solution_interior_accuracy():
    """For u=r^3*cos(theta), lap(u)=8*r*cos(theta) away from boundaries."""
    r, theta, dr, dtheta = create_polar_grid(1.0, 80, 180)
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    u = (rr**3) * np.cos(tt)
    lap = compute_laplacian_polar(u, r, dr, dtheta)
    lap_exact = 8.0 * rr * np.cos(tt)

    # Exclude first ring (centre treatment) and last ring (clamped edge row).
    err = np.abs(lap[1:-1, :] - lap_exact[1:-1, :])
    assert float(err.mean()) < 2.0e-3


def test_input_validation():
    with np.testing.assert_raises(ValueError):
        create_polar_grid(0.0, 10, 16)

    with np.testing.assert_raises(ValueError):
        create_u_history(10, 20, -1)


if __name__ == "__main__":
    test_create_polar_grid_shapes()
    test_laplacian_zero_field()
    test_laplacian_manufactured_solution_interior_accuracy()
    test_input_validation()
    print("test_discretization_free_vibration: all checks passed")
