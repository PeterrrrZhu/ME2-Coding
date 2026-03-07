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


def test_laplacian_radial_quadratic_first_ring_accuracy():
    """For u=r^2, lap(u)=4 and first-ring error should stay small."""
    r, theta, dr, dtheta = create_polar_grid(1.0, 40, 72)
    rr, _ = np.meshgrid(r, theta, indexing="ij")
    u = rr**2
    lap = compute_laplacian_polar(u, r, dr, dtheta)
    assert np.allclose(lap[0, :], 4.0, atol=5.0e-3)


def test_input_validation():
    with np.testing.assert_raises(ValueError):
        create_polar_grid(0.0, 10, 16)

    with np.testing.assert_raises(ValueError):
        create_u_history(10, 20, -1)


if __name__ == "__main__":
    test_create_polar_grid_shapes()
    test_laplacian_zero_field()
    test_laplacian_radial_quadratic_first_ring_accuracy()
    test_input_validation()
    print("test_discretization_free_vibration: all checks passed")
