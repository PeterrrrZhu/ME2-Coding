"""Small verification tests for discretization helpers."""

import numpy as np

from src.discretization import compute_laplacian_polar, create_polar_grid, create_u_history, forcing_center_gaussian


def test_create_polar_grid():
    r, theta, dr, dtheta = create_polar_grid(1.0, 10, 16)
    assert len(r) == 10
    assert len(theta) == 16
    assert np.isclose(r[-1], 1.0)
    assert dr > 0.0
    assert dtheta > 0.0


def test_forcing_shape():
    r, theta, _, _ = create_polar_grid(1.0, 8, 12)
    q = forcing_center_gaussian(r, theta, t=0.001, q0=2.0, sigma=0.3, omega=10.0)
    assert q.shape == (8, 12)


def test_laplacian_zero_field():
    r, theta, dr, dtheta = create_polar_grid(1.0, 10, 20)
    u = np.zeros((len(r), len(theta)))
    lap = compute_laplacian_polar(u, r, dr, dtheta)
    assert np.allclose(lap, 0.0)


def test_u_history_shape():
    u_history = create_u_history(10, 20, 30)
    assert u_history.shape == (10, 20, 31)


def test_laplacian_radial_quadratic_first_ring_accuracy():
    """For u=r^2, lap(u)=4 and first-ring error should stay small."""
    r, theta, dr, dtheta = create_polar_grid(1.0, 40, 72)
    rr, _ = np.meshgrid(r, theta, indexing="ij")
    u = rr**2
    lap = compute_laplacian_polar(u, r, dr, dtheta)
    assert np.allclose(lap[0, :], 4.0, atol=5.0e-3)


def test_discretization_input_validation():
    with np.testing.assert_raises(ValueError):
        create_polar_grid(0.0, 10, 16)

    with np.testing.assert_raises(ValueError):
        create_u_history(10, 20, -1)

    r, theta, _, _ = create_polar_grid(1.0, 8, 12)
    with np.testing.assert_raises(ValueError):
        forcing_center_gaussian(r, theta, t=0.0, q0=1.0, sigma=0.0, omega=1.0)


if __name__ == "__main__":
    test_create_polar_grid()
    test_forcing_shape()
    test_laplacian_zero_field()
    test_u_history_shape()
    test_laplacian_radial_quadratic_first_ring_accuracy()
    test_discretization_input_validation()
    print("test_discretization: all checks passed")
