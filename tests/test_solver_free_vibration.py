"""Small verification tests for free-vibration solver helpers."""

import numpy as np

from src.discretization_free_vibration import create_polar_grid
from src.solver_free_vibration import apply_boundary_conditions, initialize_state, run_simulation


def test_boundary_condition_clamped_rim():
    r, theta, _, _ = create_polar_grid(1.0, 12, 18)
    u = np.ones((len(r), len(theta)))
    u_bc = apply_boundary_conditions(u)
    assert np.allclose(u_bc[-1, :], 0.0)


def test_initialize_state_parabolic_profile():
    r, theta, _, _ = create_polar_grid(1.0, 10, 12)
    u0 = initialize_state(r, theta, R=1.0, A=1.0e-3)
    expected_center_ring = 1.0e-3 * (1.0 - r[0] ** 2)
    assert np.allclose(u0[0, :], expected_center_ring)
    assert np.allclose(u0[-1, :], 0.0)


def test_run_simulation_nonzero_initial_amplitude_gives_nonzero_response():
    parameters = {
        "R": 0.2,
        "n_r": 20,
        "n_theta": 36,
        "dt": 1.0e-6,
        "t_end": 2.0e-4,
        "save_every": 10,
        "T": 120.0,
        "rho_s": 0.35,
        "q0": 0.0,
        "sigma": 0.015,
        "omega": 2.0 * np.pi * 400.0,
        "initial_u_amp": 1.0e-3,
        "initial_u_width": 0.02,
    }
    results = run_simulation(parameters)
    assert float(np.max(np.abs(results["center_history"]))) > 0.0
    assert float(np.max(np.abs(results["snapshots"][-1]))) > 0.0


def test_run_simulation_raises_for_zero_initial_amplitude():
    parameters = {
        "R": 0.2,
        "n_r": 20,
        "n_theta": 36,
        "dt": 1.0e-6,
        "t_end": 2.0e-4,
        "save_every": 10,
        "T": 120.0,
        "rho_s": 0.35,
        "q0": 0.0,
        "sigma": 0.015,
        "omega": 2.0 * np.pi * 400.0,
        "initial_u_amp": 0.0,
        "initial_u_width": 0.02,
    }
    with np.testing.assert_raises(ValueError):
        run_simulation(parameters)


if __name__ == "__main__":
    test_boundary_condition_clamped_rim()
    test_initialize_state_parabolic_profile()
    test_run_simulation_nonzero_initial_amplitude_gives_nonzero_response()
    test_run_simulation_raises_for_zero_initial_amplitude()
    print("test_solver_free_vibration: all checks passed")
