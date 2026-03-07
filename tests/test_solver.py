"""Small verification tests for solver helpers."""

import numpy as np

from src.discretization import create_polar_grid
from src.solver import apply_boundary_conditions, forward_euler_step, initialize_state, run_simulation


def test_boundary_condition_clamped_rim():
    r, theta, _, _ = create_polar_grid(1.0, 12, 18)
    u = np.ones((len(r), len(theta)))
    v = np.ones((len(r), len(theta)))
    u_bc, v_bc = apply_boundary_conditions(u, v)
    assert np.allclose(u_bc[-1, :], 0.0)
    assert np.allclose(v_bc[-1, :], 0.0)


def test_zero_state_stays_zero_without_forcing():
    r, theta, dr, dtheta = create_polar_grid(1.0, 12, 18)
    u, v = initialize_state(r, theta)
    q = np.zeros_like(u)
    u_next, v_next = forward_euler_step(
        u,
        v,
        r,
        dr,
        dtheta,
        dt=1.0e-4,
        c=10.0,
        rho_s=1.0,
        q=q,
    )
    assert np.allclose(u_next, 0.0)
    assert np.allclose(v_next, 0.0)


def test_run_simulation_stores_full_u_history():
    parameters = {
        "R": 0.1,
        "n_r": 8,
        "n_theta": 12,
        "dt": 1.0e-5,
        "t_end": 5.0e-5,
        "save_every": 1,
        "T": 120.0,
        "rho_s": 0.35,
        "q0": 0.0,
        "sigma": 0.015,
        "omega": 2.0 * np.pi * 400.0,
        "initial_u_amp": 0.0,
        "initial_u_width": 0.02,
    }
    results = run_simulation(parameters)
    assert results["u_history"].shape == (8, 12, 6)
    assert results["time"].shape == (6,)


if __name__ == "__main__":
    test_boundary_condition_clamped_rim()
    test_zero_state_stays_zero_without_forcing()
    test_run_simulation_stores_full_u_history()
    print("test_solver: all checks passed")
