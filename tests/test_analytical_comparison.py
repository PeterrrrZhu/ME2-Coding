"""Analytical-comparison checks for core numerical components."""

import numpy as np

from src.discretization import compute_laplacian_polar, create_polar_grid, forcing_center_gaussian
from src.solver import apply_boundary_conditions, forward_euler_step, initialize_state


def test_laplacian_against_manufactured_solution():
    """Compare numerical Laplacian with analytical Laplacian.

    u(r,theta) = r^3 cos(theta)
    lap(u)     = 8 r cos(theta)
    """
    r, theta, dr, dtheta = create_polar_grid(1.0, 80, 180)
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    u = (rr**3) * np.cos(tt)

    lap_num = compute_laplacian_polar(u, r, dr, dtheta)
    lap_exact = 8.0 * rr * np.cos(tt)

    # The last ring is intentionally clamped boundary in our operator.
    err = np.abs(lap_num[:-1, :] - lap_exact[:-1, :])
    assert float(err.mean()) < 2.0e-3


def test_forward_euler_against_exact_forced_ode_limit():
    """Compare time integration with exact solution when c=0.

    With c=0, each node satisfies:
    u_t = v
    v_t = A sin(omega t), A = q(r)/rho_s
    """
    R = 0.1
    n_r = 40
    n_theta = 72
    r, theta, dr, dtheta = create_polar_grid(R, n_r, n_theta)
    u, v = initialize_state(r, theta)
    u, v = apply_boundary_conditions(u, v)

    dt = 1.0e-5
    rho_s = 0.35
    q0 = 3.0
    sigma = 0.015
    omega = 2.0 * np.pi * 400.0
    steps = 1000

    for n in range(steps):
        t_n = n * dt
        q = forcing_center_gaussian(r, theta, t_n, q0, sigma, omega)
        u, v = forward_euler_step(u, v, r, dr, dtheta, dt, c=0.0, rho_s=rho_s, q=q)

    t = steps * dt
    a_node = (q0 * np.exp(-((r[0] / sigma) ** 2))) / rho_s
    u_exact = a_node * (t / omega - np.sin(omega * t) / (omega**2))
    assert abs(float(u[0, 0] - u_exact)) < 1.0e-7


if __name__ == "__main__":
    test_laplacian_against_manufactured_solution()
    test_forward_euler_against_exact_forced_ode_limit()
    print("test_analytical_comparison: all checks passed")
