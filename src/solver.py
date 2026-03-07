"""Time-stepping routines for the membrane PDE."""

import numpy as np

from src.config import add_derived_parameters
from src.discretization import (
    compute_laplacian_polar,
    create_polar_grid,
    create_u_history,
    forcing_center_gaussian,
)


def initialize_state(r, theta, initial_u_amp=0.0, initial_u_width=0.02):
    """Create initial displacement and velocity fields."""
    n_r = len(r)
    n_theta = len(theta)
    u = np.zeros((n_r, n_theta))
    v = np.zeros((n_r, n_theta))

    if initial_u_amp != 0.0:
        for i in range(n_r):
            radial_factor = np.exp(-((r[i] / initial_u_width) ** 2))
            for j in range(n_theta):
                u[i, j] = initial_u_amp * radial_factor

    return u, v


def apply_boundary_conditions(u, v):
    """Apply membrane boundary conditions."""
    u[-1, :] = 0.0
    v[-1, :] = 0.0
    return u, v


def forward_euler_step(u, v, r, dr, dtheta, dt, c, rho_s, q):
    """Advance one time step with Forward Euler on the first-order system."""
    lap_u = compute_laplacian_polar(u, r, dr, dtheta)
    u_new = u + dt * v
    v_new = v + dt * ((c**2) * lap_u + q / rho_s)
    return apply_boundary_conditions(u_new, v_new)


def run_simulation(parameters):
    """Run the full membrane simulation."""
    params = add_derived_parameters(parameters)

    r, theta, dr, dtheta = create_polar_grid(params["R"], params["n_r"], params["n_theta"])
    n_r = len(r)
    n_theta = len(theta)
    n_steps = params["n_steps"]
    dt = params["dt"]

    u, v = initialize_state(
        r,
        theta,
        initial_u_amp=params["initial_u_amp"],
        initial_u_width=params["initial_u_width"],
    )
    u, v = apply_boundary_conditions(u, v)

    u_history = create_u_history(n_r, n_theta, n_steps)
    u_history[:, :, 0] = u

    snapshots = [u.copy()]
    snapshot_times = [0.0]
    center_history = [u[0, 0]]
    center_time = [0.0]

    for step in range(1, n_steps + 1):
        t_n = (step - 1) * dt
        q = forcing_center_gaussian(r, theta, t_n, params["q0"], params["sigma"], params["omega"])
        u, v = forward_euler_step(u, v, r, dr, dtheta, dt, params["c"], params["rho_s"], q)

        u_history[:, :, step] = u

        current_time = step * dt
        center_history.append(u[0, 0])
        center_time.append(current_time)

        if step % params["save_every"] == 0:
            snapshots.append(u.copy())
            snapshot_times.append(current_time)

    return {
        "r": r,
        "theta": theta,
        "u_history": u_history,
        "time": np.arange(n_steps + 1) * dt,
        "snapshots": snapshots,
        "snapshot_times": snapshot_times,
        "center_time": np.array(center_time),
        "center_history": np.array(center_history),
        "parameters": params,
    }
