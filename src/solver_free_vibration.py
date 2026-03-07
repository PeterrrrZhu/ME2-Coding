"""Time-stepping routines for the membrane PDE using 2nd-order explicit FDM."""

import numpy as np

from src.config import add_derived_parameters
from src.discretization_free_vibration import compute_laplacian_polar, create_polar_grid, create_u_history


def estimate_stable_time_step(c, dr, dtheta, r_min, safety_factor=0.2):
    """Estimate a conservative stable dt for explicit 2D polar wave stepping."""
    if dr <= 0.0 or dtheta <= 0.0 or r_min <= 0.0:
        raise ValueError("dr, dtheta and r_min must be positive.")
    if c < 0.0:
        raise ValueError("c must be non-negative.")
    if c == 0.0:
        return np.inf
    if safety_factor <= 0.0 or safety_factor > 1.0:
        raise ValueError("safety_factor must be in (0, 1].")

    # Conservative CFL-like condition near the smallest angular arc length.
    # (c*dt/dr)^2 + (c*dt/(r_min*dtheta))^2 <= 1
    inverse_spacing_sq = (1.0 / (dr**2)) + (1.0 / ((r_min * dtheta) ** 2))
    dt_cfl = 1.0 / (c * np.sqrt(inverse_spacing_sq))
    return safety_factor * dt_cfl


def validate_time_step_for_explicit_scheme(dt, c, dr, dtheta, r):
    """Raise a clear error if dt violates the explicit-stability estimate."""
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    dt_max = estimate_stable_time_step(c, dr, dtheta, r[0])
    if dt > dt_max:
        raise ValueError(
            f"Unstable time step for explicit scheme: dt={dt:.3e}, "
            f"estimated stable limit is about {dt_max:.3e}. "
            "Reduce dt or use a coarser grid / lower wave speed."
        )
    return dt_max


def initialize_state(r, theta, R, A):
    """Apply initial shape: u(r,theta,0) = A * (1 - (r/R)^2)."""
    n_r = len(r)
    n_theta = len(theta)
    u = np.zeros((n_r, n_theta))

    for i in range(n_r):
        for j in range(n_theta):
            u[i, j] = A * (1.0 - (r[i] / R) ** 2)

    return apply_boundary_conditions(u)


def apply_boundary_conditions(u):
    """Apply clamped edge boundary condition: u(R, theta, t) = 0."""
    u[-1, :] = 0.0
    return u


def explicit_wave_step(u_curr, u_prev, r, dr, dtheta, dt, c):
    """Advance one time step with u^{n+1} = 2u^n - u^{n-1} + (c*dt)^2 * lap(u^n)."""
    lap_u = compute_laplacian_polar(u_curr, r, dr, dtheta)
    u_next = 2.0 * u_curr - u_prev + ((c * dt) ** 2) * lap_u
    return apply_boundary_conditions(u_next)


def run_simulation(parameters):
    """Run the full membrane simulation."""
    params = add_derived_parameters(parameters)

    r, theta, dr, dtheta = create_polar_grid(params["R"], params["n_r"], params["n_theta"])
    n_r = len(r)
    n_theta = len(theta)
    n_steps = params["n_steps"]
    dt = params["dt"]
    c = params["c"]
    save_every = params["save_every"]

    if save_every <= 0:
        raise ValueError("save_every must be a positive integer.")
    validate_time_step_for_explicit_scheme(dt, c, dr, dtheta, r)

    A = params.get("initial_u_amp", 1.0e-3)
    if A == 0.0:
        raise ValueError(
            "initial_u_amp is 0.0, so the free-vibration solution stays zero. "
            "Set initial_u_amp to a non-zero value."
        )

    # Initial state at n = 0
    u_prev = initialize_state(r, theta, params["R"], A)

    # Released from rest: u_t(r,theta,0) = 0
    # First step from central difference in time:
    # u^1 = u^0 + 0.5 * (c*dt)^2 * lap(u^0)
    lap_u_initial = compute_laplacian_polar(u_prev, r, dr, dtheta)
    u_curr = apply_boundary_conditions(u_prev + 0.5 * ((c * dt) ** 2) * lap_u_initial)

    u_history = create_u_history(n_r, n_theta, n_steps)
    u_history[:, :, 0] = u_prev

    snapshots = [u_prev.copy()]
    snapshot_times = [0.0]
    center_history = [u_prev[0, 0]]
    center_time = [0.0]

    # Step index matches stored time level n, with time t = n*dt.
    for step in range(1, n_steps + 1):
        u_history[:, :, step] = u_curr
        current_time = step * dt
        center_history.append(u_curr[0, 0])
        center_time.append(current_time)

        # Always keep the true final state at t_end in snapshots.
        if (step % save_every == 0) or (step == n_steps):
            snapshots.append(u_curr.copy())
            snapshot_times.append(current_time)

        # Advance only if another stored time level is needed.
        if step < n_steps:
            u_next = explicit_wave_step(u_curr, u_prev, r, dr, dtheta, dt, c)
            u_prev = u_curr
            u_curr = u_next

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
