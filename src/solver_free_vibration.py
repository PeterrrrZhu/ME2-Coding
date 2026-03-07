"""Time-stepping routines for the membrane PDE using 2nd-Order Explicit FDM."""

import numpy as np
from src.config import add_derived_parameters
from src.discretization_free_vibration import (
    compute_laplacian_polar,
    create_polar_grid,
    create_u_history
)

def initialize_state(r, theta, R, A):
    """Apply Initial Shape: Parabolic displacement u(r,theta,0) = A(1 - (r/R)^2)"""
    n_r = len(r)
    n_theta = len(theta)
    u = np.zeros((n_r, n_theta))

    for i in range(n_r):
        for j in range(n_theta):
            # Implements the parabolic initial shape from your document
            u[i, j] = A * (1.0 - (r[i] / R)**2) 
            
    return apply_boundary_conditions(u)

def apply_boundary_conditions(u):
    """Apply Clamped Edge Boundary Condition: u(R, theta, t) = 0"""
    u[-1, :] = 0.0 
    return u

def explicit_wave_step(u_curr, u_prev, r, dr, dtheta, dt, c):
    """
    Advances one time step using the 2nd-order explicit discretisation:
    u^{n+1} = 2u^n - u^{n-1} + (c*dt)^2 * Laplacian
    """
    lap_u = compute_laplacian_polar(u_curr, r, dr, dtheta)
    
    # This directly mirrors the central difference formula from your Word file
    u_next = 2.0 * u_curr - u_prev + ((c * dt)**2) * lap_u
    
    return apply_boundary_conditions(u_next)

def run_simulation(parameters):
    """Run the full membrane simulation."""
    params = add_derived_parameters(parameters)

    # 1. Setup Grid
    r, theta, dr, dtheta = create_polar_grid(params["R"], params["n_r"], params["n_theta"])
    n_r = len(r)
    n_theta = len(theta)
    n_steps = params["n_steps"]
    dt = params["dt"]
    c = params["c"]
    
    # 2. Setup Initial Conditions (n = 0)
    A = params.get("initial_u_amp", 1.0) # Maximum initial amplitude
    u_curr = initialize_state(r, theta, params["R"], A)
    
    # 3. Handle Initial Velocity (Released from rest: du/dt = 0)
    # To start the algorithm, we need a special first step. 
    # Because velocity is 0, u^{-1} effectively equals u^{1}.
    lap_u_initial = compute_laplacian_polar(u_curr, r, dr, dtheta)
    u_next = u_curr + 0.5 * ((c * dt)**2) * lap_u_initial
    u_next = apply_boundary_conditions(u_next)
    
    # Shift variables for the main loop
    u_prev = u_curr.copy()
    u_curr = u_next.copy()

    # 4. Storage for plotting
    u_history = create_u_history(n_r, n_theta, n_steps)
    u_history[:, :, 0] = u_prev
    snapshots = [u_prev.copy()]
    snapshot_times = [0.0]
    center_history = [u_prev[0, 0]]
    center_time = [0.0]

    # 5. The Main Time Loop (n >= 1)
    for step in range(1, n_steps + 1):
        # Save data
        u_history[:, :, step] = u_curr
        current_time = step * dt
        center_history.append(u_curr[0, 0])
        center_time.append(current_time)

        if step % params["save_every"] == 0:
            snapshots.append(u_curr.copy())
            snapshot_times.append(current_time)

        # Calculate the future state (n+1)
        u_next = explicit_wave_step(u_curr, u_prev, r, dr, dtheta, dt, c)
        
        # Prepare variables for the next iteration
        u_prev = u_curr.copy()
        u_curr = u_next.copy()

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