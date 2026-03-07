"""Grid and finite-difference operators for the membrane model."""

import numpy as np

def create_polar_grid(R, n_r, n_theta):
    """Create a polar grid with r in (0, R] and theta in [0, 2pi)."""
    if n_r < 3:
        raise ValueError("n_r must be at least 3.")
    if n_theta < 4:
        raise ValueError("n_theta must be at least 4.")

    # Calculate step sizes for radius (dr) and angle (dtheta) based on the number of nodes
    dr = R / n_r
    dtheta = 2.0 * np.pi / n_theta
    
    # Mathematical trick: The +1 shifts the grid so it starts at 1*dr instead of 0.
    # This intentionally avoids the exact center (r=0) so we never divide by zero in the PDE.
    r = (np.arange(n_r) + 1) * dr
    
    # Creates evenly spaced angles from 0 up to (but not including) 2*pi
    theta = np.arange(n_theta) * dtheta
    return r, theta, dr, dtheta

'''
def forcing_center_gaussian(r, theta, t, q0, sigma, omega):
    """Simulates FREE VIBRATION by returning a zero array for external forces."""
    n_r = len(r)
    n_theta = len(theta)
    
    # By returning pure zeros, we ensure there is no external driving force.
    # This allows the membrane to vibrate naturally based only on its initial shape.
    return np.zeros((n_r, n_theta))
'''

def create_u_history(n_r, n_theta, n_steps):
    """Allocate full displacement history u(r,theta,t)."""
    # Creates a 3D matrix to store the 2D grid at every single time step for later plotting
    return np.zeros((n_r, n_theta, n_steps + 1))


def compute_laplacian_polar(u, r, dr, dtheta):
    """
    Compute 2D spatial Laplacian in polar coordinates.
    This calculates the right-hand side of the Polar Form equation: 
    (d^2u/dr^2) + (1/r)*(du/dr) + (1/r^2)*(d^2u/dtheta^2)
    """
    n_r, n_theta = u.shape
    lap = np.zeros_like(u)

    # Loop through all radial rings EXCEPT the very outer edge (n_r - 1)
    for i in range(n_r - 1):
        r_i = r[i]
        
        # Loop through all angles in the current ring
        for j in range(n_theta):
            
            # --- PERIODIC BOUNDARY CONDITION ---
            # Angle wraps around: u(r, 0, t) = u(r, 2pi, t)[cite: 36].
            # The modulo operator (%) ensures that if we look "right" of the last angle,
            # we wrap back to the first angle (index 0).
            j_plus = (j + 1) % n_theta
            j_minus = (j - 1) % n_theta
            
            # Calculates the 2nd derivative with respect to theta: (d^2u/dtheta^2)
            d2u_dtheta2 = (u[i, j_plus] - 2.0 * u[i, j] + u[i, j_minus]) / (dtheta**2)

            # --- CENTER BOUNDARY CONDITION ---
            if i == 0:
                # At the centre r = 0, du/dr = 0.
                # This inherently applies the du/dr = 0 condition at the center.
                # We invent a "ghost node" (u_im1) exactly opposite the center that mirrors the value 
                # at u[i+1, j]. This mathematically forces the slope (du/dr) to be flat.
                u_im1 = u[i + 1, j]
                
                # Because u_im1 equals u[i+1, j], this du_dr calculation will perfectly equal 0
                '''改为urr, ur'''
                du_dr = (u[i + 1, j] - u_im1) / (2.0 * dr) 
                d2u_dr2 = (u[i + 1, j] - 2.0 * u[i, j] + u_im1) / (dr**2)
            else:
                # Standard central difference for nodes away from the center
                du_dr = (u[i + 1, j] - u[i - 1, j]) / (2.0 * dr)
                d2u_dr2 = (u[i + 1, j] - 2.0 * u[i, j] + u[i - 1, j]) / (dr**2)

            # --- ASSEMBLING THE LAPLACIAN ---
            # This combines the spatial derivatives matching the polar form[cite: 32].
            lap[i, j] = d2u_dr2 + (1.0 / r_i) * du_dr + (1.0 / (r_i**2)) * d2u_dtheta2

    # --- CLAMPED EDGE BOUNDARY CONDITION ---
    # If the membrane has a radius of R, the clamped edge condition is u(R, theta, t) = 0[cite: 33, 34].
    # By setting the entire last row ([-1, :]) of the Laplacian to 0, 
    # we ensure the outer edge never accelerates or moves.
    lap[-1, :] = 0.0
    return lap