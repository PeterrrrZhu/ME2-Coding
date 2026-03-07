"""Grid and finite-difference operators for the membrane model."""

import numpy as np


def create_polar_grid(R, n_r, n_theta):
    """Create a polar grid with r in (0, R] and theta in [0, 2pi)."""
    if R <= 0.0:
        raise ValueError("R must be positive.")
    if n_r < 3:
        raise ValueError("n_r must be at least 3.")
    if n_theta < 4:
        raise ValueError("n_theta must be at least 4.")

    dr = R / n_r
    dtheta = 2.0 * np.pi / n_theta
    r = (np.arange(n_r) + 1) * dr
    theta = np.arange(n_theta) * dtheta
    return r, theta, dr, dtheta


def forcing_center_gaussian(r, theta, t, q0, sigma, omega):
    """Build q(r, theta, t) = q0 * exp(-(r/sigma)^2) * sin(omega*t)."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    n_r = len(r)
    n_theta = len(theta)
    q = np.zeros((n_r, n_theta))
    harmonic = np.sin(omega * t)

    for i in range(n_r):
        radial_factor = np.exp(-((r[i] / sigma) ** 2))
        for j in range(n_theta):
            q[i, j] = q0 * radial_factor * harmonic
    return q


def create_u_history(n_r, n_theta, n_steps):
    """Allocate full displacement history u(r,theta,t)."""
    if n_r <= 0 or n_theta <= 0:
        raise ValueError("n_r and n_theta must be positive.")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")
    return np.zeros((n_r, n_theta, n_steps + 1))


def compute_laplacian_polar(u, r, dr, dtheta):
    """Compute Laplacian in polar coordinates on the membrane grid."""
    # Continuous equation:
    # lap(u) = u_rr + (1/r) u_r + (1/r^2) u_thetatheta
    #
    # Discretised equation (interior ring i >= 1):
    # u_r      ~ (u[i+1,j] - u[i-1,j]) / (2*dr)
    # u_rr     ~ (u[i+1,j] - 2*u[i,j] + u[i-1,j]) / dr^2
    # u_tt     ~ (u[i,j+1] - 2*u[i,j] + u[i,j-1]) / dtheta^2
    # lap[i,j] = u_rr + (1/r_i) u_r + (1/r_i^2) u_tt
    #
    # At the first ring (i = 0), use one-sided second-order radial formulas
    # to avoid a large near-centre truncation error from mirrored stencils.
    if u.ndim != 2:
        raise ValueError("u must be a 2D array with shape (n_r, n_theta).")
    if dr <= 0.0 or dtheta <= 0.0:
        raise ValueError("dr and dtheta must be positive.")

    n_r, n_theta = u.shape
    if len(r) != n_r:
        raise ValueError("len(r) must match u.shape[0].")
    lap = np.zeros_like(u)

    for i in range(n_r - 1):
        r_i = r[i]
        for j in range(n_theta):
            j_plus = (j + 1) % n_theta
            j_minus = (j - 1) % n_theta
            d2u_dtheta2 = (u[i, j_plus] - 2.0 * u[i, j] + u[i, j_minus]) / (dtheta**2)

            if i == 0:
                if n_r >= 4:
                    du_dr = (-3.0 * u[i, j] + 4.0 * u[i + 1, j] - u[i + 2, j]) / (2.0 * dr)
                    d2u_dr2 = (
                        2.0 * u[i, j] - 5.0 * u[i + 1, j] + 4.0 * u[i + 2, j] - u[i + 3, j]
                    ) / (dr**2)
                else:
                    # Fallback for minimum-size grids.
                    du_dr = (u[i + 1, j] - u[i, j]) / dr
                    d2u_dr2 = (u[i + 2, j] - 2.0 * u[i + 1, j] + u[i, j]) / (dr**2)
            else:
                du_dr = (u[i + 1, j] - u[i - 1, j]) / (2.0 * dr)
                d2u_dr2 = (u[i + 1, j] - 2.0 * u[i, j] + u[i - 1, j]) / (dr**2)

            lap[i, j] = d2u_dr2 + (1.0 / r_i) * du_dr + (1.0 / (r_i**2)) * d2u_dtheta2

    lap[-1, :] = 0.0
    return lap
