"""Grid and finite-difference operators for the membrane model."""

import numpy as np


def create_polar_grid(R, n_r, n_theta):
    """Create a polar grid with r in (0, R] and theta in [0, 2pi)."""
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
    return np.zeros((n_r, n_theta, n_steps + 1))


def compute_laplacian_polar(u, r, dr, dtheta):
    """Compute Laplacian in polar coordinates on the membrane grid."""
    n_r, n_theta = u.shape
    lap = np.zeros_like(u)

    for i in range(n_r - 1):
        r_i = r[i]
        for j in range(n_theta):
            j_plus = (j + 1) % n_theta
            j_minus = (j - 1) % n_theta
            d2u_dtheta2 = (u[i, j_plus] - 2.0 * u[i, j] + u[i, j_minus]) / (dtheta**2)

            if i == 0:
                u_im1 = u[i + 1, j]
                du_dr = (u[i + 1, j] - u_im1) / (2.0 * dr)
                d2u_dr2 = (u[i + 1, j] - 2.0 * u[i, j] + u_im1) / (dr**2)
            else:
                du_dr = (u[i + 1, j] - u[i - 1, j]) / (2.0 * dr)
                d2u_dr2 = (u[i + 1, j] - 2.0 * u[i, j] + u[i - 1, j]) / (dr**2)

            lap[i, j] = d2u_dr2 + (1.0 / r_i) * du_dr + (1.0 / (r_i**2)) * d2u_dtheta2

    lap[-1, :] = 0.0
    return lap
