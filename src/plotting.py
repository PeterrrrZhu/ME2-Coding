
"""Plotting helpers for membrane simulation results."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _polar_to_cartesian_mesh(r, theta):
    """Return x,y mesh for a polar grid."""
    theta_grid, r_grid = np.meshgrid(theta, r)
    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)
    return x, y


def plot_contour_snapshot(r, theta, u, output_path, title):
    """Save a contour plot of one displacement snapshot."""
    x, y = _polar_to_cartesian_mesh(r, theta)
    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(x, y, u, levels=40, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="u [m]")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_surface_snapshot(r, theta, u, output_path, title):
    """Save a 3D surface plot of one displacement snapshot."""
    x, y = _polar_to_cartesian_mesh(r, theta)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, u, cmap="plasma", linewidth=0.0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("u [m]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_center_history(time_array, center_u, output_path, title):
    """Save line plot of center displacement over time."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time_array, center_u, color="black", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("u_center [m]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def ensure_output_dir(path="outputs"):
    """Create the output folder if it does not exist."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
