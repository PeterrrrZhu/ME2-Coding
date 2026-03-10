"""Main entry point for the membrane coursework skeleton."""

from pathlib import Path

import numpy as np

from src.config import default_parameters
from src.plotting import ensure_output_dir, plot_center_history, plot_contour_snapshot, plot_surface_snapshot
from src.solver_free_vibration import run_simulation
from src.Fourier_Transform import Transform

def validate_results_for_plotting(results, growth_factor_limit=1.0e4):
    """Validate solver output before writing figures."""
    final_u = results["snapshots"][-1]
    center_history = results["center_history"]

    if not np.isfinite(final_u).all() or not np.isfinite(center_history).all():
        raise ValueError(
            "Simulation output contains NaN/Inf values. "
            "This usually means the time step is unstable for explicit integration."
        )

    initial_u = results["snapshots"][0]
    initial_max = float(np.max(np.abs(initial_u)))
    final_max = float(np.max(np.abs(final_u)))
    baseline = max(initial_max, 1.0e-12)

    if final_max > growth_factor_limit * baseline:
        raise ValueError(
            "Simulation output grew excessively before plotting "
            f"(|u| from {initial_max:.3e} to {final_max:.3e}). "
            "Reduce dt or review stability settings."
        )


def main():
    """Run simulation and produce three required plot types."""
    params = default_parameters()
    results = run_simulation(params)
    validate_results_for_plotting(results)
 
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = ensure_output_dir(repo_root / "outputs")

    final_u = results["snapshots"][-1]
    final_t = results["snapshot_times"][-1]

    contour_path = output_dir / "contour_final.png"
    surface_path = output_dir / "surface_final.png"
    line_path = output_dir / "center_history.png"

    plot_contour_snapshot(
        results["r"],
        results["theta"],
        final_u,
        contour_path,
        f"Membrane displacement contour at t={final_t:.4f} s",
    )
    plot_surface_snapshot(
        results["r"],
        results["theta"],
        final_u,
        surface_path,
        f"Membrane displacement surface at t={final_t:.4f} s",
    )
    plot_center_history(
        results["center_time"],
        results["center_history"],
        line_path,
        "Center displacement vs time",
    )

    print("Simulation finished.")
    print(f"Saved: {contour_path}")
    print(f"Saved: {surface_path}")
    print(f"Saved: {line_path}")

    Transform(results["u_history"], results["r"].shape[0], params["dt"], output_dir)

if __name__ == "__main__":
    main()
