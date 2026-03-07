"""Main entry point for the membrane coursework skeleton."""

from pathlib import Path

from src.config import default_parameters
from src.plotting import ensure_output_dir, plot_center_history, plot_contour_snapshot, plot_surface_snapshot
from src.solver import run_simulation


def main():
    """Run simulation and produce three required plot types."""
    params = default_parameters()
    results = run_simulation(params)

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


# if __name__ == "__main__":
main()

