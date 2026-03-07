"""Configuration helpers for the membrane simulation."""


def default_parameters():
    """Return default simulation parameters."""
    return {
        "R": 0.1,  # membrane radius [m]
        "n_r": 80,  # number of radial grid points (includes clamped edge)
        "n_theta": 180,  # number of angular grid points
        "dt": 1.0e-6,  # time step [s]
        "t_end": 0.01,  # end time [s]
        "save_every": 20,  # save one snapshot every N steps
        "T": 120.0,  # membrane tension per unit length [N/m]
        "rho_s": 0.35,  # surface density [kg/m^2]
        "q0": 3.0,  # forcing amplitude [N/m^2]
        "sigma": 0.015,  # forcing radial width [m]
        "omega": 2.0 * 3.141592653589793 * 400.0,  # forcing angular frequency [rad/s]
        "initial_u_amp": 0.0,  # optional free-vibration seed
        "initial_u_width": 0.02,
    }


def add_derived_parameters(parameters):
    """Return a copy of parameters with derived values."""
    params = dict(parameters)
    params["c"] = (params["T"] / params["rho_s"]) ** 0.5
    params["n_steps"] = int(round(params["t_end"] / params["dt"]))
    return params

