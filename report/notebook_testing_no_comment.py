"""[1] Imports"""

from pathlib import Path
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

"""[2] Parameters"""

R = 0.02
Nr = 80
Ntheta = 30
dt = 1e-6
t_end = 0.02
save_every = 250
T = 100.0
rho_s = 0.35
u0_amp = 1.0e-3
c = (T / rho_s) ** 0.5
Nt = int(round(t_end / dt)) 

"""[3] Grid"""

"""Create a polar grid with r in (0, R] and theta in [0, 2pi).
R must be positive.
at least 3 points needed
Ntheta must be at least 4."""
dr = R / Nr
dtheta = 2.0 * np.pi / Ntheta
r = (np.arange(Nr) + 2) * dr
theta = np.arange(Ntheta) * dtheta
Nr = len(r)
Ntheta = len(theta)



"""[4] Storage"""

"""[5] Initial Condition"""

A = u0_amp
u_p0 = np.zeros((Nr, Ntheta))
for i in range(Nr):
    u_p0[i, :] = A * (1.0 - (r[i] / R) ** 2)
u_p0[-1, :] = 0.0

"""[6] Laplacian"""

def compute_laplacian_polar(u, r, dr, dtheta):

    """Compute Laplacian in polar coordinates on the membrane grid."""
    Nr, Ntheta = u.shape
    lap = np.zeros_like(u)

    for i in range(Nr - 1):
        ri = r[i]
        for j in range(Ntheta):
            j_plus = (j + 1) % Ntheta
            j_minus = (j - 1) % Ntheta
            u_thetatheta = (u[i, j_plus] - 2.0 * u[i, j] + u[i, j_minus]) / (dtheta**2)
            if i == 0:
                u_im1 = u[i + 1, j]
                du_dr = (u[i + 1, j] - u_im1) / (2.0 * dr)
                u_rr = (u[i + 1, j] - 2.0 * u[i, j] + u_im1) / (dr**2)
            else:
                du_dr = (u[i + 1, j] - u[i - 1, j]) / (2.0 * dr)
                u_rr = (u[i + 1, j] - 2.0 * u[i, j] + u[i - 1, j]) / (dr**2)
            lap[i, j] = u_rr + (1.0 / ri) * du_dr + (1.0 / (ri**2)) * u_thetatheta
    lap[-1, :] = 0.0
    return lap
'''再次确认，ValueError是否学过，如果没学过，应该仍然保留合法性检查，还是直接删除？前面的代码已经删除了一些合法性检查了'''
"""[7] Stability and Time Step"""

if save_every <= 0:
    raise ValueError("save_every must be a positive integer.")
if dt <= 0.0:
    raise ValueError("dt must be positive.")
if dr <= 0.0 or dtheta <= 0.0 or r[0] <= 0.0:
    raise ValueError("dr, dtheta and r_min must be positive.")
if c < 0.0:
    raise ValueError("c must be non-negative.")
if c == 0.0:
    dt_max = np.inf
else:
    safety_factor = 0.2
    if safety_factor <= 0.0 or safety_factor > 1.0:
        raise ValueError("safety_factor must be in (0, 1].")
    inverse_spacing_sq = (1.0 / (dr**2)) + (1.0 / ((r[0] * dtheta) ** 2))
    dt_cfl = 1.0 / (c * np.sqrt(inverse_spacing_sq))
    dt_max = safety_factor * dt_cfl
if dt > dt_max:
    raise ValueError(
        f"Unstable time step for explicit scheme: dt={dt:.3e}, "
        f"estimated stable limit is about {dt_max:.3e}. "
        "Reduce dt or use a coarser grid / lower wave speed."
    )
lap_u0 = compute_laplacian_polar(u_p0, r, dr, dtheta)
u_p1 = u_p0 + 0.5 * ((c * dt) ** 2) * lap_u0
u_p1[-1, :] = 0.0
Nsaved = (Nt // save_every) + 1
if Nt % save_every != 0:
    Nsaved += 1
u_history = np.zeros((Nr, Ntheta, Nsaved))
u_history[:, :, 0] = u_p0
t_history = [0.0]
i_history = 1
snapshots = [u_p0.copy()]
snapshot_times = [0.0]
center_history = [u_p0[0, 0]]
t_centre = [0.0]
for step in range(1, Nt + 1):
    current_time = step * dt
    center_history.append(u_p1[0, 0])
    t_centre.append(current_time)
    if (step % save_every == 0) or (step == Nt):
        u_history[:, :, i_history] = u_p1
        t_history.append(current_time)
        i_history += 1
        snapshots.append(u_p1.copy())
        snapshot_times.append(current_time)
    if step < Nt:
        lap_u = compute_laplacian_polar(u_p1, r, dr, dtheta)
        u_p2 = 2.0 * u_p1 - u_p0 + ((c * dt) ** 2) * lap_u
        u_p2[-1, :] = 0.0
        u_p0 = u_p1
        u_p1 = u_p2

"""[8] Post Checks"""

growth_factor_limit = 1.0e4
u_final = snapshots[-1]
center_history = np.array(center_history)
if not np.isfinite(u_final).all() or not np.isfinite(center_history).all():
    raise ValueError(
        "Simulation output contains NaN/Inf values. "
        "This usually means the time step is unstable for explicit integration."
    )
u_initial = snapshots[0]
initial_max = float(np.max(np.abs(u_initial)))
final_max = float(np.max(np.abs(u_final)))
baseline = max(initial_max, 1.0e-12)
if final_max > growth_factor_limit * baseline:
    raise ValueError(
        "Simulation output grew excessively before plotting "
        f"(|u| from {initial_max:.3e} to {final_max:.3e}). "
        "Reduce dt or review stability settings."
    )
final_t = snapshot_times[-1]

"""[9] Fourier Functions"""

def DFT(yn):

    N = len(yn)
    w = 2 * np.pi / N
    FTk = np.zeros(N, dtype=complex)
    for k in range(0, N):
        for n in range(0, N):
            FTk[k] += np.exp(-1j * k * w * n) * yn[n]
    return FTk

def Transform(u_history, Nr, dt, output_dir="outputs"):

    r_idx = Nr // 2
    theta_idx = 0
    time_signal = u_history[r_idx, theta_idx, :]
    N = len(time_signal)
    print(f"Starting DFT calculation for {N} points. Please wait, this might take a moment...")
    u_fft = DFT(time_signal)
    u_amplitude = np.abs(u_fft[:N // 2])
    fs = 1.0 / dt
    freq_axis = np.linspace(0, fs / 2, N // 2)
    plt.figure(figsize=(10, 5))
    plt.plot(freq_axis, u_amplitude,label='Frequency Spectrum',linewidth=1.5)
    plt.title(rf'Frequency Spectrum at $r$={r_idx}, $\theta$={theta_idx}')
    plt.xlim(0, 1000)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    search_amplitude = u_amplitude.copy()
    if len(search_amplitude) > 0:
        search_amplitude[0] = 0.0
    dominant_idx = np.argmax(search_amplitude)
    dominant_freq = freq_axis[dominant_idx]
    plt.scatter(dominant_freq, u_amplitude[dominant_idx], color='red', zorder=5)
    plt.axvline(x=dominant_freq, color='red', linestyle='--', alpha=0.7,
                label=rf'Dominant Frequency: {dominant_freq:.2f} Hz')
    plt.legend(loc='upper right')
    print(f"Dominant Frequency: {dominant_freq:.2f} Hz")
    print(f"Peak Amplitude: {u_amplitude[dominant_idx]:.4f}")
    output_path = Path(output_dir) / "fourier_spectrum_dft.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Fourier plot successfully saved to {output_path}")
    plt.close()

"""[10] Fourier Run"""

#Transform(u_history, Nr, dt * save_every)




"""[11] Output Paths"""
"""Display plots directly; no file saving."""

"""[12] Plots"""

# Build x-y coordinates from polar grid for physical plotting in the membrane plane.
theta_closed = np.append(theta, theta[0] + 2.0 * np.pi)
final_u_closed = np.hstack((u_final, u_final[:, :1]))
theta_grid, r_grid = np.meshgrid(theta_closed, r)
x = r_grid * np.cos(theta_grid)
y = r_grid * np.sin(theta_grid)

# Contour plot: final membrane displacement field u(x, y, t_end).
plt.figure(figsize=(6, 5))
contour = plt.contourf(x, y, final_u_closed, levels=40, cmap="viridis")
plt.colorbar(contour, label="u [m]")
plt.title(f"Membrane displacement contour at t={final_t:.4f} s")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.grid(True)
plt.show()
plt.close()

# Surface plot: final membrane displacement shown as height in 3D.
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x, y, final_u_closed, cmap="plasma", linewidth=0.0, antialiased=True)
plt.title(f"Membrane displacement surface at t={final_t:.4f} s")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.show()
plt.close()

# Line plot: displacement of membrane center over time.
plt.figure(figsize=(7, 4))
plt.plot(t_centre, center_history, color="black", linewidth=1.5)
plt.title("Center displacement vs time")
plt.xlabel("t [s]")
plt.ylabel("u_center [m]")
plt.grid(True)
plt.show()
plt.close()

"""[12.5] Animation"""

# Animation of membrane shape using u_history.
# Use the same 3D surface style as the final surface plot.
N_saved_frames = u_history.shape[2]
anim_step = max(1, N_saved_frames // 300)
u_max = np.max(np.abs(u_history))
if u_max < 1.0e-12:
    u_max = 1.0e-12
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection="3d")
for saved_idx in range(0, N_saved_frames, anim_step):
    ax.clear()
    frame_u = u_history[:, :, saved_idx]
    frame_u_closed = np.hstack((frame_u, frame_u[:, :1]))
    ax.plot_surface(x, y, frame_u_closed, cmap="plasma", linewidth=0.0, antialiased=True)
    ax.set_title(f"Membrane animation at t={t_history[saved_idx]:.4f} s")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("u [m]")
    ax.set_zlim(-u_max, u_max)
    plt.pause(0.03)
plt.show()
plt.close()

"""[13] End Messages"""


print('Done')

