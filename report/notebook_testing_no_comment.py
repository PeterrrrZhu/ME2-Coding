"""[1] Imports"""

from pathlib import Path
import matplotlib
import numpy as np
'''似乎没有学过'''
import matplotlib.pyplot as plt

"""[2] Parameters"""

R = 0.02
n_r = 60 #Change from 80 to 60 for faster testing
n_theta = 30
dt = 1e-6
t_end = 0.05
save_every = 100
T = 100.0
rho_s = 0.35
initial_u_amp = 1.0e-3
c = (T / rho_s) ** 0.5
n_steps = int(round(t_end / dt))

"""[3] Grid"""

"""Create a polar grid with r in (0, R] and theta in [0, 2pi).
R must be positive.
at least 3 points needed
n_theta must be at least 4."""
dr = R / n_r
dtheta = 2.0 * np.pi / n_theta
r = (np.arange(n_r) + 2) * dr
theta = np.arange(n_theta) * dtheta
n_r = len(r)
n_theta = len(theta)
'''后面只引用到这个函数一次，可以考虑简化，不定义函数。同时怀疑不应该放到这个cell'''
'''plot时未用上完整位移历史，需要增加动态展示'''

"""[4] Storage"""

def create_u_history(n_r, n_theta, n_steps):

    """Allocate full displacement history u(r,theta,t)."""
    return np.zeros((n_r, n_theta, n_steps + 1))

"""[5] Initial Condition"""

A = initial_u_amp
u_prev = np.zeros((n_r, n_theta))
for i in range(n_r):
    u_prev[i, :] = A * (1.0 - (r[i] / R) ** 2)
u_prev[-1, :] = 0.0

"""[6] Laplacian"""

def compute_laplacian_polar(u, r, dr, dtheta):

    """Compute Laplacian in polar coordinates on the membrane grid."""
    n_r, n_theta = u.shape
    lap = np.zeros_like(u)
    '''此处类似d2u_dr2等命名需要修改为和课件统一'''
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
    inverse_spacing_sq = (1.0 / (dr**2)) + (1.0 / ((r[0] * dtheta) ** 2))
    dt_max = 1.0 / (c * np.sqrt(inverse_spacing_sq))
if dt > dt_max:
    raise ValueError(
        f"Unstable time step for explicit scheme: dt={dt:.3e}, "
        f"estimated stable limit is about {dt_max:.3e}. "
        "Reduce dt or use a coarser grid / lower wave speed."
    )
lap_u_initial = compute_laplacian_polar(u_prev, r, dr, dtheta)
u_curr = u_prev + 0.5 * ((c * dt) ** 2) * lap_u_initial
u_curr[-1, :] = 0.0
u_history = create_u_history(n_r, n_theta, n_steps)
u_history[:, :, 0] = u_prev
snapshots = [u_prev.copy()]
snapshot_times = [0.0]
center_history = [u_prev[0, 0]]
center_time = [0.0]
for step in range(1, n_steps + 1):
    u_history[:, :, step] = u_curr
    current_time = step * dt
    center_history.append(u_curr[0, 0])
    center_time.append(current_time)
    if (step % save_every == 0) or (step == n_steps):
        snapshots.append(u_curr.copy())
        snapshot_times.append(current_time)
    if step < n_steps:
        lap_u = compute_laplacian_polar(u_curr, r, dr, dtheta)
        u_next = 2.0 * u_curr - u_prev + ((c * dt) ** 2) * lap_u
        u_next[-1, :] = 0.0
        u_prev = u_curr
        u_curr = u_next

"""[8] Post Checks"""

growth_factor_limit = 1.0e4
final_u = snapshots[-1]
center_history = np.array(center_history)
if not np.isfinite(final_u).all() or not np.isfinite(center_history).all():
    raise ValueError(
        "Simulation output contains NaN/Inf values. "
        "This usually means the time step is unstable for explicit integration."
    )
initial_u = snapshots[0]
initial_max = float(np.max(np.abs(initial_u)))
final_max = float(np.max(np.abs(final_u)))
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

def Transform(u_history, n_r, dt, output_dir="outputs"):

    r_idx = n_r // 2
    theta_idx = 0
    raw_time_signal = u_history[r_idx, theta_idx, :]
    #Downsample the time signal for faster DFT calculation, by a factor of 50.
    downsample_factor = 50
    time_signal = raw_time_signal[::downsample_factor]
    dt_new = dt * downsample_factor

    N = len(time_signal)
    print(f"Starting DFT calculation for {N} points. Please wait, this might take a moment...")
    u_fft = DFT(time_signal)
    u_amplitude = np.abs(u_fft[:N // 2])
    fs = 1.0 / dt_new
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
    plt.show()
    plt.close()

"""[10] Fourier Run"""

Transform(u_history, n_r, dt)


'''需要review：
1，哪些code学过，哪些可以用
2，仍然是旧的src路径等，不确定是否有用。需要重构'''

"""[11] Output Paths"""
"""Display plots directly; no file saving."""

"""[12] Plots"""

# Build x-y coordinates from polar grid for physical plotting in the membrane plane.
theta_closed = np.append(theta, theta[0] + 2.0 * np.pi)
final_u_closed = np.hstack((final_u, final_u[:, :1]))
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
plt.plot(center_time, center_history, color="black", linewidth=1.5)
plt.title("Center displacement vs time")
plt.xlabel("t [s]")
plt.ylabel("u_center [m]")
plt.grid(True)
plt.show()
plt.close()

"""[12.5] Animation"""

# Animation of membrane shape using u_history.
# Use the same 3D surface style as the final surface plot.
anim_step = max(1, n_steps // 300)
max_u = np.max(np.abs(u_history))
if max_u < 1.0e-12:
    max_u = 1.0e-12
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection="3d")
for frame_idx in range(0, n_steps + 1, anim_step):
    ax.clear()
    frame_u = u_history[:, :, frame_idx]
    frame_u_closed = np.hstack((frame_u, frame_u[:, :1]))
    ax.plot_surface(x, y, frame_u_closed, cmap="plasma", linewidth=0.0, antialiased=True)
    ax.set_title(f"Membrane animation at t={frame_idx * dt:.4f} s")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("u [m]")
    ax.set_zlim(-max_u, max_u)
    plt.pause(0.03)
plt.show()
plt.close()

"""[13] End Messages"""

print("Simulation finished.")
print("Displayed 3 plots (contour, surface, center history).")
print('CW done: I deserve a good mark')
