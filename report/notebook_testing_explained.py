# 代码总览：
# 这个脚本用 explicit finite difference 在 polar grid 上模拟圆形 membrane 的振动，
# 控制方程是 u_tt = c^2 * (u_rr + (1/r)u_r + (1/r^2)u_thetatheta)；
# 先定义参数和网格，再计算初始位移与时间推进，之后做 DFT 频谱分析，最后绘图展示结果。

# 导入 numpy，用于数组、数学函数和数值计算。
import numpy as np
# 导入 matplotlib.pyplot，用于画图和可视化结果。
import matplotlib.pyplot as plt

# 定义圆形 membrane 的半径 R（单位 m）。
R = 0.02

# 设定径向离散点数量 Nr（radial points）。
Nr = 80
# 设定角向离散点数量 Ntheta（angular points）。
Ntheta = 30
# 设定时间步长 dt（单位 s），用于 time marching。
dt = 1e-6
# 设定总模拟时长 t_end（单位 s）。
t_end = 0.02
# 每隔 save_every 个时间步存一次场数据，减少存储量。
save_every = 250
# 设定膜面的张力 T（单位 N/m）。
T = 100.0
# 设定面密度 rho_s（单位 kg/m^2）。
rho_s = 0.35
# 设定初始位移幅值 u0_amp（单位 m）。
u0_amp = 1.0e-3
# 根据波速关系 c = sqrt(T/rho_s) 计算 membrane 波速 c。
c = (T / rho_s) ** 0.5
# 根据总时长和时间步长计算时间步数 Nt。
Nt = int(round(t_end / dt))

# 计算径向网格步长 dr = R / Nr。
dr = R / Nr
# 计算角向网格步长 dtheta = 2*pi / Ntheta。
dtheta = 2.0 * np.pi / Ntheta
# 构造径向坐标数组 r（从 2*dr 开始到靠近 R），每个元素对应一个 radial node。
r = (np.arange(Nr) + 2) * dr
# 构造角向坐标数组 theta（0 到 2*pi 之前）。
theta = np.arange(Ntheta) * dtheta
# 用数组长度重新确认 Nr，保证后续循环和数组一致。
Nr = len(r)
# 用数组长度重新确认 Ntheta，保证后续循环和数组一致。
Ntheta = len(theta)

# 把初始位移振幅赋给 A，便于后续公式书写。
A = u0_amp
# 创建 t=0 时刻位移场 u_p0，shape 为 (Nr, Ntheta)。
u_p0 = np.zeros((Nr, Ntheta))
# 遍历每个径向节点 i，按抛物线分布设置初始位移。
for i in range(Nr):
    # 这里使用 u(r,0)=A*(1-(r/R)^2)，表示中心最大、边界减小的初值形状。
    u_p0[i, :] = A * (1.0 - (r[i] / R) ** 2)
# 强制边界条件 u(R,theta,t)=0（固定边界），把最外层径向节点设为 0。
u_p0[-1, :] = 0.0


# 定义函数：在 polar coordinates 上计算离散 Laplacian。
def compute_laplacian_polar(u, r, dr, dtheta):
    # 从输入场 u 读取网格尺寸 Nr 和 Ntheta。
    Nr, Ntheta = u.shape
    # 创建与 u 同形状的数组 lap，用来存离散 Laplacian 结果。
    lap = np.zeros_like(u)

    # 径向循环只到 Nr-2，最后一层是固定边界不参与内部更新。
    for i in range(Nr - 1):
        # 取当前半径 ri，用于 (1/r)u_r 和 (1/r^2)u_thetatheta 项。
        ri = r[i]
        # 遍历每个角向节点 j。
        for j in range(Ntheta):
            # 用 periodic indexing 得到 j+1，保证 theta 方向首尾相连。
            j_plus = (j + 1) % Ntheta
            # 用 periodic indexing 得到 j-1，保证 theta 方向首尾相连。
            j_minus = (j - 1) % Ntheta
            # 用 central difference 计算角向二阶导数 u_thetatheta。
            u_thetatheta = (u[i, j_plus] - 2.0 * u[i, j] + u[i, j_minus]) / (dtheta**2)
            # 在最内层径向节点 i=0 处，用 one-sided 方式近似 u_r 和 u_rr。
            if i == 0:
                # 计算 u_r 的离散近似（forward difference）。
                du_dr = (u[i + 1, j] - u[i, j]) / dr
                # 计算 u_rr 的离散近似（基于前向 stencil）。
                u_rr = (u[i + 2, j] - 2.0 * u[i + 1, j] + u[i, j]) / (dr**2)
            # 在其余内部径向节点使用 central difference。
            else:
                # 计算 u_r 的 central difference 近似。
                du_dr = (u[i + 1, j] - u[i - 1, j]) / (2.0 * dr)
                # 计算 u_rr 的 central difference 近似。
                u_rr = (u[i + 1, j] - 2.0 * u[i, j] + u[i - 1, j]) / (dr**2)
            # 组合 polar Laplacian: u_rr + (1/r)u_r + (1/r^2)u_thetatheta。
            lap[i, j] = u_rr + (1.0 / ri) * du_dr + (1.0 / (ri**2)) * u_thetatheta
    # 返回整个网格上的离散 Laplacian。
    return lap


# 计算初始时刻位移场 u_p0 的 Laplacian。
lap_u0 = compute_laplacian_polar(u_p0, r, dr, dtheta)
# 用二阶时间精度启动公式得到第一步位移 u_p1（与 leapfrog 一致的起步）。
u_p1 = u_p0 + 0.5 * ((c * dt) ** 2) * lap_u0
# 对第一步仍施加固定边界条件。
u_p1[-1, :] = 0.0
# 估算需要保存的帧数：整除部分 + 初始帧。
Nsaved = (Nt // save_every) + 1
# 若最后一步不是整除保存点，再额外加 1 帧保存终点。
if Nt % save_every != 0:
    Nsaved += 1
# 创建三维数组 u_history 存位移历史，维度为 (r, theta, saved_time)。
u_history = np.zeros((Nr, Ntheta, Nsaved))
# 把初始位移场存到第 0 帧。
u_history[:, :, 0] = u_p0
# 用列表保存已存帧对应的时间。
t_history = [0.0]
# 历史帧写入指针从 1 开始。
i_history = 1
# 记录中心点位移历史（这里用最内层第一个角向点近似中心观测点）。
center_history = [u_p0[0, 0]]
# 记录中心位移对应的时间轴。
t_centre = [0.0]
# 时间推进循环：step 从 1 到 Nt。
for step in range(1, Nt + 1):
    # 当前物理时间 t = step * dt。
    current_time = step * dt
    # 记录当前时刻中心点位移。
    center_history.append(u_p1[0, 0])
    # 记录中心点对应时间。
    t_centre.append(current_time)
    # 到达存储步或最后一步时，把当前场写入历史。
    if (step % save_every == 0) or (step == Nt):
        # 存储当前位移场 u_p1。
        u_history[:, :, i_history] = u_p1
        # 存储对应时间。
        t_history.append(current_time)
        # 历史帧索引加 1，指向下一存储位置。
        i_history += 1
    # 只要还没到最后一步，就继续计算下一步。
    if step < Nt:
        # 计算当前位移场的 Laplacian。
        lap_u = compute_laplacian_polar(u_p1, r, dr, dtheta)
        # 使用二阶中心时间差分更新：
        # u^{n+1}=2u^n-u^{n-1}+(c*dt)^2*Laplacian(u^n)。
        u_p2 = 2.0 * u_p1 - u_p0 + ((c * dt) ** 2) * lap_u
        # 对新时刻场施加固定边界条件。
        u_p2[-1, :] = 0.0
        # 时间层推进：旧层 <- 当前层。
        u_p0 = u_p1
        # 时间层推进：当前层 <- 新层。
        u_p1 = u_p2


# 定义 DFT 函数：把 time-domain 离散信号变成 frequency-domain 复数谱。
def DFT(yn):
    # 信号长度 N，同时也是 frequency bins 数量。
    N = len(yn)
    # 基本角频率步长 w = 2*pi/N。
    w = 2 * np.pi / N

    # 创建复数数组 FTk 来存每个 k 的 DFT 结果 X_k。
    FTk = np.zeros(N, dtype=complex)

    # 外层循环遍历每个频率索引 k。
    for k in range(0, N):
        # 内层循环做求和，遍历每个时间索引 n。
        for n in range(0, N):
            # 按 DFT 定义式累加：X_k += x_n * exp(-i*k*w*n)。
            FTk[k] += np.exp(-1j * k * w * n) * yn[n]

    # 返回完整复数频谱。
    return FTk


# 定义频谱处理函数：从位移历史中取一个观测点并做 DFT。
def Transform(u_history, Nr, dt):
    # 选取观测点径向索引为中间半径位置。
    r_idx = Nr // 2
    # 选取观测点角向索引为 theta=0 对应位置。
    theta_idx = 0

    # 抽取该空间点在所有已保存时刻的 time-domain 信号。
    time_signal = u_history[r_idx, theta_idx, :]

    # 读取信号长度 N。
    N = len(time_signal)
    # 输出提示信息，提醒 DFT 计算量较大。
    print(f"Starting DFT calculation for {N} points. Please wait, this might take a moment...")

    # 调用上面的 DFT 函数获得复数频谱。
    u_fft = DFT(time_signal)

    # 取前半谱并取绝对值，得到 amplitude spectrum（实信号谱前后对称）。
    u_amplitude = np.abs(u_fft[:N // 2])

    # 采样频率 fs = 1 / 采样间隔；这里采样间隔是 dt*save_every。
    fs = 1.0 / (dt * save_every)
    # 构造频率坐标轴，从 0 到 Nyquist frequency=fs/2。
    freq_axis = np.linspace(0, fs / 2, N // 2)

    # 返回绘图和标注频谱所需变量。
    return freq_axis, u_amplitude, r_idx, theta_idx


# 执行频谱分析函数，得到频率轴、幅值谱和观测点索引。
freq_axis, u_amplitude, r_idx, theta_idx = Transform(u_history, Nr, dt)

# 为了在 x-y 平面画膜面，把 theta 做闭合（补上 2*pi 端点）以避免曲面缝隙。
theta_closed = np.append(theta, theta[0] + 2.0 * np.pi)
# 创建极坐标网格矩阵，供后续坐标变换和曲面绘图使用。
theta_grid, r_grid = np.meshgrid(theta_closed, r)
# 极坐标转直角坐标 x = r*cos(theta)。
x = r_grid * np.cos(theta_grid)
# 极坐标转直角坐标 y = r*sin(theta)。
y = r_grid * np.sin(theta_grid)

# 新建图窗用于中心点位移-时间曲线。
plt.figure(figsize=(7, 4))
# 绘制中心点位移历史，反映主振动随时间变化。
plt.plot(t_centre, center_history, color="black", linewidth=1.5)
# 设置图标题。
plt.title("Center displacement vs time")
# 设置 x 轴标签。
plt.xlabel("t [s]")
# 设置 y 轴标签。
plt.ylabel("u_center [m]")
# 显示网格线，便于读数。
plt.grid(True)
# 显示图像。
plt.show()
# 关闭当前图窗，释放资源。
plt.close()

# 提取 theta=0 切线上的 u(r,t) 数据，形成 r-t heatmap 的输入矩阵。
u_rt = u_history[:, 0, :]
# 新建图窗用于 heatmap。
plt.figure(figsize=(7, 4))
# 绘制热力图，颜色表示位移大小，横轴时间，纵轴半径。
plt.imshow(
    u_rt,
    origin="lower",
    aspect="auto",
    extent=[t_history[0], t_history[-1], r[0], r[-1]],
    cmap="viridis",
)
# 添加 colorbar 并标注位移单位。
plt.colorbar(label="u [m]")
# 设置图标题，说明这是 theta=0 处的 u(r,t)。
plt.title("Displacement heatmap u(r,t) at theta=0")
# 设置 x 轴标签。
plt.xlabel("t [s]")
# 设置 y 轴标签。
plt.ylabel("r [m]")
# 显示图像。
plt.show()
# 关闭当前图窗。
plt.close()

# 读取已保存帧数，用于 animation 循环。
N_saved_frames = u_history.shape[2]
# 控制动画抽帧步长，避免帧数过多导致播放过慢。
anim_step = max(1, N_saved_frames // 300)
# 计算全时空位移绝对值最大值，用于统一 z 轴范围。
u_max = np.max(np.abs(u_history))
# 防止全零时 z 轴范围退化为 0。
if u_max < 1.0e-12:
    u_max = 1.0e-12
# 新建 3D 图窗用于膜面动画。
fig = plt.figure(figsize=(7, 5))
# 在图窗上添加 3D 坐标轴。
ax = fig.add_subplot(111, projection="3d")
# 遍历已保存帧并按 anim_step 抽帧播放。
for saved_idx in range(0, N_saved_frames, anim_step):
    # 清空当前坐标轴内容，准备画下一帧。
    ax.clear()
    # 取当前帧位移场。
    frame_u = u_history[:, :, saved_idx]
    # 在 theta 方向补第一列实现曲面闭合，避免 0 和 2*pi 处出现裂缝。
    frame_u_closed = np.hstack((frame_u, frame_u[:, :1]))
    # 画 3D 曲面，显示当前时刻膜面形状。
    ax.plot_surface(x, y, frame_u_closed, cmap="plasma", linewidth=0.0, antialiased=True)
    # 设置当前帧标题并标出对应时间。
    ax.set_title(f"Membrane animation at t={t_history[saved_idx]:.4f} s")
    # 设置 x 轴标签。
    ax.set_xlabel("x [m]")
    # 设置 y 轴标签。
    ax.set_ylabel("y [m]")
    # 设置 z 轴标签（位移）。
    ax.set_zlabel("u [m]")
    # 固定 z 轴上下限，保证不同帧视觉尺度一致。
    ax.set_zlim(-u_max, u_max)
    # 暂停 0.03 秒以形成动画效果。
    plt.pause(0.03)
# 显示动画窗口。
plt.show()
# 关闭动画图窗。
plt.close()

# 新建图窗用于 frequency spectrum。
plt.figure(figsize=(10, 5))
# 绘制 amplitude 随频率变化的曲线。
plt.plot(freq_axis, u_amplitude, label="Frequency Spectrum", linewidth=1.5)
# 设置标题并标注观测点位置。
plt.title(rf"Frequency Spectrum at $r$={r_idx}, $\theta$={theta_idx}")

# 限制频率显示范围到 0~1000 Hz，便于看主峰。
plt.xlim(0, 1000)
# 设置 x 轴标签。
plt.xlabel("Frequency (Hz)")
# 设置 y 轴标签。
plt.ylabel("Amplitude")
# 显示网格线。
plt.grid(True)

# 在排除 0 Hz 分量后找到幅值最大的频率索引（dominant frequency）。
dominant_idx = np.argmax(u_amplitude[1:]) + 1
# 根据索引读取主频值。
dominant_freq = freq_axis[dominant_idx]

# 在主频位置画散点，突出峰值。
plt.scatter(dominant_freq, u_amplitude[dominant_idx], color="red", zorder=5)
# 画主频竖线并放入图例标签。
plt.axvline(
    x=dominant_freq,
    color="red",
    linestyle="--",
    alpha=0.7,
    label=rf"Dominant Frequency: {dominant_freq:.2f} Hz",
)

# 显示图例。
plt.legend(loc="upper right")

# 在终端输出主频数值。
print(f"Dominant Frequency: {dominant_freq:.2f} Hz")
# 在终端输出主峰幅值。
print(f"Peak Amplitude: {u_amplitude[dominant_idx]:.4f}")

# 显示频谱图。
plt.show()
# 关闭频谱图窗口。
plt.close()

# 脚本结束提示。
print("Done")
