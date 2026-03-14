# 这份脚本的总体作用：
# 1) 在 polar coordinates 上离散一个圆形 membrane 的 free vibration 问题；
# 2) 使用 explicit finite difference + central difference in time 推进位移 u(r, theta, t)；
# 3) 对稳定性做 CFL-like 检查，对结果做 NaN/Inf 与幅值增长检查；
# 4) 输出最终时刻的 contour、3D surface、以及中心点位移时程图。
# 5) 加入Fourier Transform功能 by Terry

# %%
# 这一段导入脚本需要的标准库和第三方库。

# 从 pathlib 导入 Path，用于跨平台处理文件路径与输出目录。
from pathlib import Path

# 导入 matplotlib 主包（先设后端再导入 pyplot 是常见做法）。
import matplotlib
# 导入 numpy，后续用于 array、向量化计算和数值函数。
import numpy as np

# 设置 matplotlib 后端为 Agg（非交互后端），便于在脚本环境直接保存图片。
'''似乎没有学过'''
# 导入 pyplot，用于画图 API。
import matplotlib.pyplot as plt


# %%
# 这一段定义数值模型会用到的参数（几何、网格、时间步长、材料参数等）。

# 定义圆形 membrane 半径 R，单位 m。
R = 0.02  # membrane radius [m]
# 定义径向网格点数 n_r（包含 clamped 边界点）。
n_r = 60  # number of radial grid points (includes clamped edge)
# 定义角向网格点数 n_theta（theta 方向离散点个数）。
n_theta = 30  # number of angular grid points
# 定义时间步长 dt，单位 s。
dt = 1e-6  # time step [s]
# 定义总模拟时长 t_end，单位 s。
# CHANGE 1: increase total simulation time to improve frequency resolution (df = 1 / t_end).
t_end = 0.05  # end time [s]
# 每隔多少个 time steps 保存一个 snapshot。
save_every = 100  # save one snapshot every N steps
# 定义膜张力（单位长度）T，单位 N/m。
T = 100.0  # membrane tension per unit length [N/m]
# 定义面密度 rho_s，单位 kg/m^2。
rho_s = 0.35  # surface density [kg/m^2]
# 定义外载幅值 q0（本脚本中未实际进入推进公式）。
# 定义外载径向宽度 sigma（本脚本中未实际使用）。
# 定义外载角频率 omega（本脚本中未实际使用）。
# 定义初始位移幅值 initial_u_amp，单位 m。
initial_u_amp = 1.0e-3  # free-vibration initial amplitude [m]
# 定义初始位移宽度 initial_u_width（本脚本中未实际使用）。

# 由物理参数推导波速 c = sqrt(T / rho_s)，对应 membrane wave equation 的传播速度。
c = (T / rho_s) ** 0.5
# 计算总时间步数 n_steps = round(t_end / dt)。
n_steps = int(round(t_end / dt))




# 这个 docstring 说明了网格定义：r 在 (0, R]，theta 在 [0, 2pi)。
"""Create a polar grid with r in (0, R] and theta in [0, 2pi).
R must be positive.
at least 3 points needed
n_theta must be at least 4."""

# 径向步长 dr = R / n_r。
dr = R / n_r
# 角向步长 dtheta = 2*pi / n_theta。
dtheta = 2.0 * np.pi / n_theta

# 构造径向坐标数组 r：从 dr 到 R，避免 r=0（可减少 1/r 项奇异问题）。
r = (np.arange(n_r) + 2) * dr  # Avoid r=0
# 构造角向坐标数组 theta：0, dtheta, ..., (n_theta-1)*dtheta。
theta = np.arange(n_theta) * dtheta

# 用数组长度覆盖 n_r，保证与实际网格一致。
n_r = len(r)
# 用数组长度覆盖 n_theta，保证与实际网格一致。
n_theta = len(theta)



# 这是一个独立字符串字面量表达式；Python 会执行但不产生实际计算效果（no-op）。
'''后面只引用到这个函数一次，可以考虑简化，不定义函数。同时怀疑不应该放到这个cell'''
# 定义函数：分配完整位移历史 u_history，维度为 (n_r, n_theta, n_steps+1)。
'''plot时未用上完整位移历史，需要增加动态展示'''
def create_u_history(n_r, n_theta, n_steps):
    # 函数说明：返回 3D array，存储每个时间层的位移场 u(r,theta,t)。
    """Allocate full displacement history u(r,theta,t)."""

    # 用 zeros 初始化，初值全 0，后续逐步写入每个 time step 的结果。
    return np.zeros((n_r, n_theta, n_steps + 1))




# %%
# 这一段设置初值和边界条件。

# A 是初始位移的幅值（从 initial_u_amp 读取）。
A = initial_u_amp # initial condition

# u_prev 表示上一个时间层（初始时刻 t=0）的位移场，先初始化为 0。
u_prev = np.zeros((n_r, n_theta))




# 遍历每个径向点 i。
for i in range(n_r):
    u_prev[i, :] = A * (1.0 - (r[i] / R) ** 2)
    # 赋初值：u(r,0)=A*(1-(r/R)^2)，即轴对称抛物线分布（与 theta 无关）。
  
# 对最外环施加 clamped 边界条件：u(R,theta,t)=0（此处先用于 t=0）。
u_prev[-1, :] = 0.0 # boundary condition








# %%
# 这一段实现核心离散算子和时间推进。


# 定义函数：在 polar grid 上计算 Laplacian(u)。
def compute_laplacian_polar(u, r, dr, dtheta):
    # 函数说明：输出与 u 同形状的 lap array。
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
    # At the first ring (i = 0), use a symmetry-based ghost treatment.
    # This enforces du/dr = 0 at r = 0 in a simple, stable way for long free-vibration runs.


    # 从 u 的 shape 中读出当前 n_r, n_theta。
    n_r, n_theta = u.shape

    # 初始化 lap，形状与 u 相同。
    lap = np.zeros_like(u)

    '''此处类似d2u_dr2等命名需要修改为和课件统一'''

    # 径向循环到 n_r-2（不直接更新最后一环，最后一环是边界）。
    for i in range(n_r - 1):
        # 当前半径 r_i，用于 (1/r) 与 (1/r^2) 项。
        r_i = r[i]
        # 角向遍历所有 j。
        for j in range(n_theta):
            # 周期边界下的 j+1（超出末端回到 0）。
            j_plus = (j + 1) % n_theta
            # 周期边界下的 j-1（Python 负索引也可行，这里统一取模）。
            j_minus = (j - 1) % n_theta
            # 二阶角向导数离散：u_thetatheta ~ (u[j+1]-2u[j]+u[j-1]) / dtheta^2。
            d2u_dtheta2 = (u[i, j_plus] - 2.0 * u[i, j] + u[i, j_minus]) / (dtheta**2)

            # 特殊处理最内环 i=0（靠近 r=0）。
            if i == 0:
                # Mirror across the centreline: ghost value equals the next interior value.
                # This gives du/dr = 0 at the centre for axisymmetric regularity.
                # 令 ghost 点值等于 i+1 处，等价施加中心对称条件。
                u_im1 = u[i + 1, j]
                # 中心处的一阶径向导数离散（这里会变为 0）。
                du_dr = (u[i + 1, j] - u_im1) / (2.0 * dr)
                # 中心处二阶径向导数离散。
                d2u_dr2 = (u[i + 1, j] - 2.0 * u[i, j] + u_im1) / (dr**2)
            else:
                # 内部普通网格点的一阶径向导数 central difference。
                du_dr = (u[i + 1, j] - u[i - 1, j]) / (2.0 * dr)
                # 内部普通网格点的二阶径向导数 central difference。
                d2u_dr2 = (u[i + 1, j] - 2.0 * u[i, j] + u[i - 1, j]) / (dr**2)

            # 按 polar Laplacian 公式合成离散值。
            lap[i, j] = d2u_dr2 + (1.0 / r_i) * du_dr + (1.0 / (r_i**2)) * d2u_dtheta2

    # 外边界 ring 直接设为 0，对应 clamped 边界下的处理。
    lap[-1, :] = 0.0
    # 返回 Laplacian array。
    return lap

'''再次确认，ValueError是否学过，如果没学过，应该仍然保留合法性检查，还是直接删除？前面的代码已经删除了一些合法性检查了'''
# 检查 save_every 合法性（必须是正整数语义）。
if save_every <= 0:
    # 非法时抛错。
    raise ValueError("save_every must be a positive integer.")
# 检查时间步长 dt 必须为正。
if dt <= 0.0:
    # 非法时抛错。
    raise ValueError("dt must be positive.")
# 检查空间步长与最小半径均为正。
if dr <= 0.0 or dtheta <= 0.0 or r[0] <= 0.0:
    # 非法时抛错。
    raise ValueError("dr, dtheta and r_min must be positive.")
# 检查波速 c 不可为负。
if c < 0.0:
    # 非法时抛错。
    raise ValueError("c must be non-negative.")

# 若 c=0，则系统不传播波，稳定上 dt 上限视为无穷。
if c == 0.0:
    # 直接给出无限上限。
    dt_max = np.inf
else:
    # 设置稳定裕度系数 safety_factor（小于 1 更保守）。
    safety_factor = 0.2
    # 检查 safety_factor 范围。
    if safety_factor <= 0.0 or safety_factor > 1.0:
        # 非法时抛错。
        raise ValueError("safety_factor must be in (0, 1].")

    # Conservative CFL-like condition near the smallest angular arc length.
    # (c*dt/dr)^2 + (c*dt/(r_min*dtheta))^2 <= 1
    # 组合径向和角向 spacing 的倒数平方和。
    inverse_spacing_sq = (1.0 / (dr**2)) + (1.0 / ((r[0] * dtheta) ** 2))
    # 由 CFL-like 关系得到理论时间步上限 dt_cfl。
    dt_cfl = 1.0 / (c * np.sqrt(inverse_spacing_sq))
    # 实际允许上限乘 safety_factor。
    dt_max = safety_factor * dt_cfl

# 若用户给的 dt 超过上限，终止并提示不稳定。
if dt > dt_max:
    # 抛错信息中给出 dt 与估计稳定上限。
    raise ValueError(
        f"Unstable time step for explicit scheme: dt={dt:.3e}, "
        f"estimated stable limit is about {dt_max:.3e}. "
        "Reduce dt or use a coarser grid / lower wave speed."
    )

# Released from rest: u_t(r,theta,0) = 0
# First step from central difference in time:
# u^1 = u^0 + 0.5 * (c*dt)^2 * lap(u^0)
# 用初始位移场计算初始 Laplacian。
lap_u_initial = compute_laplacian_polar(u_prev, r, dr, dtheta)
# 根据 central difference 首步公式求 u_curr (即 t=dt 时刻)。
u_curr = u_prev + 0.5 * ((c * dt) ** 2) * lap_u_initial
# 强制外边界位移为 0，维持 clamped 条件。
u_curr[-1, :] = 0.0

# 分配完整历史数组，用于保存所有时间层位移。
u_history = create_u_history(n_r, n_theta, n_steps)
# 存储 t=0 的位移场。
u_history[:, :, 0] = u_prev

# 初始化快照列表，先放入初始位移。
snapshots = [u_prev.copy()]
# 初始化快照时间列表，初始时刻为 0。
snapshot_times = [0.0]
# 记录中心点位移（这里取 i=0, j=0）。
center_history = [u_prev[0, 0]]
# 记录中心点对应时间。
center_time = [0.0]

# Step index matches stored time level n, with time t = n*dt.
# 从 step=1 到 n_steps 进行显式时间推进。
for step in range(1, n_steps + 1):
    # 保存当前时间层位移场。
    u_history[:, :, step] = u_curr
    # 计算当前物理时间。
    current_time = step * dt
    # 追加中心点位移。
    center_history.append(u_curr[0, 0])
    # 追加中心点时间。
    center_time.append(current_time)

    # Always keep the true final state at t_end in snapshots.
    # 到达采样间隔或最后一步时，保存 snapshot。
    if (step % save_every == 0) or (step == n_steps):
        # 拷贝当前位移场加入快照。
        snapshots.append(u_curr.copy())
        # 记录对应快照时间。
        snapshot_times.append(current_time)

    # Advance only if another stored time level is needed.
    # 不是最后一步时，继续计算下一层 u_next。
    if step < n_steps:
        # 计算当前层的 Laplacian。
        lap_u = compute_laplacian_polar(u_curr, r, dr, dtheta)
        # 核心时间推进公式：
        # u^{n+1} = 2u^n - u^{n-1} + (c*dt)^2 * lap(u^n)。
        u_next = 2.0 * u_curr - u_prev + ((c * dt) ** 2) * lap_u
        # 对下一层同样强制 clamped 边界。
        u_next[-1, :] = 0.0
        # 时间层滚动：旧的当前层变为上一层。
        u_prev = u_curr
        # 更新时间层：下一层变为当前层。
        u_curr = u_next

# %%
# 这一段做后处理检查，防止把数值爆炸结果直接拿去画图。


# 定义允许的最大增长倍率阈值。
growth_factor_limit = 1.0e4

# 取最后一个 snapshot 作为最终位移场。
final_u = snapshots[-1]
# 取中心点位移时程。
center_history = np.array(center_history)

# 检查是否存在 NaN/Inf（显式格式常见不稳定征兆）。
if not np.isfinite(final_u).all() or not np.isfinite(center_history).all():
    # 若存在无效数，抛出异常并提示可能是 dt 过大。
    raise ValueError(
        "Simulation output contains NaN/Inf values. "
        "This usually means the time step is unstable for explicit integration."
    )

# 取初始位移场，用于与最终幅值比较。
initial_u = snapshots[0]
# 初始最大绝对位移。
initial_max = float(np.max(np.abs(initial_u)))
# 最终最大绝对位移。
final_max = float(np.max(np.abs(final_u)))
# 设置比较基线，避免除零或极小数问题。
baseline = max(initial_max, 1.0e-12)

# 若最终幅值过度增长，视为潜在不稳定并终止。
if final_max > growth_factor_limit * baseline:
    # 给出增长前后幅值，帮助调参。
    raise ValueError(
        "Simulation output grew excessively before plotting "
        f"(|u| from {initial_max:.3e} to {final_max:.3e}). "
        "Reduce dt or review stability settings."
    )

# 读取最终快照的时间用于图标题显示。
final_t = snapshot_times[-1]


#这一段是傅里叶变化的代码， 在Jupyter Notebook中放在post-processing里
#定义 DFT 函数，接收参数 yn，也就是 time domain 下的离散信号 (discrete signal)。(这个函数来自Session 9的示例)
def DFT(yn):
    # 获取输入信号 yn 的长度 N，代表总共有多少个时间采样点。
    N = len(yn)

    # 计算基频的角频率步长 (fundamental angular step size) w。
    # 对应数学公式中的 omega = 2*pi / N。
    w = 2 * np.pi / N

    # 创建一个长度为 N 的一维 array，数据类型为 complex (复数)。
    # 傅里叶变换的结果包含实部和虚部 (real and imaginary parts)，分别对应波的 amplitude（幅值）和 phase（相位）。
    # np.zeros 会将它们全部初始化为 0 + 0j。
    FTk = np.zeros(N, dtype=complex)

    # 外层 for 循环：遍历所有的频率索引 k (从 0 到 N-1)。
    for k in range(0, N):
        
        # 内层 for 循环：对于每一个频率 k，遍历所有的时间索引 n (从 0 到 N-1)。
        for n in range(0, N):
            
            # 这是 DFT 的核心数学公式：X_k = sum( x_n * e^(-i * k * w * n) )。
            # 在 Python 中，1j 代表数学中的虚数单位 i。
            # np.exp() 计算自然常数 e 的指数。
            # 这一行将当前时间点 n 经过旋转因子运算后的值，累加到当前频率 k 的结果变量 FTk[k] 中。
            FTk[k] += np.exp(-1j * k * w * n) * yn[n]
            
    # 双重循环结束后，返回装满频域复数数据的 array FTk。
    return FTk

#定义包装函数 Transform
def Transform(u_history, n_r, dt, output_dir="outputs"):
    # --- 1. Select an observation point (选择观察点) ---

    # 我们需要在膜上选一个点来观察它的振动。这里选择半径中点。
    # // 是 Python 中的整数除法 (integer division)，比如 21 // 2 结果是 10。
    # array 的索引 (index) 必须是整数，所以不能用单斜杠 /。
    r_idx = n_r // 2 

    # 选择角度方向的第一个索引 (theta = 0 的位置)。
    theta_idx = 0 

    # 从三维矩阵 u_history 中提取这一个空间点在所有时间步的数据。
    # 语法 [r_idx, theta_idx, :] 中的冒号 ':' 表示“提取这一个维度上的所有数据”。
    # 提取出来的 time_signal 是一个一维 array，相当于这个点的位移-时间曲线。
    time_signal = u_history[r_idx, theta_idx, :]

    # --- 2. Perform Temporal Discrete Fourier Transform (执行时域离散傅里叶变换) ---

    # 计算提取出来的时域信号的总点数 N。
    N = len(time_signal)

    # 打印提示信息。因为 O(N^2) 的自定义 DFT 很慢，这行字可以让你知道程序没死机，只是在狂算。
    print(f"Starting DFT calculation for {N} points. Please wait, this might take a moment...")

    # 调用上面定义的老师的 DFT 函数，将时域信号 time_signal 转化为频域信号 u_fft。
    # u_fft 里面现在全都是 complex numbers (复数)。
    u_fft = DFT(time_signal)

    # 我们通常只关心振动幅度 (amplitude)，不关心相位。np.abs() 可以计算复数的模长 (magnitude/absolute value)。
    # 为什么只取前半部分 [:N // 2]？
    # 根据奈奎斯特采样定理 (Nyquist theorem)，对于实数信号，DFT 的后半段只是前半段的对称镜像，没有新的物理意义。
    u_amplitude = np.abs(u_fft[:N // 2]) 

    # --- 3. Frequency Mapping (频率轴映射) ---

    # 计算采样频率 (sampling frequency) fs。
    # 它等于时间步长 dt 的倒数 (1 / dt)。代表在模拟中，现实时间的每一秒钟我们采样了多少次。
    fs = 1.0 / dt 

    # 生成画图用的 x 轴坐标 (频率，单位 Hz)。
    # np.linspace(起点, 终点, 数量) 用于生成等差数列。
    # 频率范围从 0 到 奈奎斯特频率 (fs / 2)，总共生成 N // 2 个点，与上面的幅值 array 一一对应。
    freq_axis = np.linspace(0, fs / 2, N // 2)

    # --- 4. Visualization (数据可视化) ---

    # 创建一个 matplotlib 的 Figure 对象，也就是画板，设置尺寸为宽 10 英寸，高 5 英寸。
    plt.figure(figsize=(10, 5))

    # 在画板上绘制折线图 (line plot)。x 轴是刚刚生成的频率，y 轴是对应的振幅。
    plt.plot(freq_axis, u_amplitude,label='Frequency Spectrum',linewidth=1.5)

    # 给图表添加标题，标题中用到 f-string 把刚才选取的网格点索引动态打印上去。
    plt.title(rf'Frequency Spectrum at $r$={r_idx}, $\theta$={theta_idx}')

    # 限制 x 轴的显示范围为 0 到 1000 Hz。
    # 这是因为波的能量通常集中在低频的固有频率 (natural frequencies) 上，高频部分通常是数值噪音 (numerical noise)。
    plt.xlim(0, 1000)

    # 设置 x 轴和 y 轴的文字标签。
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    # 开启背景网格线 (grid)，方便读数。
    plt.grid(True)

    # --- 5. Find and Print Dominant Frequency (寻找并打印主频) ---

    # 找到振幅数组中最大值的索引 (Index)
    # CHANGE 2: ignore DC component when searching the dominant oscillation frequency.
    search_amplitude = u_amplitude.copy()
    if len(search_amplitude) > 0:
        search_amplitude[0] = 0.0
    dominant_idx = np.argmax(search_amplitude)

    # 根据索引在频率轴数组中找到对应的频率值
    dominant_freq = freq_axis[dominant_idx]

    # 在图上画一个红色的点标记出最高峰
    plt.scatter(dominant_freq, u_amplitude[dominant_idx], color='red', zorder=5)
    # 添加垂直虚线 (Vertical line)
    plt.axvline(x=dominant_freq, color='red', linestyle='--', alpha=0.7, 
                label=rf'Dominant Frequency: {dominant_freq:.2f} Hz')
    
    #设置图例
    plt.legend(loc='upper right')

    # 打印结果
    print(f"Dominant Frequency: {dominant_freq:.2f} Hz")
    print(f"Peak Amplitude: {u_amplitude[dominant_idx]:.4f}")

    # --- 6. 保存文件 ---

    # 使用 Path 对象拼接输出文件夹和文件名。跨平台兼容性好，斜杠 / 被自动处理。
    output_path = Path(output_dir) / "fourier_spectrum_dft.png"

    # 检查输出目录是否存在。如果不存在，parents=True 允许连同父目录一起创建，exist_ok=True 保证如果目录已有不报错。
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 将刚刚画好的图表保存为图片，设置分辨率 (dpi) 为 150，保证图片清晰度。
    plt.savefig(output_path, dpi=150)

    # 在终端打印保存成功的提示和路径。
    print(f"Fourier plot successfully saved to {output_path}")

    # 使用 plt.close() 关闭当前的 Figure。
    # 这一步在循环或者大量作图时极其重要，可以释放计算机内存 (release memory)。
    plt.close()

#调用傅里叶变换函数
Transform(u_history, n_r, dt)

# %%
# 这一段负责把计算结果可视化并保存成图片。
'''需要review：
1，哪些code学过，哪些可以用
2，仍然是旧的src路径等，不确定是否有用。需要重构'''

# 候选项目根目录列表（按顺序尝试）。
candidate_roots = [
    # 当前工作目录。
    Path.cwd(),
    # 当前目录的上一级。
    Path.cwd().parent,
    # 兜底的绝对路径。
    Path(r"d:/OneDrive - Imperial College London/Documents/GitHub/ME2-Coding"),
]

# 逐个尝试根目录。
for root in candidate_roots:
    # 若发现 src 目录或已经到最后兜底项，就确定输出目录。
    if (root / "src").exists() or root == candidate_roots[-1]:
        # 输出目录为 root/outputs。
        output_dir = root / "outputs"
        # 若不存在则创建 outputs 目录。
        output_dir.mkdir(parents=True, exist_ok=True)
        # 找到后退出循环。
        break

# 定义 contour 图输出路径。
contour_path = output_dir / "contour_final.png"
# 定义 3D surface 图输出路径。
surface_path = output_dir / "surface_final.png"
# 定义中心位移时程图输出路径。
line_path = output_dir / "center_history.png"

# Contour plot
# 生成二维网格：theta_grid 与 r_grid。
# CHANGE 4: close theta periodically (append theta=2pi and duplicate the first column of final_u)
# so contour/surface do not show a wedge gap at the seam.
theta_closed = np.append(theta, theta[0] + 2.0 * np.pi)
final_u_closed = np.hstack((final_u, final_u[:, :1]))
theta_grid, r_grid = np.meshgrid(theta_closed, r)
# polar -> Cartesian: x = r cos(theta)。
x = r_grid * np.cos(theta_grid)
# polar -> Cartesian: y = r sin(theta)。
y = r_grid * np.sin(theta_grid)

# 创建 2D Figure 和 Axes。
fig, ax = plt.subplots(figsize=(6, 5))
# 画 filled contour，levels=40，色图 viridis。
contour = ax.contourf(x, y, final_u_closed, levels=40, cmap="viridis")
# 添加 colorbar，标注位移单位。
fig.colorbar(contour, ax=ax, label="u [m]")
# 设定 x/y 等比例，保持圆形外观不失真。
ax.set_aspect("equal")
# 设置标题（包含最终时刻 final_t）。
ax.set_title(f"Membrane displacement contour at t={final_t:.4f} s")
# 设置 x 轴标签。
ax.set_xlabel("x [m]")
# 设置 y 轴标签。
ax.set_ylabel("y [m]")
# 自动优化布局。
fig.tight_layout()
# 保存 contour 图片。
fig.savefig(contour_path, dpi=150)
# 关闭 Figure 释放内存。
plt.close(fig)

# Surface plot
# 创建 3D 图窗口。
fig = plt.figure(figsize=(7, 5))
# 添加 3D 坐标轴。
ax = fig.add_subplot(111, projection="3d")
# 绘制位移曲面 z = final_u。
ax.plot_surface(x, y, final_u_closed, cmap="plasma", linewidth=0.0, antialiased=True)
# 设置标题。
ax.set_title(f"Membrane displacement surface at t={final_t:.4f} s")
# 设置 x 轴标签。
ax.set_xlabel("x [m]")
# 设置 y 轴标签。
ax.set_ylabel("y [m]")
# 设置 z 轴标签。
ax.set_zlabel("u [m]")
# 自动优化布局。
fig.tight_layout()
# 保存 surface 图片。
fig.savefig(surface_path, dpi=150)
# 关闭 Figure 释放内存。
plt.close(fig)

# Center-history line plot
# 创建中心点位移时程图窗口。
fig, ax = plt.subplots(figsize=(7, 4))
# 画中心点位移-时间曲线。
ax.plot(center_time, center_history, color="black", linewidth=1.5)
# 设置标题。
ax.set_title("Center displacement vs time")
# 设置 x 轴标签（时间）。
ax.set_xlabel("t [s]")
# 设置 y 轴标签（中心位移）。
ax.set_ylabel("u_center [m]")
# 打开网格线并设透明度。
ax.grid(True, alpha=0.3)
# 自动优化布局。
fig.tight_layout()
# 保存时程图。
fig.savefig(line_path, dpi=150)
# 关闭 Figure 释放内存。
plt.close(fig)

# 在终端打印仿真完成提示。
print("Simulation finished.")
# 打印 contour 图片保存路径。
print(f"Saved: {contour_path}")
# 打印 surface 图片保存路径。
print(f"Saved: {surface_path}")
# 打印时程图保存路径。
print(f"Saved: {line_path}")


# %%
# 这一段只是收尾打印，不参与数值计算。
print('CW done: I deserve a good mark')
