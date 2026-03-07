# ==============================
# 代码总结
# ==============================
# 这段脚本负责 membrane PDE 的 time-stepping，也就是让数值解随着时间一步一步往前推进。
# 它主要做四件事：
# 1. initialize_state：建立初始位移 u 和初始速度 v。
# 2. apply_boundary_conditions：施加 boundary conditions，这里是膜边缘 clamped。
# 3. forward_euler_step：用 Forward Euler method 对一阶系统做单步时间推进。
# 4. run_simulation：组织完整 simulation，包括建网格、初始化、逐步推进、保存结果。
#
# 这个脚本对应的连续方程通常是 membrane wave equation 先改写成 first-order system：
#     u_t = v
#     v_t = c^2 * lap(u) + q / rho_s
# 然后使用 Forward Euler discretisation：
#     u^(n+1) = u^n + dt * v^n
#     v^(n+1) = v^n + dt * (c^2 * lap(u^n) + q^n / rho_s)
# 其中空间项 lap(u) 由 src.discretization 里的 polar finite difference 提供。

"""Time-stepping routines for the membrane PDE."""
# module 的 docstring，说明这个脚本包含 membrane PDE 的 time-stepping 相关函数

import numpy as np
# 导入 numpy，并简写成 np
# 这个脚本中用它来创建数组、做 copy、以及把 list 转成 numpy array

from src.config import add_derived_parameters
# 从 src.config 导入 add_derived_parameters
# 这个函数通常用于根据用户输入参数，补充推导参数，例如 dt、n_steps 或 stability 相关量

from src.discretization import compute_laplacian_polar, create_polar_grid, forcing_center_gaussian
# 从 src.discretization 导入三个函数
# compute_laplacian_polar：计算 polar coordinates 下的 Laplacian
# create_polar_grid：建立 polar grid
# forcing_center_gaussian：生成 forcing term q(r, theta, t)
# 说明这个 time-stepping 脚本依赖 discretization 脚本提供的空间离散工具


def initialize_state(r, theta, initial_u_amp=0.0, initial_u_width=0.02):
    # 定义初始化函数
    # 输入 polar grid 的 r、theta，以及初始位移的 amplitude 和 width
    # 输出初始 displacement field u 和初始 velocity field v
    
    """Create initial displacement and velocity fields."""
    # docstring，说明此函数用于建立初始位移场和速度场
    
    n_r = len(r)
    # 获取 radial 方向的节点数
    
    n_theta = len(theta)
    # 获取 angular 方向的节点数
    
    u = np.zeros((n_r, n_theta))
    # 创建初始 displacement field u
    # 所有点先设为 0，shape 是 (n_r, n_theta)
    
    v = np.zeros((n_r, n_theta))
    # 创建初始 velocity field v
    # 所有点先设为 0，表示初始速度默认为零
    
    if initial_u_amp != 0.0:
        # 如果初始位移 amplitude 不等于 0，则构造一个非零初始位移场
        # 如果等于 0，就保持 u 全 0
        
        for i in range(n_r):
            # 遍历每个 radial node
            
            radial_factor = np.exp(-((r[i] / initial_u_width) ** 2))
            # 计算 radial Gaussian factor
            # 数学形式是 exp(-(r/width)^2)
            # 表示初始位移在中心附近最大，往外快速衰减
            
            for j in range(n_theta):
                # 遍历每个 angular node
                
                u[i, j] = initial_u_amp * radial_factor
                # 给当前网格点赋初始位移
                # 因为这里没有 theta dependence，所以同一个 i 下所有 j 值相同
                # 初始位移是 axisymmetric Gaussian bump
    
    return u, v
    # 返回初始化后的 displacement field 和 velocity field


def apply_boundary_conditions(u, v):
    # 定义施加 boundary conditions 的函数
    # 输入当前位移 u 和速度 v，输出施加边界条件后的 u 和 v
    
    """Apply membrane boundary conditions."""
    # docstring，说明这个函数用于施加膜的边界条件
    
    # Outer rim r=R is clamped.
    # 注释说明：最外圈边界 r=R 是 clamped boundary
    # 对 clamped membrane，边界位移为 0，边界速度也设为 0
    
    u[-1, :] = 0.0
    # 将最外层 radial 节点上所有 theta 位置的位移设为 0
    # -1 表示最后一行，也就是 r=R 的边界
    
    v[-1, :] = 0.0
    # 将最外层 radial 节点上所有 theta 位置的速度设为 0
    # 这样可保证边界始终不动
    
    return u, v
    # 返回施加了 boundary conditions 的 u 和 v


def forward_euler_step(u, v, r, dr, dtheta, dt, c, rho_s, q):
    # 定义单步时间推进函数
    # 输入当前时刻的 u、v 以及网格参数、time step、物理参数和 forcing q
    # 输出下一时刻的 u_new、v_new
    
    """Advance one time step with Forward Euler on the first-order system.

    Continuous system:
    u_t = v
    v_t = c^2 * lap(u) + q/rho_s

    Discrete Forward Euler:
    u^(n+1) = u^n + dt * v^n
    v^(n+1) = v^n + dt * (c^2 * lap(u^n) + q^n/rho_s)
    """
    # docstring 给出了连续形式和离散形式
    # 这说明原始 PDE 已被改写为 first-order system，然后用 Forward Euler 做 time discretisation
    
    lap_u = compute_laplacian_polar(u, r, dr, dtheta)
    # 计算当前时刻位移场 u 的 Laplacian
    # 这是空间离散项，对应公式中的 lap(u)
    # 具体的 finite difference discretisation 在 src.discretization 中实现
    
    u_new = u + dt * v
    # Forward Euler 更新位移
    # 对应离散公式：u^(n+1) = u^n + dt * v^n
    # 这里直接用当前速度 v 来推进位移
    
    v_new = v + dt * ((c**2) * lap_u + q / rho_s)
    # Forward Euler 更新速度
    # 对应离散公式：v^(n+1) = v^n + dt * (c^2 * lap(u^n) + q^n / rho_s)
    # (c**2) * lap_u 是 membrane 的 restoring / propagation 项
    # q / rho_s 是 external forcing per unit surface mass
    
    return apply_boundary_conditions(u_new, v_new)
    # 对更新后的解立即施加 boundary conditions
    # 并返回处理后的新位移和新速度


def run_simulation(parameters):
    # 定义完整 simulation 的主函数
    # 输入一个 parameters dictionary，输出整个 simulation 的结果
    
    """Run the full membrane simulation."""
    # docstring，说明这个函数负责运行完整的 membrane simulation
    
    params = add_derived_parameters(parameters)
    # 调用 add_derived_parameters，对输入参数进行补充和整理
    # 例如可能会自动计算 n_steps、save_every 或其他 derived parameters
    # 这样后续代码统一用 params 这个完整参数集
    
    r, theta, dr, dtheta = create_polar_grid(params["R"], params["n_r"], params["n_theta"])
    # 创建 polar grid
    # 输入半径 R、radial 节点数 n_r、angular 节点数 n_theta
    # 输出空间网格 r、theta 以及 spacing dr、dtheta
    
    u, v = initialize_state(
        r,
        theta,
        initial_u_amp=params["initial_u_amp"],
        initial_u_width=params["initial_u_width"],
    )
    # 初始化 displacement field u 和 velocity field v
    # 初始位移用 Gaussian bump 形式，初始速度默认为 0
    # 这里使用 parameters 中给定的 amplitude 和 width
    
    u, v = apply_boundary_conditions(u, v)
    # 在初始时刻先施加一次 boundary conditions
    # 保证起始状态就满足 clamped edge
    
    snapshots = [u.copy()]
    # 创建一个 list 用来保存若干时刻的 displacement snapshot
    # 一开始先保存 t=0 时刻的 u
    # 用 copy() 是为了保存当前数组的独立副本，避免后面 u 更新时影响旧结果
    
    snapshot_times = [0.0]
    # 创建一个 list 保存每个 snapshot 对应的时间
    # 初始时刻是 0.0
    
    center_history = [u[0, 0]]
    # 创建一个 list 保存中心附近某一点的位移历史
    # 这里取 u[0,0]，也就是最内层 radial node、第一个 theta node
    # 用于后续画 time history
    
    center_time = [0.0]
    # 创建一个 list 保存 center_history 对应的时间
    # 初始时刻也是 0.0
    
    for step in range(1, params["n_steps"] + 1):
        # 开始 time-stepping loop
        # step 从 1 到 n_steps，表示总共推进这么多步
        
        t_n = (step - 1) * params["dt"]
        # 计算当前这一步推进前的时间 t_n
        # 因为 step=1 时对应从第 0 步推进到第 1 步，所以当前旧解的时间是 (step-1)*dt
        
        q = forcing_center_gaussian(r, theta, t_n, params["q0"], params["sigma"], params["omega"])
        # 在当前时间 t_n 生成 forcing field q(r, theta, t_n)
        # 这个 q 会进入 v_t 方程右端项
        # 空间上是 Gaussian，时间上是 sin(omega*t)
        
        u, v = forward_euler_step(
            u,
            v,
            r,
            dr,
            dtheta,
            params["dt"],
            params["c"],
            params["rho_s"],
            q,
        )
        # 调用单步 Forward Euler 更新
        # 输入当前 u、v、网格参数、物理参数和 forcing q
        # 输出下一时刻的 u、v
        # 这一步就是数值时间推进的核心
        
        current_time = step * params["dt"]
        # 计算更新后解所在的当前时间
        # 也就是 t_(n+1)
        
        center_history.append(u[0, 0])
        # 记录当前时刻中心附近点的位移
        # 这样最终可以得到 displacement-time history
        
        center_time.append(current_time)
        # 记录与 center_history 对应的时间
        
        if step % params["save_every"] == 0:
            # 每隔 save_every 步保存一次完整 displacement field
            # % 是取余运算，等于 0 说明刚好到保存节点
            
            snapshots.append(u.copy())
            # 保存当前 displacement field 的副本
            
            snapshot_times.append(current_time)
            # 保存该 snapshot 对应的时间
    
    return {
        "r": r,
        # 返回 radial grid，后续画图或 post-processing 会用到
        
        "theta": theta,
        # 返回 angular grid
        
        "snapshots": snapshots,
        # 返回保存的 displacement snapshots 列表
        
        "snapshot_times": snapshot_times,
        # 返回每个 snapshot 对应的时间
        
        "center_time": np.array(center_time),
        # 将 center_time 从 Python list 转成 numpy array
        # 这样更方便后续数值处理和画图
        
        "center_history": np.array(center_history),
        # 将 center_history 从 Python list 转成 numpy array
        # 用于分析中心点的 time response
        
        "parameters": params,
        # 返回完整参数字典，方便结果和参数一起保存
        
    }
    # 返回一个 dictionary，包含 simulation 所需的全部主要结果