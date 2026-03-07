# ==============================
# 代码总结
# ==============================
# 这段代码是为 polar coordinates 下的 membrane / wave-type numerical model 提供基础工具。
# 一共做了三件事：
# 1. create_polar_grid：建立 polar grid，把空间离散成 r 和 theta 两个方向的网格。
# 2. forcing_center_gaussian：建立外部 forcing term q(r, theta, t)，其空间分布是 radial Gaussian，时间分布是 sinusoidal。
# 3. compute_laplacian_polar：用 finite difference method 计算 polar coordinates 下的 Laplacian，
#    也就是离散形式的 u_rr + (1/r)u_r + (1/r^2)u_thetatheta。
# 这些函数通常会被主求解器调用，用来离散 wave equation 或类似 PDE 里的空间项和 forcing term。

import numpy as np

def create_polar_grid(R, n_r, n_theta):
    # 定义函数 create_polar_grid，输入圆域半径 R、radial 方向网格数 n_r、angular 方向网格数 n_theta
    # 这个函数的作用是建立 polar grid，并返回 r、theta、dr、dtheta
    
    """Create a polar grid with r in (0, R] and theta in [0, 2pi)."""
    # docstring，说明这个函数创建的网格范围：
    # r 从 (0, R]，theta 从 [0, 2pi)
    
    '''考虑删除这两个判断'''
    if n_r < 3:
        # 检查 radial 方向网格点数是否太少
        # 因为后面要用 central difference，所以至少需要足够的点来形成前后相邻节点
        raise ValueError("n_r must be at least 3.")
        # 如果 n_r 小于 3，就抛出错误并停止程序
    
    if n_theta < 4:
        # 检查 angular 方向网格点数是否太少
        # theta 方向后面使用 periodic indexing，所以至少要有几个点才有意义
        raise ValueError("n_theta must be at least 4.")
        # 如果 n_theta 小于 4，就抛出错误
    
    dr = R / n_r
    # 计算 radial grid spacing，记作 dr = Δr
    # 这里把半径 R 平均分成 n_r 份
    
    dtheta = 2.0 * np.pi / n_theta
    # 计算 angular grid spacing，记作 dtheta = Δtheta
    # 因为完整一圈是 2pi，所以每个角度步长是 2pi / n_theta
    
    r = (np.arange(n_r) + 1) * dr
    # 生成 radial 坐标数组 r
    # np.arange(n_r) 生成 [0, 1, 2, ..., n_r-1]
    # 加 1 后变成 [1, 2, ..., n_r]
    # 再乘 dr 后得到 [dr, 2dr, ..., n_r*dr]
    # 也就是说这里故意不取 r = 0，而是从第一个非零 radial node 开始
    
    theta = np.arange(n_theta) * dtheta
    # 生成 angular 坐标数组 theta
    # 得到 [0, dtheta, 2*dtheta, ..., (n_theta-1)*dtheta]
    # 最后一个点不会等于 2pi，这样更适合 periodic grid
    
    return r, theta, dr, dtheta
    # 返回 radial grid、angular grid，以及两个 spacing
    # 这些量会被后续 forcing 和 Laplacian 计算使用


def forcing_center_gaussian(r, theta, t, q0, sigma, omega):
    # 定义函数，用来建立 forcing term q(r, theta, t)
    # 输入包括空间网格 r、theta，当前时刻 t，以及参数 q0、sigma、omega
    
    """Build q(r, theta, t) = q0 * exp(-(r/sigma)^2) * sin(omega*t)."""
    # docstring，给出 forcing 的数学表达式
    # 这个 forcing 在空间上是 centered Gaussian，在时间上是 harmonic sinusoidal
    
    n_r = len(r)
    # 读取 radial 方向节点数
    # len(r) 就是 r 数组长度
    
    n_theta = len(theta)
    # 读取 angular 方向节点数
    # len(theta) 就是 theta 数组长度
    
    q = np.zeros((n_r, n_theta))
    # 创建一个全 0 的二维数组 q
    # shape 是 (n_r, n_theta)
    # q[i, j] 表示第 i 个 radial 点、第 j 个 angular 点上的 forcing 值
    
    harmonic = np.sin(omega * t)
    # 计算时间项 sin(omega*t)
    # 这是 forcing 的 time-dependent 部分
    # 因为这个值对所有空间点都一样，所以先算一次即可
    
    for i in range(n_r):
        # 遍历所有 radial node
        
        radial_factor = np.exp(-((r[i] / sigma) ** 2))
        # 计算当前 radial 位置上的 Gaussian factor
        # 数学上对应 exp(-(r/sigma)^2)
        # r 越接近中心，这个值越大；r 越远离中心，这个值越小
        # sigma 控制 Gaussian 的 spread
        
        for j in range(n_theta):
            # 遍历所有 angular node
            
            q[i, j] = q0 * radial_factor * harmonic
            # 计算并填入 forcing 值
            # q0 是 forcing amplitude
            # radial_factor 给出空间上的 radial 分布
            # harmonic 给出时间上的振荡
            # 因为公式里没有显式 theta dependence，所以同一个 i 下所有 j 的值相同
    
    return q
    # 返回完整的 forcing field q
    # 这个数组通常会加到 PDE 的右端项里


def compute_laplacian_polar(u, r, theta, dr, dtheta):
    # 定义函数，计算 polar coordinates 下的 Laplacian
    # 输入是 field u，以及对应的 r、dr、dtheta
    
    """Compute Laplacian in polar coordinates on the membrane grid.

    Continuous form:
    lap(u) = u_rr + (1/r)u_r + (1/r^2)u_thetatheta

    Discrete form for i=1,...,n_r-2:
    u_rr     ~ (u[i+1,j] - 2u[i,j] + u[i-1,j]) / dr^2
    u_r      ~ (u[i+1,j] - u[i-1,j]) / (2dr)
    u_tt     ~ (u[i,j+1] - 2u[i,j] + u[i,j-1]) / dtheta^2

    At i=0 (near center), use symmetry u[-1,j] = u[1,j] so du/dr at r=0 is zero.
    """
    # docstring，说明这个函数计算的连续形式和离散形式
    # 连续形式是 polar coordinates 下的 Laplacian：
    # ∇²u = u_rr + (1/r)u_r + (1/r²)u_thetatheta
    # 下面代码就是按这个 discretised equation 来实现的
    
    

    n_r = len(r)
    # 读取 radial 方向节点数
    # len(r) 就是 r 数组长度
    
    n_theta = len(theta)
    # 读取 angular 方向节点数
    # len(theta) 就是 theta 数组长度

    # 从二维数组 u 的 shape 中取出 radial 和 angular 方向的节点数
    # u.shape 返回 (行数, 列数)，这里约定行对应 r，列对应 theta
    
    lap = np.zeros((n_r, n_theta))
    # 创建一个与 u 同 shape 的全 0 数组 lap
    # 用来存放计算得到的 Laplacian 数值
    
    for i in range(n_r - 1):
        # 遍历 radial 方向节点
        # 这里只遍历到 n_r - 2，因为最后一个 radial 点通常是边界点
        # 边界点在后面直接单独处理
        
        r_i = r[i]
        # 取出当前 radial node 对应的物理半径 r_i
        # 这个值会用在 (1/r)u_r 和 (1/r²)u_thetatheta 里
        
        for j in range(n_theta):
            # 遍历 angular 方向所有节点
            
            j_plus = (j + 1) % n_theta
            # 计算 j 的下一个 angular index
            # 用 modulo 实现 periodic boundary condition in theta
            # 当 j 是最后一个点时，j_plus 会回到 0
            
            j_minus = (j - 1) % n_theta
            # 计算 j 的上一个 angular index
            # 同样用 modulo 保证 theta 方向周期性
            
            d2u_dtheta2 = (u[i, j_plus] - 2.0 * u[i, j] + u[i, j_minus]) / (dtheta**2)
            # 用 central difference 计算 second derivative in theta
            # 数学上对应 u_thetatheta
            # 这是 polar Laplacian 中的 angular 部分
            
            if i == 0:
                # 单独处理最靠近中心的 radial node
                # 因为这里没有 i-1 这个真实的网格点，所以不能直接套普通 central difference
                
                u_im1 = u[i + 1, j]
                # 这里用 symmetry 来近似虚拟点
                # 根据注释里的假设，u[-1,j] = u[1,j]
                # 所以把“左边的虚拟点”取成和右边点一样
                
                du_dr = (u[i + 1, j] - u_im1) / (2.0 * dr)
                # 计算 first derivative in r
                # 因为 u_im1 被设成 u[i+1,j]，所以这里结果实际上是 0
                # 这就对应中心对称条件 du/dr = 0
                
                d2u_dr2 = (u[i + 1, j] - 2.0 * u[i, j] + u_im1) / (dr**2)
                # 计算 second derivative in r
                # 使用对称虚拟点代替 i-1 节点
                # 这样可以在接近 r = 0 的位置保持离散格式
            
            else:
                # 对于一般 interior radial nodes，使用标准 central difference
                
                du_dr = (u[i + 1, j] - u[i - 1, j]) / (2.0 * dr)
                # 计算 first derivative in r
                # 数学上对应 u_r
                
                d2u_dr2 = (u[i + 1, j] - 2.0 * u[i, j] + u[i - 1, j]) / (dr**2)
                # 计算 second derivative in r
                # 数学上对应 u_rr
            
            lap[i, j] = d2u_dr2 + (1.0 / r_i) * du_dr + (1.0 / (r_i**2)) * d2u_dtheta2
            # 把三个部分组合起来，得到 polar Laplacian
            # 对应离散形式：
            # ∇²u = u_rr + (1/r)u_r + (1/r²)u_thetatheta
            # 这是许多 PDE（例如 wave equation、diffusion equation）中的核心空间项
    
    lap[-1, :] = 0.0
    # 将最外层 radial boundary 的 Laplacian 直接设为 0
    # 这通常表示 boundary 不在这里计算 interior stencil
    # 对固定边界 membrane 来说，真实求解器一般还会另外施加 u = 0 的 boundary condition
    
    return lap
    # 返回计算好的 Laplacian array
    # 这个结果通常会被 time-stepping solver 用来更新下一时刻的解