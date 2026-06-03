"""
非奇异位错位移模块
Bertin & Cai (2018) 公式(10)

公式结构：
    u_m = b_m/(4π) * Ω(x)                                      第一项（立体角）
        - b_i*ε_{mik}/(4π) * t_k * ∫(2/Ra) ds                  第二项
        - b_i*ε_{ijk}/(8π(1-ν)) * t_k * ∫(∂²Ra/∂x'm∂x'j) ds   第三项

Ra = sqrt(R² + a²)，a 为非奇异核心宽度

用法示例：
    import numpy as np
    from displacement import displacement_loop

    b_mag = 2.55e-10          # 柏氏矢量大小（m），铜
    nu    = 0.324             # 泊松比
    a     = 6 * b_mag         # 非奇异核心宽度

    # 矩形位错环顶点（在 z=0 平面）
    side = 100.0 * b_mag
    vertices = [
        np.array([-side/2, -side/2, 0.0]),
        np.array([ side/2, -side/2, 0.0]),
        np.array([ side/2,  side/2, 0.0]),
        np.array([-side/2,  side/2, 0.0]),
    ]
    b         = np.array([0., 0., b_mag])         # 柏氏矢量（螺位错）
    pos_pt    = np.array([0., 0., 1.])            # 切割面正面的参考点
    x         = np.array([50*b_mag, 30*b_mag, 5*b_mag])  # 场点

    u = displacement_loop(x, vertices, b, nu, a, pos_pt)
    print(u)  # 位移矢量（m）

依赖：只需要 numpy。
"""

import numpy as np


# ── Levi-Civita 符号 ──────────────────────────────────────────────────
EPS = np.zeros((3, 3, 3))
EPS[0,1,2] = EPS[1,2,0] = EPS[2,0,1] =  1
EPS[0,2,1] = EPS[2,1,0] = EPS[1,0,2] = -1


# ════════════════════════════════════════════════════════════════════════
# 基础函数：四个积分原函数
# ════════════════════════════════════════════════════════════════════════

def Ra_val(s, c):
    """Ra = sqrt(c + s²)，c = d·d + a²"""
    return np.sqrt(c + s**2)

def F_1_Ra(s, c):
    """∫ 1/Ra ds 的原函数 = ln(s + Ra)"""
    return np.log(s + Ra_val(s, c))

def F_1_Ra3(s, c):
    """∫ 1/Ra³ ds 的原函数 = s / (c * Ra)"""
    return s / (c * Ra_val(s, c))

def F_s_Ra3(s, c):
    """∫ s/Ra³ ds 的原函数 = -1/Ra"""
    return -1.0 / Ra_val(s, c)

def F_s2_Ra3(s, c):
    """∫ s²/Ra³ ds 的原函数 = -s/Ra + ln(s + Ra)"""
    return -s / Ra_val(s, c) + np.log(s + Ra_val(s, c))

def defint(F, s1, s2, c):
    """定积分：F(s2, c) - F(s1, c)"""
    return F(s2, c) - F(s1, c)


# ════════════════════════════════════════════════════════════════════════
# 立体角计算（van Oosterom 公式）
# ════════════════════════════════════════════════════════════════════════

def solid_angle_triangle(a_vec, b_vec, c_vec):
    """
    用 van Oosterom 公式计算球面三角形的立体角。
    a_vec, b_vec, c_vec：从场点出发指向三角形三顶点的单位向量
    tan(Ω/2) = |a·(b×c)| / (1 + a·b + b·c + c·a)
    """
    numerator   = abs(np.dot(a_vec, np.cross(b_vec, c_vec)))
    denominator = 1.0 + np.dot(a_vec, b_vec) + np.dot(b_vec, c_vec) + np.dot(c_vec, a_vec)
    return 2.0 * np.arctan2(numerator, denominator)

def solid_angle_polygon(vertices, x, positive_side_point):
    """
    计算多边形对场点 x 张的有符号立体角。

    参数：
        vertices           : 多边形顶点列表（按顺序）
        x                  : 场点坐标
        positive_side_point: 切割面正面的任意参考点
                             （场点与该点同侧时 Ω > 0）
    """
    vertices = [np.asarray(v, float) for v in vertices]
    x = np.asarray(x, float)
    positive_side_point = np.asarray(positive_side_point, float)

    # 从场点到每个顶点的单位向量
    r_vecs = []
    for v in vertices:
        r = v - x
        r_vecs.append(r / np.linalg.norm(r))

    # 把多边形切成三角形，以第一个顶点为公共顶点
    Omega = 0.0
    for i in range(1, len(vertices) - 1):
        Omega += solid_angle_triangle(r_vecs[0], r_vecs[i], r_vecs[i+1])

    # 用法向量和参考点判断符号
    e1 = vertices[1] - vertices[0]
    e2 = vertices[2] - vertices[1]
    normal = np.cross(e1, e2)
    center = np.mean(vertices, axis=0)

    ref_side   = np.dot(positive_side_point - center, normal)
    field_side = np.dot(x - center, normal)

    if ref_side * field_side < 0:
        Omega = -Omega

    return Omega


# ════════════════════════════════════════════════════════════════════════
# 单段线积分（第二项 + 第三项）
# ════════════════════════════════════════════════════════════════════════

def disp_line_segment(x, x1, x2, b, nu, a):
    """
    单段线段对位移的贡献（第二项 + 第三项，不含立体角）。

    参数：
        x        : 场点坐标
        x1, x2   : 线段起点和终点
        b        : 柏氏矢量
        nu       : 泊松比
        a        : 非奇异核心宽度

    第二项：-b_i ε_{mik} / (4π) * t_k * I_1_Ra
    第三项：-b_i ε_{ijk} / (8π(1-ν)) * t_k * {
                δ_{mj} * I_1_Ra
              - d_m*d_j * I_1_Ra3
              - (d_m*t_j + t_m*d_j) * I_s_Ra3
              - t_m*t_j * I_s2_Ra3
            }
    """
    x, x1, x2, b = (np.asarray(v, float) for v in (x, x1, x2, b))

    seg = x2 - x1
    L   = np.linalg.norm(seg)
    if L < 1e-30:
        return np.zeros(3)

    t  = seg / L
    x0 = x1 + np.dot(x - x1, t) * t   # 垂足
    d  = x0 - x                        # 从场点到垂足的向量
    d2 = np.dot(d, d)

    s1 = np.dot(x1 - x0, t)            # 积分下限
    s2 = np.dot(x2 - x0, t)            # 积分上限
    c  = d2 + a**2                      # c = d·d + a²

    # 四个定积分
    I_1_Ra   = defint(F_1_Ra,   s1, s2, c)
    I_1_Ra3  = defint(F_1_Ra3,  s1, s2, c)
    I_s_Ra3  = defint(F_s_Ra3,  s1, s2, c)
    I_s2_Ra3 = defint(F_s2_Ra3, s1, s2, c)

    u = np.zeros(3)
    for m in range(3):

        # 第二项
        term2 = 0.0
        for i in range(3):
            for k in range(3):
                term2 += b[i] * EPS[m, i, k] * t[k]
        term2 *= -I_1_Ra / (4.0 * np.pi)

        # 第三项
        term3 = 0.0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    eps_b_t = EPS[i, j, k] * b[i] * t[k]
                    if abs(eps_b_t) < 1e-30:
                        continue
                    delta_mj = 1.0 if m == j else 0.0
                    bracket  = (  delta_mj * I_1_Ra
                                - d[m]*d[j]               * I_1_Ra3
                                - (d[m]*t[j] + t[m]*d[j]) * I_s_Ra3
                                - t[m]*t[j]               * I_s2_Ra3)
                    term3 += eps_b_t * bracket
        term3 *= -1.0 / (8.0 * np.pi * (1.0 - nu))

        u[m] = term2 + term3

    return u


# ════════════════════════════════════════════════════════════════════════
# 完整位移（三项之和）
# ════════════════════════════════════════════════════════════════════════

def displacement_loop(x, vertices, b, nu, a, positive_side_point):
    """
    闭合位错环在场点 x 处产生的完整位移。

    参数：
        x                  : 场点坐标
        vertices           : 位错环顶点列表（按顺序）
        b                  : 柏氏矢量
        nu                 : 泊松比
        a                  : 非奇异核心宽度
        positive_side_point: 切割面正面的参考点（用于立体角符号）

    返回：
        u : 位移矢量，shape (3,)，单位与 b、a 相同
    """
    x = np.asarray(x, float)
    b = np.asarray(b, float)
    N = len(vertices)

    # 第一项：立体角
    Omega = solid_angle_polygon(vertices, x, positive_side_point)
    u = b * Omega / (4.0 * np.pi)

    # 第二项 + 第三项：对每段线段求和
    for i in range(N):
        x1 = np.asarray(vertices[i],         float)
        x2 = np.asarray(vertices[(i+1) % N], float)
        u += disp_line_segment(x, x1, x2, b, nu, a)

    return u
