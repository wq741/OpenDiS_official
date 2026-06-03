"""
Cu 多源 Frank-Read 位错动力学模拟
初始构型生成 + 模拟运行

参数：
    盒子        : 5×5×5 μm，周期性边界
    材料        : Cu (b=2.55e-10 m, mu=48 GPa, nu=0.324)
    位错类型    : Frank-Read 源（两端固定 + 中间自由节点）
    初始密度    : 1e12 m^-2
    源长度      : 0.5~2 μm，对数均匀分布
    滑移系      : FCC 全部 12 个 {111}<110>
    加载方向    : [001]，常应变率 1000/s
    目标应变    : 1%
"""

import numpy as np
import sys, os

pyexadis_paths = ['../../python', '../../lib', '../../core/pydis/python', '../../core/exadis/python/']
for _p in pyexadis_paths:
    _ap = os.path.abspath(_p)
    if _ap not in sys.path:
        sys.path.append(_ap)

import pyexadis
from framework.disnet_manager import DisNetManager
from pyexadis_base import ExaDisNet, NodeConstraints
from pyexadis_base import CalForce, MobilityLaw, TimeIntegration
from pyexadis_base import Collision, Topology, Remesh, VisualizeNetwork, SimulateNetwork


# ════════════════════════════════════════════════════════════════════════
# 材料和模拟参数
# ════════════════════════════════════════════════════════════════════════

b_mag   = 2.55e-10   # Burgers 矢量大小 (m)，Cu
mu      = 48e9       # 剪切模量 (Pa)
nu      = 0.324      # 泊松比
a_over_b = 6.0       # 非奇异核心宽度 a = 6 * burgmag

L_phys  = 5e-6       # 盒子边长 (m)
Lbox    = L_phys / b_mag   # burgmag 单位，约 19608

strain_rate = 1e3    # 应变率 (1/s)
target_strain = 0.01 # 目标应变 1%

# ── FR 源参数 ─────────────────────────────────────────────────────────
rho_target  = 1e12   # 初始位错密度 (m^-2)
L_min_phys  = 0.5e-6 # 源最小长度 (m)
L_max_phys  = 2.0e-6 # 源最大长度 (m)
L_min = L_min_phys / b_mag
L_max = L_max_phys / b_mag

# 对数均匀分布的期望长度
mean_L = (L_max - L_min) / np.log(L_max / L_min)

V_phys   = L_phys ** 3
L_total  = rho_target * V_phys / b_mag   # 总位错长度（burgmag 单位）
n_sources = int(round(L_total / mean_L))

# ── Remesh 参数 ──────────────────────────────────────────────────────
# maxseg 取最短源长度的一半，保证最短源至少被离散成 2 段（加中间节点）
maxseg = (L_min_phys / 2.0) / b_mag   # 约 392 burgmag ≈ 0.1 μm
minseg = maxseg * 0.25


# ════════════════════════════════════════════════════════════════════════
# FCC 12 个滑移系（法向量和柏氏矢量均为单位向量）
#
# 加载方向 [001] 下的 Schmid 因子：
#   m = (n·ê_z)(b·ê_z) / (|n||b|)
#   8 个系统 |m| = 1/√6 ≈ 0.408（会被激活）
#   4 个系统  m  = 0（b 垂直于 z，短期静止，
#             长期因内应力可能局部激活）
# ════════════════════════════════════════════════════════════════════════

s  = 1.0 / np.sqrt(2)   # <110> 方向归一化系数
n3 = 1.0 / np.sqrt(3)   # {111} 法向归一化系数

SLIP_SYSTEMS = [
    # ── (111) 面 ──────────────────────────────────────────────────────
    (np.array([ 1,  1,  1]) * n3,  np.array([ 0,  1, -1]) * s),  # |m|=0.408
    (np.array([ 1,  1,  1]) * n3,  np.array([ 1,  0, -1]) * s),  # |m|=0.408
    (np.array([ 1,  1,  1]) * n3,  np.array([ 1, -1,  0]) * s),  # m=0
    # ── (-111) 面 ─────────────────────────────────────────────────────
    (np.array([-1,  1,  1]) * n3,  np.array([ 0,  1, -1]) * s),  # |m|=0.408
    (np.array([-1,  1,  1]) * n3,  np.array([ 1,  1,  0]) * s),  # m=0
    (np.array([-1,  1,  1]) * n3,  np.array([ 1,  0,  1]) * s),  # |m|=0.408
    # ── (1-11) 面 ─────────────────────────────────────────────────────
    (np.array([ 1, -1,  1]) * n3,  np.array([ 0,  1,  1]) * s),  # |m|=0.408
    (np.array([ 1, -1,  1]) * n3,  np.array([ 1,  1,  0]) * s),  # m=0
    (np.array([ 1, -1,  1]) * n3,  np.array([ 1,  0, -1]) * s),  # |m|=0.408
    # ── (11-1) 面 ─────────────────────────────────────────────────────
    (np.array([ 1,  1, -1]) * n3,  np.array([ 0,  1,  1]) * s),  # |m|=0.408
    (np.array([ 1,  1, -1]) * n3,  np.array([ 1,  0,  1]) * s),  # |m|=0.408
    (np.array([ 1,  1, -1]) * n3,  np.array([ 1, -1,  0]) * s),  # m=0
]

# Precompute dislocation line directions (t ⊥ b and t ⊥ n) for each slip system
_T_VECTORS = np.array([np.cross(sn, sb) for sn, sb in SLIP_SYSTEMS])
_T_VECTORS /= np.linalg.norm(_T_VECTORS, axis=1, keepdims=True)


# ════════════════════════════════════════════════════════════════════════
# 生成初始 Frank-Read 源构型
# ════════════════════════════════════════════════════════════════════════

def init_FR_sources(n_sources, seed=42):
    """
    随机生成 n_sources 个 Frank-Read 源，均匀分布在 12 个滑移系上。

    每个源由 3 个节点组成：
        PINNED ── FREE ── PINNED
    两端固定，中间节点自由，受力后向外弯曲触发增殖。

    返回：(rn, links)
        rn    : (3*n_sources, 4)，节点坐标和约束
        links : (2*n_sources, 8)，线段连接和 Burgers/法向信息
    """
    PINNED = NodeConstraints.PINNED_NODE
    FREE   = NodeConstraints.UNCONSTRAINED

    rng = np.random.default_rng(seed)

    rn_list    = []
    links_list = []
    node_offset = 0
    total_length = 0.0

    for _ in range(n_sources):
        # ── 随机选择滑移系 ────────────────────────────────────────────
        sys_idx = rng.integers(0, 12)
        slip_n, slip_b = SLIP_SYSTEMS[sys_idx]

        t = _T_VECTORS[sys_idx]

        # ── 对数均匀分布随机源长度 ────────────────────────────────────
        L = np.exp(rng.uniform(np.log(L_min), np.log(L_max)))
        total_length += L

        # ── 随机中心位置（保证源段不跨越边界）────────────────────────
        # 需要：center[k] ∈ [|t[k]|*L/2 + buf, Lbox - |t[k]|*L/2 - buf]
        buf     = 5.0   # 边界缓冲（5 burgmag）
        margin  = np.abs(t) * (L / 2) + buf
        low     = margin
        high    = Lbox - margin
        # 极端情况保护（理论上不会发生，因为 L_max << Lbox）
        low  = np.minimum(low,  Lbox * 0.1)
        high = np.maximum(high, Lbox * 0.9)
        center = rng.uniform(low, high)

        p0 = center - t * L / 2   # 起点（固定端）
        pm = center
        p2 = center + t * L / 2   # 终点（固定端）

        # ── 节点数组 ──────────────────────────────────────────────────
        rn_src = np.array([
            [p0[0], p0[1], p0[2], PINNED],
            [pm[0], pm[1], pm[2], FREE  ],
            [p2[0], p2[1], p2[2], PINNED],
        ])

        # ── 线段数组（两段，共享同一 Burgers 矢量和滑移面法向）────────
        i0 = node_offset
        links_src = np.array([
            [i0,   i0+1, slip_b[0], slip_b[1], slip_b[2], slip_n[0], slip_n[1], slip_n[2]],
            [i0+1, i0+2, slip_b[0], slip_b[1], slip_b[2], slip_n[0], slip_n[1], slip_n[2]],
        ])

        rn_list.append(rn_src)
        links_list.append(links_src)
        node_offset += 3

    rn    = np.vstack(rn_list)
    links = np.vstack(links_list)

    actual_density = (total_length * b_mag) / V_phys
    print(f"生成 FR 源：{n_sources} 个，分布在 12 个滑移系")
    print(f"  实际总长度 : {total_length * b_mag * 1e6:.2f} μm")
    print(f"  实际密度   : {actual_density:.3e} m^-2（目标 {rho_target:.1e}）")
    print(f"  总节点数   : {len(rn)}")
    print(f"  总线段数   : {len(links)}")

    return rn, links


# ════════════════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════════════════

def main(plot=True):
    global net, sim, state

    print("=" * 60)
    print("Cu Frank-Read 位错动力学模拟")
    print(f"  盒子    : {L_phys*1e6:.0f}×{L_phys*1e6:.0f}×{L_phys*1e6:.0f} μm  (Lbox={Lbox:.0f} burgmag)")
    print(f"  目标密度: {rho_target:.1e} m^-2  →  约 {n_sources} 个 FR 源")
    print(f"  源长范围: {L_min_phys*1e6:.1f}~{L_max_phys*1e6:.1f} μm（对数均匀）")
    print(f"  应变率  : {strain_rate:.0f} /s，目标应变 {target_strain*100:.0f}%")
    print("=" * 60)

    # ── 生成初始构型 ──────────────────────────────────────────────────
    rn, links = init_FR_sources(n_sources, seed=42)

    cell = pyexadis.Cell(h=Lbox * np.eye(3), is_periodic=[True, True, True])
    net  = DisNetManager(ExaDisNet(cell, rn, links))

    os.makedirs('output', exist_ok=True)
    net.write_json('output/init_config.json')
    print("初始构型已保存至 output/init_config.json")

    # ── 模拟参数 ──────────────────────────────────────────────────────
    state = {
        "burgmag" : b_mag,
        "mu"      : mu,
        "nu"      : nu,
        "a"       : a_over_b,    # 单位：burgmag
        "maxseg"  : maxseg,      # 单位：burgmag
        "minseg"  : minseg,      # 单位：burgmag
        "rann"    : 3.0,         # 碰撞检测半径（burgmag）；example 22用10但maxseg=2000，此处minseg=98故缩至3
        "rtol"    : 3.0,         # 节点每步最大位移（burgmag），与 rann 保持一致
        "nextdt"  : 1e-10,       # 初始时间步（s），Subcycling 从此开始自适应调整
        "maxdt"   : 1e-9,        # 时间步上限（s）
    }

    # ── 模拟组件（参照 example 22: fcc_Cu_15um_1e3）────────────────────
    # CalForce: SUBCYCLING_MODEL — 分短程/长程计算全场弹性相互作用，
    #           是 SUBCYCLING_MODEL + Subcycling 积分器的配套组合，
    #           效率最高，适合大规模生产模拟
    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state,
                         Ngrid=64, cell=cell)

    # MobilityLaw: FCC_0 — Cu/FCC 专用迁移率，
    #              区分刃/螺分量阻尼，有最大速度限制（声速分数）
    #              Medge/Mscrew 取自 OpenDiS Cu 参考参数
    mobility  = MobilityLaw(mobility_law='FCC_0', state=state,
                             Medge=64103.0, Mscrew=64103.0, vmax=4000.0)

    # TimeIntegration: Subcycling — 必须与 SUBCYCLING_MODEL 配套，
    #                  rgroups 是短程/长程力更新的距离分组边界（burgmag 单位）
    timeint   = TimeIntegration(integrator='Subcycling',
                                rgroups=[0.0, 100.0, 600.0, 1600.0],
                                state=state, force=calforce, mobility=mobility)

    # Collision: Retroactive — 每步结束后检测穿越事件，
    #            处理湮灭和结点形成，适合 FR 源增殖场景
    collision = Collision(collision_mode='Retroactive', state=state)

    # Topology: TopologyParallel — 处理节点分裂，
    #           FR 源弯曲时产生新节点必须靠这个模块，不能省略
    topology  = Topology(topology_mode='TopologyParallel', state=state,
                         force=calforce, mobility=mobility)

    # Remesh: LengthBased — 控制线段长度在 minseg~maxseg 之间
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    vis = VisualizeNetwork() if plot else None

    sim = SimulateNetwork(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh, vis=vis,
        state=state,
        loading_mode = 'strain_rate',
        erate        = strain_rate,
        edir         = np.array([0., 0., 1.]),
        max_strain   = target_strain,
        burgmag      = b_mag,
        print_freq   = 100,
        plot_freq    = 1000,
        plot_pause_seconds = 0.01,
        write_freq   = 50,
        write_dir    = 'output',
    )

    print(f"\n开始模拟，目标应变 {target_strain*100:.0f}%，应变率 {strain_rate:.0f}/s")
    sim.run(net, state)

    net.write_json('output/final_config.json')
    print("最终构型已保存至 output/final_config.json")

    return net.is_sane()


if __name__ == "__main__":
    pyexadis.initialize()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-plot', dest='plot', action='store_false', default=True)
    args = parser.parse_args()

    main(plot=args.plot)

    if not sys.flags.interactive:
        pyexadis.finalize()
