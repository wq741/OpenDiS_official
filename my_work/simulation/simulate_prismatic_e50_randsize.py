"""
Cu FCC single crystal dislocation dynamics simulation
Initial configuration: TRUE prismatic loops (b ⊥ loop plane), RANDOM side lengths
Framework: OpenDiS / ExaDiS (pyexadis)

与 simulate_prismatic_e50_s001.py 的区别
----------------------------------------
1. 环的边长不再固定为 4 μm / 1 μm，而是在 [R_min, R_max] 内对数均匀随机：
     R_max = 盒子边长 / 5，R_min = R_max / 5
     （5 μm 盒子：R_max = 1 μm，R_min = 0.2 μm）
2. 不再区分"主系环 / 共线系环"，所以没有 60/40 的线长分配；
   四个柏氏矢量按顺序轮流分配给各个环。
3. 最大线段长度 maxseg 取"最短的边（R_min）刚好分成 5 段"。
   insert_prismatic_loop() 对每条边分 ceil(边长 / maxseg) 段，
   所以每条边至少 5 段，边越长段数越多（1 μm 的边是 25 段）。
   minseg = maxseg / 4（和 e50 原脚本 500/2000 的比例一致）。
4. 输出文件夹按应变率和目标应变自动命名，
   init_config.json 也写进输出文件夹（不再覆盖 output/init_config.json）。
5. 环的中心之间加了最小距离检查（min_dist_factor，设成 0 就关闭，结果和不加检查时一样）：
   两个环的中心距离必须 >= min_dist_factor × (两个环的边长之和)，
   距离按周期性边界取最近的镜像。这只是近似的防重叠办法，不保证环之间不相交。

For FCC with b = 1/2<110>:
  * loop plane normal  ||  b
  * 4 sides alternate between the two {111} planes containing b
  * side length = `radius` argument of insert_prismatic_loop()
  * loop perimeter  = 4 × radius
"""

import os, sys, json
import numpy as np

pyexadis_path = os.path.join(os.path.dirname(__file__), '../../core/exadis/python/')
if pyexadis_path not in sys.path:
    sys.path.append(pyexadis_path)
try:
    import pyexadis
    from framework.disnet_manager import DisNetManager
    from pyexadis_base import (ExaDisNet, NodeConstraints,
                            CalForce, MobilityLaw, TimeIntegration,
                            Collision, Topology, Remesh)

    sys.path.append(os.path.join(os.path.dirname(__file__), '../displacement'))
    from ae_recorder import AEDisplacementRecorder
    from simulate_with_ae import SimulateNetworkWithAE
    from pyexadis_utils import insert_prismatic_loop, dislocation_density
except ImportError:
    raise ImportError('Cannot import pyexadis. Check pyexadis_path.')


def simulate_cu_fcc_prismatic():

    pyexadis.initialize()

    # ── 运行参数（做短跑测试时，改这两行）──────────────────────────────────
    erate      = 50.0     # 应变率 [/s]
    max_strain = 0.01     # 目标最大应变（到这个应变就停止）
    min_dist_factor = 0.5   # 环中心的最小距离 = 因子 × (两个环的边长之和)；0 = 不检查
    max_tries       = 1000  # 每个环最多试多少次随机位置

    # ── Material ────────────────────────────────────────────────────────────
    b_mag = 2.55e-10    # Burgers vector magnitude [m]

    # ── Simulation box: 5 × 5 × 5 μm ────────────────────────────────────────
    L_m  = 5.0e-6                   # box edge [m]
    Lbox = L_m / b_mag               # box edge [burgmag]

    # ── 环的边长范围，以及由它决定的线段长度 ─────────────────────────────────
    R_max_m = L_m / 5.0              # 最大边长 = 盒子边长的 1/5
    R_min_m = R_max_m / 5.0          # 最小边长 = 最大边长的 1/5
    R_min_b = R_min_m / b_mag        # 最小边长 [burgmag]
    # 最短的边（R_min）分成 5 段。乘 (1+1e-6) 是为了避免浮点误差
    # 让 ceil(边长/maxseg) 变成 6。
    maxseg = R_min_b / 5.0 * (1.0 + 1e-6)
    minseg = maxseg / 4.0

    state = {
        "crystal" : 'fcc',
        "burgmag" : b_mag,
        "mu"      : 54.6e9,     # shear modulus [Pa]
        "nu"      : 0.324,      # Poisson's ratio
        "a"       : 6.0,        # non-singular core width [burgmag]
        "maxseg"  : maxseg,     # max segment length [burgmag] (= R_min / 5)
        "minseg"  : minseg,     # min segment length [burgmag] (= maxseg / 4)
        "rann"    : 3.0,
        "rtol"    : 3.0,
        "nextdt"  : 1e-10,
        "maxdt"   : 1e-9,
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(
        script_dir, f'output_prismatic_e{int(erate)}_randsize_strain{max_strain:g}')
    os.makedirs(output_dir, exist_ok=True)

    cell = pyexadis.Cell(h=Lbox * np.eye(3), is_periodic=[True, True, True])
    orig = np.array(cell.origin)     # lower-left corner in burgmag

    # ── Target density ──────────────────────────────────────────────────────
    # IMPORTANT: do NOT change rho_target; it sets the initial dislocation density
    rho_target = 1.0e12              # [m⁻²]
    V_m3       = L_m ** 3            # [m³]
    L_total_m  = rho_target * V_m3   # total line length needed [m]

    # ── FCC Burgers vectors ─────────────────────────────────────────────────
    s2 = 1.0 / np.sqrt(2.0)

    burgers_vectors = [
        np.array([ 0., 1.,-1.]) * s2,   # [01-1]
        np.array([ 1., 0.,-1.]) * s2,   # [10-1]
        np.array([ 1., 0., 1.]) * s2,   # [101]
        np.array([ 0., 1., 1.]) * s2,   # [011]
    ]

    # ── 生成环的边长：对数均匀随机，累加到接近目标线长 ──────────────────────
    np.random.seed(42)
    sizes_m = []
    total_m = 0.0
    while True:
        R_m      = np.exp(np.random.uniform(np.log(R_min_m), np.log(R_max_m)))
        length_m = 4.0 * R_m          # 正方形环的周长
        if total_m + length_m > L_total_m:
            # 再加这个环会超过目标线长：只有超出量比现在的缺口更小时才收下它，然后停止
            if (total_m + length_m - L_total_m) < (L_total_m - total_m):
                sizes_m.append(R_m)
                total_m += length_m
            break
        sizes_m.append(R_m)
        total_m += length_m
    sizes_m = np.array(sizes_m)

    nseg_side = np.ceil((sizes_m / b_mag) / maxseg)   # 每条边被分成几段

    print('=' * 60)
    print(f'Cu FCC prismatic-loop simulation ({L_m*1e6:.1f} μm box, random loop size)')
    print('=' * 60)
    print(f'  Lbox        = {Lbox:.1f} burgmag  ({L_m*1e6:.1f} μm)')
    print(f'  Loop side   ~ log-uniform[{R_min_m*1e6:.2f}, {R_max_m*1e6:.2f}] μm')
    print(f'  maxseg      = {maxseg:.2f} burgmag ({maxseg*b_mag*1e6:.4f} μm)')
    print(f'  minseg      = {minseg:.2f} burgmag ({minseg*b_mag*1e6:.4f} μm)')

    # ── Build initial dislocation network ───────────────────────────────────
    # 先给每个环选中心位置：盒子内均匀随机；如果离已放好的环太近就重新抽
    centres = []
    n_failed = 0
    for i, R_m in enumerate(sizes_m):
        for attempt in range(max_tries):
            c = orig + np.random.rand(3) * Lbox
            ok = True
            for c_j, R_j in zip(centres, sizes_m[:i]):
                d = c - c_j
                d -= np.round(d / Lbox) * Lbox        # 周期性边界：取最近的镜像
                if np.linalg.norm(d) * b_mag < min_dist_factor * (R_m + R_j):
                    ok = False
                    break
            if ok:
                break
        else:
            n_failed += 1                             # 试满 max_tries 次都不行：保留最后一次的位置
        centres.append(c)

    d_closest = np.inf                                # 放好之后，最近的一对中心相距多远
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            d = centres[i] - centres[j]
            d -= np.round(d / Lbox) * Lbox
            d_closest = min(d_closest, np.linalg.norm(d) * b_mag)

    nodes, segs = [], []
    for i, R_m in enumerate(sizes_m):
        b = burgers_vectors[i % 4]                    # 四个柏氏矢量轮流分配
        nodes, segs = insert_prismatic_loop(
            'fcc', cell, nodes, segs, b, R_m / b_mag, centres[i], maxseg=maxseg)

    G   = ExaDisNet(cell, nodes, segs)
    net = DisNetManager(G)

    # ── Report initial density ───────────────────────────────────────────────
    rho_actual = dislocation_density(net, b_mag)
    print(f'\nInitial configuration:')
    print(f'  Total loops  = {len(sizes_m)}  (per b-direction: '
          f'{[int(np.sum(np.arange(len(sizes_m)) % 4 == k)) for k in range(4)]})')
    print(f'  Total nodes  = {len(nodes)}')
    print(f'  Total segs   = {len(segs)}')
    print(f'  Loop side: min={sizes_m.min()*1e6:.3f} μm, max={sizes_m.max()*1e6:.3f} μm, '
          f'mean={sizes_m.mean()*1e6:.3f} μm, median={np.median(sizes_m)*1e6:.3f} μm')
    print(f'  Segments per side: min={int(nseg_side.min())}, max={int(nseg_side.max())}')
    print(f'  Min-distance check: factor={min_dist_factor:g}, placement failures={n_failed}, '
          f'closest centre pair={d_closest*1e6:.3f} μm')
    print(f'  Target ρ     = {rho_target:.3e} m⁻²')
    print(f'  Actual ρ     = {rho_actual:.3e} m⁻²  (ratio = {rho_actual/rho_target:.3f})')

    # ── Save initial configuration as JSON ──────────────────────────────────
    init_json = os.path.join(output_dir, 'init_config.json')

    def to_list(obj):
        if isinstance(obj, np.ndarray):   return obj.tolist()
        if isinstance(obj, dict):         return {k: to_list(v) for k, v in obj.items()}
        if isinstance(obj, (list,tuple)): return [to_list(v) for v in obj]
        return obj

    with open(init_json, 'w') as f:
        json.dump(to_list(net.export_data()), f, indent=2)
    print(f'  Saved → {init_json}')

    # ── Simulation components ────────────────────────────────────────────────
    vis = None

    calforce  = CalForce(
        force_mode='SUBCYCLING_MODEL',
        state=state, Ngrid=64, cell=net.cell)

    mobility  = MobilityLaw(
        mobility_law='FCC_0',
        state=state, Medge=64103.0, Mscrew=64103.0, vmax=4000.0)

    timeint   = TimeIntegration(
        integrator='Subcycling',
        rgroups=[0.0, 100.0, 600.0, 1600.0],
        state=state, force=calforce, mobility=mobility)

    collision = Collision(
        collision_mode='Retroactive',
        state=state)

    topology  = Topology(
        topology_mode='TopologyParallel',
        state=state, force=calforce, mobility=mobility)

    remesh    = Remesh(
        remesh_rule='LengthBased',
        state=state)

    # ── AE displacement recorder ──────────────────────────────────────────
    # 场点：盒子中心，1 个
    field_points = np.array([
        orig + 0.5 * Lbox,
    ])
    ae_recorder = AEDisplacementRecorder(
        field_points=field_points,
        core_a=state["a"],
        nu=state["nu"],
        output_file=os.path.join(output_dir, 'ae_displacement.dat')
    )

    # ── Run ─────────────────────────────────────────────────────────────────
    sim = SimulateNetworkWithAE(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
        cross_slip=None, vis=vis,
        loading_mode='strain_rate',
        erate=erate,
        edir=np.array([0., 0., 1.]),
        max_strain=max_strain,
        burgmag=b_mag,
        state=state,
        print_freq=10,
        plot_freq=None,
        write_freq=10,
        write_dir=output_dir,
        ae_recorder=ae_recorder,
        restart=None)
    print(f'\nStarting simulation (erate={erate:g}/s, max_strain={max_strain:g})')
    print(f'Output → {output_dir}')
    print('=' * 60)

    sim.run(net, state)

    pyexadis.finalize()


if __name__ == '__main__':
    simulate_cu_fcc_prismatic()
