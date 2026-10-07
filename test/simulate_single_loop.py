"""
Cu FCC single crystal dislocation dynamics simulation
SINGLE prismatic loop test (验证：棱柱环在均匀加载下的行为)
Framework: OpenDiS / ExaDiS (pyexadis)

与多环脚本（erate=50, max_strain=0.01 那份）的区别
----------------------------------------
只改了三处，其余（state、力、移动律、积分、碰撞、拓扑、重划分、
cross_slip=None、erate、edir、AE 记录器等）全部保持不变：

1. 环的生成：不再随机生成很多环，只放 1 个环。
   边长、柏氏矢量、中心位置由下面"单环测试参数"指定。
2. 边长做成可调变量 loop_side_um，也可以用命令行参数覆盖：
     python simulate_single_loop.py 0.5
   这样三次运行（0.2 / 0.5 / 1.0 μm）不用改文件。
3. 输出文件夹名里加上边长和柏氏矢量序号，三次运行不会互相覆盖。

maxseg / minseg 仍然由 R_min_m（0.2 μm）算出，保持不动，
所以每条边的段数是：0.2 μm → 5 段，0.5 μm → 13 段，1.0 μm → 25 段，
和多环脚本可比。

For FCC with b = 1/2<110>:
  * loop plane normal  ||  b
  * 4 sides alternate between the two {111} planes containing b
  * side length = `radius` argument of insert_prismatic_loop()
  * loop perimeter  = 4 × radius
"""

import os, sys, json
import numpy as np

pyexadis_path = os.path.join(os.path.dirname(__file__), '../core/exadis/python/')
if pyexadis_path not in sys.path:
    sys.path.append(pyexadis_path)
try:
    import pyexadis
    from framework.disnet_manager import DisNetManager
    from pyexadis_base import (ExaDisNet, NodeConstraints,
                            CalForce, MobilityLaw, TimeIntegration,
                            Collision, Topology, Remesh)

    sys.path.append(os.path.join(os.path.dirname(__file__), '../my_work/displacement'))
    from ae_recorder import AEDisplacementRecorder
    from simulate_with_ae import SimulateNetworkWithAE
    from pyexadis_utils import insert_prismatic_loop, dislocation_density
except ImportError:
    raise ImportError('Cannot import pyexadis. Check pyexadis_path.')


def simulate_cu_fcc_prismatic():

    pyexadis.initialize()

    # ── 运行参数（和多环脚本一致）──────────────────────────────────────────
    erate      = 50.0     # 应变率 [/s]
    max_strain = 0.01     # 目标最大应变（到这个应变就停止）

    # ── 单环测试参数（这三个是本脚本新增的）────────────────────────────────
    loop_side_um = 0.5    # 环的边长 [μm]；命令行第一个参数可覆盖
    if len(sys.argv) > 1:
        loop_side_um = float(sys.argv[1])
    b_index      = 2      # 用 burgers_vectors 里的第几个：0=[01-1] 1=[10-1] 2=[101] 3=[011]
    center_frac  = 0.3    # 环中心在盒子里的位置：三个方向都取 0.3 倍盒长（避开 AE 场点）

    # ── Material ────────────────────────────────────────────────────────────
    b_mag = 2.55e-10    # Burgers vector magnitude [m]

    # ── Simulation box: 5 × 5 × 5 μm ────────────────────────────────────────
    L_m  = 5.0e-6                   # box edge [m]
    Lbox = L_m / b_mag               # box edge [burgmag]

    # ── 环的边长范围，以及由它决定的线段长度（保持不动，用来定 maxseg）──────
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
        script_dir,
        f'output_single_loop_e{int(erate)}_strain{max_strain:g}_side{loop_side_um:g}um_b{b_index}')
    os.makedirs(output_dir, exist_ok=True)

    cell = pyexadis.Cell(h=Lbox * np.eye(3), is_periodic=[True, True, True])
    orig = np.array(cell.origin)     # lower-left corner in burgmag

    # ── FCC Burgers vectors ─────────────────────────────────────────────────
    s2 = 1.0 / np.sqrt(2.0)

    burgers_vectors = [
        np.array([ 0., 1.,-1.]) * s2,   # [01-1]
        np.array([ 1., 0.,-1.]) * s2,   # [10-1]
        np.array([ 1., 0., 1.]) * s2,   # [101]
        np.array([ 0., 1., 1.]) * s2,   # [011]
    ]

    # ── 单环：边长、柏氏矢量、中心位置 ──────────────────────────────────────
    sizes_m = np.array([loop_side_um * 1e-6])            # 只有 1 个环
    nseg_side = np.ceil((sizes_m / b_mag) / maxseg)      # 每条边被分成几段
    b = burgers_vectors[b_index]
    centre = orig + center_frac * Lbox * np.ones(3)      # 盒子里 (0.3, 0.3, 0.3) 倍盒长处

    print('=' * 60)
    print(f'Cu FCC SINGLE prismatic-loop test ({L_m*1e6:.1f} μm box)')
    print('=' * 60)
    print(f'  Lbox        = {Lbox:.1f} burgmag  ({L_m*1e6:.1f} μm)')
    print(f'  Loop side   = {loop_side_um:g} μm  (b index = {b_index}, b = {np.round(b / s2, 3)} × 1/√2)')
    print(f'  Loop centre = {np.round(centre, 1)} burgmag  (frac = {center_frac:g} of box)')
    print(f'  maxseg      = {maxseg:.2f} burgmag ({maxseg*b_mag*1e6:.4f} μm)')
    print(f'  minseg      = {minseg:.2f} burgmag ({minseg*b_mag*1e6:.4f} μm)')

    # ── Build initial dislocation network ───────────────────────────────────
    nodes, segs = [], []
    nodes, segs = insert_prismatic_loop(
        'fcc', cell, nodes, segs, b, sizes_m[0] / b_mag, centre, maxseg=maxseg)

    G   = ExaDisNet(cell, nodes, segs)
    net = DisNetManager(G)

    # ── Report initial density ───────────────────────────────────────────────
    rho_actual = dislocation_density(net, b_mag)
    print(f'\nInitial configuration:')
    print(f'  Total loops  = {len(sizes_m)}')
    print(f'  Total nodes  = {len(nodes)}')
    print(f'  Total segs   = {len(segs)}')
    print(f'  Segments per side: {int(nseg_side[0])}')
    print(f'  Actual ρ     = {rho_actual:.3e} m⁻²')

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
        write_freq=100,
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
