"""
Cu FCC single crystal dislocation dynamics simulation
Initial configuration: prismatic (glide) loops on 8 specific slip systems
Framework: OpenDiS / ExaDiS (pyexadis)
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
                                SimulateNetworkPerf,
                                CalForce, MobilityLaw, TimeIntegration,
                                Collision, Topology, Remesh)
    from pyexadis_utils import dislocation_density
except ImportError:
    raise ImportError('Cannot import pyexadis. Check pyexadis_path.')

# ---------------------------------------------------------------------------
# Helper: rectangular glide loop strictly in a given slip plane
# ---------------------------------------------------------------------------

def insert_glide_loop(cell, nodes, segs, burg, plane, center, L, maxseg=-1):
    """Insert a closed rectangular glide loop in the slip plane defined by `plane`.

    All four corners satisfy  n · (v - center) = 0.
    Every segment carries the same Burgers vector `burg`.
    The loop is traversed so that Burgers vector conservation holds at each node.

    Args:
        cell    : pyexadis.Cell object (used only for future PBC checks)
        nodes   : list of node arrays [x, y, z, constraint]
        segs    : list of segment arrays [n1, n2, bx, by, bz, nx, ny, nz]
        burg    : unit Burgers vector (1/sqrt(2) * <110>)
        plane   : unit slip-plane normal (<111>/sqrt(3))
        center  : loop centre position in burgmag units (3-vector)
        L       : side length of the square loop in burgmag units
        maxseg  : maximum segment discretisation length (burgmag); -1 = no limit
    """
    plane_n = plane / np.linalg.norm(plane)
    b_hat   = burg  / np.linalg.norm(burg)

    # Two orthogonal in-plane directions
    t1 = b_hat                               # along Burgers vector
    t2 = np.cross(plane_n, b_hat)
    t2 = t2 / np.linalg.norm(t2)

    half = 0.5 * L
    # Four corners visited counter-clockwise when viewed from +plane_n
    corners = [
        center - half*t1 - half*t2,
        center + half*t1 - half*t2,
        center + half*t1 + half*t2,
        center - half*t1 + half*t2,
    ]

    # Number of segments per side (enforce maxseg)
    nseg = max(1, int(np.ceil(L / maxseg))) if maxseg > 0 else 1

    istart      = len(nodes)
    total_nodes = 4 * nseg   # one node at the START of each sub-segment

    for side in range(4):
        c0 = corners[side]
        c1 = corners[(side + 1) % 4]
        for j in range(nseg):
            p = c0 + (j / nseg) * (c1 - c0)
            nodes.append(np.array([p[0], p[1], p[2],
                                   NodeConstraints.UNCONSTRAINED]))

    # Closed loop: segment k connects node k → node (k+1) % total_nodes
    for k in range(total_nodes):
        n1 = istart + k
        n2 = istart + (k + 1) % total_nodes
        segs.append(np.array([n1, n2,
                               burg[0], burg[1], burg[2],
                               plane_n[0], plane_n[1], plane_n[2]]))

    return nodes, segs


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def simulate_cu_fcc_prismatic():

    pyexadis.initialize()

    # ── Material / state ────────────────────────────────────────────────────
    b_mag = 2.55e-10    # Burgers vector magnitude [m]

    state = {
        "crystal" : 'fcc',
        "burgmag" : b_mag,
        "mu"      : 54.6e9,     # shear modulus [Pa]
        "nu"      : 0.324,      # Poisson's ratio
        "a"       : 6.0,        # non-singular core width [burgmag]
        "maxseg"  : 2000.0,     # max segment length [burgmag] ≈ 0.51 μm
        "minseg"  : 500.0,      # min segment length [burgmag] ≈ 0.13 μm
        "rann"    : 3.0,
        "rtol"    : 3.0,
        "nextdt"  : 1e-10,
        "maxdt"   : 1e-9,
    }

    output_dir = os.path.join(os.path.dirname(__file__), 'output_prismatic')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'output'), exist_ok=True)

    # ── Simulation box: 10 × 10 × 10 μm ────────────────────────────────────
    L_m   = 10.0e-6                 # box edge [m]
    Lbox  = L_m / b_mag             # box edge [burgmag] ≈ 39215.7
    cell  = pyexadis.Cell(h=Lbox * np.eye(3), is_periodic=[True, True, True])
    orig  = np.array(cell.origin)   # lower-left corner in burgmag

    # ── Target density & loop counts ────────────────────────────────────────
    rho_target = 1.0e12             # [m⁻²]
    V_m3       = L_m ** 3           # [m³]
    L_total_m  = rho_target * V_m3  # total line length needed [m]

    Lp = 4.0e-6 / b_mag             # primary loop side [burgmag]
    Lc = 1.0e-6 / b_mag             # co-linear loop side [burgmag]

    # 4 primary systems share 60 %, 4 co-linear systems share 40 %
    frac_p = 0.60 / 4.0             # 15 % per primary system
    frac_c = 0.40 / 4.0             # 10 % per co-linear system

    Np = max(1, round(frac_p * L_total_m / (4.0 * Lp * b_mag)))
    Nc = max(1, round(frac_c * L_total_m / (4.0 * Lc * b_mag)))

    print('='*60)
    print('Cu FCC prismatic-loop simulation (10 μm box)')
    print('='*60)
    print(f'  Lbox          = {Lbox:.1f} burgmag  ({L_m*1e6:.1f} μm)')
    print(f'  Primary  loop = {Lp:.1f} burgmag  ({4.0:.1f} μm)  ×{Np} per system')
    print(f'  Co-linear loop= {Lc:.1f} burgmag  ({1.0:.1f} μm)  ×{Nc} per system')

    # ── FCC slip systems ─────────────────────────────────────────────────────
    # Burgers vectors are 1/2 <110>  (unit form = <110>/√2)
    # Plane normals are {111}        (unit form = <111>/√3)
    # Verify b · n = 0 for all systems (glide condition)

    s2 = 1.0 / np.sqrt(2.0)
    s3 = 1.0 / np.sqrt(3.0)

    # Primary systems (60 % of density): sys 0, 1, 5, 6
    primary_systems = [
        # sys 0: (111) / [01-1]
        dict(b=np.array([ 0., 1.,-1.]) * s2,
             n=np.array([ 1., 1., 1.]) * s3),
        # sys 1: (111) / [10-1]
        dict(b=np.array([ 1., 0.,-1.]) * s2,
             n=np.array([ 1., 1., 1.]) * s3),
        # sys 5: (-111) / [101]
        dict(b=np.array([ 1., 0., 1.]) * s2,
             n=np.array([-1., 1., 1.]) * s3),
        # sys 6: (1-11) / [011]
        dict(b=np.array([ 0., 1., 1.]) * s2,
             n=np.array([ 1.,-1., 1.]) * s3),
    ]

    # Co-linear systems (40 % of density): sys 3, 8, 9, 10
    colinear_systems = [
        # sys 3: (-111) / [01-1]  — co-linear with sys 0
        dict(b=np.array([ 0., 1.,-1.]) * s2,
             n=np.array([-1., 1., 1.]) * s3),
        # sys 8: (1-11) / [10-1]  — co-linear with sys 1
        dict(b=np.array([ 1., 0.,-1.]) * s2,
             n=np.array([ 1.,-1., 1.]) * s3),
        # sys 9: (11-1) / [011]   — co-linear with sys 6
        dict(b=np.array([ 0., 1., 1.]) * s2,
             n=np.array([ 1., 1.,-1.]) * s3),
        # sys 10: (11-1) / [101]  — co-linear with sys 5
        dict(b=np.array([ 1., 0., 1.]) * s2,
             n=np.array([ 1., 1.,-1.]) * s3),
    ]

    # Sanity-check: b · n = 0 for every system
    for i, sys in enumerate(primary_systems + colinear_systems):
        dot = abs(np.dot(sys['b'], sys['n']))
        assert dot < 1e-10, f"Slip system {i}: b·n = {dot:.2e} ≠ 0"

    # ── Build initial dislocation network ───────────────────────────────────
    np.random.seed(42)
    nodes, segs = [], []
    maxseg = state['maxseg']

    # Primary-system loops
    for sidx, slip in enumerate(primary_systems):
        centres = orig + np.random.rand(Np, 3) * Lbox
        for c in centres:
            nodes, segs = insert_glide_loop(
                cell, nodes, segs,
                slip['b'], slip['n'], c, Lp, maxseg=maxseg)

    # Co-linear-system loops
    for sidx, slip in enumerate(colinear_systems):
        centres = orig + np.random.rand(Nc, 3) * Lbox
        for c in centres:
            nodes, segs = insert_glide_loop(
                cell, nodes, segs,
                slip['b'], slip['n'], c, Lc, maxseg=maxseg)

    G   = ExaDisNet(cell, nodes, segs)
    net = DisNetManager(G)

    # ── Report initial density ───────────────────────────────────────────────
    rho_actual = dislocation_density(net, b_mag)
    total_loops = 4*Np + 4*Nc
    print(f'\nInitial configuration:')
    print(f'  Total loops   = {total_loops}  '
          f'({4*Np} primary + {4*Nc} co-linear)')
    print(f'  Total nodes   = {len(nodes)}')
    print(f'  Total segs    = {len(segs)}')
    print(f'  Target density = {rho_target:.3e} m⁻²')
    print(f'  Actual density = {rho_actual:.3e} m⁻²')
    print(f'  Ratio          = {rho_actual/rho_target:.3f}')

    # ── Save initial configuration as JSON ──────────────────────────────────
    init_json = os.path.join(os.path.dirname(__file__), 'output', 'init_config.json')

    def ndarray_to_list(obj):
        """Recursively convert numpy arrays to nested Python lists."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: ndarray_to_list(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ndarray_to_list(v) for v in obj]
        return obj

    with open(init_json, 'w') as f:
        json.dump(ndarray_to_list(net.export_data()), f, indent=2)
    print(f'\nInitial configuration saved → {init_json}')

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

    # ── Run ─────────────────────────────────────────────────────────────────
    # erate = 50 /s for verification; change to 1000 /s for production
    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
        cross_slip=None, vis=vis,
        loading_mode='strain_rate',
        erate=50.0,                         # [/s] — set to 1000.0 for production
        edir=np.array([0., 0., 1.]),        # [001] uniaxial loading
        max_strain=0.01,                    # stop at 1 % strain
        burgmag=b_mag,
        state=state,
        print_freq=100,
        plot_freq=1000,
        plot_pause_seconds=0.0001,
        write_freq=10,
        write_dir=output_dir,
        restart=None)

    print(f'\nStarting simulation  (erate = 50 /s, max_strain = 1 %)')
    print(f'Output directory     : {output_dir}')
    print('='*60)

    sim.run(net, state)

    pyexadis.finalize()


if __name__ == '__main__':
    simulate_cu_fcc_prismatic()
