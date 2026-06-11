"""
Cu FCC single crystal dislocation dynamics simulation
Initial configuration: TRUE prismatic loops (b ⊥ loop plane)
Framework: OpenDiS / ExaDiS (pyexadis)

Physics note
------------
Glide loops (b in loop plane) expand freely under applied shear stress and
inevitably annihilate via PBC images.  True prismatic loops (b perpendicular
to the loop plane, sides alternating on two {111} planes) are translationally
stable: they translate along b as a unit rather than expanding, and do not
rapidly self-annihilate.  ExaDiS insert_prismatic_loop() creates this kind.

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
                                SimulateNetworkPerf,
                                CalForce, MobilityLaw, TimeIntegration,
                                Collision, Topology, Remesh)
    from pyexadis_utils import insert_prismatic_loop, dislocation_density
except ImportError:
    raise ImportError('Cannot import pyexadis. Check pyexadis_path.')


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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output_prismatic')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(script_dir, 'output'), exist_ok=True)

    # ── Simulation box: 10 × 10 × 10 μm ────────────────────────────────────
    L_m  = 10.0e-6                   # box edge [m]
    Lbox = L_m / b_mag               # box edge [burgmag] ≈ 39215.7
    cell = pyexadis.Cell(h=Lbox * np.eye(3), is_periodic=[True, True, True])
    orig = np.array(cell.origin)     # lower-left corner in burgmag

    # ── Target density & loop counts ────────────────────────────────────────
    # IMPORTANT: do NOT change rho_target; it sets the initial dislocation density
    rho_target = 1.0e12              # [m⁻²]
    V_m3       = L_m ** 3            # [m³]
    L_total_m  = rho_target * V_m3   # total line length needed [m]

    # For insert_prismatic_loop(): `radius` = side length of the square loop
    # (perimeter = 4 × radius)
    R_p = 4.0e-6 / b_mag    # primary loop side [burgmag] ≈ 15686
    R_c = 1.0e-6 / b_mag    # co-linear loop side [burgmag] ≈  3922

    # Density fractions: 4 primary sys (60 %) + 4 co-linear sys (40 %)
    frac_p = 0.60 / 4.0     # 15 % per primary system
    frac_c = 0.40 / 4.0     # 10 % per co-linear system

    Np = max(1, round(frac_p * L_total_m / (4.0 * R_p * b_mag)))
    Nc = max(1, round(frac_c * L_total_m / (4.0 * R_c * b_mag)))

    print('=' * 60)
    print('Cu FCC prismatic-loop simulation (10 μm box)')
    print('=' * 60)
    print(f'  Lbox        = {Lbox:.1f} burgmag  ({L_m*1e6:.1f} μm)')
    print(f'  Primary  R  = {R_p:.1f} burgmag  ({4.0:.1f} μm side)  ×{Np}/system')
    print(f'  Co-linear R = {R_c:.1f} burgmag  ({1.0:.1f} μm side)  ×{Nc}/system')

    # ── FCC Burgers vectors ──────────────────────────────────────────────────
    # Each prismatic loop is characterised by its Burgers vector alone:
    #   b ⊥ loop plane → loop plane normal || b
    #   for b = [01-1]/√2 the loop sides alternate on (111) and (-111)
    #   i.e. one loop type covers BOTH sys-0 ((111)/[01-1]) and sys-3 ((-111)/[01-1])
    #
    # Four unique b vectors correspond to the user's 8 systems:
    #   b=[01-1]: sys 0 (primary, (111)) + sys 3 (co-linear, (-111))
    #   b=[10-1]: sys 1 (primary, (111)) + sys 8 (co-linear, (1-11))
    #   b=[101]:  sys 5 (primary, (-111))+ sys10 (co-linear, (11-1))
    #   b=[011]:  sys 6 (primary, (1-11))+ sys 9 (co-linear, (11-1))

    s2 = 1.0 / np.sqrt(2.0)

    burgers_vectors = [
        np.array([ 0., 1.,-1.]) * s2,   # [01-1]  → sys 0 + sys 3
        np.array([ 1., 0.,-1.]) * s2,   # [10-1]  → sys 1 + sys 8
        np.array([ 1., 0., 1.]) * s2,   # [101]   → sys 5 + sys10
        np.array([ 0., 1., 1.]) * s2,   # [011]   → sys 6 + sys 9
    ]

    # ── Build initial dislocation network ───────────────────────────────────
    np.random.seed(42)
    nodes, segs = [], []
    maxseg = state['maxseg']

    # Large prismatic loops (primary systems, 4 μm side)
    for b in burgers_vectors:
        centres = orig + np.random.rand(Np, 3) * Lbox
        for c in centres:
            nodes, segs = insert_prismatic_loop(
                'fcc', cell, nodes, segs, b, R_p, c, maxseg=maxseg)

    # Small prismatic loops (co-linear systems, 1 μm side)
    for b in burgers_vectors:
        centres = orig + np.random.rand(Nc, 3) * Lbox
        for c in centres:
            nodes, segs = insert_prismatic_loop(
                'fcc', cell, nodes, segs, b, R_c, c, maxseg=maxseg)

    G   = ExaDisNet(cell, nodes, segs)
    net = DisNetManager(G)

    # ── Report initial density ───────────────────────────────────────────────
    rho_actual  = dislocation_density(net, b_mag)
    total_loops = 4*Np + 4*Nc
    print(f'\nInitial configuration:')
    print(f'  Total loops  = {total_loops}  '
          f'({4*Np} primary + {4*Nc} co-linear)')
    print(f'  Total nodes  = {len(nodes)}')
    print(f'  Total segs   = {len(segs)}')
    print(f'  Target ρ     = {rho_target:.3e} m⁻²')
    print(f'  Actual ρ     = {rho_actual:.3e} m⁻²  (ratio = {rho_actual/rho_target:.3f})')

    # ── Save initial configuration as JSON ──────────────────────────────────
    init_json = os.path.join(script_dir, 'output', 'init_config.json')

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

    # ── Run ─────────────────────────────────────────────────────────────────
    # erate = 50 /s for verification; change to 1000 /s for production
    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
        cross_slip=None, vis=vis,
        loading_mode='strain_rate',
        erate=50.0,                          # [/s] → set 1000.0 for production
        edir=np.array([0., 0., 1.]),         # [001] uniaxial loading
        max_strain=0.01,                     # stop at 1 % strain
        burgmag=b_mag,
        state=state,
        print_freq=100,
        plot_freq=1000,
        plot_pause_seconds=0.0001,
        write_freq=10,
        write_dir=output_dir,
        restart=None)

    print(f'\nStarting simulation (erate=50/s, max_strain=1%)')
    print(f'Output → {output_dir}')
    print('=' * 60)

    sim.run(net, state)

    pyexadis.finalize()


if __name__ == '__main__':
    simulate_cu_fcc_prismatic()
