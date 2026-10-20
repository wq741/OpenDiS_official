"""Run FCC prismatic-loop strain hardening to 1% total strain at 1000/s.

Run generate_loops.py first. Default loading direction: crystal [001].
Outputs are relative to this script, independent of the current directory.
"""
import argparse
import json
import os
from pathlib import Path

from generate_loops import HERE, B_MAG, CELL_M, MAXSEG, MINSEG, setup_exadis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='output_fcc_prismatic_5um')
    args = parser.parse_args()
    setup_exadis()
    import numpy as np
    import pyexadis
    from pyexadis_base import (ExaDisNet, DisNetManager, SimulateNetworkPerf,
                              CalForce, MobilityLaw, TimeIntegration, Collision,
                              Topology, Remesh, CrossSlip)
    source = HERE / 'prismatic_loops.data'
    if not source.is_file():
        raise FileNotFoundError(f'Run generate_loops.py first: missing {source}')
    output = Path(args.output_dir).expanduser()
    if not output.is_absolute():
        output = HERE / output
    output = output.resolve()
    # The native driver can remove its output directory on initialization.
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    if output == HERE or output in HERE.parents:
        raise ValueError('Output directory cannot contain the input scripts.')
    active_inclusion = os.environ.get('INCLUSION_A', '').strip()
    if active_inclusion and float(active_inclusion) != 0:
        raise RuntimeError('Unset INCLUSION_A for this inclusion-free case.')
    state = {
        'crystal': 'fcc', 'burgmag': B_MAG, 'mu': 54.6e9, 'nu': 0.324,
        'a': 1.0, 'maxseg': MAXSEG, 'minseg': MINSEG,
        'rann': 0.5, 'rtol': 0.25, 'nextdt': 1e-10, 'maxdt': 1e-9,
    }
    pyexadis.initialize()
    try:
        graph = ExaDisNet().read_paradis(str(source))
        if not graph.is_sane():
            raise RuntimeError('Input network sanity check failed.')
        if not np.allclose(np.asarray(graph.cell.h), np.eye(3)*CELL_M/B_MAG,
                           rtol=1e-8, atol=1e-8):
            raise ValueError('Input cell is not the prescribed 5 micrometre cube.')
        net = DisNetManager(graph)
        vis = None
        calforce = CalForce(force_mode='SUBCYCLING_MODEL', state=state,
                            Ngrid=64, cell=net.cell)
        mobility = MobilityLaw(mobility_law='FCC_0', state=state,
                               Medge=64103.0, Mscrew=64103.0, vmax=4000.0)
        timeint = TimeIntegration(integrator='Subcycling',
                                  rgroups=[0.0, 100.0, 600.0, 1600.0],
                                  state=state, force=calforce, mobility=mobility)
        collision = Collision(collision_mode='Retroactive', state=state)
        topology = Topology(topology_mode='TopologyParallel', state=state,
                            force=calforce, mobility=mobility)
        remesh = Remesh(remesh_rule='LengthBased', state=state)
        cross_slip = CrossSlip(cross_slip_mode='ForceBasedParallel',
                               state=state, force=calforce)
        print('  Cross-slip   : ON, mode = ForceBasedParallel')
        sim = SimulateNetworkPerf(
            state=state, calforce=calforce, mobility=mobility, timeint=timeint,
            collision=collision, topology=topology, remesh=remesh,
            cross_slip=cross_slip, vis=vis, loading_mode='strain_rate',
            erate=1000.0, edir=np.array([0., 0., 1.]), max_strain=0.01,
            burgmag=B_MAG, print_freq=10, write_freq=100,
            write_dir=str(output), restart=None)
        final_state = sim.run(net, state)
        strain = float(final_state['strain'])
        if not np.isfinite(strain) or abs(strain) < 0.01:
            raise RuntimeError(f'Target total strain not reached: {strain}')
        (output / 'completion.json').write_text(json.dumps({
            'final_total_strain': strain, 'target_total_strain': 0.01,
            'strain_rate_s': 1000.0, 'loading_direction': [0, 0, 1],
        }, indent=2)+'\n', encoding='utf-8')
        print(f'Completed: total strain = {strain:.8g}; results: {output}')
    finally:
        pyexadis.finalize()


if __name__ == '__main__':
    main()
