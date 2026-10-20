"""Generate native FCC prismatic loops. Run before run_simulation.py.

Move both scripts into exadis/examples/<case>/ before running, or set
EXADIS_ROOT to the repository root. All lengths passed to ExaDiS are in b.
"""
import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
B_MAG = 2.55e-10
CELL_M = 5e-6
RHO = 1e12
MAXSEG = 0.2e-6 / 5 / B_MAG
MINSEG = MAXSEG / 4


def setup_exadis():
    roots = ([Path(os.environ['EXADIS_ROOT']).expanduser().resolve()]
             if os.environ.get('EXADIS_ROOT') else list(HERE.parents))
    root = next((p for p in roots if (p / 'python' / 'pyexadis_base.py').is_file()), None)
    if root is None:
        raise RuntimeError('Place this directory under exadis/examples/, or set EXADIS_ROOT.')
    # Match the repository examples; optionally include its compiled module path.
    sys.path.insert(0, str(root / 'python'))
    sys.path.insert(1, str(root / 'build' / 'python'))
    return root


def sample_sides(rng, count, total_um):
    """Uniform samples conditioned on a fixed sum; units: micrometres.

    The final value enforces density and equal slip-system line lengths.
    These are constrained samples, not independent uniform draws.
    """
    import numpy as np
    for _ in range(100000):
        first = rng.uniform(0.2, 1.0, count - 1)
        last = total_um - first.sum()
        if 0.2 <= last <= 1.0:
            return rng.permutation(np.append(first, last))
    raise RuntimeError('Could not sample the prescribed side-length sum.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    setup_exadis()
    import numpy as np
    import pyexadis
    from pyexadis_base import ExaDisNet
    from pyexadis_utils import insert_prismatic_loop

    data_path = HERE / 'prismatic_loops.data'
    manifest_path = HERE / 'initial_network.json'
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError('Initial files already exist; move them before regenerating.')
    rng = np.random.default_rng(args.seed)
    length_b = CELL_M / B_MAG
    # 54 loops, 9 per Burgers direction; each direction activates two systems.
    n_per_b = 9
    nloops = 6 * n_per_b
    total_line_m = RHO * CELL_M**3
    per_b_side_sum_um = total_line_m / (4 * 6) * 1e6
    sides = [sample_sides(rng, n_per_b, per_b_side_sum_um) for _ in range(6)]
    burgers = np.array([[1, 1, 0], [-1, 1, 0], [1, 0, 1],
                        [-1, 0, 1], [0, 1, 1], [0, -1, 1]], dtype=float) / np.sqrt(2)
    # Randomly occupied, jittered 4x4x4 strata, following native uniform placement.
    grid = np.array(list(np.ndindex(4, 4, 4)), dtype=float)
    centers = (grid[rng.permutation(len(grid))[:nloops]] + 0.5
               + rng.uniform(-0.25, 0.25, (nloops, 3))) / 4 * length_b
    order = rng.permutation(nloops)
    pyexadis.initialize()
    try:
        cell = pyexadis.Cell(h=np.eye(3)*length_b, is_periodic=[True]*3)
        nodes, segs, records = [], [], []
        for k, index in enumerate(order):
            ib, j = divmod(int(index), n_per_b)
            side_um = float(sides[ib][j])
            # Native FCC 'radius' equals its rhombus side length, not circumradius.
            nodes, segs = insert_prismatic_loop(
                'fcc', cell, nodes, segs, burgers[ib],
                side_um*1e-6/B_MAG, centers[k], maxseg=MAXSEG)
            records.append({'burgers_group': ib, 'side_um': side_um,
                            'center_b': centers[k].tolist()})
        nodes, segs = np.asarray(nodes), np.asarray(segs)
        nodes[:, :3] %= length_b
        ids = segs[:, :2].astype(int)
        delta = nodes[ids[:, 1], :3] - nodes[ids[:, 0], :3]
        delta -= length_b*np.rint(delta/length_b)
        lengths = np.linalg.norm(delta, axis=1)*B_MAG
        density = float(lengths.sum()/CELL_M**3)
        if not np.isclose(density, RHO, rtol=1e-10):
            raise RuntimeError(f'Density mismatch: {density}')
        if lengths.max() > MAXSEG*B_MAG*(1+1e-10) or lengths.min() <= 0:
            raise RuntimeError('Invalid segment lengths.')
        # Audit all 12 unsigned (Burgers, plane) slip systems by actual line length.
        slip_lengths = {}
        def unsigned(v):
            v = v / np.linalg.norm(v)
            if v[np.flatnonzero(np.abs(v)>1e-8)[0]] < 0:
                v = -v
            return tuple(np.round(v, 8))
        for seg, length in zip(segs, lengths):
            key = str((unsigned(seg[2:5]), unsigned(seg[5:8])))
            slip_lengths[key] = slip_lengths.get(key, 0.0) + float(length)
        if len(slip_lengths) != 12 or not np.allclose(
                list(slip_lengths.values()), total_line_m/12, rtol=1e-10, atol=0):
            raise RuntimeError('Slip-system balance check failed.')
        network = ExaDisNet(cell, nodes, segs)
        if not network.is_sane():
            raise RuntimeError('Native network sanity check failed.')
        network.write_data(str(data_path))
        manifest = dict(seed=args.seed, crystal='fcc', burgmag_m=B_MAG,
                        cell_m=CELL_M, target_density_m2=RHO, actual_density_m2=density,
                        num_loops=nloops, num_nodes=len(nodes), num_segments=len(segs),
                        maxseg_b=MAXSEG, minseg_b=MINSEG,
                        slip_system_line_lengths_m=slip_lengths, loops=records)
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        print(f'Wrote {data_path}\nLoops: {nloops}; density: {density:.12g} m^-2')
        print('12 slip systems have equal total line length; positions are randomized.')
    finally:
        pyexadis.finalize()


if __name__ == '__main__':
    main()
