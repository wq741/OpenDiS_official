"""
Cu 位错动力学模拟
使用 example 10 标准初始构型（180chains_16.10e.data）
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
from pyexadis_base import ExaDisNet
from pyexadis_base import CalForce, MobilityLaw, TimeIntegration
from pyexadis_base import Collision, Topology, Remesh, SimulateNetworkPerf
from pyexadis_utils import read_paradis

# ════════════════════════════════════════════════════════════════════════
# 材料和模拟参数
# ════════════════════════════════════════════════════════════════════════

b_mag  = 2.55e-10
mu     = 54.6e9
nu     = 0.324

strain_rate   = 1e3
target_strain = 0.01

state = {
    "crystal" : 'fcc',
    "burgmag" : b_mag,
    "mu"      : mu,
    "nu"      : nu,
    "a"       : 6.0,
    "maxseg"  : 2000.0,
    "minseg"  : 300.0,
    "rann"    : 10.0,
    "rtol"    : 10.0,
    "nextdt"  : 1e-10,
    "maxdt"   : 1e-9,
}

def main():
    # 读取标准初始构型
    data_file = '../../examples/10_strain_hardening/180chains_16.10e.data'
    print(f"读取初始构型: {data_file}")
    net = read_paradis(data_file)

    os.makedirs('output', exist_ok=True)

    cell = net.get_disnet(ExaDisNet).cell

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state,
                         Ngrid=64, cell=cell)
    mobility  = MobilityLaw(mobility_law='FCC_0', state=state,
                            Medge=64103.0, Mscrew=64103.0, vmax=4000.0)
    timeint   = TimeIntegration(integrator='Subcycling',
                                rgroups=[0.0, 100.0, 600.0, 1600.0],
                                state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state,
                         force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    sim = SimulateNetworkPerf(
        calforce=calforce, mobility=mobility, timeint=timeint,
        collision=collision, topology=topology, remesh=remesh,
        state=state,
        loading_mode='strain_rate',
        erate=strain_rate,
        edir=np.array([0., 0., 1.]),
        max_strain=target_strain,
        burgmag=b_mag,
        print_freq=10,
        write_freq=50,
        write_dir='output',
    )

    print(f"开始模拟，目标应变 {target_strain*100:.0f}%")
    sim.run(net, state)
    print("模拟完成")

if __name__ == "__main__":
    pyexadis.initialize()
    main()
    pyexadis.finalize()