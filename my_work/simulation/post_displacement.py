"""
位错位移后处理脚本

从模拟输出的 .data 文件中读取位错网络，用 displacement.py 计算：
    - 闭合位错环：调用 displacement_loop（含立体角，完整三项）
    - 开放线段（FR 源弓出中）：调用 disp_line_segment（第二+第三项）

输出（每个时间步）：
    output/displacement/
        config.XXXXX_per_seg.npy   — shape (N_segs, 3)，每段的贡献（m）
        config.XXXXX_total.npy     — shape (3,)，总位移（m）
        summary.txt                — 每步汇总：段数、闭合环数、总位移大小
"""

import numpy as np
import os
import sys

# pyexadis 路径
_sim_dir = os.path.dirname(__file__)
for _p in ['../../python', '../../lib', '../../core/exadis/python/']:
    _ap = os.path.abspath(os.path.join(_sim_dir, _p))
    if _ap not in sys.path:
        sys.path.append(_ap)

import pyexadis
from pyexadis_base import ExaDisNet

# displacement.py 路径
sys.path.insert(0, os.path.join(_sim_dir, '..', 'displacement'))
from displacement import disp_line_segment, displacement_loop

# ════════════════════════════════════════════════════════════════════════
# 参数
# ════════════════════════════════════════════════════════════════════════

b_mag  = 2.55e-10   # Cu Burgers 矢量大小 (m)
nu     = 0.324      # 泊松比
a      = 6 * b_mag  # 非奇异核心宽度 (m)
L_phys = 5e-6       # 盒子边长 (m)

# ── 观测点（可修改）────────────────────────────────────────────────────
x_obs = np.array([L_phys/2, L_phys/2, L_phys/2])


# ════════════════════════════════════════════════════════════════════════
# 网络拓扑分析：识别闭合环与开放线段
# ════════════════════════════════════════════════════════════════════════

def partition_network(segs_arr):
    """
    将网络线段分成两类：闭合环的线段 和 开放链的线段。

    算法：叶剪枝（leaf pruning）——
      反复删去度为 1 的节点（即 FR 源固定端等死端），
      剩余线段全部属于闭合环。

    返回：
        loops     : list of (ordered_vertex_indices, seg_indices, b_unit, n_unit)
        open_segs : list of segment indices（开放链线段）
    """
    if len(segs_arr) == 0:
        return [], []

    n_nodes = int(segs_arr[:, :2].max()) + 1

    # 构建邻接表
    adj = [[] for _ in range(n_nodes)]
    for si in range(len(segs_arr)):
        n1, n2 = int(segs_arr[si, 0]), int(segs_arr[si, 1])
        adj[n1].append([n2, si])
        adj[n2].append([n1, si])

    active = [True] * len(segs_arr)
    degree = np.array([len(a) for a in adj], dtype=int)

    # 叶剪枝：反复删去度=1 的节点
    changed = True
    while changed:
        changed = False
        for node in range(n_nodes):
            if degree[node] != 1:
                continue
            for entry in adj[node]:
                nb, si = entry
                if not active[si]:
                    continue
                active[si] = False
                degree[node] -= 1
                degree[nb]   -= 1
                changed = True
                break

    loop_seg_idx = [si for si in range(len(segs_arr)) if     active[si]]
    open_seg_idx = [si for si in range(len(segs_arr)) if not active[si]]

    if not loop_seg_idx:
        return [], open_seg_idx

    # 构建闭合环子图的邻接表
    loop_nodes = set()
    for si in loop_seg_idx:
        loop_nodes.add(int(segs_arr[si, 0]))
        loop_nodes.add(int(segs_arr[si, 1]))

    ladj = {n: [] for n in loop_nodes}
    for si in loop_seg_idx:
        n1, n2 = int(segs_arr[si, 0]), int(segs_arr[si, 1])
        ladj[n1].append((n2, si))
        ladj[n2].append((n1, si))

    # 逐条追踪闭合环
    visited = set()
    loops = []

    for start in sorted(loop_nodes):
        if start in visited:
            continue

        path_v = [start]
        path_s = []
        prev, curr = -1, start

        while True:
            candidates = [(nb, si) for nb, si in ladj[curr] if nb != prev]
            if not candidates:
                break
            nb, si = candidates[0]
            if nb == start:
                path_s.append(si)
                break
            if nb in visited:
                break
            path_v.append(nb)
            path_s.append(si)
            prev, curr = curr, nb

        if path_s:
            b_unit = segs_arr[path_s[0], 2:5]
            n_unit = segs_arr[path_s[0], 5:8]
            loops.append((path_v, path_s, b_unit, n_unit))
            visited.update(path_v)

    return loops, open_seg_idx


# ════════════════════════════════════════════════════════════════════════
# 读取 .data 文件，提取节点坐标和线段信息
# ════════════════════════════════════════════════════════════════════════

def load_network(datafile):
    """读取 .data 文件，返回 (pos_m, nodes_arr, segs_arr)。
    pos_m : (N_nodes, 3) 节点坐标，物理单位 m
    """
    G = ExaDisNet().read_paradis(datafile, verbose=False)
    data = G.export_data()

    nodes_arr = np.array(data['nodes'], dtype=float)  # (N_nodes, 6)
    segs_arr  = np.array(data['segs'],  dtype=float)  # (N_segs,  8)

    if len(nodes_arr) == 0 or len(segs_arr) == 0:
        return None, nodes_arr, segs_arr

    pos_m = nodes_arr[:, 2:5] * b_mag   # burgmag → m
    return pos_m, nodes_arr, segs_arr


# ════════════════════════════════════════════════════════════════════════
# 计算位移（闭合环用完整公式，开放段用弹性项）
# ════════════════════════════════════════════════════════════════════════

def compute_displacement(pos_m, segs_arr, x_obs, nu, a):
    """
    计算观测点 x_obs 处的位移。

    闭合环：displacement_loop（立体角 + 第二项 + 第三项）
    开放线段：disp_line_segment（第二项 + 第三项）

    返回：
        u_per_seg  : (N_segs, 3)，每段贡献
        u_total    : (3,)，总位移
        n_loops    : 识别出的闭合环数量
    """
    n_segs = len(segs_arr)
    u_per_seg = np.zeros((n_segs, 3))

    loops, open_seg_idx = partition_network(segs_arr)

    # ── 闭合环（完整三项）────────────────────────────────────────────
    for path_v, path_s, b_unit, n_unit in loops:
        vertices = [pos_m[n] for n in path_v]
        b_vec    = b_unit * b_mag

        # 参考点：环中心沿滑移面法向偏移一个 b，决定立体角符号
        center              = np.mean(vertices, axis=0)
        positive_side_point = center + n_unit * b_mag

        u_loop = displacement_loop(x_obs, vertices, b_vec, nu, a,
                                   positive_side_point)

        # 将环的总贡献均分到各线段（便于 per_seg 输出分析）
        contrib = u_loop / len(path_s)
        for si in path_s:
            u_per_seg[si] = contrib

    # ── 开放线段（第二+第三项）──────────────────────────────────────
    for si in open_seg_idx:
        seg = segs_arr[si]
        n1, n2 = int(seg[0]), int(seg[1])
        bv = seg[2:5] * b_mag
        u_per_seg[si] = disp_line_segment(x_obs, pos_m[n1], pos_m[n2],
                                           bv, nu, a)

    return u_per_seg, u_per_seg.sum(axis=0), len(loops)


# ════════════════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════════════════

def main():
    output_dir = 'output'
    disp_dir   = os.path.join(output_dir, 'displacement')
    os.makedirs(disp_dir, exist_ok=True)

    data_files = sorted([
        f for f in os.listdir(output_dir)
        if f.startswith('config.') and f.endswith('.data')
    ])

    if not data_files:
        print("未找到 .data 文件，请先运行模拟。")
        return

    pyexadis.initialize()

    print(f"观测点 x_obs = {x_obs / b_mag} × b")
    print(f"找到 {len(data_files)} 个时间步文件\n")

    summary_lines = ["step, n_segs, n_loops, |u_total|/b, u_x/b, u_y/b, u_z/b"]

    for fname in data_files:
        fpath = os.path.join(output_dir, fname)
        stem  = fname.replace('.data', '')

        pos_m, nodes_arr, segs_arr = load_network(fpath)

        if pos_m is None:
            print(f"{fname}: 空网络，跳过")
            continue

        u_per_seg, u_total, n_loops = compute_displacement(
            pos_m, segs_arr, x_obs, nu, a
        )

        np.save(os.path.join(disp_dir, f'{stem}_per_seg.npy'), u_per_seg)
        np.save(os.path.join(disp_dir, f'{stem}_total.npy'),   u_total)

        u_mag   = np.linalg.norm(u_total)
        n_segs  = len(segs_arr)
        summary_lines.append(
            f"{stem}, {n_segs}, {n_loops}, "
            f"{u_mag/b_mag:.6f}, "
            f"{u_total[0]/b_mag:.6f}, "
            f"{u_total[1]/b_mag:.6f}, "
            f"{u_total[2]/b_mag:.6f}"
        )
        print(f"{fname}: {n_segs} 段, {n_loops} 个闭合环, |u| = {u_mag/b_mag:.4f} b")

    pyexadis.finalize()

    summary_path = os.path.join(disp_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"\n汇总已保存至 {summary_path}")


if __name__ == "__main__":
    main()
