"""
位移场后处理脚本
计算底面（z=0）上的位移场，对应声发射传感器位置
"""

import numpy as np
import sys, os, glob

sys.path.insert(0, '/data/home/dg000246c/OpenDis/my_work/displacement')
sys.path.insert(0, '/data/home/dg000246c/OpenDis/core/exadis/python')

from displacement import disp_line_segment

# ════════════════════════════════════════════════════════════════════════
# 物理参数
# ════════════════════════════════════════════════════════════════════════

b_mag = 2.55e-10   # m
nu    = 0.324
a     = 6 * b_mag

L_phys = 5e-6      # m
Lbox   = L_phys / b_mag  # burgmag 单位


# ════════════════════════════════════════════════════════════════════════
# 读取 .data 文件
# ════════════════════════════════════════════════════════════════════════

def read_data_file(datafile):
    """读取 ParaDiS .data 文件，返回线段列表 [(x1, x2, b), ...]"""
    nodes = {}
    node_arms = {}

    with open(datafile) as f:
        lines = f.readlines()

    # 找到 nodalData 开始位置
    start = 0
    for i, line in enumerate(lines):
        if 'nodalData' in line:
            start = i + 3
            break

    i = start
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 5 and ',' in parts[0]:
            try:
                domain_tag = parts[0].rstrip(',')
                domain, tag = map(int, domain_tag.split(','))
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                num_arms = int(parts[4])
                key = (domain, tag)
                nodes[key] = np.array([x, y, z]) * b_mag  # 转换为物理单位(m)

                arms = []
                for _ in range(num_arms):
                    i += 1
                    arm_line = lines[i].strip().split()
                    arm_dt = arm_line[0].rstrip(',')
                    arm_domain, arm_tag = map(int, arm_dt.split(','))
                    bx, by, bz = float(arm_line[1]), float(arm_line[2]), float(arm_line[3])
                    i += 1  # 跳过法向量行
                    arms.append(((arm_domain, arm_tag), np.array([bx, by, bz]) * b_mag))
                node_arms[key] = arms
            except Exception:
                pass
        i += 1

    # 组装线段（避免重复）
    segs = []
    for key, arms in node_arms.items():
        x1 = nodes.get(key)
        if x1 is None:
            continue
        for (arm_domain, arm_tag), b in arms:
            n2_key = (arm_domain, arm_tag)
            x2 = nodes.get(n2_key)
            if x2 is None:
                continue
            if key < n2_key:
                segs.append((x1, x2, b))

    return segs


# ════════════════════════════════════════════════════════════════════════
# 计算底面位移场
# ════════════════════════════════════════════════════════════════════════

def compute_bottom_surface(segs, N_grid=20):
    """计算底面（z=0）上 N_grid×N_grid 网格的位移场"""

    # 场点：底面 z=0，x和y从盒子10%到90%均匀分布
    coords = np.linspace(0.1 * L_phys, 0.9 * L_phys, N_grid)
    pts = []
    for x in coords:
        for y in coords:
            pts.append(np.array([x, y, 0.0]))

    u_field = np.zeros((len(pts), 3))
    for idx, xf in enumerate(pts):
        u = np.zeros(3)
        for x1, x2, b in segs:
            u += disp_line_segment(xf, x1, x2, b, nu, a)
        u_field[idx] = u
        if (idx + 1) % 100 == 0:
            print(f"    场点 {idx+1}/{len(pts)} 完成")

    return np.array(pts), u_field


# ════════════════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════════════════

def main():
    input_dir  = '/data/home/dg000246c/OpenDis/my_work/simulation/output_FR_v2'
    output_dir = '/data/home/dg000246c/OpenDis/my_work/postprocess/disp_output'
    os.makedirs(output_dir, exist_ok=True)

    # 每50步取一个文件
    all_files = sorted(
        glob.glob(os.path.join(input_dir, 'config.*.data')),
        key=lambda f: int(os.path.basename(f).split('.')[1])
    )
    selected = [f for f in all_files
                if int(os.path.basename(f).split('.')[1]) % 50 == 0]

    print(f"找到 {len(all_files)} 个文件，选取 {len(selected)} 个处理")

    for datafile in selected:
        step = int(os.path.basename(datafile).split('.')[1])
        print(f"\n处理第 {step} 步...")

        segs = read_data_file(datafile)
        print(f"  读取到 {len(segs)} 条线段")

        if len(segs) == 0:
            print("  没有线段，跳过")
            continue

        pts, u_field = compute_bottom_surface(segs, N_grid=20)

        # 保存结果
        np.save(os.path.join(output_dir, f'pts_bottom.npy'), pts)
        np.save(os.path.join(output_dir, f'disp_bottom_step{step:05d}.npy'), u_field)
        print(f"  保存完成，位移最大值 = {np.max(np.abs(u_field)):.3e} m")

    print("\n全部完成！")


if __name__ == "__main__":
    main()