#!/usr/bin/env python3
"""总览图：应力-应变、位错密度、AE位移。

用法：python plot_overview.py <输出文件夹>
      （不写文件夹就用当前目录）
读取：stress_strain_dens.dat、ae_displacement.dat
保存：<输出文件夹>/plots/ 下的两张 PNG
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # 无显示器环境也能存图（集群终端没有窗口）
import matplotlib.pyplot as plt
import numpy as np

folder = sys.argv[1] if len(sys.argv) > 1 else "."
plot_dir = os.path.join(folder, "plots")
os.makedirs(plot_dir, exist_ok=True)

# ---------- 图1：应力-应变、位错密度-应变 ----------
# 列：Step Strain Stress Density Walltime（表头以 # 开头，loadtxt 会自动跳过）
ss = np.loadtxt(os.path.join(folder, "stress_strain_dens.dat"), comments="#")
strain, stress, density = ss[:, 1], ss[:, 2], ss[:, 3]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(strain, stress / 1e6, "b-", linewidth=1.0)
ax1.set_xlabel("Strain")
ax1.set_ylabel("Stress (MPa)")
ax1.set_title("Stress-Strain")
ax1.grid(True, alpha=0.3)

ax2.plot(strain, density, "g-", linewidth=1.0)
ax2.set_xlabel("Strain")
ax2.set_ylabel("Dislocation density (m$^{-2}$)")
ax2.set_title("Dislocation Density vs Strain")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "stress_strain_density.png"), dpi=200)
plt.close(fig)

# ---------- 图2：AE 位移 ----------
# 列：step，然后每个场点 6 列：du_x du_y du_z u_x u_y u_z
ae = np.loadtxt(os.path.join(folder, "ae_displacement.dat"), comments="#")
n_points = (ae.shape[1] - 1) // 6
steps = ae[:, 0]

for p in range(n_points):
    base = 1 + 6 * p
    du = ae[:, base:base + 3]      # 这一步的位移增量
    u = ae[:, base + 3:base + 6]   # 累积位移
    du_mag = np.linalg.norm(du, axis=1)  # 增量的大小 |du|

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 第0步的增量特别大，放进对数图会把其他点压得很小，所以这里不画它
    ax1.plot(steps[1:], du_mag[1:], "k-", linewidth=0.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("|du| per step (code units)")
    ax1.set_title(f"Field point {p}: increment size per step (step 0 not shown)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, u[:, 0], label="u_x", linewidth=0.8)
    ax2.plot(steps, u[:, 1], label="u_y", linewidth=0.8)
    ax2.plot(steps, u[:, 2], label="u_z", linewidth=0.8)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Cumulative displacement (code units)")
    ax2.set_title(f"Field point {p}: cumulative displacement")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"ae_displacement_point{p}.png"), dpi=200)
    plt.close(fig)

print("图已保存到：", plot_dir)
print("应力文件数据点数：", len(strain), "；AE文件行数：", len(steps), "；场点数：", n_points)
