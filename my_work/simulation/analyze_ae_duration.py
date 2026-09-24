#!/usr/bin/env python3
"""
从 ae_displacement.dat 估算 AE 事件的持续时间，并和弹性波传播时间比较。

用途：判断"静态位移公式"够不够用。
  事件持续时间 >> 波传播时间  -> 静态近似合理
  事件持续时间 ~  波传播时间  -> 静态近似不可靠，需要弹性动力学解

用法：
  python analyze_ae_duration.py <ae_displacement.dat> <stress_strain_dens.dat> --erate 50

做了什么：
  1. 时间轴：应力文件每 10 步给一个应变，时间 = 应变 / 应变率，
     中间的步用线性插值（这是近似）。
  2. 速度：|v| = |位移增量| / 这一步的时间间隔。第 0 步不参与（增量特别大）。
  3. 事件：|v| 超过 k × 中位数 的连续步算一个事件（--gap 可以允许中间断开几步）。
  4. 输出：事件数、持续时间的分布、和波传播时间的对比，并存两张图。
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MEAN_DIST_FRAC = 0.4803   # 立方体中心到盒内均匀随机点的平均距离，约 0.4803 倍边长


def find_events(v_mag, thr, gap):
    """返回每个事件的 (起始下标, 结束下标)。超过 thr 的步算事件，中间断开不超过 gap 步的并成一个。"""
    idx = np.flatnonzero(v_mag > thr)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > gap + 1) + 1
    return [(g[0], g[-1]) for g in np.split(idx, breaks)]


def main():
    ap = argparse.ArgumentParser(description="估算 AE 事件的持续时间")
    ap.add_argument("ae_file", help="ae_displacement.dat")
    ap.add_argument("stress_file", help="stress_strain_dens.dat（用来补时间轴）")
    ap.add_argument("--erate", type=float, required=True, help="应变率 [/s]，e50 是 50，e1000 是 1000")
    ap.add_argument("--point", type=int, default=0, help="用第几个场点，默认 0")
    ap.add_argument("--k", type=float, default=10.0, help="阈值 = k × |v| 的中位数，默认 10")
    ap.add_argument("--gap", type=int, default=0, help="允许事件中间断开几步，默认 0")
    ap.add_argument("--mu", type=float, default=54.6e9, help="剪切模量 [Pa]")
    ap.add_argument("--rho", type=float, default=8960.0, help="密度 [kg/m^3]，脚本里没有，这是铜的通用值")
    ap.add_argument("--box_um", type=float, default=5.0, help="盒子边长 [um]")
    ap.add_argument("--plot_window", type=int, nargs=2, default=[0, 5000], help="检查图画哪一段步数")
    ap.add_argument("--out", default=None, help="图的输出文件夹，默认在 ae 文件所在文件夹下的 plots/")
    args = ap.parse_args()

    # ---------- 读数据 ----------
    ae = np.loadtxt(args.ae_file, comments="#")
    ss = np.loadtxt(args.stress_file, comments="#")
    n_points = (ae.shape[1] - 1) // 6
    if args.point >= n_points:
        raise SystemExit(f"文件里只有 {n_points} 个场点，--point {args.point} 超出范围")

    steps_all = ae[:, 0].astype(int)
    base = 1 + 6 * args.point
    du_all = ae[:, base:base + 3]                       # 每步位移增量的三个分量

    # ---------- 时间轴 ----------
    steps_s = np.concatenate(([0.0], ss[:, 0]))          # 应力文件的步数（补上第 0 步）
    time_s = np.concatenate(([0.0], ss[:, 1])) / args.erate   # 时间 = 应变 / 应变率
    last_step = int(steps_s[-1])
    t_full = np.interp(np.arange(0, last_step + 1), steps_s, time_s)   # 每一步的时间
    dt_full = np.diff(t_full)                                           # dt_full[n-1] = 第 n 步的时间间隔

    keep = (steps_all >= 1) & (steps_all <= last_step)   # 去掉第 0 步，以及应力文件之外的最后几步
    steps = steps_all[keep]
    du = du_all[keep]
    dt = dt_full[steps - 1]
    n_dropped = int(np.sum(steps_all > last_step))

    # ---------- 速度 ----------
    v_mag = np.linalg.norm(du, axis=1) / dt              # 单位：代码位移单位 / s，只看相对大小

    # ---------- 波传播时间 ----------
    c_T = np.sqrt(args.mu / args.rho)                    # 横波速度
    d_max = np.sqrt(3.0) / 2.0 * args.box_um * 1e-6      # 场点在盒子中心：到盒角的最远距离
    d_mean = MEAN_DIST_FRAC * args.box_um * 1e-6
    t_wave_max = d_max / c_T
    t_wave_mean = d_mean / c_T

    # ---------- 主分析 ----------
    med = np.median(v_mag)
    thr = args.k * med
    events = find_events(v_mag, thr, args.gap)
    n_ev = len(events)
    len_steps = np.array([e - s + 1 for s, e in events])
    dur_ns = np.array([dt[s:e + 1].sum() * 1e9 for s, e in events])

    print(f"读入 {len(steps_all)} 行 AE 数据，分析第 {steps[0]} 到 {steps[-1]} 步（丢掉第 0 步和应力文件之外的 {n_dropped} 步）")
    print(f"平均每步 dt = {dt.mean()*1e9:.3f} ns（最小 {dt.min()*1e9:.3f}，最大 {dt.max()*1e9:.3f}）")
    print(f"|v| 的中位数 = {med:.3e}，阈值 = {args.k:g} × 中位数 = {thr:.3e}")
    print(f"横波速度 c_T = {c_T:.0f} m/s；波传播时间：到盒角（最远）{t_wave_max*1e9:.2f} ns，到盒内平均位置 {t_wave_mean*1e9:.2f} ns")
    print()
    if n_ev == 0:
        print("没有找到事件，试试减小 --k")
        return

    p10, p50, p90 = np.percentile(dur_ns, [10, 50, 90])
    print(f"事件数：{n_ev}")
    print(f"持续时间：中位数 {p50:.2f} ns（10% 分位 {p10:.2f}，90% 分位 {p90:.2f}，最长 {dur_ns.max():.2f}）")
    print(f"持续步数：中位数 {np.median(len_steps):.0f}，只有 1 步的占 {100*np.mean(len_steps == 1):.0f}%，最长 {len_steps.max()}")
    print(f"持续时间 / 最远波传播时间：中位数 {p50 / (t_wave_max*1e9):.1f}")
    print(f"  持续时间 < 1 倍波传播时间（静态近似不可靠）：{100*np.mean(dur_ns < t_wave_max*1e9):.0f}%")
    print(f"  持续时间 >= 10 倍波传播时间（静态近似合理）：{100*np.mean(dur_ns >= 10*t_wave_max*1e9):.0f}%")

    print("\n阈值敏感性（k 不同时结果变多少）：")
    print(f"{'k':>5} {'事件数':>8} {'中位持续(ns)':>14} {'中位步数':>9} {'>=10倍波时间':>13}")
    for kk in (5, 10, 20, 50):
        ev = find_events(v_mag, kk * med, args.gap)
        if not ev:
            print(f"{kk:>5} {0:>8}")
            continue
        d = np.array([dt[s:e + 1].sum() * 1e9 for s, e in ev])
        ls = np.array([e - s + 1 for s, e in ev])
        print(f"{kk:>5} {len(ev):>8} {np.median(d):>14.2f} {np.median(ls):>9.0f} {100*np.mean(d >= 10*t_wave_max*1e9):>12.0f}%")

    # ---------- 图 ----------
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ae_file)), "plots")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.geomspace(max(dur_ns.min(), 1e-3), dur_ns.max() * 1.01, 30) if dur_ns.max() > dur_ns.min() else 10
    ax.hist(dur_ns, bins=bins, color="tab:blue", alpha=0.8)
    ax.set_xscale("log")
    ax.axvline(t_wave_max * 1e9, color="tab:red", linestyle="--", label=f"wave travel time to corner ({t_wave_max*1e9:.2f} ns)")
    ax.axvline(10 * t_wave_max * 1e9, color="tab:green", linestyle="--", label="10 x wave travel time")
    ax.set_xlabel("Event duration (ns)")
    ax.set_ylabel("Number of events")
    ax.set_title(f"Event durations (threshold = {args.k:g} x median |v|, {n_ev} events)")
    ax.legend()
    fig.tight_layout()
    f1 = os.path.join(out_dir, "event_duration_hist.png")
    fig.savefig(f1, dpi=200)
    plt.close(fig)

    lo, hi = args.plot_window
    m = (steps >= lo) & (steps <= hi)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(steps[m], v_mag[m], "k-", linewidth=0.5)
    ax.axhline(thr, color="tab:red", linestyle="--", label=f"threshold ({args.k:g} x median)")
    for s, e in events:
        if steps[e] >= lo and steps[s] <= hi:
            ax.axvspan(steps[s] - 0.5, steps[e] + 0.5, color="tab:orange", alpha=0.4)
    ax.set_xlabel("Step")
    ax.set_ylabel("|v| (code units / s)")
    ax.set_title(f"Threshold check, steps {lo} to {hi} (shaded = detected events)")
    ax.legend()
    fig.tight_layout()
    f2 = os.path.join(out_dir, "event_threshold_check.png")
    fig.savefig(f2, dpi=200)
    plt.close(fig)
    print(f"\n图已保存：\n  {f1}\n  {f2}")


if __name__ == "__main__":
    main()
