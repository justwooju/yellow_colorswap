#!/usr/bin/env python3
"""fig/*.png 를 data/*.csv 에서 다시 만든다.

    python3 tools/make_figures.py

색은 Okabe & Ito (2008) 팔레트. 좌표는 로봇 베이스 기준 mm (x 좌우, y 앞).
"""
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "data", ROOT / "fig"

OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000"}
INK, INK2, GRID, RACK = "#222222", "#555555", "#DDDDDD", "#9DB9C9"
VC = {"success": OI["green"], "partial": OI["orange"], "fail": OI["vermillion"]}
VK = {"success": "성공", "partial": "부분", "fail": "실패"}
VM = {"success": ("o", OI["green"]), "partial": ("D", OI["orange"]), "fail": ("X", OI["black"])}
RACK_L, RACK_W, PITCH = 170.0, 40.0, 37.2

plt.rcParams.update({
    "font.family": ["Noto Sans CJK KR", "Noto Sans CJK JP", "DejaVu Sans"],
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold",
    "axes.titlesize": 11, "legend.frameon": False,
})


def rd(name):
    p = DATA / name
    if not p.is_file():
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rack(ax, x, y, yaw, alpha=1.0, holes=True):
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    a, b = RACK_L / 2, RACK_W / 2
    P = [(x + c * u - s * v, y + s * u + c * v) for u, v in ((-a, -b), (a, -b), (a, b), (-a, b))]
    ax.add_patch(Polygon(P, closed=True, fc="none", ec=RACK, lw=1.4, alpha=alpha, zorder=1))
    if holes:
        for k in range(4):
            u = (k - 1.5) * PITCH
            ax.add_patch(Circle((x + c * u, y + s * u), 15, fc="none", ec=RACK, lw=0.8, alpha=alpha, zorder=1))


def tubes(ax, t, size=1.0, label=False):
    tx, ty, ox, oy, rx, ry = (fl(t[k]) for k in ("target_x", "target_y", "other_x", "other_y", "ring_x", "ring_y"))
    yel = t["task"] == "yellow"
    tc, oc = (OI["yellow"], OI["vermillion"]) if yel else (OI["vermillion"], OI["yellow"])
    if rx is not None:
        ax.add_patch(Circle((rx, ry), 19 * size, fc="none", ec=OI["orange"] if yel else OI["vermillion"], lw=3 * size, zorder=2))
    if tx is not None:
        ax.add_patch(Circle((tx, ty), 14 * size, fc=tc, ec=INK, lw=0.8, zorder=3))
    if ox is not None:
        ax.add_patch(Circle((ox, oy), 14 * size, fc=oc, ec=INK, lw=0.8, alpha=0.9, zorder=3))
    if label:
        if tx is not None:
            ax.annotate("노란 튜브(목표)" if yel else "빨간 튜브(목표)", (tx, ty), (tx - 10, ty - 48), fontsize=9, color=INK, ha="center",
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
        if ox is not None:
            ax.annotate("빨간 튜브(방해물)" if yel else "노란 튜브", (ox, oy), (ox + 25, oy - 52), fontsize=9, color=INK, ha="center",
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
        if rx is not None:
            ax.annotate("노란 링 칸" if yel else "빨간 링 칸", (rx, ry), (rx + 34, ry - 34), fontsize=9, color=INK, ha="left",
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))


# ------------------------------------------------------------------ 1. 11-2 한 판: 위에서 본 손끝 경로 · 높이·폭
def fig_trial(T, R, tag="11-2"):
    t = next((x for x in T if x["task"] == "yellow" and f"{x['layout']}-{x['k']}" == tag), None)
    P = rd(f"trial_{tag}_tcp.csv")
    if t is None or not P:
        return
    A = np.array([[float(r[k]) for k in ("t_s", "x_mm", "y_mm", "z_mm", "width_mm")] for r in P])
    tc, to = fl(t["close_t"]), fl(t["open_t"])
    pre, held, post = A[:, 0] <= tc, (A[:, 0] >= tc) & (A[:, 0] <= to), A[:, 0] >= to
    fig = plt.figure(figsize=(12.5, 4.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.12, wspace=0.18)
    ax = fig.add_subplot(gs[:, 0])
    for r in R:
        if r["stamp"] == t["stamp"]:
            rack(ax, float(r["x"]), float(r["y"]), float(r["yaw"]))
    tubes(ax, t, label=True)
    for m, col, lab in ((pre, OI["sky"], "잡기까지"), (held, OI["blue"], "쥐고 옮기기"), (post, "#999999", "놓은 뒤")):
        ax.plot(A[m, 1], A[m, 2], color=col, lw=2.2, label=lab, zorder=4)
    cx, cy, ox_, oy_ = (fl(t[k]) for k in ("close_x", "close_y", "open_x", "open_y"))
    ax.plot(cx, cy, "X", ms=9, color=INK, zorder=5)
    ax.plot(ox_, oy_, "v", ms=9, color=INK, zorder=5)
    ax.annotate(f"닫은 곳 — 목표 중심에서 {float(t['d_close_target']):.0f} mm", (cx, cy), (cx - 100, cy + 72), fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.annotate(f"놓은 곳 — 링 칸 중심에서 {float(t['d_open_ring']):.0f} mm", (ox_, oy_), (ox_ + 20, oy_ + 74), fontsize=9, ha="right",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.plot(A[0, 1], A[0, 2], "o", ms=6, color=INK2, zorder=5)
    ax.annotate("시작", (A[0, 1], A[0, 2]), (A[0, 1] + 14, A[0, 2] - 6), fontsize=9, color=INK2)
    ax.set_aspect("equal")
    ax.set_xlim(-170, 330)
    ax.set_ylim(395, 690)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"배치 {tag} 손끝 경로(위에서 본 모습)")
    ax.legend(loc="lower left", fontsize=9)
    az, aw = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    az.plot(A[:, 0], A[:, 3], color=INK, lw=1.6)
    aw.plot(A[:, 0], A[:, 4], color=OI["blue"], lw=1.6)
    for a in (az, aw):
        a.axvline(tc, color=OI["vermillion"], ls="--", lw=1.1)
        a.axvline(to, color=OI["blue"], ls="--", lw=1.1)
        a.set_xlim(0, A[-1, 0])
    az.text(tc + 0.4, A[:, 3].max() - 8, f"닫음 {tc:.1f} s", color=OI["vermillion"], fontsize=9, va="top")
    az.text(to - 0.4, A[:, 3].max() - 8, f"놓음 {to:.1f} s", color=OI["blue"], fontsize=9, va="top", ha="right")
    az.set_ylabel("손끝 높이 z (mm)")
    az.set_xticklabels([])
    aw.set_ylabel("손가락 사이 (mm)")
    aw.set_xlabel("실행 시작 뒤 시간 (s)")
    az.set_title("손끝 높이와 그리퍼 폭")
    fig.savefig(FIG / f"trial_{tag}_path.png")
    plt.close(fig)


# ------------------------------------------------------------------ 2. 배치별 결과
def fig_outcomes(T):
    Y = [t for t in T if t["task"] == "yellow"]
    lays = sorted({int(t["layout"]) for t in Y})
    groups = [(f"L{l}", [t for t in Y if int(t["layout"]) == l]) for l in lays]
    Rc = [t for t in T if t["task"] == "red_check"]
    if Rc:
        groups.append(("빨강 과제\n확인", Rc))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for v in ("success", "partial", "fail"):
        n = np.array([sum(1 for t in g if t["verdict"] == v) for _, g in groups])
        ax.bar(x, n, 0.62, bottom=bottom, color=VC[v], edgecolor="white", linewidth=1.5, label=VK[v])
        for i, k in enumerate(n):
            if k:
                ax.text(x[i], bottom[i] + k / 2, str(k), ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        bottom += n
    ax.set_xticks(x, [g for g, _ in groups])
    for i, (g, _) in enumerate(groups):
        if g == "L11":
            ax.annotate("튜브를 꽉 찬 색으로\n칠한 조건", (x[i], bottom[i]), (x[i] - 1.2, bottom[i] + 1.2), fontsize=8.5, color=INK2, ha="center",
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    if Rc:
        ax.axvline(len(groups) - 1.5, color=GRID, lw=1.2)
    ax.set_ylabel("유효 시행 수")
    ax.set_ylim(0, max(bottom) + 2.2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_title("배치별 결과 (사람 판정, 무효 판 제외)")
    ax.legend(ncol=3, loc="upper left", fontsize=9)
    fig.savefig(FIG / "outcomes_by_layout.png")
    plt.close(fig)


# ------------------------------------------------------------------ 3. 닫은 자리: 목표(노랑)까지 vs 방해물(빨강)까지
def fig_close(T):
    Y = [t for t in T if t["task"] == "yellow" and fl(t["d_close_target"]) is not None and fl(t["d_close_other"]) is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True, sharey=True)
    for ax, sel, title in ((axes[0], [t for t in Y if t["solid"] != "1"], "배치 1–10 (기본 색 바꾸기)"),
                           (axes[1], [t for t in Y if t["solid"] == "1"], "배치 11 (튜브를 꽉 찬 색으로 바꾸기)")):
        lim = 300
        ax.fill_between([0, lim], [0, lim], [lim, lim], color=OI["yellow"], alpha=0.12, lw=0)
        ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls="--")
        for v in ("success", "partial", "fail"):
            pts = [(float(t["d_close_target"]), float(t["d_close_other"])) for t in sel if t["verdict"] == v]
            if pts:
                m, c = VM[v]
                ax.scatter(*zip(*pts), marker=m, s=46, c=c, edgecolors="white" if v != "fail" else None, linewidths=0.6, zorder=3,
                           label=f"{VK[v]} {len(pts)}")
        n_t = sum(1 for t in sel if float(t["d_close_target"]) < float(t["d_close_other"]))
        ax.text(0.97, 0.04, f"노랑 쪽에서 닫음 {n_t}/{len(sel)}", transform=ax.transAxes, ha="right", fontsize=9.5, color=INK)
        ax.text(0.04, 0.95, "노랑에 더 가까움", transform=ax.transAxes, fontsize=9, color="#8a7a00", va="top")
        ax.text(0.96, 0.62, "빨강에 더 가까움", transform=ax.transAxes, fontsize=9, color=INK2, ha="right")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("닫은 곳 → 노란 튜브(목표) (mm)")
        ax.legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.0, 0.9))
    axes[0].set_ylabel("닫은 곳 → 빨간 튜브(방해물) (mm)")
    fig.suptitle("처음 그리퍼를 닫은 자리 (유효 시행)", fontweight="bold", y=1.0)
    fig.savefig(FIG / "close_target_vs_distractor.png")
    plt.close(fig)


# ------------------------------------------------------------------ 4. 배치별 위에서 본 장면과 닫은 자리
def groups_of(S, tol=20.0):
    """같은 배치 번호 안에서도 튜브 자리가 바뀐 판이 있다 — 목표·방해물 자리가 둘 다 tol mm 안이면 한 장면으로 묶는다."""
    G = []
    for t in S:
        a = (fl(t["target_x"]), fl(t["target_y"]), fl(t["other_x"]), fl(t["other_y"]))
        for g in G:
            b = g["key"]
            if None not in a and None not in b and math.dist(a[:2], b[:2]) < tol and math.dist(a[2:], b[2:]) < tol:
                g["S"].append(t)
                break
        else:
            G.append({"key": a, "S": [t]})
    return G


def fig_layouts(T, R):
    Y = [t for t in T if t["task"] == "yellow"]
    panels = []
    for l in sorted({int(t["layout"]) for t in Y}):
        G = groups_of([t for t in Y if int(t["layout"]) == l])
        for g in G:
            a = g["key"]
            g["swap"] = any(h is not g and None not in a and None not in h["key"] and math.dist(a[:2], h["key"][2:]) < 20
                            and math.dist(a[2:], h["key"][:2]) < 20 for h in G[:G.index(g)])
            panels.append((l, g))
    cols = 4
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.9 * cols, 2.05 * rows + 0.9))
    axes = np.atleast_1d(axes).ravel()
    for ax, (l, g) in zip(axes, panels):
        S = g["S"]
        t0 = S[0]
        for r in R:
            if r["stamp"] == t0["stamp"]:
                rack(ax, float(r["x"]), float(r["y"]), float(r["yaw"]), alpha=0.8, holes=False)
        tubes(ax, t0, size=0.75)
        for t in S:
            cx, cy = fl(t["close_x"]), fl(t["close_y"])
            if cx is None:
                continue
            m, c = VM[t["verdict"]]
            ax.plot([cx, float(t["target_x"])], [cy, float(t["target_y"])], color=INK2, lw=0.5, alpha=0.6, zorder=3)
            ax.scatter([cx], [cy], marker=m, s=34, c=c, edgecolors="white" if t["verdict"] != "fail" else None, linewidths=0.5, zorder=5)
        k = sum(1 for t in S if t["verdict"] == "success")
        ids = "·".join(t["k"] for t in S)
        ax.set_title(f"L{l}-{ids}  성공 {k}/{len(S)}" + ("  · 꽉 찬 색" if t0["solid"] == "1" else ""), fontsize=9.5)
        if g["swap"]:
            ax.text(0.5, 0.04, "앞 장면과 노랑·빨강 자리를 맞바꿈", transform=ax.transAxes, ha="center", fontsize=8, color=OI["vermillion"])
        ax.set_aspect("equal")
        ax.set_xlim(-200, 320)
        ax.set_ylim(455, 665)
        ax.tick_params(labelsize=7)
    for ax in axes[len(panels):]:
        ax.axis("off")
    hs = [Line2D([], [], marker="o", ls="", mfc=OI["yellow"], mec=INK, ms=9, label="노란 튜브(목표)"),
          Line2D([], [], marker="o", ls="", mfc=OI["vermillion"], mec=INK, ms=9, label="빨간 튜브(방해물)"),
          Line2D([], [], marker="o", ls="", mfc="none", mec=OI["orange"], mew=2.5, ms=11, label="노란 링 칸")]
    hs += [Line2D([], [], marker=VM[v][0], ls="", color=VM[v][1], ms=7, label=f"처음 닫은 곳 — {VK[v]}") for v in ("success", "partial", "fail")]
    fig.legend(handles=hs, loc="lower center", ncol=6, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("장면별 배치와 처음 닫은 자리 (위에서 본 모습, 로봇 베이스 기준 mm · 회색 선은 닫은 곳 → 목표)", fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(FIG / "layouts_topview.png")
    plt.close(fig)


# ------------------------------------------------------------------ 5. 학습 데이터 자리와 이번 시험 자리
def fig_train(T):
    D = rd("train_positions.csv")
    Y = [t for t in T if t["task"] == "yellow"]
    if not D or not Y:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    tr = np.array([[float(r["red_x"]), float(r["red_y"])] for r in D])
    ts = np.array([[float(r["slot_x"]), float(r["slot_y"])] for r in D])
    yt = np.array([[float(t["target_x"]), float(t["target_y"])] for t in Y if fl(t["target_x"]) is not None])
    yr = np.array([[float(t["ring_x"]), float(t["ring_y"])] for t in Y if fl(t["ring_x"]) is not None])
    for ax, P, Q, a, b, title in ((axes[0], tr, yt, "학습: 빨간 튜브를 잡은 자리", "시험: 노란 튜브(목표) 자리", "잡는 자리"),
                                  (axes[1], ts, yr, "학습: 꽂은 링 칸", "시험: 노란 링 칸", "꽂는 자리")):
        ax.scatter(P[:, 0], P[:, 1], s=10, c=OI["sky"], alpha=0.55, lw=0, label=f"{a} ({len(P)}판)")
        ax.scatter(Q[:, 0], Q[:, 1], s=90, marker="*", c=OI["orange"], edgecolors=INK, linewidths=0.5, zorder=3,
                   label=f"{b} ({len(Q)}회)")
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.legend(loc="upper left", fontsize=8.5)
        ax.set_xlim(-260, 340)
        ax.set_ylim(400, 700)
    axes[0].set_ylabel("y (mm)")
    fig.savefig(FIG / "train_vs_test_positions.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    T, R = rd("trials.csv"), rd("racks.csv")
    fig_trial(T, R)
    if len(T) > 1:
        fig_outcomes(T)
        fig_close(T)
        fig_layouts(T, R)
        fig_train(T)
    print("fig:", ", ".join(sorted(p.name for p in FIG.glob("*.png"))))


if __name__ == "__main__":
    main()
