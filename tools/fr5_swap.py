"""실행 중 색 바꾸기 — 노란 튜브 → 노란 링을 빨강 과제(200판) 정책으로. fr5_policy_try.py --swap 이 쓴다(같은 폴더에 둔다).

실물: 노란 튜브(목표) · 빨간 튜브(방해) · 노란 링(목표 칸). 정책·fr5auto 에 주는 영상에서 그 셋을 학습 장면처럼
빨간 튜브 · 노란 튜브 · 빨간 링으로 바꾼다. 마스크는 색이 아니라 자리로(시작 장면 인식 + 3인칭 보정 + 손목 자리·관절 FK,
쥔 튜브는 TCP 를 따라감). 바꿀 목표 모습(색조·채도·밝기 분포)은 학습 200판에서 잰다(work/colorswap_stats.json 에 저장).
오프라인 시험(fr5_colorswap_test.py, 2026-11-11): 바꿨다 되돌린 영상에 대한 정책 반응 = 잡음 바닥(120 질의)."""
from __future__ import annotations

import json
import math
import os
import threading
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from fr5auto import camera as CM
from fr5auto import kin as K
from fr5auto import rackcal as RCAL
from fr5auto import rackvis as RV
from fr5auto import rackwrist as WR

RED, YEL = "red", "yellow"
SOLID = int(os.environ.get("FR5_SWAP_SOLID", "0") or 0)             # 1: 3인칭 튜브를 꽉 찬 색으로 · 2: 손목 튜브도 (패치3)
S_MIN = {"third": (0.28, 0.20), "wrist": (0.40, 0.40)}                # (확실한 색 채도, 링·손목 몸통 채도) — 손목은 책상(나무색)이 채도 0.2–0.3 이라 높게
DEF = {"red_tube": {"h": 7.0, "s": 0.55, "v": 0.48, "ss": 0.08, "sv": 0.10},       # 실물 ep_177 에서 잰 값(학습 통계가 없을 때만)
       "yellow_tube": {"h": 56.0, "s": 0.38, "v": 0.60, "ss": 0.08, "sv": 0.10},
       "red_ring": {"h": 4.0, "s": 0.55, "v": 0.40, "ss": 0.10, "sv": 0.10},
       "yellow_ring": {"h": 52.0, "s": 0.60, "v": 0.70, "ss": 0.10, "sv": 0.10}}


def geom(cfg):
    rc = cfg["rack"]
    zt = float(rc["z_top_mm"])
    dz = float(rc["grasp_dz_mm"]) + float(rc.get("grasp_dz_adj_mm") or 0.0)
    return {"z_top": zt, "pitch": float(rc["pitch_mm"]), "n_holes": int(rc["n_holes"]), "tube_r": float(rc["tube_d_mm"]) / 2,
            "z_table": zt - float(rc["top_h_mm"]), "z_body_top": zt + float(rc["tube_above_mm"]) - float(rc["cap_h_mm"]),
            "body_len": float(rc["tube_len_mm"]) - float(rc["cap_h_mm"]),
            "tcp_to_bottom": dz + float(rc["tube_len_mm"]) - float(rc["tube_above_mm"]),
            "ring_h": float((rc.get("ring") or {}).get("h_mm") or 5.0), "ring_ri": 17.0, "ring_ro": 20.5}


# ------------------------------------------------------------------ 자리 → 마스크
def hull(P):
    P = sorted(map(tuple, P))
    if len(P) < 3:
        return P

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def poly(wh, uv, grow=0, off=(0.0, 0.0)):
    m = Image.new("L", wh, 0)
    uv = uv[np.all(np.isfinite(uv), axis=1)] - np.asarray(off, float)
    if len(uv) >= 3:
        pts = [tuple(map(float, p)) for p in hull(uv)]
        if grow:
            try:                                                        # 테두리를 굵게 그려 넓힌다(MaxFilter 보다 훨씬 빠름)
                ImageDraw.Draw(m).polygon(pts, fill=255, outline=255, width=2 * grow + 1)
                return np.asarray(m) > 0
            except TypeError:                                           # 오래된 Pillow
                ImageDraw.Draw(m).polygon(pts, fill=255)
                return np.asarray(m.filter(ImageFilter.MaxFilter(2 * grow + 1))) > 0
        ImageDraw.Draw(m).polygon(pts, fill=255)
    return np.asarray(m) > 0


def cyl_uv(cam, T, xy, z0, z1, r, n=36):
    th = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    c = np.c_[xy[0] + r * np.cos(th), xy[1] + r * np.sin(th)]
    P = np.vstack([np.c_[c, np.full(n, z)] for z in np.linspace(z0, z1, 5)])
    return CM.project_world(cam, T, P)


def tube_masks(cam, T, wh, xy, z0, z1, G_):
    """(넓은 마스크, 몸통 마스크) — 몸통 원기둥 z0~z1."""
    return (poly(wh, cyl_uv(cam, T, xy, z0, z1, G_["tube_r"] + 1.5), grow=2), poly(wh, cyl_uv(cam, T, xy, z0, z1, G_["tube_r"])))


def ring_masks(cam, T, wh, xy, G_):
    z0, z1 = G_["z_top"], G_["z_top"] + G_["ring_h"]
    inn = poly(wh, cyl_uv(cam, T, xy, z1, z1, G_["ring_ri"] - 1.0))
    wide = poly(wh, cyl_uv(cam, T, xy, z0, z1, G_["ring_ro"] + 1.0), grow=2) & ~inn
    tight = poly(wh, cyl_uv(cam, T, xy, z0, z1, G_["ring_ro"])) & ~poly(wh, cyl_uv(cam, T, xy, z1, z1, G_["ring_ri"]))
    return wide, tight


def erode(m, r=1):
    return np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(2 * r + 1))) > 0


def hsv(a):
    x = np.asarray(Image.fromarray(a).convert("HSV")).astype(float)
    return x[..., 0] * 360.0 / 255.0, x[..., 1] / 255.0, x[..., 2] / 255.0


def strict(h, s, v, color, s_min=0.28):
    if color == RED:
        hw = np.where(h > 180, h - 360, h)
        return (hw >= -25) & (hw <= 22) & (s > s_min) & (v > 0.12)
    return (h >= 30) & (h <= 75) & (s > s_min) & (v > 0.12)


def stat(h, s, v, m, color, ref):
    if m.sum() < 25:
        return dict(ref), False
    hh = np.where(h > 180, h - 360, h) if color == RED else h
    q = lambda x: float(np.subtract(*np.percentile(x, [75, 25]))) / 1.35   # noqa: E731
    return {"h": float(np.median(hh[m])), "s": float(np.median(s[m])), "v": float(np.median(v[m])),
            "ss": max(0.02, q(s[m])), "sv": max(0.02, q(v[m]))}, True


def recolor(a, objs, tgt_stats, s_min=(0.28, 0.20)):
    """objs = [{kind: tube|ring, src: red|yellow, dst: red_tube|yellow_tube|red_ring, wide, tight, d}] — 먼 것부터 칠하고 가까운 것이 덮는다.
    튜브(3인칭): 몸통 안쪽(1 px 깎음)에서 파랑·청록(뚜껑·받침)·검정(손가락)이 아니면 그 튜브 화소, 가장자리는 원래 색이 확실한 화소만.
    링·손목 튜브(strict_only): 원래 색(채도 0.2 넘음)인 화소만. 채도·밝기는 z 점수 맞춤(원래 = 그 영상에서 잰 분포, 목표 = 학습 200판 분포)."""
    h, s, v = hsv(a)
    not_bg = ~((((h >= 150) & (h <= 270)) & (s > 0.15)) | (v < 0.2))
    own = []
    for o in sorted(objs, key=lambda o: -o["d"]):
        st_src = strict(h, s, v, o["src"], s_min[0])
        st_oth = strict(h, s, v, RED if o["src"] == YEL else YEL, s_min[0])
        if o["kind"] == "tube" and not o.get("strict_only"):
            m = (erode(o["tight"], 1) & not_bg & ~st_oth) | (o["wide"] & st_src)
        else:
            m = (o["tight"] & strict(h, s, v, o["src"], s_min[1])) | (o["wide"] & st_src)
        for p in own:
            p[1] &= ~m
        own.append([o, m])
    H, S, V = h.copy(), s.copy(), v.copy()
    info = []
    for o, m in own:
        if not m.any():
            info.append((o["dst"], 0))
            continue
        sref = DEF[("yellow_" if o["src"] == YEL else "red_") + o["kind"]]
        a_, _own = stat(h, s, v, m if o.get("solid") else m & strict(h, s, v, o["src"], s_min[1]), o["src"], sref)   # 꽉 찬 색: 원래 분포를 마스크 전체에서
        b_ = tgt_stats.get(o["dst"]) or DEF[o["dst"]]
        hh = np.where(h > 180, h - 360, h) if o["src"] == RED else h
        H[m] = (b_["h"] + 0.3 * np.clip(hh[m] - a_["h"], -20, 20)) % 360
        zs, zv = (s[m] - a_["s"]) / a_["ss"], (v[m] - a_["v"]) / a_["sv"]
        if o.get("solid"):                                              # 옅은·반짝이는 화소가 흰 점으로 남지 않게
            zs, zv = np.clip(zs, -1.5, 1.5), np.clip(zv, -2.5, 2.5)
        S[m] = b_["s"] + zs * b_["ss"]
        V[m] = b_["v"] + zv * b_["sv"]
        info.append((o["dst"], int(m.sum())))
    out = np.asarray(Image.fromarray(np.stack([H / 360.0 * 255.0, np.clip(S, 0, 1) * 255.0, np.clip(V, 0, 1) * 255.0], -1).astype(np.uint8), "HSV").convert("RGB"))
    u = np.zeros(h.shape, bool)
    for _o, m in own:
        u |= m
    w = np.clip(np.asarray(Image.fromarray((u * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))).astype(float) / 255.0 * 2.0, 0, 1)
    w = np.where(u, 1.0, w)[..., None]
    return (out.astype(float) * w + a.astype(float) * (1 - w)).astype(np.uint8), info


# ------------------------------------------------------------------ 학습 200판의 모습(목표 분포)
def train_stats(ds, cfg, wd, n_eps=16, offs=(35, 15)):
    """학습 판 프레임에서 빨간 튜브·노란 튜브·빨간 링의 색 분포(3인칭·손목). 판 꼬리표 + 지금 보정으로 자리 마스크 → 확실한 색 화소.
    한 번 재서 work/colorswap_stats.json 에 둔다(데이터셋·보정 파일이 같으면 다시 쓴다)."""
    ds = Path(ds).expanduser()
    G_ = geom(cfg)
    rcal = RCAL.load(wd) or {}
    th = rcal.get("third") or {}
    wcal = WR.load(wd)
    key = {"ds": str(ds), "T_cb": th.get("T_cb"), "T_bw": wcal.get("T_bw"), "v": 2}
    cache = Path(wd) / "colorswap_stats.json"
    try:
        c = json.loads(cache.read_text())
        if c.get("key") == key:
            return c["stats"]
    except (OSError, ValueError):
        pass
    T3 = CM.inv(np.asarray(th["T_cb"], float)) if th.get("T_cb") else None
    flip = bool(th.get("flip"))
    Tw, hq = (np.asarray(wcal["T_bw"], float), wcal.get("home_q")) if wcal.get("T_bw") else (None, None)
    eps = sorted(p for p in ds.glob("ep_*") if (p / "meta.json").is_file())
    pick_i = np.unique(np.linspace(0, len(eps) - 1, min(n_eps, len(eps))).round().astype(int)) if eps else []
    acc = {cam: {k: [] for k in ("red_tube", "yellow_tube", "red_ring")} for cam in ("third", "wrist")}
    for i in pick_i:
        m = json.loads((eps[i] / "meta.json").read_text(encoding="utf-8"))
        tag = (m.get("fr5rec") or {}).get("tag") or {}
        if not (tag.get("racks") and tag.get("src") and tag.get("yellow_at") and tag.get("dst")):
            continue
        H = [RV._holes_pitch(float(F["x"]), float(F["y"]), float(F["yaw"]), G_["z_top"], G_["n_holes"], G_["pitch"]) for F in tag["racks"]]
        red_xy = np.asarray(tag.get("red_xy") or [np.nan, np.nan], float)
        hp = lambda at, rev: np.asarray(H[int(at[0])][(len(H[int(at[0])]) - 1 - int(at[1])) if rev else int(at[1])], float)   # noqa: E731
        rev = min((False, True), key=lambda rv: float(np.linalg.norm(hp(tag["src"], rv)[:2] - red_xy)))
        if float(np.linalg.norm(hp(tag["src"], rev)[:2] - red_xy)) > 10.0:
            continue
        g = np.array([float(f.get("grip_raw", 1000.0)) for f in m.get("frames") or []])
        kk = np.flatnonzero(g < 800.0)
        if not len(kk):
            continue
        intr = {c_: (((m.get("fr5rec") or {}).get("camera") or {}).get(c_) or {}).get("intrinsics") for c_ in ("third", "wrist")}
        for off in offs:
            k = int(kk[0]) - off
            if k < 0:
                continue
            q = m["frames"][k].get("joints_deg")
            for cam in ("third", "wrist"):
                p = eps[i] / ("rgb" if cam == "third" else "wrist") / f"{k:05d}.png"
                if not p.is_file() or not intr.get(cam):
                    continue
                T = T3 if cam == "third" else (WR.home_delta(q, hq) @ Tw if (Tw is not None and q) else None)
                if T is None:
                    continue
                cm = CM.Cam(intr[cam], flip)
                a = np.asarray(Image.open(p).convert("RGB"))
                h, s, v = hsv(a)
                wh = (a.shape[1], a.shape[0])
                for nm, at, col in (("red_tube", tag["src"], RED), ("yellow_tube", tag["yellow_at"], YEL)):
                    best = None
                    for rv in ((rev,) if int(at[0]) == int(tag["src"][0]) else (False, True)):
                        _w, t_ = tube_masks(cm, T, wh, hp(at, rv)[:2], G_["z_table"], G_["z_body_top"], G_)
                        mm = t_ & strict(h, s, v, col, S_MIN[cam][0])
                        if best is None or mm.sum() > best.sum():
                            best = mm
                    if best is not None and best.sum():
                        acc[cam][nm].append(np.c_[h[best], s[best], v[best]])
                _w, t_ = ring_masks(cm, T, wh, hp(tag["dst"], rev)[:2], G_)
                mm = t_ & strict(h, s, v, RED, S_MIN[cam][1])
                if mm.sum():
                    acc[cam]["red_ring"].append(np.c_[h[mm], s[mm], v[mm]])
    out = {}
    for cam, D in acc.items():
        out[cam] = {}
        for nm, L in D.items():
            if not L:
                continue
            X = np.vstack(L)
            if len(X) < 200:
                continue
            col = RED if nm.startswith("red") else YEL
            st_, _ok = stat(X[:, 0], X[:, 1], X[:, 2], np.ones(len(X), bool), col, DEF[nm])
            st_["n"] = int(len(X))
            out[cam][nm] = st_
    for nm in ("red_tube", "yellow_tube", "red_ring"):                  # 손목에 거의 안 보이거나 색조가 3인칭과 15° 넘게 다르면(책상이 섞임) 3인칭 값으로
        w_, t_ = (out.get("wrist") or {}).get(nm), (out.get("third") or {}).get(nm)
        if t_ is not None and (w_ is None or abs(((w_["h"] - t_["h"]) + 180.0) % 360.0 - 180.0) > 15.0):
            out.setdefault("wrist", {})[nm] = dict(t_, from_third=True, wrist_h=None if w_ is None else round(w_["h"], 1))
    try:
        cache.write_text(json.dumps({"key": key, "stats": out, "t": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False))
    except OSError:
        pass
    return out


# ------------------------------------------------------------------ 실행 중
class Swapper:
    def __init__(self, cfg, wd, intr3, intrw, stats):
        self.G = geom(cfg)
        self.wd = Path(wd)
        rcal = RCAL.load(wd) or {}
        th = rcal.get("third") or {}
        if not th.get("T_cb"):
            raise RuntimeError("3인칭 보정(rack_calib.json)이 없다")
        self.flip = bool(th.get("flip"))
        self.T3 = CM.inv(np.asarray(th["T_cb"], float))
        self.cam3 = CM.Cam(intr3, self.flip)
        wcal = WR.load(wd)
        self.Tw, self.hq = (np.asarray(wcal["T_bw"], float), wcal.get("home_q")) if wcal.get("T_bw") else (None, None)
        self.camw = CM.Cam(intrw, self.flip) if (intrw and self.Tw is not None) else None
        self.stats = stats or {}
        self.tgt = self.oth = self.ring = None                          # 베이스 xy(목표 노랑 · 방해 빨강 · 노란 링 칸)
        self.held, self.lock, self.ms = False, threading.Lock(), []
        self.sc = None
        self.slot = None
        self.snap_dir, self.n_snap, self.log = None, 0, {}
        self.ring_red, self.ring_mode = {}, "yellow"

    def snap(self, third_png, wrist_png):
        """정책에 준 바꾼 영상(원래 크기)을 처음과 20번째 질의마다 남긴다 — 눈으로 확인."""
        if self.snap_dir is None:
            return
        self.n_snap += 1
        if self.n_snap == 1 or self.n_snap % 20 == 0:
            try:
                (self.snap_dir / f"q{self.n_snap:03d}_third.png").write_bytes(third_png)
                (self.snap_dir / f"q{self.n_snap:03d}_wrist.png").write_bytes(wrist_png)
            except OSError:
                pass

    # 장면 ----------------------------------------------------------
    def find_ring(self, img, sc, dbg=None, wimg=None, q=None):
        """노란 링 칸. 빈 구멍마다 링 자리(고리, 보정 오차 감안해 안 −2 · 밖 +3 mm)에서 '확실한 노랑'(채도 0.35 넘음) 비율을
        손목(초기 자세 — 링이 진하게 보임)과 3인칭에서 잰다. 손목에서 그 칸이 보이면 손목 비율, 아니면 3인칭 비율로 고른다.
        받아들임: 0.10 넘고 (둘째의 1.5배 · 둘째보다 0.08 높음 · fr5auto 링 인식 칸과 같음) 중 하나. 못 고르면 fr5auto 링 인식 칸.
        환경 변수 FR5_SWAP_SLOT=받침:구멍 이면 그 칸(시험용). 그림: work/swap_ringfind.png(3인칭) · swap_ringfind_wrist.png(손목)."""
        import os
        a3 = np.asarray(img)
        h, s, v = hsv(a3)
        wh = (a3.shape[1], a3.shape[0])
        self.ring_red, self.ring_mode = {}, "yellow"
        G2 = dict(self.G, ring_ri=self.G["ring_ri"] - 2.0, ring_ro=self.G["ring_ro"] + 3.0)
        Tw = None
        if wimg is not None and q is not None and self.camw is not None:
            aw = np.asarray(wimg)
            hw_, sw_, vw_ = hsv(aw)
            whw = (aw.shape[1], aw.shape[0])
            Tw = WR.home_delta(q, self.hq) @ self.Tw
        rows, o3, ow = [], [], []
        for i, F in enumerate(sc.get("racks") or []):
            for j, st in enumerate(F["state"]):
                if st not in ("empty", "unseen"):
                    continue
                xy = np.asarray(F["holes"][j], float)[:2]
                _w, t_ = ring_masks(self.cam3, self.T3, wh, xy, G2)
                f3 = float((t_ & strict(h, s, v, YEL, 0.35)).sum()) / max(1, int(t_.sum()))
                f3l = float((t_ & strict(h, s, v, YEL, 0.20)).sum()) / max(1, int(t_.sum()))
                self.ring_red[(i, j)] = float((t_ & strict(h, s, v, RED, 0.35)).sum()) / max(1, int(t_.sum()))
                o3.append((t_, (i, j)))
                fw = None
                if Tw is not None:
                    _w2, tw = ring_masks(self.camw, Tw, whw, xy, G2)
                    if tw.sum() >= 50:
                        fw = float((tw & strict(hw_, sw_, vw_, YEL, 0.35)).sum()) / int(tw.sum())
                        ow.append((tw, (i, j)))
                rows.append({"ij": (i, j), "f3": f3, "f3l": f3l, "fw": fw})
        use_w = any(r_["fw"] is not None for r_ in rows)
        key = lambda r_: max(r_["f3"], r_["fw"] if r_["fw"] is not None else 0.0)   # noqa: E731 — 손목에 안 보이는 칸은 3인칭 값
        rows.sort(key=key, reverse=True)
        cand = [(key(r_), r_["ij"]) for r_ in rows]
        def fmt(r_):
            fw = "-" if r_["fw"] is None else "%.2f" % r_["fw"]
            return "받침%d:%d 손목 %s/3인칭 %.2f(%.2f)" % (r_["ij"][0], r_["ij"][1], fw, r_["f3"], r_["f3l"])
        txt = " · ".join(fmt(r_) for r_ in rows[:4])
        for path, items, base in ((dbg or self.wd / "swap_ringfind.png", o3, a3),
                                  (self.wd / "swap_ringfind_wrist.png", ow, np.asarray(wimg) if wimg is not None else None)):
            if base is None:
                continue
            try:
                a_ = base.copy()
                for t_, ij in items:
                    a_[t_ & ~erode(t_, 1)] = (0, 255, 0) if cand and ij == cand[0][1] else (255, 0, 255)
                Image.fromarray(a_).save(path)
            except Exception:                                           # noqa: BLE001
                pass
        print(f"  노란 링 칸 찾기({'손목·3인칭 중 큰 값' if use_w else '3인칭'} · 그림 {self.wd / 'swap_ringfind.png'}, swap_ringfind_wrist.png): {txt}")
        env = os.environ.get("FR5_SWAP_SLOT")
        if env:
            try:
                return tuple(int(x) for x in env.split(":")), cand[:3]
            except ValueError:
                pass
        if cand and cand[0][0] >= 0.10:
            b, n2 = cand[0][0], (cand[1][0] if len(cand) > 1 else 0.0)
            if b >= 1.5 * max(n2, 0.0) or b - n2 >= 0.08 or (sc.get("slot") is not None and tuple(sc["slot"]) == cand[0][1]):
                return cand[0][1], cand[:3]
        if sc.get("slot") is not None:                                  # 못 고르면 fr5auto 링 인식 칸(링 화소 중 노랑만 빨강으로 — 빨간 링이면 그대로)
            ij = tuple(sc["slot"])
            loose = next((r_["f3l"] for r_ in rows if r_["ij"] == ij), 0.0)
            self.ring_mode = "yellow" if loose > 0.02 else "red"
            print(f"  fr5auto 링 인식 칸 받침{ij[0]}:{ij[1]} 을 쓴다 · 빨강 {self.ring_red.get(ij, 0):.2f}"
                  + (" → 노란 링으로 본다" if self.ring_mode == "yellow" else " → 빨간 링(색 그대로)"))
            return ij, cand[:3]
        return None, cand[:3]

    def set_scene(self, sc, slot):
        hole = lambda ij: np.asarray(sc["racks"][ij[0]]["holes"][ij[1]], float)[:2]   # noqa: E731
        self.sc, self.slot = sc, tuple(slot)
        self.tgt, self.oth, self.ring = hole(sc["yellow"]), hole(sc["red"]), hole(slot)
        self.held = False

    def relabel(self, sc):
        """fr5auto 장면의 빨강↔노랑 이름을 바꾸고 링 칸을 노란 링 칸으로 — 계획·판정이 '빨강 → 링 칸'으로 노랑을 옮긴다."""
        import copy
        out = copy.deepcopy({k: v for k, v in sc.items() if not str(k).startswith("_")})
        sw = {"red": "yellow", "yellow": "red"}
        for F in out.get("racks") or []:
            F["state"] = [sw.get(x, x) for x in F["state"]]
        out["red"], out["yellow"] = sc.get("yellow"), sc.get("red")
        if self.ring is not None:
            best = None
            for i, F in enumerate(out.get("racks") or []):
                for j, hh in enumerate(F["holes"]):
                    d = math.dist(hh[:2], self.ring)
                    if best is None or d < best[0]:
                        best = (d, (i, j))
            if best and best[0] < 12.0:
                out["slot"] = best[1]
                out["slot_xy"] = [float(self.ring[0]), float(self.ring[1])]
        return out

    def on_close(self):
        self.held = True

    def on_open(self, tcp_xy):
        self.held = False
        self.tgt = np.asarray(tcp_xy, float)[:2]                        # 놓은 자리(링 칸)에 선다

    # 영상 ----------------------------------------------------------
    def objects(self, cam_name, q):
        """이 카메라에서 바꿀 물체들의 투영(원기둥 점 uv) — 마스크는 apply_np 가 잘라 낸 영역 안에서 만든다(빠르게)."""
        if cam_name == "third":
            cam, T = self.cam3, self.T3
        else:
            if self.camw is None or q is None:
                return []
            cam, T = self.camw, WR.home_delta(q, self.hq) @ self.Tw
        G_ = self.G
        out = []

        def dist(xy, z):
            return float(np.linalg.norm(CM.apply(CM.inv(T), np.array([[xy[0], xy[1], z]]))[0]))

        def tube(xy, z0, z1, src, dst, d):
            out.append({"kind": "tube", "src": src, "dst": dst, "d": d, "strict_only": cam_name == "wrist" and SOLID < 2, "solid": SOLID >= (1 if cam_name == "third" else 2),   # 손목: 몸통이 뚜껑에 가려 거의 안 보인다 — 확실한 색 화소만
                        "uv": {"wide": cyl_uv(cam, T, xy, z0, z1, G_["tube_r"] + 1.5), "tight": cyl_uv(cam, T, xy, z0, z1, G_["tube_r"])}})
        if self.held and q is not None:
            p = K.fk(np.asarray(q, float)[:6])[1]
            z0 = float(p[2]) - G_["tcp_to_bottom"]
            tube(p[:2], z0, z0 + G_["body_len"], YEL, "red_tube", dist(p[:2], z0 + 50))
        elif self.tgt is not None:
            tube(self.tgt, G_["z_table"], G_["z_body_top"], YEL, "red_tube", dist(self.tgt, G_["z_top"]))
        if self.oth is not None:
            tube(self.oth, G_["z_table"], G_["z_body_top"], RED, "yellow_tube", dist(self.oth, G_["z_top"]))
        if self.ring is not None:
            z0, z1 = G_["z_top"], G_["z_top"] + G_["ring_h"]
            out.append({"kind": "ring", "src": YEL, "dst": "red_ring", "d": dist(self.ring, G_["z_top"]) + 30.0,
                        "uv": {"wide": cyl_uv(cam, T, self.ring, z0, z1, G_["ring_ro"] + (8.0 if cam_name == "wrist" else 1.0)), "hole": cyl_uv(cam, T, self.ring, z1, z1, G_["ring_ri"] - 1.0),
                               "tight": cyl_uv(cam, T, self.ring, z0, z1, G_["ring_ro"]), "tight_in": cyl_uv(cam, T, self.ring, z1, z1, G_["ring_ri"])}})
        return out

    def apply_np(self, a, cam_name, q):
        t0 = time.perf_counter()
        objs = self.objects(cam_name, q)
        Hh, Ww = a.shape[:2]
        pts = [o["uv"]["wide"] for o in objs]
        pts = [x[np.all(np.isfinite(x), axis=1)] for x in pts]
        pts = [x for x in pts if len(x)]
        if not pts:
            return a
        P = np.vstack(pts)
        x0, y0 = int(max(0, np.floor(P[:, 0].min()) - 10)), int(max(0, np.floor(P[:, 1].min()) - 10))
        x1, y1 = int(min(Ww, np.ceil(P[:, 0].max()) + 11)), int(min(Hh, np.ceil(P[:, 1].max()) + 11))
        if x1 - x0 < 3 or y1 - y0 < 3:
            return a
        wh, off = (x1 - x0, y1 - y0), (x0, y0)
        ms = []
        for o in objs:
            u = o["uv"]
            if o["kind"] == "tube":
                wide, tight = poly(wh, u["wide"], 2, off), poly(wh, u["tight"], 0, off)
            else:
                wide = poly(wh, u["wide"], 2, off) & ~poly(wh, u["hole"], 0, off)
                tight = poly(wh, u["tight"], 0, off) & ~poly(wh, u["tight_in"], 0, off)
            ms.append(dict(o, wide=wide, tight=tight))
        out = a.copy()
        crop, _info = recolor(np.ascontiguousarray(a[y0:y1, x0:x1]), ms, self.stats.get(cam_name) or {}, S_MIN.get(cam_name, S_MIN["third"]))
        out[y0:y1, x0:x1] = crop
        self.ms.append((time.perf_counter() - t0) * 1000)
        return out

    def apply_png(self, raw, cam_name, q):
        with Image.open(BytesIO(raw)) as im:
            a = np.asarray(im.convert("RGB"))
        b = BytesIO()
        Image.fromarray(self.apply_np(a, cam_name, q)).save(b, "PNG", compress_level=1)
        return b.getvalue()


def wrap_frame(r, sw):
    """r.frame(fr5auto 의 내려가기 전 맞춤·넣는 중 미세조정·손목 확인이 쓰는 사진)도 바꾼 영상으로 — 쥔 노란 튜브를 '빨강'으로 잰다."""
    f0 = r.frame

    def frame(cam, fresh_ms=None, tries=4, ext="jpg"):
        fr = f0(cam, fresh_ms=fresh_ms, tries=tries, ext=ext)
        if fr is None or cam not in ("third", "wrist"):
            return fr
        q = list(r.sender.target[0]) if r.sender is not None else None
        try:
            return (sw.apply_np(np.asarray(fr[0]), cam, q),) + tuple(fr[1:])
        except Exception:                                               # noqa: BLE001 — 바꾸기가 고장 나면 원래 사진(그 판 기록에 남는다)
            return fr
    r.frame = frame
    return f0
