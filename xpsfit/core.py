# -*- coding: utf-8 -*-
"""XPS 拟合核心：背景、峰型、带约束的最小二乘拟合、定量。与界面无关，可单独测试。"""
import copy
import json
import math
import os

import numpy as np
from scipy.optimize import least_squares

LN2 = math.log(2.0)
PARAMS = ("center", "fwhm", "area", "lor")   # lor: 洛伦兹比例 %，0=纯高斯


# ------------------------------------------------------------------ 峰型
def gl_peak(x, center, fwhm, area, lor):
    """面积归一的 Gaussian–Lorentzian 混合 (pseudo-Voigt, GL sum)。lor 为洛伦兹 %。"""
    eta = min(max(lor / 100.0, 0.0), 1.0)
    w = max(fwhm, 1e-6)
    dx = x - center
    g = math.sqrt(4 * LN2 / math.pi) / w * np.exp(-4 * LN2 * dx * dx / (w * w))
    l = (2 / math.pi) * w / (4 * dx * dx + w * w)
    return area * ((1 - eta) * g + eta * l)


# ------------------------------------------------------------------ 背景
def _ends(x, y, navg):
    """区间内 (x 升序) 两端的平均强度。"""
    n = max(1, min(navg, len(x) // 4 or 1))
    return float(np.mean(y[:n])), float(np.mean(y[-n:]))


def smooth(y, n=5):
    if len(y) < n:
        return y.copy()
    k = np.ones(n) / n
    ys = np.convolve(np.pad(y, n // 2, mode="edge"), k, mode="valid")
    return ys[:len(y)]


def points_background(x, pts, interp="smooth"):
    """穿过用户点选的点 [(x, y), ...]。smooth = 单调保形三次（PCHIP，不会过冲），linear = 折线。"""
    p = sorted((float(a), float(b)) for a, b in pts)
    xs = np.array([a for a, _ in p]); ys = np.array([b for _, b in p])
    keep = np.concatenate([[True], np.diff(xs) > 1e-9])
    xs, ys = xs[keep], ys[keep]
    if len(xs) == 1:
        return np.full_like(np.asarray(x, float), ys[0])
    if interp == "linear" or len(xs) == 2:
        return np.interp(x, xs, ys)
    from scipy.interpolate import PchipInterpolator
    f = PchipInterpolator(xs, ys, extrapolate=False)
    out = f(np.clip(x, xs[0], xs[-1]))
    return out


def background(x, y, kind="Shirley", navg=3, tougaard_B=2866.0, tougaard_C=1643.0, ylo=None, yhi=None,
               points=None, interp="smooth"):
    """x 为升序 BE。返回与 y 同长的背景。
    ylo / yhi：手动指定低 BE 端 / 高 BE 端的背景强度（None = 端点若干点平均）。
    Smart：Shirley，但任何位置都不高于（平滑后的）数据。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if kind in (None, "", "None", "无"):
        return np.zeros_like(y)
    if kind == "Points":
        if points and len(points) >= 1:
            return points_background(x, points, interp)
        kind = "Linear"
    yl, yh = _ends(x, y, navg)          # 低 BE 端、高 BE 端
    if ylo is not None: yl = float(ylo)
    if yhi is not None: yh = float(yhi)
    if kind in ("Linear", "线性"):
        return yl + (yh - yl) * (x - x[0]) / (x[-1] - x[0] or 1)
    if kind in ("Shirley", "Smart"):
        ys = smooth(y) if kind == "Smart" else None
        bg = np.full_like(y, yl)
        for _ in range(200):
            # 只累积高于背景的部分：数据低于背景时（端点落在别的峰上等）不会出现负面积导致背景失控
            s = np.clip(y - bg, 0, None)
            # 累积面积：从低 BE 端积到 x 处
            cum = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(x))])
            tot = cum[-1]
            new = yl + (yh - yl) * (cum / tot if tot else 0)
            if ys is not None:
                new = np.minimum(new, ys)
            if np.max(np.abs(new - bg)) < 1e-6 * max(1.0, abs(yh)):
                bg = new; break
            bg = new
        return bg
    if kind == "Tougaard":
        # 通用 Tougaard 两参数：B·T/(C+T²)²，T = 能量损失。高 BE 侧的强度来自低 BE 侧的非弹性散射
        s = y - yl
        bg = np.zeros_like(y)
        dx = np.gradient(x)
        for i in range(len(x)):
            T = x[i] - x[:i]
            bg[i] = np.sum(tougaard_B * T / (tougaard_C + T * T) ** 2 * s[:i] * dx[:i])
        # 按高 BE 端标定 B，让背景在高 BE 端接上数据
        if bg[-1] > 0:
            bg *= (yh - yl) / bg[-1]
        return yl + bg
    raise ValueError("未知背景类型 %s" % kind)


# ------------------------------------------------------------------ 模型数据结构
ROLES = ("main", "satellite", "unknown")   # 主峰 / 卫星峰 / 未知重叠峰


CENTER_WINDOW = 2.0     # 手动加峰：峰位默认只能在初始位置 ±2 eV 内拟合
TEMPLATE_WINDOW = 1.0   # 套模板：±1 eV，保证各化学态不互换位置


def bound_center(p, window=CENTER_WINDOW):
    if p["center"]["min"] is None and p["center"]["max"] is None and "center" not in linked_params(p):
        c = p["center"]["value"]
        p["center"]["min"], p["center"]["max"] = c - window, c + window
    return p


def new_peak(name, center, fwhm=1.2, area=1000.0, lor=30.0, role="main"):
    p = {"name": name, "link": {}, "role": role}
    for k, v in zip(PARAMS, (center, fwhm, area, lor)):
        p[k] = {"value": float(v), "fixed": k == "lor", "min": None, "max": None}
    p["fwhm"]["min"], p["fwhm"]["max"] = 0.4, 6.0
    p["area"]["min"] = 0.0
    p["lor"]["min"], p["lor"]["max"] = 0.0, 100.0
    return p
    # link 例：{"ref": "Al2p3/2", "dE": 0.44, "area_ratio": 0.5, "fwhm_ratio": 1.0}
    # 每项可省略；ref 为另一个峰的名字。dE/ratio 为 None 时该参数不绑定。


def resolve(peaks, free_vec=None, layout=None):
    """返回每个峰的实际参数 [(center,fwhm,area,lor)]，处理固定/绑定。"""
    vals = [{k: p[k]["value"] for k in PARAMS} for p in peaks]
    lnk = [dict(p.get("link") or {}) for p in peaks]
    if free_vec is not None:
        for (i, k), v in zip(layout, free_vec):
            if k.startswith("L:"):
                lnk[i][k[2:]] = v
            else:
                vals[i][k] = v
    names = {p["name"]: i for i, p in enumerate(peaks)}
    done = [False] * len(peaks)

    def fin(i, depth=0):
        if done[i] or depth > 20:
            return
        ln = lnk[i]
        r = names.get(ln.get("ref"))
        if r is not None and r != i:
            fin(r, depth + 1)
            if ln.get("dE") is not None:
                vals[i]["center"] = vals[r]["center"] + ln["dE"]
            if ln.get("area_ratio") is not None:
                vals[i]["area"] = vals[r]["area"] * ln["area_ratio"]
            if ln.get("fwhm_ratio") is not None:
                vals[i]["fwhm"] = vals[r]["fwhm"] * ln["fwhm_ratio"]
            if ln.get("lor_same"):
                vals[i]["lor"] = vals[r]["lor"]
        done[i] = True

    for i in range(len(peaks)):
        fin(i)
    return vals


LINK_KEYS = ("dE", "area_ratio", "fwhm_ratio")


def link_bounds(k, v):
    """绑定量自由拟合时的默认范围。"""
    if k == "dE":
        d = max(0.2, 0.1 * abs(v))
        return v - d, v + d
    return v * 0.5, v * 1.5


def linked_params(p):
    ln = p.get("link") or {}
    if not ln.get("ref"):
        return set()
    s = set()
    if ln.get("dE") is not None: s.add("center")
    if ln.get("area_ratio") is not None: s.add("area")
    if ln.get("fwhm_ratio") is not None: s.add("fwhm")
    if ln.get("lor_same"): s.add("lor")
    return s


def model(x, peaks, vals=None):
    vals = vals or resolve(peaks)
    comps = [gl_peak(x, v["center"], v["fwhm"], v["area"], v["lor"]) for v in vals]
    return comps, (np.sum(comps, axis=0) if comps else np.zeros_like(x))


def init_areas(x, s, peaks, w=None):
    """固定峰位/峰宽/L%，用非负最小二乘求各峰面积（面积是线性参数）。作为非线性拟合的起点。
    面积绑定（如双峰副峰）的峰并入其参考峰的基函数。"""
    from scipy.optimize import nnls
    vals = resolve(peaks)
    names = {p["name"]: i for i, p in enumerate(peaks)}
    free = [i for i, p in enumerate(peaks) if not p["area"]["fixed"] and "area" not in linked_params(p)]
    if not free:
        return
    w = np.ones_like(s) if w is None else w
    fixed_part = np.zeros_like(s)
    cols = {i: np.zeros_like(s) for i in free}
    for i, (p, v) in enumerate(zip(peaks, vals)):
        g = gl_peak(x, v["center"], v["fwhm"], 1.0, v["lor"])
        # 找到面积归属：沿面积绑定链找到自由面积的根
        j, ratio, guard = i, 1.0, 0
        while "area" in linked_params(peaks[j]) and guard < 20:
            ln = peaks[j]["link"]; ratio *= ln["area_ratio"]; j = names[ln["ref"]]; guard += 1
        if j in cols:
            cols[j] += ratio * g
        else:
            fixed_part += vals[i]["area"] * g if i == j else 0
    A = np.column_stack([cols[i] * w for i in free])
    a, _ = nnls(A, (s - fixed_part) * w)
    for i, ai in zip(free, a):
        mn = peaks[i]["area"]["min"] if peaks[i]["area"]["min"] is not None else 0.0
        mx = peaks[i]["area"]["max"] if peaks[i]["area"]["max"] is not None else np.inf
        floor = 1e-3 * max(float(np.max(s)), 1.0)   # 避免恰好为 0 卡在边界
        peaks[i]["area"]["value"] = float(min(max(ai, mn, floor), mx))


def fit(x, y, bg, peaks, max_nfev=4000):
    """原地更新 peaks 中自由参数。返回 dict(chi2, rchi2, success, message)。"""
    x = np.asarray(x, float); s = np.asarray(y, float) - bg
    try:
        init_areas(x, s, peaks, 1.0 / np.sqrt(np.clip(np.asarray(y, float), 1.0, None)))
    except Exception:
        pass
    layout, x0, lo, hi = [], [], [], []
    for i, p in enumerate(peaks):
        lk = linked_params(p)
        for k in PARAMS:
            par = p[k]
            if par["fixed"] or k in lk:
                continue
            mn = -np.inf if par["min"] is None else par["min"]
            mx = np.inf if par["max"] is None else par["max"]
            if k == "center":
                mn = max(mn, x.min()) if par["min"] is None else mn
                mx = min(mx, x.max()) if par["max"] is None else mx
            v = min(max(par["value"], mn), mx)
            if mx - mn < 1e-12:
                continue
            if v <= mn: v = mn + 1e-9 * (abs(mn) + 1)
            if v >= mx: v = mx - 1e-9 * (abs(mx) + 1)
            layout.append((i, k)); x0.append(v); lo.append(mn); hi.append(mx)
        ln = p.get("link") or {}
        if ln.get("ref"):
            for k in LINK_KEYS:
                if ln.get(k) is None or not ln.get(k + "_fit"):
                    continue
                dmn, dmx = link_bounds(k, ln[k])
                mn = ln.get(k + "_min", dmn); mx = ln.get(k + "_max", dmx)
                mn = dmn if mn is None else mn; mx = dmx if mx is None else mx
                if mx - mn < 1e-12:
                    continue
                v = min(max(ln[k], mn + 1e-9), mx - 1e-9)
                layout.append((i, "L:" + k)); x0.append(v); lo.append(mn); hi.append(mx)
    # 泊松权重：sigma ~ sqrt(counts)
    w = 1.0 / np.sqrt(np.clip(np.asarray(y, float), 1.0, None))

    def resid(v):
        _, env = model(x, peaks, resolve(peaks, v, layout))
        return (s - env) * w

    if layout:
        r = least_squares(resid, np.array(x0), bounds=(np.array(lo), np.array(hi)),
                          max_nfev=max_nfev, x_scale="jac")
        for (i, k), v in zip(layout, r.x):
            if k.startswith("L:"):
                peaks[i]["link"][k[2:]] = float(v)
            else:
                peaks[i][k]["value"] = float(v)
        ok, msg = bool(r.success), r.message
    else:
        ok, msg = True, "没有自由参数"
    vals = resolve(peaks)
    for p, v in zip(peaks, vals):     # 把绑定参数的结果写回，便于显示
        for k in PARAMS:
            p[k]["value"] = float(v[k])
    cur = lambda i, k: peaks[i]["link"][k[2:]] if k.startswith("L:") else peaks[i][k]["value"]
    res = resid(np.array([cur(i, k) for i, k in layout])) if layout else (s - model(x, peaks)[1]) * w
    chi2 = float(np.sum(res ** 2))
    dof = max(1, len(x) - len(layout))
    return {"chi2": chi2, "rchi2": chi2 / dof, "success": ok, "message": str(msg), "nfree": len(layout)}


# ------------------------------------------------------------------ 谱
class Spectrum:
    def __init__(self, name, be, counts, source="", hv=1486.68):
        be = np.asarray(be, float); counts = np.asarray(counts, float)
        o = np.argsort(be)
        self.name, self.source, self.hv = name, source, hv
        self.be_raw, self.y = be[o], counts[o]          # 升序存储
        self.shift = 0.0                                # 荷电校正（加到 BE 上）
        self.bg = {"type": "Shirley", "lo": None, "hi": None, "navg": 3}
        self.peaks = []
        self.last_fit = None
        self.element = guess_orbital(name)
        self.sample = os.path.basename(os.path.dirname(os.path.abspath(source))) if source else ""

    @property
    def be(self):
        return self.be_raw + self.shift

    def region(self):
        lo = self.bg["lo"] if self.bg["lo"] is not None else self.be[0]
        hi = self.bg["hi"] if self.bg["hi"] is not None else self.be[-1]
        pts = self.bg.get("points") or []
        if self.bg.get("type") == "Points" and len(pts) >= 2:
            lo, hi = min(p[0] for p in pts), max(p[0] for p in pts)
        lo, hi = min(lo, hi), max(lo, hi)
        m = (self.be >= lo - 1e-9) & (self.be <= hi + 1e-9)
        if m.sum() < 5:
            m[:] = True
        return m

    def compute(self):
        """返回 (x, y, bg, comps, envelope) —— 仅背景区间内。"""
        m = self.region()
        x, y = self.be[m], self.y[m]
        bg = background(x, y, self.bg["type"], int(self.bg.get("navg", 3)),
                        ylo=self.bg.get("ylo"), yhi=self.bg.get("yhi"),
                        points=self.bg.get("points"), interp=self.bg.get("interp", "smooth"))
        comps, env = model(x, self.peaks)
        return x, y, bg, comps, env

    def do_fit(self):
        m = self.region()
        x, y = self.be[m], self.y[m]
        bg = background(x, y, self.bg["type"], int(self.bg.get("navg", 3)),
                        ylo=self.bg.get("ylo"), yhi=self.bg.get("yhi"),
                        points=self.bg.get("points"), interp=self.bg.get("interp", "smooth"))
        self.last_fit = fit(x, y, bg, self.peaks)
        return self.last_fit

    def raw_area(self):
        x, y, bg, _, _ = self.compute()
        return float(np.trapezoid(y - bg, x)) if hasattr(np, "trapezoid") else float(np.trapz(y - bg, x))

    def to_dict(self):
        return {"name": self.name, "source": self.source, "hv": self.hv,
                "be": self.be_raw.tolist(), "y": self.y.tolist(), "shift": self.shift,
                "bg": self.bg, "peaks": self.peaks, "element": self.element, "sample": self.sample}

    @classmethod
    def from_dict(cls, d):
        s = cls(d["name"], d["be"], d["y"], d.get("source", ""), d.get("hv", 1486.68))
        s.shift, s.bg, s.peaks = d.get("shift", 0.0), d["bg"], d["peaks"]
        s.element = d.get("element", s.element)
        s.sample = d.get("sample", s.sample)
        return s


def guess_orbital(name):
    import re
    m = re.search(r"([A-Z][a-z]?)\s*(\d[spdf])(\d/2)?", name or "")
    return (m.group(1) + m.group(2)) if m else ""


# ------------------------------------------------------------------ 参数表 / 定量
def counted(p):
    """未知重叠峰不计入面积% 与定量。"""
    return p.get("role", "main") != "unknown"


def area_percents(peaks):
    tot = sum(p["area"]["value"] for p in peaks if counted(p)) or 1.0
    return [100.0 * p["area"]["value"] / tot if counted(p) else None for p in peaks]


def peak_table(spec):
    """每个峰：名称、类型、位置、FWHM、面积、L%、面积%（未知重叠峰不计）。"""
    return [{"spectrum": spec.name, "peak": p["name"], "role": p.get("role", "main"),
             "BE_eV": p["center"]["value"], "FWHM_eV": p["fwhm"]["value"], "area": p["area"]["value"],
             "L_percent": p["lor"]["value"], "area_percent": pct}
            for p, pct in zip(spec.peaks, area_percents(spec.peaks))]


def state_name(name):
    import re
    n = re.sub(r"\s*\d/2\s*$", "", name).strip()
    return n or name


def chemical_states(spec):
    """把双峰副峰、卫星峰并入所属主峰，返回 [(化学态名, 面积)]；未知重叠峰不计。"""
    by = {p["name"]: p for p in spec.peaks}
    order, area = [], {}
    for p in spec.peaks:
        if not counted(p):
            continue
        owner = p
        ln = p.get("link") or {}
        if (p.get("pair") or p.get("role") == "satellite") and ln.get("ref") in by:
            owner = by[ln["ref"]]
        key = state_name(owner["name"])
        if key not in area:
            order.append(key); area[key] = 0.0
        area[key] += p["area"]["value"]
    return [(k, area[k]) for k in order]


def quantify_samples(specs, sf_table, use_fit=True, ecf_exp=0.6, include=None):
    """按样品分组定量。返回 [{sample, elements:{orb: at%}, states:{"orb: 态": at%}, rows:[...]}]。
    同一样品同一轨道有多条谱时只用第一条（其余标记为重复）。"""
    groups = {}
    for k, s in enumerate(specs):
        if include is not None and not include[k]:
            continue
        groups.setdefault(s.sample or "(未分组)", []).append(s)
    out = []
    for sample, ss in groups.items():
        seen, use, dup = set(), [], []
        for s in ss:
            (dup if s.element in seen else use).append(s); seen.add(s.element)
        rows = quantify(use, sf_table, use_fit, ecf_exp)
        elements, states = {}, {}
        for s, r in zip(use, rows):
            at = r["atomic_percent"]
            if at is None:
                continue
            elements[s.element or s.name] = at
            st = chemical_states(s) if (use_fit and s.peaks) else []
            tot = sum(a for _, a in st) or 1.0
            for nm, a in st:
                states["%s: %s" % (s.element or s.name, nm)] = at * a / tot
        out.append({"sample": sample, "elements": elements, "states": states, "rows": rows,
                    "duplicates": [s.name for s in dup]})
    return out


def quantify(specs, sf_table, use_fit=True, ecf_exp=0.6):
    """原子百分比：A / (SF · KE^ecf_exp)。KE = hν − BE(峰中心)。ecf_exp=0 不做能量补偿。"""
    rows = []
    for s in specs:
        sf = sf_table.get(s.element)
        if use_fit and s.peaks:
            pk = [p for p in s.peaks if counted(p)]
            area = sum(p["area"]["value"] for p in pk)
            be = sum(p["center"]["value"] * p["area"]["value"] for p in pk) / (area or 1)
        else:
            area = s.raw_area()
            x, y, bg, _, _ = s.compute()
            be = float(x[np.argmax(y - bg)])
        ke = max(s.hv - be, 1.0)
        corr = area / (sf * ke ** ecf_exp) if sf else None
        rows.append({"spectrum": s.name, "orbital": s.element, "BE_eV": be, "area": area,
                     "SF": sf, "corrected": corr})
    tot = sum(r["corrected"] for r in rows if r["corrected"])
    for r in rows:
        r["atomic_percent"] = 100.0 * r["corrected"] / tot if r["corrected"] and tot else None
    return rows


# ------------------------------------------------------------------ 模板
def doublet(name, center, split, ratio, fwhm=1.2, area=1000.0, lor=30.0, labels=None,
            fit_split=False, fit_ratio=False):
    """自旋轨道双峰：返回 [主峰, 副峰]，副峰位置/面积/FWHM 绑定主峰。"""
    a, b = labels or (name + " 3/2", name + " 1/2")
    p1 = new_peak(a, center, fwhm, area, lor)
    p2 = new_peak(b, center + split, fwhm, area * ratio, lor)
    p2["link"] = {"ref": a, "dE": split, "area_ratio": ratio, "fwhm_ratio": 1.0, "lor_same": True}
    if fit_split:
        p2["link"]["dE_fit"] = True
    if fit_ratio:
        p2["link"]["area_ratio_fit"] = True
    p2["pair"] = True   # 双峰的副峰：删除主峰时一起删除
    return [p1, p2]


def _same_fwhm(peaks):
    for p in peaks[1:]:
        p["link"] = dict(p.get("link") or {}, ref=peaks[0]["name"], fwhm_ratio=1.0)
    return peaks


def builtin_templates():
    t = {}
    t["Al2p: Al 金属 + Al2O3"] = (doublet("Al0 2p", 72.7, 0.44, 0.5, 0.8, lor=20)
                                + doublet("Al2O3 2p", 74.6, 0.44, 0.5, 1.6, lor=20))
    t["Al2p: 单一 Al2O3"] = doublet("Al2O3 2p", 74.6, 0.44, 0.5, 1.6, lor=20)
    t["C1s: 碳化物 + C-C / C-O / C=O / O-C=O"] = [new_peak("M-C 碳化物", 282.8, 1.0)] + _same_fwhm([
        new_peak("C-C", 284.8, 1.2), new_peak("C-O", 286.3, 1.2), new_peak("C=O", 287.8, 1.2), new_peak("O-C=O", 288.9, 1.2)])
    t["C1s: C-C / C-O / C=O / O-C=O"] = _same_fwhm([new_peak("C-C", 284.8, 1.2), new_peak("C-O", 286.3, 1.2),
                                                    new_peak("C=O", 287.8, 1.2), new_peak("O-C=O", 288.9, 1.2)])
    t["O1s: 晶格 / 缺陷·OH / 吸附"] = [new_peak("O lattice", 530.2, 1.2), new_peak("O defect/OH", 531.6, 1.5),
                                    new_peak("O adsorbed", 532.8, 1.6)]
    t["Ti2p: Ti0 / Ti2+ / Ti3+ / Ti4+"] = (doublet("Ti0 2p", 454.0, 6.1, 0.5, 1.0)
                                          + doublet("Ti2+ 2p", 455.5, 5.8, 0.5, 1.6)
                                          + doublet("Ti3+ 2p", 457.3, 5.7, 0.5, 1.6)
                                          + doublet("Ti4+ 2p", 458.8, 5.7, 0.5, 1.2))
    return t


def satellite(name, parent, dE, area_ratio=None, bind_dE=True, fwhm=None, lor=None):
    """卫星峰（shake-up 等）：位置相对主峰；ΔE 可绑定或在 ±1.5 eV 内自由；面积比可绑定或自由。"""
    c = parent["center"]["value"] + dE
    w = fwhm or parent["fwhm"]["value"] * 1.5
    p = new_peak(name, c, w, parent["area"]["value"] * (area_ratio or 0.1),
                 parent["lor"]["value"] if lor is None else lor, role="satellite")
    ln = {"ref": parent["name"]}
    if bind_dE:
        ln["dE"] = dE
    else:
        p["center"]["min"], p["center"]["max"] = c - 1.5, c + 1.5
    if area_ratio is not None:
        ln["area_ratio"] = area_ratio
    p["link"] = ln if len(ln) > 1 else {}
    if not p["link"]:
        p["link"] = {"ref": parent["name"]}   # 只记录归属，不约束
    return p


def apply_template(spec, peaks):
    """套用模板：面积按数据高度自动估算，位置保持模板值。"""
    x, y, bg, _, _ = spec.compute()
    s = y - bg
    new = copy.deepcopy(peaks)
    for p in new:
        bound_center(p, TEMPLATE_WINDOW)
        if "area" in linked_params(p):
            continue
        c, w = p["center"]["value"], p["fwhm"]["value"]
        h = float(np.interp(c, x, s)) if x[0] <= c <= x[-1] else 0.0
        p["area"]["value"] = max(h, 0.05 * max(s.max(), 1)) * w * 1.064
    spec.peaks = new


# ------------------------------------------------------------------ 灵敏度因子
def load_sf(path=None):
    here = os.path.dirname(os.path.abspath(__file__))
    for p in [path, os.path.join(os.getcwd(), "sensitivity_factors.json"),
              os.path.join(here, "sensitivity_factors.json")]:
        if p and os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            return {k: float(v) for k, v in d.get("factors", d).items()}
    return {}
