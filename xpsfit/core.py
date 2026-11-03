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


def background(x, y, kind="Shirley", navg=3, tougaard_B=2866.0, tougaard_C=1643.0):
    """x 为升序 BE。返回与 y 同长的背景。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if kind in (None, "", "None", "无"):
        return np.zeros_like(y)
    yl, yh = _ends(x, y, navg)          # 低 BE 端、高 BE 端
    if kind in ("Linear", "线性"):
        return yl + (yh - yl) * (x - x[0]) / (x[-1] - x[0] or 1)
    if kind == "Shirley":
        bg = np.full_like(y, yl)
        for _ in range(100):
            s = y - bg
            # 累积面积：从低 BE 端积到 x 处
            cum = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(x))])
            tot = cum[-1]
            new = yl + (yh - yl) * (cum / tot if tot else 0)
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
def new_peak(name, center, fwhm=1.2, area=1000.0, lor=30.0):
    p = {"name": name, "link": {}}
    for k, v in zip(PARAMS, (center, fwhm, area, lor)):
        p[k] = {"value": float(v), "fixed": k == "lor", "min": None, "max": None}
    p["fwhm"]["min"], p["fwhm"]["max"] = 0.2, 6.0
    p["area"]["min"] = 0.0
    p["lor"]["min"], p["lor"]["max"] = 0.0, 100.0
    return p
    # link 例：{"ref": "Al2p3/2", "dE": 0.44, "area_ratio": 0.5, "fwhm_ratio": 1.0}
    # 每项可省略；ref 为另一个峰的名字。dE/ratio 为 None 时该参数不绑定。


def resolve(peaks, free_vec=None, layout=None):
    """返回每个峰的实际参数 [(center,fwhm,area,lor)]，处理固定/绑定。"""
    vals = [{k: p[k]["value"] for k in PARAMS} for p in peaks]
    if free_vec is not None:
        for (i, k), v in zip(layout, free_vec):
            vals[i][k] = v
    names = {p["name"]: i for i, p in enumerate(peaks)}
    done = [False] * len(peaks)

    def fin(i, depth=0):
        if done[i] or depth > 20:
            return
        ln = peaks[i].get("link") or {}
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


def fit(x, y, bg, peaks, max_nfev=4000):
    """原地更新 peaks 中自由参数。返回 dict(chi2, rchi2, success, message)。"""
    x = np.asarray(x, float); s = np.asarray(y, float) - bg
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
    # 泊松权重：sigma ~ sqrt(counts)
    w = 1.0 / np.sqrt(np.clip(np.asarray(y, float), 1.0, None))

    def resid(v):
        _, env = model(x, peaks, resolve(peaks, v, layout))
        return (s - env) * w

    if layout:
        r = least_squares(resid, np.array(x0), bounds=(np.array(lo), np.array(hi)),
                          max_nfev=max_nfev, x_scale="jac")
        for (i, k), v in zip(layout, r.x):
            peaks[i][k]["value"] = float(v)
        ok, msg = bool(r.success), r.message
    else:
        ok, msg = True, "没有自由参数"
    vals = resolve(peaks)
    for p, v in zip(peaks, vals):     # 把绑定参数的结果写回，便于显示
        for k in PARAMS:
            p[k]["value"] = float(v[k])
    res = resid(np.array([peaks[i][k]["value"] for i, k in layout])) if layout else (s - model(x, peaks)[1]) * w
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

    @property
    def be(self):
        return self.be_raw + self.shift

    def region(self):
        lo = self.bg["lo"] if self.bg["lo"] is not None else self.be[0]
        hi = self.bg["hi"] if self.bg["hi"] is not None else self.be[-1]
        lo, hi = min(lo, hi), max(lo, hi)
        m = (self.be >= lo - 1e-9) & (self.be <= hi + 1e-9)
        if m.sum() < 5:
            m[:] = True
        return m

    def compute(self):
        """返回 (x, y, bg, comps, envelope) —— 仅背景区间内。"""
        m = self.region()
        x, y = self.be[m], self.y[m]
        bg = background(x, y, self.bg["type"], int(self.bg.get("navg", 3)))
        comps, env = model(x, self.peaks)
        return x, y, bg, comps, env

    def do_fit(self):
        m = self.region()
        x, y = self.be[m], self.y[m]
        bg = background(x, y, self.bg["type"], int(self.bg.get("navg", 3)))
        self.last_fit = fit(x, y, bg, self.peaks)
        return self.last_fit

    def raw_area(self):
        x, y, bg, _, _ = self.compute()
        return float(np.trapezoid(y - bg, x)) if hasattr(np, "trapezoid") else float(np.trapz(y - bg, x))

    def to_dict(self):
        return {"name": self.name, "source": self.source, "hv": self.hv,
                "be": self.be_raw.tolist(), "y": self.y.tolist(), "shift": self.shift,
                "bg": self.bg, "peaks": self.peaks, "element": self.element}

    @classmethod
    def from_dict(cls, d):
        s = cls(d["name"], d["be"], d["y"], d.get("source", ""), d.get("hv", 1486.68))
        s.shift, s.bg, s.peaks = d.get("shift", 0.0), d["bg"], d["peaks"]
        s.element = d.get("element", s.element)
        return s


def guess_orbital(name):
    import re
    m = re.search(r"([A-Z][a-z]?)\s*(\d[spdf])(\d/2)?", name or "")
    return (m.group(1) + m.group(2)) if m else ""


# ------------------------------------------------------------------ 参数表 / 定量
def peak_table(spec):
    """每个峰：名称、位置、FWHM、面积、L%、面积%。"""
    tot = sum(p["area"]["value"] for p in spec.peaks) or 1.0
    return [{"spectrum": spec.name, "peak": p["name"], "BE_eV": p["center"]["value"],
             "FWHM_eV": p["fwhm"]["value"], "area": p["area"]["value"], "L_percent": p["lor"]["value"],
             "area_percent": 100.0 * p["area"]["value"] / tot} for p in spec.peaks]


def quantify(specs, sf_table, use_fit=True, ecf_exp=0.6):
    """原子百分比：A / (SF · KE^ecf_exp)。KE = hν − BE(峰中心)。ecf_exp=0 不做能量补偿。"""
    rows = []
    for s in specs:
        sf = sf_table.get(s.element)
        if use_fit and s.peaks:
            area = sum(p["area"]["value"] for p in s.peaks)
            be = sum(p["center"]["value"] * p["area"]["value"] for p in s.peaks) / (area or 1)
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
def doublet(name, center, split, ratio, fwhm=1.2, area=1000.0, lor=30.0, labels=None):
    """自旋轨道双峰：返回 [主峰, 副峰]，副峰位置/面积/FWHM 绑定主峰。"""
    a, b = labels or (name + " 3/2", name + " 1/2")
    p1 = new_peak(a, center, fwhm, area, lor)
    p2 = new_peak(b, center + split, fwhm, area * ratio, lor)
    p2["link"] = {"ref": a, "dE": split, "area_ratio": ratio, "fwhm_ratio": 1.0, "lor_same": True}
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
    t["C1s: C-C / C-O / C=O / O-C=O"] = _same_fwhm([new_peak("C-C", 284.8, 1.2), new_peak("C-O", 286.3, 1.2),
                                                    new_peak("C=O", 287.8, 1.2), new_peak("O-C=O", 288.9, 1.2)])
    t["O1s: 晶格 / 缺陷·OH / 吸附"] = [new_peak("O lattice", 530.2, 1.2), new_peak("O defect/OH", 531.6, 1.5),
                                    new_peak("O adsorbed", 532.8, 1.6)]
    t["Ti2p: Ti0 / Ti2+ / Ti3+ / Ti4+"] = (doublet("Ti0 2p", 454.0, 6.1, 0.5, 1.0)
                                          + doublet("Ti2+ 2p", 455.5, 5.8, 0.5, 1.6)
                                          + doublet("Ti3+ 2p", 457.3, 5.7, 0.5, 1.6)
                                          + doublet("Ti4+ 2p", 458.8, 5.7, 0.5, 1.2))
    return t


def apply_template(spec, peaks):
    """套用模板：面积按数据高度自动估算，位置保持模板值。"""
    x, y, bg, _, _ = spec.compute()
    s = y - bg
    new = copy.deepcopy(peaks)
    for p in new:
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
