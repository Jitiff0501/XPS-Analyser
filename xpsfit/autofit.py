# -*- coding: utf-8 -*-
"""一键拟合：自动选背景 → 按元素建模 → 残差检查并区分 重叠峰 / 卫星峰 / 新化学态 → BIC 判断取舍。
另含微调工具：局部微调、多起点全局拟合、参数不确定度、质量检查。

模型与规则来自对真实数据（Al2p, C1s, Co2p, Cr2p, Cu2p, O1s）逐条核对时采用的文献约束：
Thermo XPS Knowledge、NIST SRD 20、Biesinger 等对 3d 过渡金属 2p 的系统研究。数值均为常用参考值。
"""
import copy
import math
import re

import numpy as np

from . import core

# ------------------------------------------------------------------ 其他元素的谱线（用于识别重叠峰）
# 俄歇峰：最强几条线的动能 KE (eV)；在 BE 轴上位于 hν − KE（近似值，±1 eV）
AUGER_KE = {
    "C": [263], "N": [379], "O": [510, 490], "F": [656], "Na": [990],
    "Ti": [418, 383], "V": [473, 437], "Cr": [529, 489], "Mn": [589, 542], "Fe": [703, 651, 598],
    "Co": [775, 716, 656], "Ni": [848, 775, 716], "Cu": [918, 838], "Zn": [992, 1013],
    "Mg": [1186], "Al": [1393],
}
# 其他能级的结合能 BE (eV)（近似）
CORE_BE = {
    "Cu": {"Cu3p": [75.1, 77.4], "Cu3s": [122.5], "Cu2p": [932.6, 952.4]},
    "Cr": {"Cr3s": [74.1], "Cr3p": [42.4], "Cr2p": [574.2, 583.5]},
    "Co": {"Co3p": [59.5], "Co3s": [101.0], "Co2p": [778.2, 793.2]},
    "Ni": {"Ni3p": [66.3], "Ni3s": [111.0], "Ni2p": [852.6, 870.0]},
    "Fe": {"Fe3p": [52.8], "Fe3s": [91.3], "Fe2p": [706.7, 719.8]},
    "Mn": {"Mn3p": [47.5], "Mn3s": [83.0], "Mn2p": [638.7, 650.3]},
    "Ti": {"Ti3p": [32.6], "Ti3s": [58.7], "Ti2p": [454.0, 460.1]},
    "V": {"V3p": [37.5], "V3s": [66.0], "V2p": [512.4, 519.8]},
    "Zn": {"Zn3d": [10.0], "Zn3p": [88.6], "Zn2p": [1021.8, 1044.9]},
    "Al": {"Al2p": [72.7], "Al2s": [117.9]},
    "Si": {"Si2p": [99.4], "Si2s": [150.5]},
    "O": {"O2s": [23.0], "O1s": [530.5]},
    "C": {"C1s": [284.8]}, "N": {"N1s": [398.0]},
    "Mo": {"Mo3d": [228.0, 231.1], "Mo3p": [394.0, 411.5]},
    "Zr": {"Zr3d": [179.0, 181.4], "Zr3p": [330.5, 344.2]},
    "Ta": {"Ta4f": [21.8, 23.7], "Ta4p": [404.0]}, "Pt": {"Pt4f": [71.1, 74.4]},
    "K": {"K2p": [293.0, 295.8]}, "Ca": {"Ca2p": [347.0, 350.5]}, "Ru": {"Ru3d": [280.1, 284.2]},
    "S": {"S2p": [162.0], "S2s": [227.0]}, "Cl": {"Cl2p": [199.0]}, "Na": {"Na1s": [1071.5]},
}
# 有明显卫星 / 损失结构的元素：卫星位于主峰高 BE 侧的范围 (eV)
SAT_RULES = {
    "Co": (2.5, 12.0), "Cu": (6.0, 12.0), "Ni": (5.0, 8.0), "Fe": (5.0, 9.0), "Mn": (4.5, 7.0),
    "C": (5.5, 7.5), "Ti": (12.0, 14.0), "Cr": (10.0, 12.0),
}
SPLIT = {"Al2p": 0.44, "Si2p": 0.61, "P2p": 0.87, "S2p": 1.18, "Cl2p": 1.6, "K2p": 2.8, "Ca2p": 3.55,
         "Ti2p": 5.7, "V2p": 7.4, "Cr2p": 9.3, "Mn2p": 11.7, "Fe2p": 13.6, "Co2p": 15.0, "Ni2p": 17.3,
         "Cu2p": 19.8, "Zn2p": 23.1, "Zr3d": 2.43, "Nb3d": 2.72, "Mo3d": 3.13, "Ag3d": 6.0,
         "Hf4f": 1.68, "Ta4f": 1.91, "W4f": 2.18, "Pt4f": 3.33, "Au4f": 3.67}
RATIO = {"p": 0.5, "d": 2 / 3, "f": 0.75}
# 本元素常见的其他化学态（主线 BE 范围，eV）——残差峰落在这里时作为新化学态加入
KNOWN_STATES = {
    # 只列能与常见主组分分辨（≳1.5 eV）的态；Cr(OH)3/Cr2O3、Al(OH)3/Al2O3、Co3+/Co2+ 在常规分辨率下分不开，不列
    "Cr2p": [("Cr6+", 579.0, 580.6)],
    "Cu2p": [("Cu2+", 933.3, 934.6)],
    "Ti2p": [("Ti4+", 458.4, 459.2), ("Ti3+", 456.7, 457.7), ("Ti2+", 455.0, 455.9), ("TiN", 454.8, 455.3)],
    "Fe2p": [("Fe3+", 710.5, 711.5), ("Fe2+", 709.2, 710.2)],
    "Ni2p": [("NiO", 853.5, 854.3), ("Ni(OH)2", 855.2, 856.2)],
    "Mn2p": [("Mn2+", 640.6, 641.4), ("Mn4+", 642.0, 642.6)],
    "C1s": [("CO3", 289.5, 290.5)],
    "N1s": [("nitride", 396.4, 397.4), ("C-N", 399.0, 400.5)],
    "O1s": [("O lattice", 529.3, 530.6), ("OH / defect", 531.0, 532.0)],
}
# 必须伴随卫星峰的化学态：卫星区间（eV）内没有信号就不能加入（如 Cu²⁺：940–946 eV）
REQUIRES_SAT = {"Cu2+": (940.0, 946.0), "Ni2+": (859.0, 864.0)}
# 哪些化学态可以带卫星峰（正则匹配主峰名）；未列出的元素：任何主峰都可以
SAT_PARENT = {"Cu": r"2\+|CuO|Cu\(OH\)", "C": r"C-C|graph|sp2"}


# ------------------------------------------------------------------ 构件
def _P(name, c, w, lor=30, shape="GL", role="main", lo=None, hi=None, wmin=1.0, wmax=3.0):
    p = core.new_peak(name, c, w, 1000, lor, role=role, shape=shape, tail=0.2, tau=1.5)
    p["center"]["min"], p["center"]["max"] = (c - 0.6 if lo is None else lo), (c + 0.6 if hi is None else hi)
    p["fwhm"]["min"], p["fwhm"]["max"] = wmin, wmax
    p["fwhm"]["value"] = min(max(w, wmin), wmax)
    # 金属峰拖尾的常见范围：过长的拖尾会“吃掉”相邻氧化物 / 高价态的信号
    p["tail"]["min"], p["tail"]["max"] = 0.05, 0.45
    p["tau"]["min"], p["tau"]["max"] = 0.5, 2.0
    return p


def _D(name, c, split, ratio, w, sub="2p", shape="GL", lo=None, hi=None, split_rng=None, ratio_rng=None,
       fwr=(1.0, 1.3), wmin=1.0, wmax=3.0, lor=30):
    j = {"p": ("3/2", "1/2"), "d": ("5/2", "3/2"), "f": ("7/2", "5/2")}[sub[-1]]
    a, b = "%s %s%s" % (name, sub, j[0]), "%s %s%s" % (name, sub, j[1])
    p1 = _P(a, c, w, lor, shape, lo=lo, hi=hi, wmin=wmin, wmax=wmax)
    p2 = _P(b, c + split, w, lor, shape, wmin=wmin, wmax=wmax * fwr[1])
    ln = {"ref": a, "dE": split, "area_ratio": ratio, "fwhm_ratio": fwr[0], "lor_same": True}
    if split_rng:
        ln.update(dE_fit=True, dE_min=split_rng[0], dE_max=split_rng[1])
    if ratio_rng:
        ln.update(area_ratio_fit=True, area_ratio_min=ratio_rng[0], area_ratio_max=ratio_rng[1])
    if fwr[1] > fwr[0]:
        ln.update(fwhm_ratio=(fwr[0] + fwr[1]) / 2, fwhm_ratio_fit=True, fwhm_ratio_min=fwr[0], fwhm_ratio_max=fwr[1])
    p2["link"], p2["pair"] = ln, True
    return [p1, p2]


def _SAT(name, ref, c, w, lo, hi, wmax=8.0, ratio=None, wmin=None):
    """卫星峰；ratio=(min,max) 时面积与主峰之比限制在该范围内拟合。"""
    p = _P(name, c, w, 0, role="satellite", lo=lo, hi=hi, wmin=wmin or (3.0 if wmax >= 5 else 1.0), wmax=wmax)
    p["link"] = {"ref": ref}
    if ratio:
        p["link"].update(area_ratio=(ratio[0] + ratio[1]) / 2, area_ratio_fit=True,
                         area_ratio_min=ratio[0], area_ratio_max=ratio[1])
    return p


def _signal(s, lo, hi):
    """区间内扣背景后的信号面积 / 全谱信号面积。"""
    x, y, bg, _, _ = s.compute()
    sig = np.clip(core.smooth(y, 5) - bg, 0, None)
    m = (x >= lo) & (x <= hi)
    tot = float(np.trapezoid(sig, x)) if hasattr(np, "trapezoid") else float(np.trapz(sig, x))
    part = float(np.trapezoid(sig[m], x[m])) if m.sum() > 2 else 0.0
    return part / tot if tot > 0 else 0.0


# ------------------------------------------------------------------ 按元素的起始模型（文献约束）
def recipe(s, log):
    el = s.element
    if el == "Al2p":
        log("Al2p：金属 Al⁰ 用不对称双峰（Δ0.44 锁定）；氧化物 Al³⁺ 双峰 FWHM ≤ 2.0 eV（Thermo）")
        return _D("Al0", 72.6, 0.44, 0.5, 0.8, shape="Asym", lo=71.8, hi=73.0, wmin=0.6, wmax=1.6, fwr=(1.0, 1.0)) + \
            _D("Al3+", 74.6, 0.44, 0.5, 1.6, lo=73.8, hi=75.6, wmax=2.0, fwr=(1.0, 1.0))
    if el == "Co2p":
        log("Co2p：Co⁰ 不对称双峰 Δ14.9–15.1；Co²⁺ 双峰 Δ15.2–15.7 + 卫星 ~786 / ~802 eV")
        return (_D("Co0", 778.2, 14.99, 0.5, 1.0, shape="Asym", lo=777.9, hi=778.5, split_rng=(14.9, 15.1), ratio_rng=(0.45, 0.55), fwr=(1.0, 1.4))
                + _D("Co2+", 780.6, 15.4, 0.5, 2.6, lo=779.9, hi=781.5, split_rng=(15.2, 15.7))
                + [_SAT("Co2+ sat. 2p3/2", "Co2+ 2p3/2", 786.0, 5.0, 783.5, 788.5, wmax=6.0, ratio=(0.3, 0.9)),
                   _SAT("Co2+ sat. 2p1/2", "Co2+ 2p1/2", 802.3, 5.0, 800.0, 804.5, wmax=6.0, ratio=(0.3, 0.9))])
    if el == "Cr2p":
        log("Cr2p：Cr⁰ 不对称双峰 Δ9.2–9.4；Cr³⁺ 双峰 Δ9.6–9.9（多重分裂以宽峰近似）")
        return (_D("Cr0", 574.2, 9.3, 0.5, 1.2, shape="Asym", lo=573.6, hi=574.7, split_rng=(9.2, 9.4), ratio_rng=(0.45, 0.55), fwr=(1.0, 1.5))
                + _D("Cr3+", 576.4, 9.75, 0.5, 2.6, lo=575.6, hi=577.3, split_rng=(9.6, 9.9)))
    if el == "Cu2p":
        m = _D("Cu0/Cu+", 932.6, 19.8, 0.5, 1.2, shape="Asym", lo=932.3, hi=932.9, split_rng=(19.7, 20.0), ratio_rng=(0.42, 0.55))
        f = _signal(s, 940.0, 946.0)
        if f > 0.02:
            cu2 = _D("Cu2+", 933.8, 19.9, 0.5, 2.5, lo=933.4, hi=934.4, wmax=3.5) + [_SAT("Cu2+ sat.", "Cu2+ 2p3/2", 942.5, 4.5, 940.0, 945.5, wmax=7.0, ratio=(0.4, 0.75))]
            for p in cu2:
                p["optional"] = "Cu2+"
            log("Cu2p：940–946 eV 有卫星信号（%.1f%%）→ Cu²⁺ 及其卫星峰作为候选，拟合后用 BIC 判断是否保留" % (100 * f))
            m += cu2
        else:
            log("Cu2p：940–946 eV 无卫星信号（%.1f%%）→ 不加 Cu²⁺，只用 Cu⁰/Cu⁺" % (100 * f))
        return m
    if el == "O1s":
        log("O1s：晶格 O / 缺陷·羟基 / C–O·吸附水 三组分")
        return [_P("O lattice (M-O)", 530.2, 1.3, lo=529.4, hi=531.0), _P("O defect / M-OH", 531.5, 1.6, lo=530.9, hi=532.2),
                _P("C-O / H2O ads.", 532.8, 1.8, lo=532.1, hi=533.8)]
    if el == "C1s":
        m = []
        f = _signal(s, 282.0, 283.7)
        if f > 0.08:
            log("C1s：282–283.7 eV 有明显信号（%.0f%%）→ 加入金属碳化物" % (100 * f))
            m.append(_P("M-C carbide", 283.0, 1.1, lo=282.3, hi=283.6))
        return m + [_P("C-C/C-H", 284.8, 1.3, lo=284.3, hi=285.3), _P("C-O", 286.3, 1.4, lo=285.8, hi=287.0),
                    _P("C=O / O-C=O", 288.6, 1.6, lo=287.6, hi=289.6)]
    return None


def generic_model(s, log, ref=None):
    """没有内置模型：在数据中找峰，p/d/f 轨道按分裂加双峰；名称尽量用 NIST 最近的化合物。"""
    from scipy.signal import find_peaks
    x, y, bg, _, _ = s.compute()
    sig = core.smooth(y - bg, 7)
    k, pr = find_peaks(sig, prominence=0.08 * float(np.ptp(sig) or 1))
    pos = sorted((float(x[j]) for j in k), key=lambda v: -sig[np.argmin(abs(x - v))])
    l_ = (s.element or "x")[-1]
    split = SPLIT.get(s.element)
    used, peaks = [], []
    for c in pos:
        if split and any(abs(c - (u + split)) < 0.8 for u in used):
            continue                       # 已被某双峰的 1/2 分量覆盖
        name = "P%.1f" % c
        if ref and not ref.get("missing"):
            st = min(ref["states"], key=lambda t: abs(t["median"] - c))
            if abs(st["median"] - c) < 1.0:
                name = st["formula"]
        if split and l_ in RATIO:
            peaks += _D(name, c, split, RATIO[l_], 1.5, sub=s.element[-2:], lo=c - 0.8, hi=c + 0.8)
        else:
            peaks.append(_P(name, c, 1.5, lo=c - 0.8, hi=c + 0.8))
        used.append(c)
        if len(used) >= 4:
            break
    log("通用模型：在数据中找到 %d 个主峰（%s）" % (len(used), ", ".join("%.1f" % u for u in used)))
    return peaks


# ------------------------------------------------------------------ 背景自动选择
def auto_anchors(s, x, y):
    """自定义点背景的锚点：各峰之间、两端平坦区的局部最低点。"""
    from scipy.signal import find_peaks, peak_widths
    ys = core.smooth(y, 7)
    k, _ = find_peaks(ys, prominence=0.05 * float(np.ptp(ys) or 1))
    step = float(np.mean(np.diff(x))) if len(x) > 1 else 0.1
    wid = peak_widths(ys, k, rel_height=0.5)[0] * step if len(k) else []
    # 锚点要离峰至少 1.2×FWHM（≥1.5 eV），不能落在峰尾上
    margins = [max(1.5, 1.2 * float(w)) for w in wid]
    edges = [(float(x[0]), 0.4)] + [(float(x[j]), m) for j, m in zip(k, margins)] + [(float(x[-1]), 0.4)]
    pts = []
    for (a, ma), (b, mb) in zip(edges[:-1], edges[1:]):
        seg = [(a, b, ma, mb)] if b - a < 10 else [(a, (a + b) / 2, ma, 0.0), ((a + b) / 2, b, 0.0, mb)]
        for lo, hi, mlo, mhi in seg:
            m = (x >= lo + mlo) & (x <= hi - mhi)
            if m.sum() < 3:
                continue
            j = np.where(m)[0][np.argmin(ys[m])]
            pts.append([float(x[j]), float(ys[j])])
    for j in (0, len(x) - 1):            # 两端
        pts.append([float(x[j]), float(ys[j])])
    pts.sort()
    out = []
    for p in pts:
        if not out or p[0] - out[-1][0] > 0.6:
            out.append(p)
    return out


def choose_background(s, log):
    x0 = s.be[s.region()]
    s.bg.update({"type": "Shirley", "navg": 5})
    x, y, bg, _, _ = s.compute()
    ys = core.smooth(y, 5)
    noise = float(np.std(y - ys)) or 1.0
    n = max(3, len(x) // 40)
    lo_end, hi_end = float(np.mean(ys[:n])), float(np.mean(ys[-n:]))
    over = float(np.max(bg - ys))
    if lo_end - hi_end > 2 * noise:
        pts = auto_anchors(s, x, y)
        s.bg.update({"type": "Points", "points": pts, "interp": "smooth"})
        log("背景：低 BE 端（%.0f）高于高 BE 端（%.0f），本底反向，Shirley 不适用 → 自定义点背景（%d 个平坦区最低点）"
            % (lo_end, hi_end, len(pts)))
    elif over > 5 * noise:
        pts = auto_anchors(s, x, y)
        s.bg.update({"type": "Points", "points": pts, "interp": "smooth"})
        log("背景：Shirley 会高于数据 %.0f（%.0fσ），窗口内可能有其他谱线 → 自定义点背景（%d 个平坦区最低点）" % (over, over / noise, len(pts)))
    elif over > 1e-6 * max(abs(float(np.max(ys))), 1.0):   # 背景任何一处高于数据都不用 Shirley
        s.bg["type"] = "Smart"
        log("背景：Shirley 会高于数据 %.0f（噪声 σ≈%.0f）→ Smart（不高于数据）" % (over, noise))
    else:
        log("背景：Shirley（未高于数据，噪声 σ≈%.0f）" % noise)
    return noise


# ------------------------------------------------------------------ 残差检查：重叠 / 卫星 / 新组分
def other_lines(sample_elements, own_element, hv):
    """同一样品其他元素可能落在当前窗口的谱线：[(BE, 名称, 类型, 容差)]。"""
    own = re.match(r"[A-Z][a-z]?", own_element or "")
    own = own.group(0) if own else ""
    out = []
    for el in sample_elements:
        if el == own:
            # 自己的俄歇也可能落在窗口内
            for ke in AUGER_KE.get(el, []):
                out.append((hv - ke, "%s LMM/KLL 俄歇 (KE≈%d)" % (el, ke), "auger", 1.5))
            continue
        for ke in AUGER_KE.get(el, []):
            out.append((hv - ke, "%s 俄歇 (KE≈%d)" % (el, ke), "auger", 1.5))
        for lvl, bes in CORE_BE.get(el, {}).items():
            for b in bes:
                out.append((b, "%s" % lvl, "core", 0.9))
    return out


def classify(c, width, s, sample_elements, ref, log_ctx):
    """返回 (类别, 名称, 参考峰名或None, 说明)。类别：overlap / satellite / main / unknown。"""
    el = re.match(r"[A-Z][a-z]?", s.element or "")
    el = el.group(0) if el else ""
    # 0) 已识别的俄歇重叠峰很宽：其附近 ±5 eV 的残差也归于同一来源
    for p in s.peaks:
        if p.get("role") == "unknown" and "俄歇" in p["name"] and abs(p["center"]["value"] - c) <= 5.0:
            return ("overlap", p["name"] + "（续）", None,
                    "%.2f eV 的残差峰在已识别的 %s 附近（%.1f eV 内），俄歇峰很宽 → 同一重叠来源" % (c, p["name"].replace("overlap: ", ""), abs(p["center"]["value"] - c)))
    # 1) 其他元素的谱线
    best = None
    for be, name, kind, tol in other_lines(sample_elements, s.element, s.hv):
        d = abs(be - c)
        if d <= tol and (best is None or d < best[0]):
            best = (d, be, name, kind)
    if best:
        return ("overlap", "overlap: %s" % best[2], None,
                "%.2f eV 处的残差峰与 %s（%.1f eV）相差 %.1f eV，且样品含该元素 → 重叠峰（不计入定量）" % (c, best[2], best[1], best[0]))
    # 2) 卫星：本元素的主峰高 BE 侧
    rule = SAT_RULES.get(el)
    mains = [p for p in s.peaks if p.get("role", "main") == "main"]
    if el in SAT_PARENT:
        mains = [p for p in mains if re.search(SAT_PARENT[el], p["name"])]
    if rule and mains:
        cand = [(c - p["center"]["value"], p) for p in mains if rule[0] <= c - p["center"]["value"] <= rule[1]]
        if cand:
            # 优先氧化态（名称含 + 或 O）；其次距离最近
            cand.sort(key=lambda t: (0 if re.search(r"\d\+|O|ox", t[1]["name"]) else 1, t[0]))
            d, par = cand[0]
            return ("satellite", "%s sat." % re.sub(r"\s*\d?[spdf]?\d/2\s*$", "", par["name"]), par["name"],
                    "%.2f eV 处的残差峰位于 %s 高 BE 侧 +%.1f eV（%s 卫星区间 %.0f–%.0f eV）→ 卫星峰" % (c, par["name"], d, el, rule[0], rule[1]))
    # 3) 本元素常见的其他化学态
    for name, a, b in KNOWN_STATES.get(s.element, []):
        if name in REQUIRES_SAT and _signal(s, *REQUIRES_SAT[name]) < 0.04:
            continue
        if a - 0.2 <= c <= b + 0.2 and not any(p["name"].startswith(name) for p in s.peaks):
            return ("main", name, None, "%.2f eV 落在 %s 的常见范围（%.1f–%.1f eV）→ 新化学态" % (c, name, a, b))
    # 4) 参考库里的化学态
    if ref and not ref.get("missing"):
        st = min(ref["states"], key=lambda t: abs(t["median"] - c))
        if abs(st["median"] - c) <= max(0.6, st["q3"] - st["q1"]):
            return ("main", st["formula"], None, "%.2f eV 接近 NIST 中 %s（%.2f eV）→ 新化学态" % (c, st["formula"], st["median"]))
    return ("unknown", "unknown @%.1f" % c, None, "%.2f eV 的残差峰无法归属（不是同样品其他元素的谱线，也不在卫星区间）→ 作为未知峰，不计入定量，请确认" % c)


def _pinned(peaks):
    """卡在约束边界上的峰位 / FWHM 参数个数（非绑定、非固定）。"""
    n = 0
    for p in peaks:
        for k in ("center", "fwhm"):
            par = p[k]
            if k in core.linked_params(p) or par["fixed"] or par["min"] is None or par["max"] is None:
                continue
            if min(abs(par["value"] - par["min"]), abs(par["value"] - par["max"])) < 1e-3 * max(1.0, abs(par["value"]) * 1e-3):
                n += 1
    return n


def co_alternative(s, r, log):
    """Co2p：金属 Co 在 +3~+5 eV 有固有损失结构，与 Co²⁺ 卫星区重叠，无法由谱唯一确定。
    同时拟合模型 B（加入 Co⁰ 损失峰，计入 Co⁰），报告 Co²⁺ 比例的范围，默认保留 BIC 更好的模型。"""
    import copy as _c

    def frac(peaks):
        t = _c.deepcopy(s); t.peaks = peaks
        st = core.chemical_states(t); tot = sum(a for _, a in st) or 1
        return sum(a for n, a in st if n.startswith("Co2+")) / tot
    A = _c.deepcopy(s.peaks); bA = bic(s, r); fA = frac(A)
    loss = [_SAT("Co0 loss +3eV", "Co0 2p3/2", 781.2, 2.5, 780.6, 782.2, wmax=4.0, ratio=(0.03, 0.3)),
            _SAT("Co0 loss +5eV", "Co0 2p3/2", 783.3, 3.0, 782.6, 784.3, wmax=5.0, ratio=(0.03, 0.3))]
    s.peaks = s.peaks + loss
    rB = staged_fit(s); bB = bic(s, rB); fB = frac(s.peaks)
    lo_, hi_ = sorted((fA, fB))
    log("Co2p 模型对比：A（780–787 eV 肩峰全归 Co²⁺）Co²⁺ %.0f%%；B（含金属 Co 的 +3/+5 eV 损失峰）Co²⁺ %.0f%%。"
        "两者都与数据一致，报告 Co²⁺ 范围 %.0f–%.0f%%" % (100 * fA, 100 * fB, 100 * lo_, 100 * hi_))
    if bB < bA - 6:
        log("默认显示模型 B（BIC 更好）")
        return rB
    s.peaks = A
    log("默认显示模型 A（B 没有明显更好）")
    return s.do_fit()


def staged_fit(s):
    """两步拟合：① 不对称峰的拖尾固定在典型值（T=0.25, τ=1.2 eV），先让各组分定位；② 再放开拖尾（在常见范围内）。"""
    asym = [p for p in s.peaks if p.get("shape") == "Asym" and "tail" not in core.linked_params(p)]
    saved = [(p["tail"]["fixed"], p["tau"]["fixed"]) for p in asym]
    for p in asym:
        p["tail"]["value"], p["tau"]["value"] = 0.25, 1.2
        p["tail"]["fixed"] = p["tau"]["fixed"] = True
    s.do_fit()
    for p, (a, b) in zip(asym, saved):
        p["tail"]["fixed"], p["tau"]["fixed"] = a, b
    return s.do_fit()


def test_known_states(s, b0, r, log):
    """主动测试本元素常见的其他化学态（如 Cr⁶⁺）：逐个试加，BIC 明显下降且面积 ≥2% 才保留。"""
    x = s.be[s.region()]
    lo, hi = float(x.min()), float(x.max())
    l_ = (s.element or "x")[-1]
    split = SPLIT.get(s.element)
    for name, a, b in KNOWN_STATES.get(s.element, []):
        c = (a + b) / 2
        if not (lo + 1 <= c <= hi - 1):
            continue
        if name in REQUIRES_SAT and _signal(s, *REQUIRES_SAT[name]) < 0.04:
            continue                                   # 没有卫星就不可能有该价态
        if any(re.sub(r"\s.*", "", p["name"]) == name for p in s.peaks):
            continue
        # 只测试能分辨的态：与已有组分（含双峰分量）相距 ≥1.5 eV；太近的无法区分，只提示
        near = [p for p in s.peaks if p.get("role", "main") == "main" and
                abs(p["center"]["value"] - c) < max(1.5, 0.8 * p["fwhm"]["value"])]
        if near:
            log("提示：%s（%.1f–%.1f eV）与 %s 相距不到 1.5 eV，此分辨率下无法区分，未单独拟合（可能包含在该组分中）"
                % (name, a, b, re.sub(r"\s*\d[spdf]\d/2$", "", near[0]["name"])))
            continue
        trial = s.peaks
        if split and l_ in RATIO and split > 1.0:
            new = _D(name, c, split, RATIO[l_], 1.8, sub=s.element[-2:], lo=a - 0.2, hi=b + 0.2)
        else:
            new = [_P(name, c, 1.8, lo=a - 0.2, hi=b + 0.2)]
        import copy as _c
        saved = _c.deepcopy(s.peaks)
        s.peaks = s.peaks + new
        r2 = staged_fit(s)
        b2 = bic(s, r2)
        tot = sum(p["area"]["value"] for p in s.peaks if core.counted(p)) or 1.0
        part = sum(p["area"]["value"] for p in s.peaks if p["name"].startswith(name)) / tot
        if b2 < b0 - 10 and part >= 0.02:
            b0, r = b2, r2
            log("化学态测试：加入 %s（%.1f–%.1f eV）使拟合明显改善，占 %.1f%% → 新化学态，保留" % (name, a, b, 100 * part))
        else:
            s.peaks = saved
            r = s.do_fit()
    return b0, r


def prior_overlaps(s, sample_elements, log):
    """同一样品其他元素的谱线若落在窗口内，预先加入为重叠峰（不计入定量）。
    与本元素主峰相距 <1 eV 的（如 Al2p 窗口里的 Cr3s 与 Al³⁺）无法区分 → 不加，只提示。"""
    x, y, bg, _, _ = s.compute()
    lo, hi = float(x.min()) + 0.5, float(x.max()) - 0.5
    mains = [p["center"]["value"] for p in s.peaks if p.get("role", "main") == "main"]
    own = re.match(r"[A-Z][a-z]?", s.element or "")
    own = own.group(0) if own else ""
    for el in sample_elements:
        if el == own:
            continue
        for lvl, bes in CORE_BE.get(el, {}).items():
            inside = [b for b in bes if lo <= b <= hi]
            if not inside or lvl.endswith(("2p", "3d", "4f")) and len(bes) > 1 and not all(lo <= b <= hi for b in bes[:1]):
                continue
            if all(any(abs(b - m) < 1.0 for m in mains) for b in inside):
                log("⚠ %s（%s eV）与本谱主峰几乎重合，无法拟合区分，可能使比例偏高；请结合该元素含量判断" % (
                    lvl, "、".join("%.1f" % b for b in inside)))
                continue
            if len(inside) == 2:
                # 双峰能级（如 Cu3p）：按锁定间距/面积比加入，远离主峰的分量把整对峰“锚定”住
                p1 = _P("overlap: %s" % lvl, inside[0], 2.0, role="unknown", lo=inside[0] - 0.6, hi=inside[0] + 0.6, wmax=3.0)
                p2 = _P("overlap: %s（分量）" % lvl, inside[1], 2.0, role="unknown", wmax=3.0)
                p2["link"] = {"ref": p1["name"], "dE": inside[1] - inside[0], "area_ratio": 0.5, "fwhm_ratio": 1.0, "lor_same": True}
                p2["pair"] = True
                new = [p1, p2]
            else:
                new = [_P("overlap: %s" % lvl, b, 2.0, role="unknown", lo=b - 0.8, hi=b + 0.8, wmax=3.0) for b in inside]
            for p in new:
                p["prior"] = True
                s.peaks.append(p)
            log("预置重叠峰：样品含 %s，%s（%s eV）落在窗口内 → 预先加入为重叠峰（不计入定量）" % (
                el, lvl, "、".join("%.1f" % b for b in inside)))
        for ke in AUGER_KE.get(el, []):
            b = s.hv - ke
            if lo <= b <= hi:
                if any(abs(b - m) < 1.0 for m in mains):
                    continue
                p = _P("overlap: %s 俄歇 (KE≈%d)" % (el, ke), b, 5.0, role="unknown", lo=b - 2.0, hi=b + 2.0, wmax=8.0)
                p["prior"] = True
                s.peaks.append(p)
                log("预置重叠峰：样品含 %s，其俄歇峰（KE≈%d → BE≈%.1f eV）落在窗口内 → 预先加入为重叠峰" % (el, ke, b))


def residual_bumps(s, noise):
    from scipy.signal import find_peaks, peak_widths
    x, y, bg, comps, env = s.compute()
    r = core.smooth(y - bg - env, 7)
    sig_tot = float(np.trapezoid(np.clip(env, 0, None), x)) if hasattr(np, "trapezoid") else float(np.trapz(np.clip(env, 0, None), x))
    k, pr = find_peaks(r, height=4 * noise, prominence=3 * noise)
    out = []
    if not len(k):
        return out
    w = peak_widths(r, k, rel_height=0.5)[0] * float(np.mean(np.diff(x)))
    for j, wj in zip(k, w):
        m = np.abs(x - x[j]) <= max(wj, 0.5)
        area = float(np.trapezoid(np.clip(r[m], 0, None), x[m])) if hasattr(np, "trapezoid") else float(np.trapz(np.clip(r[m], 0, None), x[m]))
        if sig_tot > 0 and area / sig_tot >= 0.015:
            out.append((float(x[j]), float(wj), area / sig_tot))
    return sorted(out, key=lambda t: -t[2])


def bic(s, r):
    x = s.be[s.region()]
    n = len(x)
    k = max(1, r.get("nfree", 1))
    chi2 = max(r["chi2"], 1e-9)
    return n * math.log(chi2 / n) + k * math.log(n)


# ------------------------------------------------------------------ 一键拟合
def run(s, sample_elements=(), ref=None, max_add=4):
    report = []
    log = report.append
    if s.window:
        s.bg["lo"], s.bg["hi"] = s.window[0] + s.shift, s.window[1] + s.shift
    s.bg.pop("ylo", None); s.bg.pop("yhi", None)
    # 文件名与谱图是否一致
    try:
        from . import identify
        cands = identify.identify_region(s.be_raw, s.y)
        if cands:
            if not s.element:
                s.element = cands[0][0]
                log("轨道：文件名中没有轨道信息，根据谱图识别为 %s（%s）" % (s.element, cands[0][2]))
            elif cands[0][0] != s.element and cands[0][1] - next((c[1] for c in cands if c[0] == s.element), -9) > 1.0:
                log("⚠ 文件名为 %s，但谱图更像 %s（%s）；如果文件名有误，请重命名后再拟合" % (s.element, cands[0][0], cands[0][2]))
    except Exception:
        pass
    noise = choose_background(s, log)
    peaks = recipe(s, log)
    if peaks is None:
        peaks = generic_model(s, log, ref)
    else:
        # 未校正时整体偏移：以金属组分（X0）为参照——在其参考位置 ±1 eV 内找数据的局部峰顶
        from scipy.signal import find_peaks
        x, y, bg, _, _ = s.compute()
        sig = core.smooth(y - bg, 5)
        k, pr = find_peaks(sig, prominence=0.05 * float(np.ptp(sig) or 1))
        best = None
        for p in peaks:
            if p.get("pair") or not re.match(r"^[A-Z][a-z]?0\b", p["name"]):
                continue
            c0 = p["center"]["value"]
            near = [j for j in k if abs(x[j] - c0) <= 1.0]
            if near:
                j = max(near, key=lambda j: sig[j])
                if best is None or sig[j] > best[1]:
                    best = (float(x[j]) - c0, float(sig[j]), float(x[j]), p["name"])
        if best and 0.15 < abs(best[0]) <= 1.0:
            d = best[0]
            for p in peaks:
                for f in ("value", "min", "max"):
                    if p["center"][f] is not None:
                        p["center"][f] += d
            log("能量：%s 的峰顶在 %.2f eV，与参考值相差 %+.2f eV，模型已整体平移；建议做能量校正（如金属峰或 C1s）"
                % (re.sub(r"\s*\d[spdf]\d/2$", "", best[3]), best[2], d))
    if not peaks:
        log("没有找到峰，停止")
        return report
    s.peaks = []
    core.apply_template(s, peaks)
    for p, q in zip(s.peaks, peaks):        # 保留模型自己的峰位范围（apply_template 只在没有范围时加 ±1 eV）
        p["center"]["min"], p["center"]["max"] = q["center"]["min"], q["center"]["max"]
    prior_overlaps(s, sample_elements, log)
    r = staged_fit(s)
    b0 = bic(s, r)
    log("初次拟合：约化 χ² %.3g（先固定金属拖尾定位各组分，再放开拖尾）" % r["rchi2"])
    # 候选组分：与去掉它们的模型比较 BIC
    for tag in sorted({p.get("optional") for p in s.peaks if p.get("optional")}):
        import copy as _c
        keep = _c.deepcopy(s.peaks)
        s.peaks = [p for p in s.peaks if p.get("optional") != tag]
        r_wo = staged_fit(s)
        b_wo = bic(s, r_wo)
        if b_wo <= b0 - 2:
            b0, r = b_wo, r_wo
            log("候选 %s：去掉后拟合同样好或更好（BIC），不保留" % tag)
        else:
            s.peaks = keep
            r = s.do_fit()
            pinned = _pinned([p for p in s.peaks if p.get("optional") == tag])
            if pinned >= 2:
                s.peaks = [p for p in s.peaks if p.get("optional") != tag]
                r = staged_fit(s); b0 = bic(s, r)
                log("候选 %s：BIC 虽更好，但 %d 个参数卡在约束边界（在填补背景/其他信号，不像真实的 %s）→ 不保留" % (tag, pinned, tag))
            else:
                for p in s.peaks:
                    p.pop("optional", None)
                log("候选 %s：保留后拟合明显更好（BIC），参数都在合理范围内 → 保留" % tag)
    b0, r = test_known_states(s, b0, r, log)
    # 预置的重叠峰若拟合后面积很小（<1%），去掉
    tot = sum(p["area"]["value"] for p in s.peaks) or 1.0
    small = [p for p in s.peaks if p.get("prior") and p["area"]["value"] / tot < 0.01]
    if small:
        names_ = {p["name"] for p in small}
        s.peaks = [p for p in s.peaks if p["name"] not in names_ and (p.get("link") or {}).get("ref") not in names_]
        r = s.do_fit(); b0 = bic(s, r)
        log("预置重叠峰 %s 拟合后面积 <1%%，去掉" % "、".join(sorted(names_)))
    added = 0
    tried = set()
    while added < max_add:
        bumps = [b for b in residual_bumps(s, noise) if round(b[0], 0) not in tried]
        if not bumps:
            break
        c, w, frac = bumps[0]
        tried.add(round(c, 0))
        kind, name, refname, why = classify(c, w, s, sample_elements, ref, log)
        trial = copy.deepcopy(s.peaks)
        wv = min(max(w, 1.0), 8.0 if kind == "satellite" else 3.0)
        if kind == "satellite":
            p = _SAT(name, refname, c, wv, c - 1.5, c + 1.5)
        elif kind in ("overlap", "unknown"):
            auger = "俄歇" in name                 # 俄歇峰很宽：允许 FWHM 到 8 eV，尽量一个峰描述
            p = _P(name, c, max(wv, 3.0) if auger else wv, role="unknown", lo=c - (2.0 if auger else 1.0),
                   hi=c + (2.0 if auger else 1.0), wmax=8.0 if auger else 4.0)
        else:
            l_ = (s.element or "x")[-1]
            split = SPLIT.get(s.element)
            if split and l_ in RATIO:
                pp = _D(name, c, split, RATIO[l_], wv, sub=s.element[-2:], lo=c - 0.6, hi=c + 0.6)
                s.peaks = s.peaks + pp
                p = None
            else:
                p = _P(name, c, wv, lo=c - 0.6, hi=c + 0.6)
        if p is not None:
            s.peaks = s.peaks + [p]
        r2 = staged_fit(s)
        b2 = bic(s, r2)
        if b2 < b0 - 6:                     # BIC 显著下降才保留
            b0, r = b2, r2
            added += 1
            log("残差峰 %.2f eV（占 %.1f%%）：%s" % (c, 100 * frac, why))
        else:
            s.peaks = trial
            r = s.do_fit()
            log("残差峰 %.2f eV（占 %.1f%%）：加入后拟合没有显著改善（BIC），不保留" % (c, 100 * frac))
    if s.element == "Co2p" and any(p["name"].startswith("Co2+") for p in s.peaks):
        r = co_alternative(s, r, log)
    r_ms = multistart(s, 6)                   # 最后用多起点避开局部最优
    if r_ms["rchi2"] < r["rchi2"] * 0.999:
        log("多起点拟合：约化 χ² %.3g → %.3g" % (r["rchi2"], r_ms["rchi2"]))
    r = r_ms
    q = quality(s)
    for w_ in q["warnings"]:
        log("⚠ " + w_)
    log("完成：约化 χ² %.3g，残差 RMS / 峰高 %.1f%%" % (r["rchi2"], 100 * q["rel_rms"]))
    return report


# ------------------------------------------------------------------ 微调与质量检查
def quality(s):
    x, y, bg, comps, env = s.compute()
    ys = core.smooth(y, 5)
    noise = float(np.std(y - ys)) or 1.0
    over = float(np.max(bg - ys))
    warns = []
    if over > 0.5 * noise:
        warns.append("背景在某处高于数据 %.0f（噪声 σ≈%.0f）" % (over, noise))
    for p in s.peaks:
        for k in ("center", "fwhm"):
            par = p[k]
            if k in core.linked_params(p) or par["fixed"] or par["min"] is None or par["max"] is None:
                continue
            if min(abs(par["value"] - par["min"]), abs(par["value"] - par["max"])) < 1e-3:
                warns.append("%s 的%s卡在边界（%.2f）" % (p["name"], "峰位" if k == "center" else " FWHM ", par["value"]))
        ln = p.get("link") or {}
        if p.get("pair") and ln.get("area_ratio") is not None:
            el = (s.element or "x")[-1]
            th = RATIO.get(el)
            if th and abs(ln["area_ratio"] - th) / th > 0.15:
                warns.append("%s 面积比 %.2f 偏离理论值 %.2f 超过 15%%（背景或重叠可能有问题）" % (p["name"], ln["area_ratio"], th))
    rel = float(np.sqrt(np.mean((y - bg - env) ** 2)) / max(float(np.max(y - bg)), 1.0)) if s.peaks else 0.0
    return {"warnings": warns, "bg_over": over, "noise": noise, "rel_rms": rel}


def refine(s, window=0.3):
    """局部微调：峰位限制在当前值 ±window 内重拟合（不改变用户设定的其他约束）。"""
    saved = []
    for p in s.peaks:
        c = p["center"]
        saved.append((c["min"], c["max"]))
        if not c["fixed"] and "center" not in core.linked_params(p):
            lo = c["value"] - window if c["min"] is None else max(c["min"], c["value"] - window)
            hi = c["value"] + window if c["max"] is None else min(c["max"], c["value"] + window)
            c["min"], c["max"] = lo, hi
    r = s.do_fit()
    for p, (a, b) in zip(s.peaks, saved):
        p["center"]["min"], p["center"]["max"] = a, b
    return r


def multistart(s, n=12, seed=0):
    """多起点：随机扰动峰位（±0.4 eV，限制在范围内）与 FWHM（×0.75–1.3），取约化 χ² 最小的结果。"""
    rng = np.random.default_rng(seed)
    best = copy.deepcopy(s.peaks)
    r_best = s.do_fit()
    best = copy.deepcopy(s.peaks)
    base = copy.deepcopy(s.peaks)
    for _ in range(n):
        trial = copy.deepcopy(base)
        for p in trial:
            lk = core.linked_params(p)
            c, w = p["center"], p["fwhm"]
            if not c["fixed"] and "center" not in lk:
                v = c["value"] + rng.uniform(-0.4, 0.4)
                if c["min"] is not None: v = max(v, c["min"])
                if c["max"] is not None: v = min(v, c["max"])
                c["value"] = v
            if not w["fixed"] and "fwhm" not in lk:
                v = w["value"] * rng.uniform(0.75, 1.3)
                if w["min"] is not None: v = max(v, w["min"])
                if w["max"] is not None: v = min(v, w["max"])
                w["value"] = v
        s.peaks = trial
        try:
            r = s.do_fit()
        except Exception:
            continue
        if r["rchi2"] < r_best["rchi2"] * (1 - 1e-6):
            r_best, best = r, copy.deepcopy(s.peaks)
    s.peaks = best
    s.last_fit = r_best
    return r_best


def uncertainties(s):
    """由雅可比矩阵估计各自由参数的 1σ 不确定度，写入 p[k]['err']。"""
    from scipy.optimize import least_squares  # noqa: F401
    x, y, bg, _, _ = s.compute()
    sdat = y - bg
    w = 1.0 / np.sqrt(np.clip(y, 1.0, None))
    layout, x0 = [], []
    for i, p in enumerate(s.peaks):
        lk = core.linked_params(p)
        shp = core.resolve(s.peaks)[i]["shape"]
        for k in core.PARAMS:
            if p[k]["fixed"] or k in lk or (k in core.SHAPE_PARAMS and shp != "Asym"):
                continue
            layout.append((i, k)); x0.append(p[k]["value"])
    if not layout:
        return
    x0 = np.array(x0, float)

    def resid(v):
        _, env = core.model(x, s.peaks, core.resolve(s.peaks, v, layout))
        return (sdat - env) * w
    r0 = resid(x0)
    J = np.empty((len(r0), len(x0)))
    for j in range(len(x0)):
        h = 1e-4 * max(abs(x0[j]), 1e-2)
        v = x0.copy(); v[j] += h
        J[:, j] = (resid(v) - r0) / h
    dof = max(1, len(r0) - len(x0))
    s2 = float(np.sum(r0 ** 2)) / dof
    try:
        cov = np.linalg.pinv(J.T @ J) * s2
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:
        return
    for p in s.peaks:
        for k in core.PARAMS:
            p[k].pop("err", None)
    for (i, k), e in zip(layout, err):
        s.peaks[i][k]["err"] = float(e)
