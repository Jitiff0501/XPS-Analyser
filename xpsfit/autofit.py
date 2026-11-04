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
    "Ni2p": [("NiO", 853.5, 854.3), ("Ni(OH)2", 855.2, 856.2)],
    "Mn2p": [("Mn2+", 640.6, 641.4), ("Mn4+", 642.0, 642.6)],
    "C1s": [("CO3", 289.5, 290.5)],
    "N1s": [("nitride", 396.4, 397.4), ("C-N", 399.0, 400.5)],
    "O1s": [("O lattice", 529.3, 530.6), ("OH / defect", 531.0, 532.0)],
}
# 必须伴随卫星峰的化学态：卫星区间（eV）内没有信号就不能加入（如 Cu²⁺：940–946 eV）
REQUIRES_SAT = {"Cu2+": (940.0, 946.0), "Ni2+": (859.0, 864.0)}
# 哪些化学态可以带卫星峰（正则匹配主峰名）；未列出的元素：任何主峰都可以
SAT_PARENT = {"Cu": r"2\+|CuO|Cu\(OH\)", "C": r"C-C|graph|sp2", "Fe": r"Fe[23]\+|FeO|Fe2O3|Fe3O4|FeOOH"}


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
       fwr=(1.0, 1.3), wmin=1.0, wmax=3.0, lor=30, orb=None):
    """自旋轨道双峰。给出 orb（如 "Fe2p"）且该轨道在 core.SPIN_ORBIT 中时，ΔE / 面积比 / FWHM 比一律用文献范围
    （金属态与化合物分开；分不开的小分裂锁定），忽略 split / ratio / *_rng / fwr 参数。"""
    rule = core.so_rule(orb, core.is_metal_name(name) or shape == "Asym") if orb else None
    if rule:
        sub = orb[-2:]
        split, ratio, fwr = rule["dE"], rule["ratio"], rule["fwr_rng"]
        split_rng = None if rule["lock"] else rule["dE_rng"]
        rr = rule["ratio_rng"]
        ratio_rng = None if rule["lock"] or rr[1] <= rr[0] else rr
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


def _SATR(name, ref, c_ref, dE, w, dE_rng, ratio, wmin=2.0, wmax=6.0):
    """位置跟随主峰的卫星峰：ΔE 与面积比都在给定范围内拟合（如 Fe²⁺ +6 eV、Fe³⁺ +8 eV）。"""
    p = _P(name, c_ref + dE, w, 0, role="satellite", wmin=wmin, wmax=wmax)
    p["link"] = {"ref": ref, "dE": dE, "dE_fit": True, "dE_min": dE_rng[0], "dE_max": dE_rng[1],
                 "area_ratio": (ratio[0] + ratio[1]) / 2, "area_ratio_fit": True,
                 "area_ratio_min": ratio[0], "area_ratio_max": ratio[1]}
    return p


# Fe 氧化物按“物相”建模：三种物相的 Fe2p 包络位置与卫星特征不同（Biesinger 2011；Yamashita & Hayes 2008；
# Grosvenor 2004；McIntyre & Zetaruk 1977）。
# 每个物相的 2p3/2 多重分裂包络用两条关联的 GL 线近似（主线 + 高 BE 多重分裂线：间距 1.0–2.6 eV、
# 面积比 0.25–1.2、宽度比 1.0–1.8），低 BE 边陡、高 BE 侧宽；2p1/2 受 Coster–Kronig 展宽，用一个宽峰。
# 卫星：FeO（+6 eV）与 Fe₂O₃（+8 eV）是清楚的独立峰；Fe₃O₄ 的 Fe²⁺/Fe³⁺ 卫星弱且连成一片平台（宽而弱）。
# 名称: (主线峰位, 范围, 卫星 ΔE（相对主线）, ΔE 范围, 卫星/主线面积比范围, 卫星 FWHM 范围, Fe²⁺ 占比)
FE_PHASES = {
    "FeO":   (709.3, (708.7, 709.9), 6.2, (5.4, 7.0), (0.2, 1.0), (2.0, 4.5), 1.0),
    "Fe3O4": (710.1, (709.6, 710.6), 7.0, (5.5, 8.8), (0.0, 0.2), (3.0, 6.5), 1.0 / 3),
    "Fe2O3": (710.6, (710.1, 711.3), 8.4, (7.6, 9.2), (0.12, 0.7), (2.0, 4.5), 0.0),
}
FE_COMBOS = [("Fe2O3",), ("Fe3O4",), ("FeO",), ("Fe3O4", "Fe2O3"), ("FeO", "Fe3O4")]


def fe_phase_peaks(name):
    c, (lo, hi), dsat, drng, rr, (swl, swh), _ = FE_PHASES[name]
    main = _P("%s 2p3/2" % name, c, 1.8, lo=lo, hi=hi, wmin=1.2, wmax=2.4)
    mult = _P("%s 2p3/2 m" % name, c + 1.6, 2.2, wmin=1.2, wmax=3.4)
    mult["link"] = {"ref": main["name"], "dE": 1.6, "dE_fit": True, "dE_min": 1.0, "dE_max": 2.2,
                    "area_ratio": 0.55, "area_ratio_fit": True, "area_ratio_min": 0.3, "area_ratio_max": 0.8,
                    "fwhm_ratio": 1.2, "fwhm_ratio_fit": True, "fwhm_ratio_min": 1.0, "fwhm_ratio_max": 1.4, "lor_same": True}
    # 2p1/2：ΔE 与面积比相对整个 2p3/2 包络（主线 + 多重分裂线的重心 / 总面积），直接使用文献范围（core.SPIN_ORBIT）；
    # 宽度相对主线（1/2 分量是一个宽峰，对应整个 2p3/2 包络的宽度再加 Coster–Kronig 展宽）
    rule = core.so_rule("Fe2p", metal=False)
    half = _P("%s 2p1/2" % name, c + 0.6 + rule["dE"], 3.2, wmin=1.6, wmax=5.0)
    half["link"] = {"ref": main["name"], "with": [mult["name"]], "dE": rule["dE"], "dE_fit": True,
                    "dE_min": rule["dE_rng"][0], "dE_max": rule["dE_rng"][1],
                    "area_ratio": rule["ratio"], "area_ratio_fit": True,
                    "area_ratio_min": rule["ratio_rng"][0], "area_ratio_max": rule["ratio_rng"][1],
                    "fwhm_ratio": 1.8, "fwhm_ratio_fit": True, "fwhm_ratio_min": 1.2, "fwhm_ratio_max": 2.5, "lor_same": True}
    mult["pair"] = half["pair"] = True
    s1 = _SATR("%s sat. 2p3/2" % name, main["name"], c, dsat, (swl + swh) / 2, drng, rr, wmin=swl, wmax=swh)
    s2 = _P("%s sat. 2p1/2" % name, c + 13.6 + dsat, (swl + swh) / 2, 0, role="satellite", wmin=swl, wmax=swh)
    s2["link"] = {"ref": s1["name"], "dE": 13.6, "dE_fit": True, "dE_min": 13.3, "dE_max": 13.9,
                  "area_ratio": 0.5, "fwhm_ratio": 1.0}
    g = [main, mult, half, s1, s2]
    for p in g:
        p["fe_phase"] = name
    return g


def fe_model(metal=True, phases=("Fe3O4",)):
    m = []
    if metal:
        m += _D("Fe0", 706.7, None, None, 1.0, shape="Asym", lo=706.3, hi=707.2, wmin=0.7, wmax=1.8, orb="Fe2p")
    for ph in phases:
        m += fe_phase_peaks(ph)
    return m


def _seed(s, peaks):
    """新加入的峰：按数据高度估算初始面积（与套用模板相同）。"""
    x, y, bg, _, _ = s.compute()
    sig = y - bg
    out = copy.deepcopy(peaks)
    for p in out:
        core.ensure_peak(p)
        if "area" in core.linked_params(p):
            continue
        c, w = p["center"]["value"], p["fwhm"]["value"]
        h = float(np.interp(c, x, sig)) if x[0] <= c <= x[-1] else 0.0
        p["area"]["value"] = max(h, 0.05 * max(sig.max(), 1)) * w * 1.064
    return out


def fe_select(s, log):
    """Fe 物相选择。① 三个单相模型按 BIC 选主物相（默认显示）；② 两相模型（Fe₃O₄+Fe₂O₃、FeO+Fe₃O₄）
    因包络形状是近似，几乎总能拟合得更好，不能单凭拟合优度确认混合——只用来给出 Fe²⁺ 比例的范围。"""
    base = [p for p in s.peaks if not p.get("fe_phase")]
    bg0 = copy.deepcopy(s.bg)                 # 主动 Shirley 的斜坡也是拟合参数：每个模型都从斜坡 0 开始，并各自保存
    if bg0.get("type") == "SlopeShirley":
        bg0["tilt"] = 0.0
    res = []
    for combo in FE_COMBOS:
        start = copy.deepcopy(base) + _seed(s, [q for ph in combo for q in fe_phase_peaks(ph)])
        # 主动 Shirley 的斜坡与卫星峰耦合，存在多个局部最优，结果取决于优化路径。单相模型用两条路径
        # （① 直接完整拟合；② 先限迭代粗拟合再完整收敛）各拟合一次，取 BIC 更好的；两相模型只用于给范围，限迭代
        paths = (lambda: staged_fit(s), lambda: (staged_fit(s, max_nfev=400), s.do_fit())[1]) if len(combo) == 1 \
            else (lambda: staged_fit(s, max_nfev=400),)
        best_t = None
        for run_path in paths:
            s.bg, s.peaks = copy.deepcopy(bg0), copy.deepcopy(start)
            r = run_path()
            b_ = bic(s, r)
            if best_t is None or b_ < best_t["bic"]:
                d = fe_phases(s)
                best_t = {"bic": b_, "combo": combo, "peaks": copy.deepcopy(s.peaks), "bg": copy.deepcopy(s.bg),
                          "x2": d["x2"], "phases": d["phases"]}
        res.append(best_t)
    singles = sorted((t for t in res if len(t["combo"]) == 1), key=lambda t: t["bic"])
    best = singles[0]
    s.peaks, s.bg = best["peaks"], best["bg"]
    r = s.do_fit()
    b = bic(s, r)
    x2 = fe_phases(s)["x2"]
    log("Fe 单相模型（BIC，越小越好）：%s → 主物相 %s（Fe²⁺ 占氧化 Fe 的 %.0f%%）"
        % ("；".join("%s %.0f" % (t["combo"][0], t["bic"]) for t in singles), best["combo"][0], 100 * x2))
    alts = [t for t in res if len(t["combo"]) == 2 and best["combo"][0] in t["combo"]
            and min(t["phases"].get(k, 0) for k in t["combo"]) >= 0.1]
    s.fe_alt = None
    if alts:
        a = min(alts, key=lambda t: t["bic"])
        lo, hi = sorted((x2, a["x2"]))
        s.fe_alt = {"default": best["combo"], "combo": a["combo"], "x2": a["x2"], "range": (lo, hi),
                    "phases": a["phases"]}
        log("两相模型 %s（%s）：Fe²⁺ 占 %.0f%%。两相模型多出一整套峰，几乎总能拟合得更好，不能据此确认混合"
            " → 默认显示单相；Fe²⁺ 比例报告为 %.0f–%.0f%%"
            % (" + ".join(a["combo"]), " + ".join("%s %.0f%%" % (k, 100 * a["phases"][k]) for k in a["combo"]),
               100 * a["x2"], 100 * lo, 100 * hi))
    return b, r


def fe_phases(s):
    """各 Fe 物相 / 氧化态在氧化 Fe 中的占比，推算 Fe²⁺ 占比 x2、金属 Fe⁰ 占比 f0。"""
    st = core.chemical_states(s)
    tot = sum(a for _, a in st) or 1.0
    f0 = sum(a for n, a in st if n.startswith("Fe0")) / tot
    share = {}
    for n, a in st:
        for k in ("Fe3O4", "Fe2O3", "FeO", "Fe2+", "Fe3+"):   # Fe3O4 / Fe2O3 要先于 FeO 匹配
            if n.startswith(k):
                share[k] = share.get(k, 0.0) + a
                break
    ox = sum(share.values())
    if ox <= 0:
        return {"phases": {}, "x2": None, "f0": f0, "ox": 0.0}
    fe2 = {"FeO": 1.0, "Fe3O4": 1.0 / 3, "Fe2O3": 0.0, "Fe2+": 1.0, "Fe3+": 0.0}
    x2 = sum(a * fe2[k] for k, a in share.items()) / ox
    return {"phases": {k: a / ox for k, a in share.items()}, "x2": x2, "f0": f0, "ox": ox / tot}


PHASE_LABEL = {"FeO": "FeO（Fe₁₋ₓO）", "Fe3O4": "Fe₃O₄", "Fe2O3": "Fe₂O₃（或 FeOOH）", "Fe2+": "Fe²⁺", "Fe3+": "Fe³⁺"}


def interpret(s):
    """按拟合结果给出化合物解读（目前用于 Fe2p：FeO / Fe3O4 / Fe2O3·FeOOH）。返回文字或 None。"""
    if s.element != "Fe2p" or not s.peaks:
        return None
    d = fe_phases(s)
    if d["x2"] is None:
        return "Fe：只有金属 Fe⁰" if d["f0"] > 0 else None
    ph = sorted(d["phases"].items(), key=lambda kv: -kv[1])
    x2 = d["x2"]
    out = "物相 %s；推算 Fe²⁺:Fe³⁺ = %s（Fe²⁺ 占氧化 Fe 的 %.0f%%）" % (
        " + ".join("%s %.0f%%" % (PHASE_LABEL[k], 100 * v) for k, v in ph),
        "1:%.1f" % ((1 - x2) / x2) if x2 > 0.02 else "0:1", 100 * x2)
    if any(k in d["phases"] for k in ("Fe2+", "Fe3+")):     # 手动按氧化态拟合时，按比例给出参考物相
        what = ("Fe₂O₃ 或 FeOOH" if x2 < 0.12 else "Fe₂O₃ + Fe₃O₄" if x2 < 0.22 else "Fe₃O₄" if x2 <= 0.45
                else "Fe₃O₄ + FeO" if x2 < 0.75 else "FeO")
        out += " → 与 %s 一致" % what
    alt = getattr(s, "fe_alt", None)
    if alt and tuple(sorted(d["phases"])) == tuple(sorted(alt["default"])):
        out += "；若为 %s 混合，Fe²⁺ 约 %.0f%%（报告范围 %.0f–%.0f%%）" % (
            " + ".join(PHASE_LABEL[k].split("（")[0] for k in alt["combo"]), 100 * alt["x2"],
            100 * alt["range"][0], 100 * alt["range"][1])
    if "Fe2O3" in d["phases"]:
        out += "；Fe₂O₃ 与 FeOOH 的 Fe2p 几乎相同，请看 O1s 中 ~531.2 eV 的 OH 组分"
    if d["f0"] > 0.02:
        out += "；另有金属 Fe⁰ %.0f%%（占 Fe2p 总面积）" % (100 * d["f0"])
    return out


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
        return _D("Al0", 72.6, None, None, 0.8, shape="Asym", lo=71.8, hi=73.0, wmin=0.6, wmax=1.6, orb="Al2p") + \
            _D("Al3+", 74.6, None, None, 1.6, lo=73.8, hi=75.6, wmax=2.0, orb="Al2p")
    if el == "Co2p":
        log("Co2p：Co⁰ 不对称双峰 Δ14.9–15.1；Co²⁺ 双峰 Δ15.2–15.7 + 卫星 ~786 / ~802 eV")
        return (_D("Co0", 778.2, None, None, 1.0, shape="Asym", lo=777.9, hi=778.5, orb="Co2p")
                + _D("Co2+", 780.6, None, None, 2.6, lo=779.9, hi=781.5, orb="Co2p")
                + [_SAT("Co2+ sat. 2p3/2", "Co2+ 2p3/2", 786.0, 5.0, 783.5, 788.5, wmax=6.0, ratio=(0.3, 0.9)),
                   _SAT("Co2+ sat. 2p1/2", "Co2+ 2p1/2", 802.3, 5.0, 800.0, 804.5, wmax=6.0, ratio=(0.3, 0.9))])
    if el == "Cr2p":
        log("Cr2p：Cr⁰ 不对称双峰 Δ9.2–9.4；Cr³⁺ 双峰 Δ9.6–9.9（多重分裂以宽峰近似）")
        return (_D("Cr0", 574.2, None, None, 1.2, shape="Asym", lo=573.6, hi=574.7, orb="Cr2p")
                + _D("Cr3+", 576.4, None, None, 2.6, lo=575.6, hi=577.3, orb="Cr2p"))
    if el == "Cu2p":
        m = _D("Cu0/Cu+", 932.6, None, None, 1.2, shape="Asym", lo=932.3, hi=932.9, orb="Cu2p")
        f = _signal(s, 940.0, 946.0)
        if f > 0.02:
            cu2 = _D("Cu2+", 933.8, None, None, 2.5, lo=933.4, hi=934.4, wmax=3.5, orb="Cu2p") + [_SAT("Cu2+ sat.", "Cu2+ 2p3/2", 942.5, 4.5, 940.0, 945.5, wmax=7.0, ratio=(0.4, 0.75))]
            for p in cu2:
                p["optional"] = "Cu2+"
            log("Cu2p：940–946 eV 有卫星信号（%.1f%%）→ Cu²⁺ 及其卫星峰作为候选，拟合后用 BIC 判断是否保留" % (100 * f))
            m += cu2
        else:
            log("Cu2p：940–946 eV 无卫星信号（%.1f%%）→ 不加 Cu²⁺，只用 Cu⁰/Cu⁺" % (100 * f))
        return m
    if el == "Fe2p":
        f0 = _signal(s, 705.8, 707.6)
        fox = _signal(s, 708.6, 713.0)
        metal = f0 > 0.03
        if fox < 0.03:
            log("Fe2p：708.6–713 eV 几乎没有氧化态信号（%.1f%%）→ 只用金属 Fe⁰（不对称双峰 Δ12.95–13.3）" % (100 * fox))
            return fe_model(metal=True, phases=())
        log("Fe2p：%s氧化物按物相建模——FeO ~709.6（卫星 +6 eV）、Fe₃O₄ ~710.5（几乎无卫星）、Fe₂O₃/FeOOH ~711.0"
            "（卫星 +8 eV）；2p3/2 多重分裂包络用两条关联峰近似，2p1/2 用一个宽峰"
            % ("Fe⁰ 不对称双峰 Δ12.95–13.3；" if metal else "705.8–707.6 eV 无明显金属峰，不加 Fe⁰；"))
        return fe_model(metal=metal)
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
            peaks += _D(name, c, split, RATIO[l_], 1.5, sub=s.element[-2:], lo=c - 0.8, hi=c + 0.8, orb=s.element)
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
        mid = (a + b) / 2
        seg = [(a, b, ma, mb)] if b - a < 10 else [(a, mid, ma, 0.0), (mid, b, 0.0, mb)]
        for lo, hi, mlo, mhi in seg:
            m = (x >= lo + mlo) & (x <= hi - mhi)
            if m.sum() < 3:
                continue
            j = np.where(m)[0][np.argmin(ys[m])]
            # 长空当被一分为二时，最低点若落在分割处，说明那里是斜坡（如卫星峰的侧坡）而不是谷底 → 不取
            if len(seg) == 2 and abs(float(x[j]) - mid) < 3 * step:
                continue
            pts.append([float(x[j]), float(ys[j])])
    for j in (0, len(x) - 1):            # 两端
        pts.append([float(x[j]), float(ys[j])])
    pts.sort()
    out = []
    for p in pts:
        if not out or p[0] - out[-1][0] > 0.6:
            out.append(p)
    return out


def _el(s):
    m = re.match(r"[A-Z][a-z]?", s.element or "")
    return m.group(0) if m else ""


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
    elif s.element == "Fe2p":
        # 主峰之间是卫星结构，不能用点背景，也不宜让 Smart 贴着数据走（会吃掉卫星）。
        # 本底在 Shirley 台阶之外常有线性斜坡 → 主动 Shirley：斜坡改正量随峰一起拟合，且背景不高于数据
        s.bg.update({"type": "SlopeShirley", "tilt": 0.0})
        log("背景：%sFe2p 主峰之间是卫星结构 → 主动 Shirley（Shirley 台阶 + 线性斜坡，斜坡随峰一起拟合，背景不高于数据）"
            % ("Shirley 会高于数据 %.0f（%.0fσ）；" % (over, over / noise) if over > noise else ""))
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


def staged_fit(s, max_nfev=4000):
    """两步拟合：① 不对称峰的拖尾固定在典型值（T=0.25, τ=1.2 eV），先让各组分定位；② 再放开拖尾（在常见范围内）。"""
    asym = [p for p in s.peaks if p.get("shape") == "Asym" and "tail" not in core.linked_params(p)]
    saved = [(p["tail"]["fixed"], p["tau"]["fixed"]) for p in asym]
    for p in asym:
        p["tail"]["value"], p["tau"]["value"] = p.get("tail0", 0.25), p.get("tau0", 1.2)
        p["tail"]["fixed"] = p["tau"]["fixed"] = True
    s.do_fit(max_nfev)
    for p, (a, b) in zip(asym, saved):
        p["tail"]["fixed"], p["tau"]["fixed"] = a, b
    return s.do_fit(max_nfev)


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
            new = _D(name, c, split, RATIO[l_], 1.8, sub=s.element[-2:], lo=a - 0.2, hi=b + 0.2, orb=s.element)
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
    s.relax_level, s.relax_hist, s.relax_last = 0, [], None
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
        if not any(p.get("optional") == tag for p in s.peaks):
            continue
        keep, keep_bg = _c.deepcopy(s.peaks), _c.deepcopy(s.bg)
        if not any(p.get("role", "main") == "main" and p.get("optional") != tag for p in s.peaks):
            continue                          # 去掉后没有主峰了：不测试
        s.peaks = [p for p in s.peaks if p.get("optional") != tag]
        r_wo = staged_fit(s)
        b_wo = bic(s, r_wo)
        if b_wo <= b0 - 2:
            b0, r = b_wo, r_wo
            log("候选 %s：去掉后拟合同样好或更好（BIC），不保留" % tag)
        else:
            s.peaks, s.bg = keep, keep_bg
            r = s.do_fit()
            pinned = _pinned([p for p in s.peaks if p.get("optional") == tag])
            if pinned >= 2:
                s.peaks = [p for p in s.peaks if p.get("optional") != tag]
                r = staged_fit(s); b0 = bic(s, r)
                log("候选 %s：BIC 虽更好，但 %d 个参数卡在约束边界（在填补背景/其他信号，不像真实的 %s）→ 不保留" % (tag, pinned, tag))
            else:
                for p in s.peaks:
                    if p.get("optional") == tag:
                        p.pop("optional")
                log("候选 %s：保留后拟合明显更好（BIC），参数都在合理范围内 → 保留" % tag)
    if s.element == "Fe2p" and any(p.get("fe_phase") for p in s.peaks):
        b0, r = fe_select(s, log)
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
        trial, trial_bg = copy.deepcopy(s.peaks), copy.deepcopy(s.bg)
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
                pp = _D(name, c, split, RATIO[l_], wv, sub=s.element[-2:], lo=c - 0.6, hi=c + 0.6, orb=s.element)
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
            s.peaks, s.bg = trial, trial_bg
            r = s.do_fit()
            log("残差峰 %.2f eV（占 %.1f%%）：加入后拟合没有显著改善（BIC），不保留" % (c, 100 * frac))
    if s.element == "Co2p" and any(p["name"].startswith("Co2+") for p in s.peaks):
        r = co_alternative(s, r, log)
    r_ms = multistart(s, 6)                   # 最后用多起点避开局部最优
    if r_ms["rchi2"] < r["rchi2"] * 0.999:
        log("多起点拟合：约化 χ² %.3g → %.3g" % (r["rchi2"], r_ms["rchi2"]))
    r = r_ms
    q = quality(s)
    it = interpret(s)
    if it:
        log("解读：" + it)
    for w_ in q["warnings"]:
        log("⚠ " + w_)
    log("完成：约化 χ² %.3g，残差 RMS / 峰高 %.1f%%" % (r["rchi2"], 100 * q["rel_rms"]))
    return report


# ------------------------------------------------------------------ 逐步放宽约束
# 默认模型的约束（L% 固定、FWHM 1–3 eV、化合物对称、双峰按文献范围…）偏保守。拟合不理想时按风险从低到高逐级放开，
# 每放一级都重新拟合，并用 BIC（已计入多出的参数）、新出现的物理警告、化学态比例的变化来判断是否保留。
def _free_params(s):
    for p in s.peaks:
        yield p, core.linked_params(p)


def _rx_lor(s):
    n = 0
    for p, lk in _free_params(s):
        if p.get("role", "main") != "main" or "lor" in lk or not p["lor"]["fixed"]:
            continue
        p["lor"]["fixed"] = False
        p["lor"]["min"], p["lor"]["max"] = 0.0, max(60.0, p["lor"]["value"])
        n += 1
    return ["%d 个峰的 Lorentzian 比例（L%%）改为在 0–60%% 内拟合（双峰副峰跟随）" % n] if n else []


def _rx_bg(s):
    if s.bg.get("type") not in ("Shirley", "Smart"):
        return []
    old = s.bg["type"]
    s.bg.update({"type": "SlopeShirley", "tilt": 0.0})
    return ["背景 %s → 主动 Shirley（台阶 + 线性斜坡，斜坡随峰拟合，仍不高于数据）" % old]


def _rx_fwhm(s):
    n = 0
    for p, lk in _free_params(s):
        f = p["fwhm"]
        if "fwhm" not in lk and not f["fixed"] and f["max"] is not None:
            f["min"] = max(0.4, round(0.7 * (f["min"] if f["min"] is not None else 1.0), 2))
            f["max"] = round(1.4 * f["max"], 2)
            n += 1
    return ["%d 个峰的 FWHM 范围放宽为原来的 0.7–1.4 倍（双峰的 FWHM 倍率仍按文献范围）" % n] if n else []


def _rx_tail(s):
    n = 0
    for p, lk in _free_params(s):
        if p.get("shape") != "Asym" or "tail" in lk or p.get("fe_phase"):
            continue
        p["tail"]["max"] = max(p["tail"]["max"] or 0.45, 0.8)
        p["tau"]["max"] = max(p["tau"]["max"] or 2.0, 3.5)
        n += 1
    return ["%d 个金属态峰的拖尾范围放宽（T ≤ 0.8，τ ≤ 3.5 eV）" % n] if n else []


def _rx_doublet(s):
    n = 0
    for p in s.peaks:
        ln = p.get("link") or {}
        R = ln.get("ref", "")
        if not p.get("pair") or p["name"].startswith(R + " ") or p.get("role") == "unknown":
            continue
        rule = core.so_rule(s.element, core.is_metal_name(R))
        if rule and rule["lock"]:
            continue                          # 分裂远小于峰宽（Al2p、Si2p）：放开也拟合不出来
        if ln.get("fwhm_ratio_fit") and ln.get("fwhm_ratio_max") is not None:
            ln["fwhm_ratio_min"] = max(0.7, ln["fwhm_ratio_min"] - 0.1)
            ln["fwhm_ratio_max"] = ln["fwhm_ratio_max"] + 0.3
        for k, d, lo_cap in (("dE", 0.1, None), ("area_ratio", 0.05, 0.2)):
            if ln.get(k) is None:
                continue
            if ln.get(k + "_fit") and ln.get(k + "_min") is not None:
                ln[k + "_min"] -= d
                ln[k + "_max"] += d
            else:
                ln.update({k + "_fit": True, k + "_min": ln[k] - d, k + "_max": ln[k] + d})
            if lo_cap is not None:
                ln[k + "_min"] = max(lo_cap, ln[k + "_min"])
        n += 1
    return ["%d 对自旋轨道双峰：分裂范围各放宽 0.1 eV、面积比各放宽 0.05、FWHM 倍率下限 −0.1 / 上限 +0.3（可超出文献范围，超出时会提示）" % n] if n else []


def _rx_comp_asym(s):
    n = 0
    for p, lk in _free_params(s):
        if p.get("role", "main") != "main" or p.get("shape") == "Asym" or "tail" in lk or core.is_metal_name(p["name"]):
            continue
        p["shape"] = "Asym"
        p["tail"].update(value=0.05, min=0.0, max=0.25, fixed=False)
        p["tau"].update(value=1.0, min=0.5, max=2.0, fixed=False)
        n += 1
    return ["%d 个化合物峰允许轻微不对称（T ≤ 0.25）" % n] if n else []


def _rx_center(s):
    n = 0
    for p, lk in _free_params(s):
        c = p["center"]
        if "center" in lk or c["fixed"] or c["min"] is None or c["max"] is None:
            continue
        c["min"] -= 0.5
        c["max"] += 0.5
        n += 1
    return ["%d 个峰的峰位范围两侧各放宽 0.5 eV" % n] if n else []


RELAX_LEVELS = [
    {"key": "lor", "name": "Lorentzian 比例", "risk": "低", "apply": _rx_lor,
     "why": "仪器展宽（高斯）与寿命展宽（洛伦兹）的比例因谱仪和化学态而异，放开它物理上最没有争议"},
    {"key": "bg", "name": "背景斜坡", "risk": "低", "apply": _rx_bg,
     "why": "本底在 Shirley 台阶之外常有线性斜坡；随峰拟合，背景仍不高于数据"},
    {"key": "fwhm", "name": "FWHM 范围", "risk": "中", "apply": _rx_fwhm,
     "why": "默认 1–3 eV 适合多数单色 Al Kα 高分辨谱；窄的金属峰或宽的多重分裂峰可能需要更宽的范围"},
    {"key": "tail", "name": "金属拖尾", "risk": "中", "apply": _rx_tail,
     "why": "金属的拖尾长度因元素和导电电子而异；过长的拖尾会吃掉相邻氧化态"},
    {"key": "doublet", "name": "双峰关系超出文献范围", "risk": "高", "apply": _rx_doublet,
     "why": "1/2 分量常受背景、重叠和 Coster–Kronig 效应影响；超出文献范围时更可能是背景或重叠的问题，而不是物理"},
    {"key": "asym", "name": "化合物不对称", "risk": "高", "apply": _rx_comp_asym,
     "why": "多重分裂或未分辨的成分会让化合物峰看起来不对称；但拖尾很容易把相邻化学态的面积吃掉"},
    {"key": "center", "name": "峰位范围", "risk": "最高", "apply": _rx_center,
     "why": "峰位放开后，化学态可能互相交换位置；只在能量校正可靠时使用"},
]


def relax_state(s):
    lvl = getattr(s, "relax_level", 0)
    kept = {h["level"] for h in getattr(s, "relax_hist", [])}
    ladder = [{"name": L["name"], "risk": L["risk"], "why": L["why"],
               "status": "kept" if k + 1 in kept else ("skipped" if k < lvl else "pending")}
              for k, L in enumerate(RELAX_LEVELS)]
    return {"level": lvl, "total": len(RELAX_LEVELS), "ladder": ladder, "last": getattr(s, "relax_last", None)}


def _fractions(s):
    st = core.chemical_states(s)
    tot = sum(a for _, a in st) or 1.0
    return {n: a / tot for n, a in st}


def relax_next(s):
    """放开下一级约束并重新拟合，返回这一步的评价（不自动撤销；由用户或 relax_auto 决定）。"""
    if not s.peaks:
        raise ValueError("请先拟合（一键拟合或添加峰）")
    lvl = getattr(s, "relax_level", 0)
    s.relax_hist = getattr(s, "relax_hist", [])
    if not s.last_fit:
        s.do_fit()
    while lvl < len(RELAX_LEVELS):
        L = RELAX_LEVELS[lvl]
        r0 = s.last_fit
        b0, f0, q0 = bic(s, r0), _fractions(s), set(quality(s)["warnings"])
        snap = (copy.deepcopy(s.peaks), copy.deepcopy(s.bg), copy.deepcopy(s.last_fit))
        changes = L["apply"](s)
        lvl += 1
        s.relax_level = lvl
        if not changes:
            continue                          # 这一级不适用（如已经是自定义点背景），直接看下一级
        r = s.do_fit()
        r_ms = multistart(s, 3)
        r = r_ms if r_ms["rchi2"] <= r["rchi2"] else s.do_fit()
        b, f1, q1 = bic(s, r), _fractions(s), set(quality(s)["warnings"])
        db = b - b0
        new_w = sorted(q1 - q0)
        phys = [w for w in new_w if "不在文献范围" in w or "高于数据" in w]
        big = max((abs(f1.get(k, 0.0) - f0.get(k, 0.0)) for k in set(f0) | set(f1)), default=0.0)
        if db < -10 and not phys and big > 0.10:
            verdict, msg = "check", "BIC 下降 %.0f，但化学态比例变化达 %.0f 个百分点 → 结果对这项约束很敏感，请复核后决定" % (-db, 100 * big)
        elif db < -10 and not phys:
            verdict, msg = "keep", "BIC 下降 %.0f（显著改善），没有新的物理问题 → 建议保留" % -db
        elif db < -10:
            verdict, msg = "check", "BIC 下降 %.0f，但出现新的物理问题（见下）→ 请判断后决定是否撤销" % -db
        elif db < 0:
            verdict, msg = "marginal", "BIC 只下降 %.0f（不显著，多出的参数不值得）→ 建议撤销" % -db
        else:
            verdict, msg = "worse", "BIC 没有下降（%+.0f）→ 建议撤销" % db
        moved = [[k, 100 * f0.get(k, 0.0), 100 * f1.get(k, 0.0)] for k in sorted(set(f0) | set(f1))
                 if abs(f1.get(k, 0.0) - f0.get(k, 0.0)) >= 0.005]
        res = {"level": lvl, "name": L["name"], "risk": L["risk"], "changes": changes,
               "before": {"rchi2": r0["rchi2"], "bic": b0}, "after": {"rchi2": r["rchi2"], "bic": b},
               "dbic": db, "verdict": verdict, "msg": msg, "fractions": moved, "new_warnings": new_w}
        s.relax_hist.append({"level": lvl, "snap": snap})
        s.relax_last = res
        return res
    s.relax_last = {"done": True, "msg": "所有约束都已放开过（共 %d 级）" % len(RELAX_LEVELS)}
    return s.relax_last


def relax_undo(s):
    hist = getattr(s, "relax_hist", [])
    if not hist:
        raise ValueError("没有可撤销的放宽步骤")
    h = hist.pop()
    s.peaks, s.bg, s.last_fit = h["snap"]
    s.relax_level = h["level"] - 1
    s.relax_last = {"undone": True, "msg": "已撤销第 %d 级（%s）" % (h["level"], RELAX_LEVELS[h["level"] - 1]["name"])}
    return s.relax_last


AUTO_MAX = 4      # 自动模式只走低、中风险的前 4 级；高风险的需手动逐级放开


def relax_auto(s):
    """从当前级开始逐级尝试（只到第 AUTO_MAX 级）：BIC 显著下降、没有新的物理问题、比例变化不大才保留，否则撤回并跳过。"""
    steps = []
    while getattr(s, "relax_level", 0) < AUTO_MAX:
        res = relax_next(s)
        if res.get("done"):
            break
        if res["verdict"] == "keep":
            steps.append([res["level"], res["name"], "保留", res["msg"], res["fractions"]])
        else:
            relax_undo(s)
            s.relax_level = res["level"]          # 跳过这一级，继续下一级
            steps.append([res["level"], res["name"], "撤回", res["msg"], res["fractions"]])
    kept = [st for st in steps if st[2] == "保留"]
    s.relax_last = {"auto": True, "steps": steps,
                    "msg": "自动放宽完成（低、中风险的前 %d 级）：保留 %d 级%s；高风险的级别请手动逐级放开" % (
                        AUTO_MAX, len(kept), "（%s）" % "、".join(k[1] for k in kept) if kept else "，默认约束已经合适")}
    return s.relax_last


# ------------------------------------------------------------------ 微调与质量检查
def quality(s):
    x, y, bg, comps, env = s.compute()
    ys = core.smooth(y, 5)
    noise = float(np.std(y - ys)) or 1.0
    over = float(np.max(bg - ys))
    warns = []
    by = {p["name"]: p for p in s.peaks}
    idx = {p["name"]: k for k, p in enumerate(s.peaks)}
    rv = core.resolve(s.peaks)
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
        R = ln.get("ref", "")
        if not (p.get("pair") and R in by) or p["name"].startswith(R + " "):
            continue                          # 只检查自旋轨道副峰（多重分裂分量不查）
        orb = s.element if (s.element or "")[-2:] in ("2p", "3d", "4f", "3p") else None
        if p.get("role") == "unknown":       # 重叠峰按它自己的能级查（如 Cu3p）
            m_ = re.search(r"([A-Z][a-z]?\d[spdf])", R)
            orb = m_.group(1) if m_ else None
        rule = core.so_rule(orb, core.is_metal_name(R) or by[R].get("shape") == "Asym") if orb else None
        k12, kR = idx[p["name"]], idx[R]
        grp = [kR] + [idx[n] for n in (ln.get("with") or []) if n in idx]
        A = sum(rv[g]["area"] for g in grp)
        c3 = sum(rv[g]["center"] * rv[g]["area"] for g in grp) / A if A > 0 else rv[kR]["center"]
        dE, ratio = rv[k12]["center"] - c3, rv[k12]["area"] / max(A, 1e-9)
        fwr = None if len(grp) > 1 else rv[k12]["fwhm"] / max(rv[kR]["fwhm"], 1e-9)
        if rule:
            tol = 1e-3
            lo, hi = rule["dE_rng"]
            if not (lo - tol <= dE <= hi + tol):
                warns.append("%s 的自旋轨道分裂 %.2f eV 不在文献范围 %.2f–%.2f eV" % (p["name"], dE, lo, hi))
            lo, hi = rule["ratio_rng"]
            if not (lo - 0.02 <= ratio <= hi + 0.02):
                warns.append("%s 面积比 %.2f 不在文献范围 %.2f–%.2f（背景或重叠可能有问题）" % (p["name"], ratio, lo, hi))
            lo, hi = rule["fwr_rng"]
            if fwr is not None and not (lo - 0.02 <= fwr <= hi + 0.02):
                warns.append("%s 的 FWHM 倍率 %.2f 不在文献范围 %.2f–%.2f" % (p["name"], fwr, lo, hi))
            # 拟合把参数推到文献范围的边界：约束在起作用，说明 1/2 区域的背景或重叠可能有问题
            at = []
            for k, lab in (("dE", "分裂"), ("area_ratio", "面积比"), ("fwhm_ratio", "FWHM 倍率")):
                if ln.get(k + "_fit") and ln.get(k + "_min") is not None and ln.get(k + "_max") is not None \
                        and ln[k + "_max"] > ln[k + "_min"] and min(abs(ln[k] - ln[k + "_min"]), abs(ln[k] - ln[k + "_max"])) < 1e-3:
                    at.append("%s %.2f" % (lab, ln[k]))
            if at:
                warns.append("%s：%s 到达文献范围边界（约束在起作用，1/2 区域的背景或重叠值得检查）" % (p["name"], "；".join(at)))
        else:
            th = RATIO.get((orb or "x")[-1])
            if th and abs(ratio - th) / th > 0.15:
                warns.append("%s 面积比 %.2f 偏离理论值 %.2f 超过 15%%（背景或重叠可能有问题）" % (p["name"], ratio, th))
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
    r_best = s.do_fit()
    best, best_bg = copy.deepcopy(s.peaks), copy.deepcopy(s.bg)    # 主动 Shirley 的斜坡随峰一起保存
    base, base_bg = copy.deepcopy(s.peaks), copy.deepcopy(s.bg)
    for _ in range(n):
        trial = copy.deepcopy(base)
        s.bg = copy.deepcopy(base_bg)
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
            r_best, best, best_bg = r, copy.deepcopy(s.peaks), copy.deepcopy(s.bg)
    s.peaks, s.bg = best, best_bg
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
