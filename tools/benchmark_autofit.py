# -*- coding: utf-8 -*-
"""一键拟合基准测试：用已知答案的模拟谱反复运行 autofit，统计正确度。

    python tools/benchmark_autofit.py [每类次数=12] [输出.md]

每条模拟谱的背景 = 常数 + 斜率（含反向本底）+ 与各峰面积成正比的 Shirley 型台阶；泊松噪声。
评估：背景（是否高于数据、相对真实背景的误差）、化学态比例、双峰间距/面积比、
不对称拖尾（金属态）、残差峰归类（重叠 / 卫星 / 新化学态 / 未知）、全谱元素识别。
"""
import copy
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xpsfit import core, autofit, identify  # noqa: E402

HV = 1486.68


def shirley_true(x, signal, k):
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (signal[1:] + signal[:-1]) * np.diff(x))])
    return k * cum


def W(peaks, *ws):
    """真实谱的峰宽不受程序默认 FWHM 范围限制：按给定值强制设置（与 peaks 顺序对应，双峰两个分量同宽）。"""
    for p, w in zip(peaks, ws):
        p["fwhm"]["value"] = float(w)
    return peaks


def true_frac(peaks, prefix, exclude=("CrLMM", "Cu3p")):
    """真实比例：按 resolve 后的实际面积求和（双峰两个分量 + 该态的卫星），排除重叠峰。"""
    vals = core.resolve(peaks)
    tot = sum(v["area"] for p, v in zip(peaks, vals) if not p["name"].startswith(exclude))
    part = sum(v["area"] for p, v in zip(peaks, vals) if p["name"].startswith(prefix))
    return part / tot


def make(x, peaks, rng, base, slope, k_step, counts):
    """返回 (y, 真实背景, 真实信号)。counts 控制信噪比（主峰高度约为 counts）。"""
    _, sig = core.model(x, peaks)
    scale = counts / max(float(sig.max()), 1e-9)
    sig = sig * scale
    bg = base + slope * (x - x[0]) + shirley_true(x, sig, k_step)
    y = rng.poisson(np.clip(bg + sig, 1, None)).astype(float)
    return y, bg, sig, scale


def frac_of(s, prefix):
    st = core.chemical_states(s); tot = sum(a for _, a in st) or 1
    return sum(a for n, a in st if n.startswith(prefix)) / tot


def eval_bg(s, true_bg):
    x, y, bg, _, env = s.compute()
    tb = np.interp(x, s.be_raw + s.shift, true_bg)
    peak_h = float(np.max(env)) or 1.0
    over = float(np.max(bg - core.smooth(y, 5)))
    err = float(np.sqrt(np.mean((bg - tb) ** 2)) / peak_h)
    return over, err


# ------------------------------------------------------------------ 场景
def case_co(rng):
    """Co⁰ 不对称 + Co²⁺ + 卫星；Shirley 台阶；Co²⁺ 比例随机。"""
    x = np.arange(770, 810.01, 0.1)
    f2 = rng.uniform(0.1, 0.7)
    co0 = W(core.doublet("Co0", 778.2 + rng.normal(0, 0.05), 14.99, 0.5, 1.05, 1 - f2, 20, shape="Asym"), 1.05, 1.3)
    co0[0]["tail"]["value"], co0[0]["tau"]["value"] = rng.uniform(0.2, 0.45), rng.uniform(1.0, 2.5)
    co2 = core.doublet("Co2+", 780.5 + rng.normal(0, 0.1), 15.5, 0.5, 2.6, f2 * 0.62, 30)
    sat = W([core.new_peak("Co2+ s1", 786.3, 5.5, f2 * 0.38 * 0.72, 0, role="satellite"),
             core.new_peak("Co2+ s2", 802.6, 5.5, f2 * 0.38 * 0.28, 0, role="satellite")], 5.5, 5.5)
    f2 = true_frac(co0 + co2 + sat, "Co2+")
    y, bg, sig, _ = make(x, co0 + co2 + sat, rng, rng.uniform(5e4, 1e5), rng.uniform(-30, 30), rng.uniform(0.05, 0.2), rng.uniform(5e4, 1.5e5))
    s = core.Spectrum("Co2p Scan", x, y)
    return s, bg, ["Co", "O"], {"frac": ("Co2+", f2), "split": ("Co0", 14.99), "asym": "Co0", "expect": set()}


def case_cu(rng, cr_overlap, cu2):
    """Cu⁰/Cu⁺（不对称）± Cu²⁺(含卫星)；可选 Cr LMM 俄歇重叠；本底可能反向（低 BE 侧更高）。"""
    x = np.arange(925, 965.01, 0.1)
    f2 = rng.uniform(0.2, 0.5) if cu2 else 0.0
    cu = W(core.doublet("Cu0", 932.6, 19.8, 0.5, 1.15, 1 - f2, 25, shape="Asym"), 1.15, 1.45)
    cu[0]["tail"]["value"], cu[0]["tau"]["value"] = 0.2, 1.2
    pk = cu
    if cu2:
        # CuO：2p3/2 卫星 / 主峰 ≈ 0.55（Biesinger）；主峰双峰面积 = 1.5×(3/2)
        m32 = f2 / (1.5 + 0.55)
        pk += W(core.doublet("Cu2+", 934.0, 19.9, 0.5, 2.8, m32, 30), 2.8, 2.8) + W([core.new_peak("Cu2+ sat", 942.5, 5.0, 0.55 * m32, 0, role="satellite")], 5.0)
        f2 = true_frac(pk, "Cu2+")
    if cr_overlap:
        pk = pk + W([core.new_peak("CrLMM", HV - 529 + rng.normal(0, 0.4), 6.0, 0.35, 0, role="unknown")], 6.0)
    slope = rng.uniform(-80, 20)                       # 多数为反向本底
    y, bg, sig, _ = make(x, pk, rng, 1.2e5, slope, rng.uniform(0.0, 0.05), rng.uniform(5e4, 8e4))
    s = core.Spectrum("Cu2p Scan", x, y)
    expect = {"overlap"} if cr_overlap else set()
    els = ["Cu", "O"] + (["Cr"] if cr_overlap else [])
    return s, bg, els, {"frac": ("Cu2+", f2), "split": ("Cu0/Cu+", 19.8), "asym": "Cu0", "expect": expect}


def case_al(rng, cu_overlap):
    x = np.arange(65, 85.01, 0.1)
    f3 = rng.uniform(0.3, 0.9)
    al0 = W(core.doublet("Al0", 72.6, 0.44, 0.5, 0.75, 1 - f3, 20, shape="Asym"), 0.75, 0.75)
    al0[0]["tail"]["value"], al0[0]["tau"]["value"] = 0.15, 1.0
    al3 = core.doublet("Al3+", 74.7, 0.44, 0.5, 1.7, f3, 30)
    pk = al0 + al3
    f3 = true_frac(pk, "Al3+")
    if cu_overlap:
        pk += core.doublet("Cu3p", 75.1, 2.3, 0.5, 2.2, 0.6, 30)
    y, bg, sig, _ = make(x, pk, rng, 2e4, rng.uniform(-20, 20), rng.uniform(0.02, 0.15), rng.uniform(5e3, 1.5e4))
    s = core.Spectrum("Al2p Scan", x, y)
    return s, bg, ["Al", "O"] + (["Cu"] if cu_overlap else []), {"frac": ("Al3+", f3), "asym": "Al0",
                                                                    "expect": {"overlap"} if cu_overlap else set()}


def case_cr(rng, cr6):
    x = np.arange(568, 594.01, 0.1)
    f3 = rng.uniform(0.1, 0.6)
    f6 = 0.08 if cr6 else 0.0
    cr0 = core.doublet("Cr0", 574.2, 9.3, 0.5, 1.25, 1 - f3 - f6, 25, shape="Asym")
    cr0[0]["tail"]["value"], cr0[0]["tau"]["value"] = 0.35, 1.5
    pk = cr0 + core.doublet("Cr3+", 576.5, 9.75, 0.5, 2.6, f3, 30)
    if cr6:
        pk += core.doublet("Cr6+", 579.8, 9.6, 0.5, 1.8, f6, 30)
    f3 = true_frac(pk, "Cr3+")
    y, bg, sig, _ = make(x, pk, rng, 2.5e4, rng.uniform(-20, 40), rng.uniform(0.05, 0.25), rng.uniform(3e4, 6e4))
    s = core.Spectrum("Cr2p Scan", x, y)
    return s, bg, ["Cr", "O"], {"frac": ("Cr3+", f3), "split": ("Cr0", 9.3), "asym": "Cr0",
                                "expect": {"main"} if cr6 else set()}


def case_o(rng):
    x = np.arange(525, 545.01, 0.1)
    fr = rng.dirichlet([4, 3, 1.5])
    pk = [core.new_peak("O lattice", 530.2, 1.3, fr[0], 30), core.new_peak("O defect", 531.6, 1.6, fr[1], 30),
          core.new_peak("C-O", 532.9, 1.8, fr[2], 30)]
    y, bg, sig, _ = make(x, pk, rng, 1e4, rng.uniform(-10, 10), rng.uniform(0.02, 0.1), rng.uniform(1.5e4, 8e4))
    s = core.Spectrum("O1s Scan", x, y)
    return s, bg, ["O", "Co", "Cr"], {"frac": ("O lattice", fr[0]), "expect": set()}


def case_c(rng, carbide):
    x = np.arange(279, 298.01, 0.1)
    fc = rng.uniform(0.3, 0.6) if carbide else 0.0
    rest = rng.dirichlet([5, 2, 1]) * (1 - fc)
    pk = [core.new_peak("C-C", 284.8, 1.3, rest[0], 30), core.new_peak("C-O", 286.3, 1.3, rest[1], 30),
          core.new_peak("O-C=O", 288.7, 1.3, rest[2], 30)]
    if carbide:
        pk.append(W([core.new_peak("carbide", 283.0, 0.9, fc, 30)], 0.9)[0])
    fc = true_frac(pk, "carbide")
    y, bg, sig, _ = make(x, pk, rng, 1.3e4, rng.uniform(-10, 10), rng.uniform(0.0, 0.08), rng.uniform(2e3, 6e3))
    s = core.Spectrum("C1s Scan", x, y)
    return s, bg, ["C", "O", "Cr"], {"frac": ("M-C carbide", fc), "expect": set()}


# Fe 氧化态的“真实”谱：Gupta–Sen 型多重分裂包络（5 条窄线，强度递减、向高 BE 拖尾），
# 刻意与拟合模型（每个氧化态一个宽峰）不同，用来检验模型在线形不匹配时的可靠性。
FE_MULT = {"Fe2+": ([-1.2, -0.3, 0.6, 1.6, 2.8], [0.30, 0.30, 0.20, 0.12, 0.08]),
           "Fe3+": ([-1.1, -0.2, 0.7, 1.7, 2.9], [0.27, 0.30, 0.22, 0.13, 0.08])}
FE_POS = {"Fe2+": (709.6, 6.0), "Fe3+": (711.0, 8.0)}     # 表观峰位（重心）、卫星 ΔE


def fe_state(tag, A, r_sat, rng):
    """一个 Fe 氧化态：2p3/2 多重分裂 + 2p1/2 宽峰 + 两个卫星；A 为该态总面积（含卫星）。"""
    off, w = FE_MULT[tag]
    c0, dsat = FE_POS[tag]
    c0 += rng.normal(0, 0.1); dsat += rng.normal(0, 0.25)
    cen = c0 - float(np.dot(off, w))                   # 让重心 = c0
    M = A / (1.5 * (1 + r_sat))
    pk = []
    for k, (o, f) in enumerate(zip(off, w)):
        pk.append(core.new_peak("%s m%d" % (tag, k), cen + o, 1.3, M * f, 30))
    pk.append(core.new_peak("%s 1/2" % tag, c0 + 13.6, 3.6, 0.5 * M, 30))
    pk.append(core.new_peak("%s sat a" % tag, c0 + dsat, 3.8, r_sat * M, 0, role="satellite"))
    pk.append(core.new_peak("%s sat b" % tag, c0 + 13.6 + dsat, 3.8, 0.5 * r_sat * M, 0, role="satellite"))
    return W(pk, *([1.3] * 5 + [3.6, 3.8, 3.8]))


def case_fe(rng, kind):
    """kind: Fe2O3 / Fe+Fe2O3 / Fe3O4 / Fe+Fe3O4 / FeO。返回真实 Fe²⁺/(Fe²⁺+Fe³⁺)、Fe⁰ 比例与应得的化合物。"""
    x = np.arange(700, 740.01, 0.1)
    f0 = rng.uniform(0.2, 0.6) if kind.startswith("Fe+") else 0.0
    ox = kind.replace("Fe+", "")
    pk = []
    if ox == "Fe3O4+Fe2O3":                               # 真实混合：Fe₃O₄ 份额 m，其余 Fe₂O₃
        m = rng.uniform(0.35, 0.65)
        pk += fe_state("Fe2+", m / 3, rng.uniform(0.12, 0.2), rng)
        pk += fe_state("Fe3+", 2 * m / 3, rng.uniform(0.06, 0.1), rng)
        pk += [dict(p, name=p["name"].replace("Fe3+", "Fe3+b")) for p in fe_state("Fe3+", 1 - m, rng.uniform(0.15, 0.25), rng)]
        y, bg, sig, _ = make(x, pk, rng, rng.uniform(2e4, 5e4), rng.uniform(-20, 20), rng.uniform(0.1, 0.3), rng.uniform(2e4, 6e4))
        s = core.Spectrum("Fe2p Scan", x, y)
        return s, bg, ["Fe", "O"], {"fe": (m / 3, 0.0, "mix"), "expect": set()}
    x2 = {"Fe2O3": 0.0, "Fe3O4": 1 / 3, "FeO": rng.uniform(0.85, 0.95)}[ox]
    weak = ox == "Fe3O4"                                  # 磁铁矿的卫星不明显
    if f0:
        fe0 = W(core.doublet("Fe0", 706.7 + rng.normal(0, 0.05), 13.1, 0.5, 0.95, f0, 20, shape="Asym"), 0.95, 1.3)
        fe0[0]["tail"]["value"], fe0[0]["tau"]["value"] = rng.uniform(0.2, 0.4), rng.uniform(1.0, 2.0)
        pk += fe0
    if x2 > 0:
        pk += fe_state("Fe2+", (1 - f0) * x2, rng.uniform(0.12, 0.2) if weak else rng.uniform(0.3, 0.45), rng)
    if x2 < 1:
        pk += fe_state("Fe3+", (1 - f0) * (1 - x2), rng.uniform(0.06, 0.1) if weak else rng.uniform(0.15, 0.25), rng)
    y, bg, sig, _ = make(x, pk, rng, rng.uniform(2e4, 5e4), rng.uniform(-20, 20), rng.uniform(0.1, 0.3), rng.uniform(2e4, 6e4))
    s = core.Spectrum("Fe2p Scan", x, y)
    expect = ox
    return s, bg, ["Fe", "O"], {"fe": (x2, f0, expect), "expect": set(), **({"split": ("Fe0", 13.1), "asym": "Fe0"} if f0 else {})}


def case_survey(rng):
    x = np.arange(0, 1350.01, 0.5)
    present = {"C": [(284.8, 1.0)], "O": [(531.0, 3.0)], "Al": [(74.5, 0.25), (118.5, 0.18)],
               "Cr": [(574.3, 2.5), (583.6, 1.25), (42.5, 0.2)], "Co": [(778.2, 3.5), (793.2, 1.75), (60, 0.3)]}
    if rng.random() < 0.5:
        present["Ni"] = [(852.6, 2.5), (870.0, 1.25), (66.5, 0.2)]
    if rng.random() < 0.5:
        present["Cu"] = [(932.6, 3.0), (952.4, 1.5)]
    pk = []
    for el, ls in present.items():
        for c, a in ls:
            pk.append(core.new_peak(el + str(c), c, 2.2, a, 30))
    for ke, a in ((510, 1.5), (775, 0.8), (529, 0.6)):
        pk.append(W([core.new_peak("A%d" % ke, HV - ke, 8.0, a, 0)], 8.0)[0])
    y, bg, sig, scale = make(x, pk, rng, 2e4, 25.0, 0.0, rng.uniform(2e5, 5e5))
    # 非弹性台阶：每个峰在高 BE 侧带一个与面积成正比的台阶，并向高 BE 缓慢衰减
    steps = np.zeros_like(x)
    for p in pk:
        c, A = p["center"]["value"], p["area"]["value"] * scale
        steps += 0.08 * A / 2.2 / (1 + np.exp(-(x - c) / 1.5)) * np.exp(-np.clip(x - c, 0, None) / 400)
    y = np.random.default_rng(int(rng.integers(1e9))).poisson(np.clip(bg + sig + steps, 1, None)).astype(float)
    true_bg = bg + steps
    s = core.Spectrum("Survey", x, y)
    return s, true_bg, sorted(present), {}


# ------------------------------------------------------------------ 运行
def run_region(builder, n, rng, label):
    rows = []
    for _ in range(n):
        s, true_bg, els, truth = builder(rng)
        t0 = time.time()
        rep = autofit.run(s, els)
        dt = time.time() - t0
        over, err = eval_bg(s, true_bg)
        r = {"case": label, "time": dt, "bg_over": over, "bg_err": err, "bg_type": s.bg["type"]}
        if "frac" in truth:
            pre, f = truth["frac"]
            r["frac_err"] = abs(frac_of(s, pre) - f)
        if "split" in truth:
            pre, d0 = truth["split"]
            pair = [p for p in s.peaks if p.get("pair") and (p.get("link") or {}).get("ref", "").startswith(pre)]
            if pair:
                r["split_err"] = abs(pair[0]["link"]["dE"] - d0)
                r["ratio"] = pair[0]["link"]["area_ratio"]
        if "asym" in truth:
            m = [p for p in s.peaks if p["name"].startswith(truth["asym"]) and not p.get("pair")]
            r["asym_ok"] = bool(m and m[0].get("shape") == "Asym" and m[0]["tail"]["value"] > 0.03)
            ox = [p for p in s.peaks if p.get("role", "main") == "main" and not p.get("pair") and not p["name"].startswith(truth["asym"])]
            # Fe²⁺/Fe³⁺ 是多重分裂包络，按设计用不对称线形；其余元素的氧化物应为对称峰
            r["oxide_sym"] = all(p.get("shape", "GL") == "GL" for p in ox if not re.match(r"Fe[23]\+", p["name"]))
        if "fe" in truth:
            x2, f0, expect = truth["fe"]
            d = autofit.fe_phases(s)
            r["x2_err"] = abs((d["x2"] if d["x2"] is not None else 0.0) - x2)
            r["f0_err"] = abs(d["f0"] - f0)
            main = max(d["phases"].items(), key=lambda kv: kv[1])[0] if d["phases"] else "Fe0"
            r["compound_ok"] = main in ("Fe3O4", "Fe2O3") if expect == "mix" else main == expect
            alt = getattr(s, "fe_alt", None)
            lo, hi = alt["range"] if alt else (d["x2"] or 0.0, d["x2"] or 0.0)
            r["range_ok"] = lo - 0.05 <= x2 <= hi + 0.05
            r["interp"] = autofit.interpret(s) or ""
            r["false_metal"] = (f0 == 0 and d["f0"] > 0.02)
        added = {("overlap" if ("重叠峰" in t or "同一重叠来源" in t) else "satellite" if "→ 卫星峰" in t else
                  "main" if "新化学态" in t else "unknown")
                 for t in rep if (t.startswith("残差峰") or t.startswith("预置重叠") or t.startswith("化学态测试")) and "不保留" not in t and "去掉" not in t}
        added.discard(None)
        exp = truth.get("expect", set())
        r["class_ok"] = (exp <= added) and not ({"overlap"} & added and "overlap" not in exp) \
            and not ("main" in added and "main" not in exp and label.startswith("Cr"))
        rows.append(r)
    return rows


def run_survey(n, rng):
    rows = []
    for _ in range(n):
        s, true_bg, present, _ = case_survey(rng)
        s.bg.update({"type": "SNIP", "lo": None, "hi": None})
        x, y, bg, _, _ = s.compute()
        over = float(np.max(bg - core.smooth(y, 5)))
        err = float(np.sqrt(np.mean((bg - np.interp(x, s.be_raw, true_bg)) ** 2)) / float(np.max(y - bg)))
        r = identify.identify_survey(s.be_raw, s.y)
        conf = {e["el"] for e in r["elements"] if e["confirmed"]}
        rows.append({"case": "Survey", "bg_over": over, "bg_err": err, "found_all": set(present) <= conf,
                     "false_pos": sorted(conf - set(present))})
    return rows


def summarize(rows):
    import collections
    g = collections.OrderedDict()
    for r in rows:
        g.setdefault(r["case"], []).append(r)
    out = []
    out.append("| 场景 | 次数 | 背景高于数据 | 背景误差(相对峰高) | 化学态比例误差 | 双峰间距误差 | 1/2 面积比 | 金属拖尾 / 氧化物对称 | 残差峰归类 | 背景类型 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, rs in g.items():
        def m(key, f=np.mean, fmt="%.3f"):
            v = [r[key] for r in rs if key in r and r[key] is not None]
            return (fmt % f(v)) if v else "—"
        over_cnt = sum(1 for r in rs if r["bg_over"] > 1e-6 * 1e5)
        types = {}
        for r in rs:
            types[r.get("bg_type", "SNIP")] = types.get(r.get("bg_type", "SNIP"), 0) + 1
        asym = [r for r in rs if "asym_ok" in r]
        out.append("| %s | %d | %d 次 | %s（最大 %s） | %s（最大 %s） | %s | %s | %s | %s | %s |" % (
            k, len(rs), over_cnt, m("bg_err"), m("bg_err", np.max), m("frac_err"), m("frac_err", np.max),
            m("split_err"), m("ratio"),
            ("%d/%d · %d/%d" % (sum(r["asym_ok"] for r in asym), len(asym), sum(r["oxide_sym"] for r in asym), len(asym))) if asym else "—",
            ("%d/%d" % (sum(r.get("class_ok", True) for r in rs), len(rs))) if any("class_ok" in r for r in rs) else "—",
            "、".join("%s×%d" % kv for kv in types.items())))
    fe = [r for r in rows if "x2_err" in r]
    if fe:
        out.append("")
        out.append("| Fe 场景 | 次数 | 主物相正确 | Fe²⁺/(Fe²⁺+Fe³⁺) 误差（默认模型） | 真值落在报告范围内（±0.05） | Fe⁰ 比例误差 | 误报金属 Fe⁰ |")
        out.append("|---|---|---|---|---|---|---|")
        for k in dict.fromkeys(r["case"] for r in fe):
            rs = [r for r in fe if r["case"] == k]
            out.append("| %s | %d | %d/%d | %.3f（最大 %.3f） | %d/%d | %.3f（最大 %.3f） | %d 次 |" % (
                k, len(rs), sum(r["compound_ok"] for r in rs), len(rs),
                np.mean([r["x2_err"] for r in rs]), np.max([r["x2_err"] for r in rs]),
                sum(r["range_ok"] for r in rs), len(rs),
                np.mean([r["f0_err"] for r in rs]), np.max([r["f0_err"] for r in rs]), sum(r["false_metal"] for r in rs)))
    sv = [r for r in rows if r["case"] == "Survey"]
    if sv:
        out.append("")
        out.append("全谱：%d 次，元素全部找到 %d 次；误报 %s；背景高于数据 %d 次；背景误差均值 %.3f。" % (
            len(sv), sum(r["found_all"] for r in sv),
            ("、".join(sorted({e for r in sv for e in r["false_pos"]})) or "无"),
            sum(1 for r in sv if r["bg_over"] > 1e-3), float(np.mean([r["bg_err"] for r in sv]))))
    return "\n".join(out)


def main(n=12, out=None, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    rows += run_region(case_co, n, rng, "Co2p（Co⁰+Co²⁺+卫星）")
    rows += run_region(lambda r: case_cu(r, True, False), n, rng, "Cu2p + Cr LMM 重叠")
    rows += run_region(lambda r: case_cu(r, False, True), n, rng, "Cu2p 含 Cu²⁺ 卫星")
    rows += run_region(lambda r: case_cu(r, False, False), n, rng, "Cu2p 纯 Cu⁰/Cu⁺")
    rows += run_region(lambda r: case_al(r, True), n, rng, "Al2p + Cu3p 重叠")
    rows += run_region(lambda r: case_al(r, False), n, rng, "Al2p")
    rows += run_region(lambda r: case_cr(r, True), n, rng, "Cr2p 含 Cr⁶⁺")
    rows += run_region(lambda r: case_cr(r, False), n, rng, "Cr2p")
    rows += run_region(case_o, n, rng, "O1s")
    rows += run_region(lambda r: case_c(r, True), n, rng, "C1s 含碳化物")
    for kind in ("Fe2O3", "Fe+Fe2O3", "Fe3O4", "Fe+Fe3O4", "FeO", "Fe3O4+Fe2O3"):
        rows += run_region(lambda r, k=kind: case_fe(r, k), n, rng, "Fe2p " + kind)
    rows += run_survey(n, rng)
    md = summarize(rows)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(md + "\n")
    print(md)
    return rows


def main_fe(n=10, seed=3, kinds=("Fe2O3", "Fe+Fe2O3", "Fe3O4", "Fe+Fe3O4", "FeO", "Fe3O4+Fe2O3"), verbose=False):
    rng = np.random.default_rng(seed)
    rows = []
    for kind in kinds:
        rr = run_region(lambda r, k=kind: case_fe(r, k), n, rng, "Fe2p " + kind)
        if verbose:
            for r in rr:
                print(kind, "x2err %.3f f0err %.3f ok=%s range=%s | %s" % (r["x2_err"], r["f0_err"], r["compound_ok"], r["range_ok"], r["interp"][:150]))
        rows += rr
    print(summarize(rows))
    return rows


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fe":
        main_fe(int(sys.argv[2]) if len(sys.argv) > 2 else 10, verbose=True)
        sys.exit()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    main(n, sys.argv[2] if len(sys.argv) > 2 else None)
