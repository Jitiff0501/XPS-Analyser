# -*- coding: utf-8 -*-
"""v1.1：没有内置模型的轨道（Ti2p、Zr3d、Nb3d、Ta4f、Hf4f、Mo3d、Si2p、Ni2p、Mn2p）——
比较“在数据中找峰”的通用模型（v1.0）与“按 NIST 数据库建模”（v1.1）。

    python tools/benchmark_nist.py [每个轨道次数=10] [seed=7] [输出.md]

模拟方法（比 v1.0 的基准更接近真实数据）：
  - 真实峰位不固定：按 NIST 中该化合物的文献分布抽取（四分位范围内均匀），即包含不同文献 / 荷电参考之间的离散；
  - 自旋轨道分裂按金属 / 化合物的文献范围随机；面积比在理论值 ±0.03 内随机；
  - 峰宽、Lorentzian 比例、金属拖尾（T、τ）、背景（常数、斜率、Shirley 台阶）、计数（信噪比）、金属比例都随机；
  - 有的场景含第二种化合物（低价氧化物或氮化物），检验多化学态时的取舍；Ni2p / Mn2p 含卫星或多重分裂展宽。
评估：金属态比例误差、真实化学态是否都被保留、是否多出不存在的化学态、背景是否高于数据。
"""
import copy
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xpsfit import core, autofit, reference  # noqa: E402

# 场景：轨道 → (窗口, 金属, [化合物（主）], [可能出现的第二化合物], 金属 FWHM 范围, 化合物 FWHM 范围, 1/2 的 FWHM 倍率范围（化合物）)
SCEN = {
    "Ti2p": ((450, 470), "Ti", ["TiO2"], ["Ti2O3", "TiN"], (0.8, 1.2), (1.1, 1.6), (1.4, 2.0)),
    "Zr3d": ((175, 190), "Zr", ["ZrO2"], [], (0.6, 0.9), (1.1, 1.5), (1.0, 1.05)),
    "Nb3d": ((198, 214), "Nb", ["Nb2O5"], ["NbO", "NbO2"], (0.6, 0.9), (1.1, 1.5), (1.0, 1.05)),
    "Ta4f": ((18, 32), "Ta", ["Ta2O5"], [], (0.5, 0.8), (1.1, 1.6), (1.0, 1.05)),
    "Hf4f": ((11, 24), "Hf", ["HfO2"], [], (0.5, 0.8), (1.1, 1.6), (1.0, 1.05)),
    "Mo3d": ((224, 240), "Mo", ["MoO3"], ["MoO2"], (0.6, 0.9), (1.1, 1.6), (1.0, 1.05)),
    "Si2p": ((96, 108), "Si", ["SiO2"], ["SiC", "Si3N4"], (0.6, 0.9), (1.2, 1.8), (1.0, 1.0)),
    "Ni2p": ((845, 885), "Ni", ["NiO"], ["Ni(OH)2"], (0.9, 1.3), (1.6, 2.4), (1.0, 1.4)),
    "Mn2p": ((632, 660), "Mn", ["MnO2"], ["MnO"], (0.9, 1.3), (2.0, 3.0), (1.0, 1.4)),
}


def nist_pos(states, formula, rng):
    t = states[formula]
    return float(rng.uniform(t["q1"], t["q3"])) if t["q3"] > t["q1"] else float(t["median"] + rng.normal(0, 0.1))


def build(orb, rng):
    win, met, main, second, wm, wo, fwr = SCEN[orb]
    ref = reference.for_orbital(orb)
    states = {t["formula"]: t for t in ref["states"]}
    l_ = orb[-1]
    ratio_th = {"p": 0.5, "d": 2 / 3, "f": 0.75}[l_]
    x = np.arange(win[0], win[1] + 1e-9, 0.1 if win[1] - win[0] > 20 else 0.05)
    f_m = rng.uniform(0.15, 0.85)
    comps = [(met, f_m, "metal")]
    rest = 1 - f_m
    if second and rng.random() < 0.4:
        sec = second[rng.integers(len(second))]
        f2 = rest * rng.uniform(0.2, 0.5)
        comps += [(main[0], rest - f2, "oxide"), (sec, f2, "oxide")]
    else:
        comps += [(main[0], rest, "oxide")]
    peaks = []
    for name, frac, kind in comps:
        c = nist_pos(states, name, rng)
        so = core.so_rule(orb, kind == "metal")
        dE = float(rng.uniform(*so["dE_rng"])) if so and so["dE_rng"][1] > so["dE_rng"][0] else (so["dE"] if so else ref["split"]["median"])
        w = float(rng.uniform(*(wm if kind == "metal" else wo)))
        fr = float(rng.uniform(*((1.0, 1.3) if kind == "metal" else fwr)))
        a = 1000.0 * frac
        shape = "Asym" if kind == "metal" else "GL"
        d = core.doublet(name, c, dE, ratio_th + rng.uniform(-0.03, 0.03), w, a, rng.uniform(15, 40), shape=shape)
        d[1]["link"]["fwhm_ratio"] = fr
        for p in d:
            p["tail"]["value"], p["tau"]["value"] = rng.uniform(0.15, 0.4), rng.uniform(0.8, 2.0)
            p["truth_kind"] = kind
        peaks += d
        if orb == "Ni2p" and kind == "metal":            # 金属 Ni 的 +6 eV 卫星
            peaks.append(dict(core.new_peak("Ni sat", c + 6.0, 2.5, a * 0.12, 0), truth_kind="metal"))
        if orb == "Ni2p" and name == "NiO":               # NiO 的多重分裂肩峰 +1.5 eV 与卫星 +7 eV
            peaks.append(dict(core.new_peak("NiO shoulder", c + 1.6, 2.2, a * 0.45, 0), truth_kind="oxide"))
            peaks.append(dict(core.new_peak("NiO sat", c + 7.0, 4.0, a * 0.55, 0), truth_kind="oxide"))
    _, sig = core.model(x, peaks)
    counts = rng.uniform(4e3, 6e4)
    sig = sig / sig.max() * counts
    scale = counts / max(float(core.model(x, peaks)[1].max()), 1e-9)
    base = rng.uniform(0.5, 2.0) * counts
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (sig[1:] + sig[:-1]) * np.diff(x))])
    bg = base + rng.uniform(-0.002, 0.002) * counts * (x - x[0]) + rng.uniform(0.03, 0.12) * cum
    y = rng.poisson(np.clip(bg + sig, 1, None)).astype(float)
    vals = core.resolve(peaks)
    tot = sum(v["area"] for v in vals)
    fm = sum(v["area"] for p, v in zip(peaks, vals) if p["truth_kind"] == "metal") / tot
    truth = {c[0] for c in comps}
    return core.Spectrum("%s Scan" % orb, x, y), bg, fm, truth


def metal_fraction(s):
    st = core.chemical_states(s)
    tot = sum(a for _, a in st) or 1.0
    kinds = {core.state_name(s.peaks[core.group_root(s.peaks, j)]["name"]): p.get("state_kind") for j, p in enumerate(s.peaks)}
    el = s.element[:-2]

    def is_metal(n):                                  # 通用模型按最近的 NIST 化学式命名，单质写作元素符号（如 "Ti 2p"）
        return kinds.get(n) == "metal" or core.is_metal_name(n) or n.split(" ")[0] == el
    return sum(a for n, a in st if is_metal(n)) / tot


def kept_states(s):
    return {core.state_name(p["name"]).rsplit(" ", 1)[0] for p in s.peaks if p.get("role", "main") == "main" and not p.get("pair")}


def run_one(orb, rng, use_nist):
    s, bg, fm, truth = build(orb, rng)
    saved = autofit.nist_model
    if not use_nist:
        autofit.nist_model = lambda *a, **k: None
    try:
        t0 = time.time()
        autofit.run(s, [orb[:-2], "O", "C", "N"])     # 样品元素（全谱通常能看到的）
        dt = time.time() - t0
    finally:
        autofit.nist_model = saved
    x, y, sbg, _, _ = s.compute()
    m = (s.be >= x.min()) & (s.be <= x.max())
    over = float(np.max(sbg - np.interp(x, s.be[::-1], s.y[::-1]))) if len(x) else 0.0
    err = abs(metal_fraction(s) - fm)
    kept = kept_states(s)
    met = SCEN[orb][1]
    truth_n = {(met + "0") if t == met else t for t in truth}
    found = sum(1 for t in truth_n if any(t in k for k in kept))
    extra = len([k for k in kept if not any(t in k for t in truth_n)])
    return {"err": err, "found": found, "n_truth": len(truth_n), "extra": extra, "time": dt}


def main(n=10, seed=7, out=None):
    lines = ["| 轨道 | 模型 | 金属比例误差 中位 | 90% 以内 | 真实化学态全部保留 | 多出的化学态（平均） | 每条用时 s |", "|---|---|---|---|---|---|---|"]
    summary = {}
    for orb in SCEN:
        for use_nist in (False, True):
            rng = np.random.default_rng(seed + hash(orb) % 1000)
            R = [run_one(orb, rng, use_nist) for _ in range(n)]
            e = np.array([r["err"] for r in R])
            row = (orb, "NIST（v1.1）" if use_nist else "通用找峰（v1.0）", 100 * np.median(e), 100 * np.percentile(e, 90),
                   sum(r["found"] == r["n_truth"] for r in R), np.mean([r["extra"] for r in R]), np.mean([r["time"] for r in R]))
            summary[(orb, use_nist)] = row
            lines.append("| %s | %s | %.1f | %.1f | %d/%d | %.1f | %.1f |" % (row[0], row[1], row[2], row[3], row[4], n, row[5], row[6]))
            print(lines[-1], flush=True)
    txt = "\n".join(lines)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
    return summary


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 10, int(a[1]) if len(a) > 1 else 7, a[2] if len(a) > 2 else None)
