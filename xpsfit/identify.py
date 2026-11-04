# -*- coding: utf-8 -*-
"""根据谱图本身识别元素/轨道（不依赖文件名），以及全谱（survey）元素识别。

谱线表为 Al Kα 激发下常用的结合能范围（覆盖金属到常见化合物的化学位移，近似值）。
"""
import math
import re

import numpy as np

from . import core

# 轨道：(主线 BE 范围 lo, hi)、自旋轨道分裂、面积比、元素常见度权重（越常见越大）
LINES = {
    "C1s": ((281.5, 290.5), None, None, 3.0), "N1s": ((396.0, 407.5), None, None, 2.0),
    "O1s": ((528.5, 534.5), None, None, 3.0), "F1s": ((684.0, 690.0), None, None, 1.2),
    "Na1s": ((1070.5, 1073.0), None, None, 1.2), "Mg2p": ((49.0, 51.5), None, None, 1.0),
    "Mg1s": ((1302.5, 1305.0), None, None, 0.8), "Al2p": ((72.2, 75.8), 0.44, 0.5, 2.0),
    "Al2s": ((117.5, 120.5), None, None, 0.6), "Si2p": ((98.8, 104.0), 0.61, 0.5, 2.0),
    "Si2s": ((150.0, 155.0), None, None, 0.6), "P2p": ((128.5, 134.5), 0.87, 0.5, 1.0),
    "S2p": ((161.0, 169.5), 1.18, 0.5, 1.2), "Cl2p": ((198.0, 201.0), 1.6, 0.5, 1.0),
    "K2p": ((292.5, 294.0), 2.8, 0.5, 0.8), "Ca2p": ((346.5, 348.0), 3.55, 0.5, 1.0),
    "Ti2p": ((453.8, 459.5), 5.7, 0.5, 1.5), "V2p": ((512.0, 517.6), 7.4, 0.5, 1.0),
    "Cr2p": ((573.6, 580.0), 9.3, 0.5, 1.5), "Mn2p": ((638.5, 642.8), 11.7, 0.5, 1.0),
    "Fe2p": ((706.5, 711.5), 13.6, 0.5, 1.5), "Co2p": ((777.8, 781.5), 15.0, 0.5, 1.3),
    "Ni2p": ((852.4, 856.5), 17.3, 0.5, 1.3), "Cu2p": ((932.2, 935.5), 19.8, 0.5, 1.3),
    "Zn2p": ((1021.5, 1022.8), 23.1, 0.5, 1.0), "Zr3d": ((178.7, 183.0), 2.43, 2 / 3, 1.0),
    "Nb3d": ((202.0, 207.8), 2.72, 2 / 3, 0.8), "Mo3d": ((227.8, 233.0), 3.13, 2 / 3, 1.0),
    "Ru3d": ((280.0, 281.5), 4.17, 2 / 3, 0.5), "Pd3d": ((335.0, 338.0), 5.26, 2 / 3, 0.6),
    "Ag3d": ((367.5, 368.5), 6.0, 2 / 3, 0.8), "In3d": ((443.5, 445.0), 7.54, 2 / 3, 0.6),
    "Sn3d": ((484.8, 487.0), 8.41, 2 / 3, 0.8), "Ba3d": ((779.0, 781.0), 15.3, 2 / 3, 0.5),
    "La3d": ((834.5, 836.0), 16.8, 2 / 3, 0.5), "Ce3d": ((881.5, 885.0), 18.6, 2 / 3, 0.5),
    "Hf4f": ((14.0, 17.4), 1.68, 0.75, 0.8), "Ta4f": ((21.6, 27.0), 1.91, 0.75, 0.8),
    "W4f": ((31.1, 36.0), 2.18, 0.75, 0.9), "Pt4f": ((70.9, 75.0), 3.33, 0.75, 0.8),
    "Au4f": ((83.9, 86.0), 3.67, 0.75, 0.8), "Pb4f": ((136.8, 139.0), 4.86, 0.75, 0.5),
    "Bi4f": ((156.8, 159.5), 5.3, 0.75, 0.5), "Y3d": ((155.8, 158.5), 2.05, 2 / 3, 0.4),
    "Sr3d": ((132.5, 134.0), 1.79, 2 / 3, 0.4), "B1s": ((187.0, 193.5), None, None, 0.6),
    "Li1s": ((54.5, 56.5), None, None, 0.4),
    # 次要能级（全谱里用来确认元素）
    "Cr3p": ((41.8, 44.0), None, None, 0.3), "Co3p": ((59.0, 61.0), None, None, 0.3),
    "Ni3p": ((66.0, 68.0), None, None, 0.3), "Cu3p": ((74.8, 77.5), None, None, 0.3),
    "Fe3p": ((52.5, 56.0), None, None, 0.3), "Ti3p": ((32.5, 38.0), None, None, 0.3),
    "Zr3p": ((330.0, 333.0), None, None, 0.3), "Mo3p": ((393.5, 399.0), None, None, 0.3),
    "O2s": ((22.0, 26.0), None, None, 0.2), "Zn3d": ((9.5, 11.0), None, None, 0.3),
}
# 每个元素在全谱中最强的光电子线（用于判断元素是否存在）
STRONGEST = {
    "C": "C1s", "N": "N1s", "O": "O1s", "F": "F1s", "Na": "Na1s", "Mg": "Mg1s", "Al": "Al2p", "Si": "Si2p",
    "P": "P2p", "S": "S2p", "Cl": "Cl2p", "K": "K2p", "Ca": "Ca2p", "Ti": "Ti2p", "V": "V2p", "Cr": "Cr2p",
    "Mn": "Mn2p", "Fe": "Fe2p", "Co": "Co2p", "Ni": "Ni2p", "Cu": "Cu2p", "Zn": "Zn2p", "Zr": "Zr3d",
    "Nb": "Nb3d", "Mo": "Mo3d", "Ru": "Ru3d", "Pd": "Pd3d", "Ag": "Ag3d", "In": "In3d", "Sn": "Sn3d",
    "Ba": "Ba3d", "La": "La3d", "Ce": "Ce3d", "Hf": "Hf4f", "Ta": "Ta4f", "W": "W4f", "Pt": "Pt4f",
    "Au": "Au4f", "Pb": "Pb4f", "Bi": "Bi4f", "Y": "Y3d", "Sr": "Sr3d", "B": "B1s", "Li": "Li1s",
}
AUGER_KE = {"C": 263, "N": 379, "O": 510, "F": 656, "Na": 990, "Ti": 418, "V": 473, "Cr": 529,
            "Mn": 589, "Fe": 703, "Co": 775, "Ni": 848, "Cu": 918, "Zn": 992}


def _peaks(x, y, prom_frac):
    from scipy.signal import find_peaks
    ys = core.smooth(y, 7)
    k, pr = find_peaks(ys, prominence=prom_frac * float(np.ptp(ys) or 1))
    return [(float(x[j]), float(pr["prominences"][n])) for n, j in enumerate(k)], ys


# ------------------------------------------------------------------ 高分辨谱：识别轨道
def identify_region(be, y):
    """返回候选 [(轨道, 得分, 说明)]，得分高者在前。"""
    x = np.asarray(be, float); o = np.argsort(x); x, y = x[o], np.asarray(y, float)[o]
    lo, hi = float(x[0]), float(x[-1])
    mid, width = (lo + hi) / 2, hi - lo
    pk, ys = _peaks(x, y, 0.08)
    if not pk:
        return []
    top = max(p for _, p in pk)
    cands = []
    for orb, ((a, b), split, ratio, w) in LINES.items():
        if b < lo - 1 or a > hi + 1:
            continue
        inside = [(c, p) for c, p in pk if a - 0.4 <= c <= b + 0.4]
        if not inside:
            continue
        c, p = max(inside, key=lambda t: t[1])
        score = 2.0 * p / top                                    # 主峰强度
        why = ["%.1f eV 有峰（%s 范围 %.1f–%.1f）" % (c, orb, a, b)]
        # Avantage 采集窗口通常以该谱线为中心：越靠近窗口中心越可能
        score += 1.5 * math.exp(-((((a + b) / 2) - mid) / max(width * 0.35, 3)) ** 2)
        if split:
            c2 = c + split
            if lo <= c2 <= hi:
                near = [q for cc, q in pk if abs(cc - c2) <= max(0.5, 0.06 * split)]
                if near:
                    score += 1.5
                    why.append("+%.2f eV 处有自旋轨道分量" % split)
                elif split > 1.0:
                    score -= 1.5
                    why.append("缺少 +%.2f eV 的自旋轨道分量" % split)
        score += 0.3 * math.log(w + 0.5)
        cands.append((orb, round(score, 2), "；".join(why)))
    cands.sort(key=lambda t: -t[1])
    return cands


# ------------------------------------------------------------------ 全谱：识别元素
def identify_survey(be, y, hv=1486.68):
    """返回 {"elements": [{el, lines:[(名, BE)], score}], "peaks": [(BE, 归属)]}。"""
    x = np.asarray(be, float); o = np.argsort(x); x, y = x[o], np.asarray(y, float)[o]
    # 在扣除 SNIP 背景后的信号上找峰：陡峭的非弹性本底上，弱峰也能被发现
    bgs = core.snip_background(x, y)
    pk, ys = _peaks(x, core.smooth(y, 3) - bgs + float(np.min(bgs)), 0.004)
    d = np.diff(y)                                   # 稳健噪声：相邻差值的中位绝对偏差（不受峰形影响）
    noise = float(np.median(np.abs(d - np.median(d))) / 0.6745 / math.sqrt(2)) or 1.0
    pk = [(c, p) for c, p in pk if p > 6 * noise]
    # 每个峰的所有可能归属
    opts = []
    for c, p in pk:
        o = []
        for orb, ((a, b), split, ratio, w) in LINES.items():
            el = re.match(r"[A-Z][a-z]?", orb).group(0)
            if a - 0.8 <= c <= b + 0.8:
                o.append({"el": el, "name": orb, "kind": "main" if STRONGEST.get(el) == orb else "minor", "w": w})
            elif split and split > 1.0 and a + split - 0.8 <= c <= b + split + 0.8:
                jl = {"p": "1/2", "d": "3/2", "f": "5/2"}.get(orb[-1], "")
                o.append({"el": el, "name": orb + jl, "kind": "partner", "w": w})
        for el, ke in AUGER_KE.items():
            if abs((hv - ke) - c) <= 4.0:
                o.append({"el": el, "name": "%s 俄歇" % el, "kind": "auger", "w": 0.5})
        opts.append((c, p, o))
    # 第 1 轮：确认元素——最强线存在；若该线有 >1 eV 的自旋轨道分裂且分量在谱范围内，分量也要存在
    have = {}
    for c, p, o in opts:
        for t in o:
            have.setdefault((t["el"], t["kind"]), []).append(c)
    confirmed = set()
    for el, main in STRONGEST.items():
        if (el, "main") not in have:
            continue
        (a, b), split, ratio, w = LINES[main]
        need_partner = bool(split and split > 1.0 and x[0] <= a + split and b + split <= x[-1])
        if not need_partner or (el, "partner") in have:
            confirmed.add(el)
    # 第 2 轮：每个峰优先归给已确认元素（光电子主线 > 分量 > 俄歇 > 次要能级），否则归给可能的元素（待确认）
    rank = {"main": 0, "partner": 1, "auger": 2, "minor": 3}
    found, assigned, tentative = {}, [], {}
    for c, p, o in opts:
        good = sorted([t for t in o if t["el"] in confirmed], key=lambda t: (rank[t["kind"]], -t["w"]))
        if good:
            t = good[0]
            found.setdefault(t["el"], []).append((t["name"], c, p)); assigned.append((c, t["name"]))
        elif o:
            t = sorted(o, key=lambda t: (rank[t["kind"]], -t["w"]))[0]
            tentative.setdefault(t["el"], []).append((t["name"], c, p)); assigned.append((c, t["name"] + "（待确认）"))
        else:
            assigned.append((c, "?"))
    elements = []
    for el, lines in found.items():
        elements.append({"el": el, "lines": [(l[0], round(l[1], 1)) for l in sorted(lines, key=lambda t: t[1])],
                         "score": round(sum(l[2] for l in lines) / (noise or 1), 1), "main": True, "confirmed": True})
    for el, lines in tentative.items():
        if el in found:
            continue
        elements.append({"el": el, "lines": [(l[0], round(l[1], 1)) for l in sorted(lines, key=lambda t: t[1])],
                         "score": round(sum(l[2] for l in lines) / (noise or 1), 1), "main": False, "confirmed": False})
    elements.sort(key=lambda e: (not e["confirmed"], -e["score"]))
    return {"elements": elements, "peaks": assigned}


def is_survey(be):
    be = np.asarray(be, float)
    return float(be.max() - be.min()) > 300
