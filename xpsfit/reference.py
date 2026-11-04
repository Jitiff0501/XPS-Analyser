# -*- coding: utf-8 -*-
"""推荐峰位参考库：由 NIST XPS Database (SRD 20) 导出的记录生成。

数据来源：NIST X-ray Photoelectron Spectroscopy Database, SRD 20, https://srdata.nist.gov/xps/
NIST 记录的荷电参考各不相同（污染碳、导体、内标等），因此这里给出的是文献分布（中位数、四分位范围、条数），
不是通用的拟合常数。

生成方式：
  - 程序内“导入 NIST 数据”（zip 或 nist_xps_all_records.csv）→ 保存到 %USERPROFILE%/XPSFitStudio/reference/<元素>.json
  - 命令行：python -m xpsfit.reference NIST_XPS_Cu.zip  → 写入 xpsfit/reference/<元素>.json（随程序发布）
"""
import csv
import io
import json
import os
import re
import statistics
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLED = os.path.join(HERE, "reference")
MIN_N = 3            # 至少几条记录才作为“推荐化学态”列出

# 每类轨道用于定位化学态的主线，以及双峰面积比（副/主）
MAIN_LINE = {"s": ("1s", None), "p": ("3/2", "1/2"), "d": ("5/2", "3/2"), "f": ("7/2", "5/2")}
RATIO = {"p": 0.5, "d": 2 / 3, "f": 0.75}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _quart(vals):
    v = sorted(vals)
    if len(v) == 1:
        return v[0], v[0]
    q = statistics.quantiles(v, n=4, method="inclusive")
    return q[0], q[2]


def _simple(formula):
    """排除混合物/表面层/掺杂等复杂体系（含 / 或括号下标、长链有机物），便于推荐。"""
    if not formula or "/" in formula or "(" in formula and ")0." in formula:
        return False
    return len(formula) <= 24


def _read_tables(path):
    """返回 (all_records 行列表, doublet 行列表)。path 可为 zip 或 csv。"""
    def rows(text):
        return list(csv.DictReader(io.StringIO(text)))
    recs, dbl = [], []
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                base = os.path.basename(n).lower()
                if not base.endswith(".csv"):
                    continue
                text = z.read(n).decode("utf-8-sig", "replace")
                if "doublet" in base:
                    dbl += rows(text)
                elif "record" in base:
                    recs += rows(text)
    else:
        with open(path, encoding="utf-8-sig") as f:
            text = f.read()
        r = rows(text)
        if r and "separation_eV" in r[0]:
            dbl = r
        else:
            recs = r
        side = os.path.join(os.path.dirname(path), "nist_xps_doublet_separation.csv")
        if not dbl and os.path.exists(side):
            with open(side, encoding="utf-8-sig") as f:
                dbl = rows(f.read())
    if not recs:
        raise ValueError("没有找到 NIST 记录表（nist_xps_all_records.csv）")
    need = {"element", "line", "binding_energy_eV", "formula"}
    if not need <= set(recs[0]):
        raise ValueError("记录表缺少列：%s" % ", ".join(sorted(need - set(recs[0]))))
    return recs, dbl


def build(recs, dbl):
    """按元素生成参考数据 {元素: dict}。"""
    out = {}
    for el in sorted({r["element"] for r in recs if r.get("element")}):
        R = [r for r in recs if r.get("element") == el]
        be_rows = [r for r in R if _num(r.get("binding_energy_eV")) is not None and str(r.get("is_auger", "")).lower() != "true"]
        lines = sorted({r["line"] for r in be_rows})
        orbitals = {}
        for line in lines:
            m = re.match(r"^(\d)([spdf])(\d/2)?$", line.strip())
            if not m:
                continue
            n_, l_, j_ = m.groups()
            if l_ != "s" and j_ != MAIN_LINE[l_][0]:
                continue                                 # 只用主线（如 2p3/2）定位化学态
            orb = "%s%s%s" % (el, n_, l_)
            groups = {}
            for r in be_rows:
                if r["line"] == line:
                    groups.setdefault(r.get("formula") or "?", []).append(r)
            states = []
            for formula, rs in groups.items():
                v = [_num(r["binding_energy_eV"]) for r in rs]
                if len(v) < MIN_N or not _simple(formula):
                    continue
                q1, q3 = _quart(v)
                states.append({
                    "formula": formula,
                    "name": (rs[0].get("chemical_state_name") or "")[:60],
                    "median": round(statistics.median(v), 2), "q1": round(q1, 2), "q3": round(q3, 2),
                    "min": min(v), "max": max(v), "n": len(v),
                    "refs": [r.get("record_url") for r in rs[:3] if r.get("record_url")],
                })
            states.sort(key=lambda s: s["median"])
            # 双峰间距：直接 DS 记录 + 成对计算，取中位数
            split = None
            if l_ != "s":
                key = "%s%s" % (n_, l_)
                sv = [_num(d.get("separation_eV")) for d in dbl if d.get("element") == el and
                      (d.get("doublet", "").endswith("DS-" + key) or d.get("doublet", "").startswith(key))]
                sv = [x for x in sv if x]
                if sv:
                    split = {"median": round(statistics.median(sv), 3), "n": len(sv), "min": min(sv), "max": max(sv)}
            # 卫星峰
            sat = []
            satg = {}
            for r in be_rows:
                if r["line"].startswith(line) and "sat" in r["line"]:
                    satg.setdefault(r.get("formula") or "?", []).append(_num(r["binding_energy_eV"]))
            for f_, v in satg.items():
                sat.append({"formula": f_, "median": round(statistics.median(v), 2), "min": min(v), "max": max(v), "n": len(v)})
            orbitals[orb] = {"line": line, "l": l_, "ratio": RATIO.get(l_), "split": split,
                             "states": states, "satellites": sorted(sat, key=lambda s: s["median"]),
                             "n_records": sum(len(g) for g in groups.values())}
        # 俄歇：最常见的一条线，按化合物给出动能中位数；与主线结合能合成修正俄歇参数 α'
        aug = [r for r in R if str(r.get("is_auger", "")).lower() == "true" and _num(r.get("kinetic_energy_eV")) is not None]
        auger = None
        if aug:
            top = max({r["line"] for r in aug}, key=lambda L: sum(1 for r in aug if r["line"] == L))
            g = {}
            for r in aug:
                if r["line"] == top:
                    g.setdefault(r.get("formula") or "?", []).append(_num(r["kinetic_energy_eV"]))
            rows = []
            for f_, v in g.items():
                if len(v) < MIN_N or not _simple(f_):
                    continue
                row = {"formula": f_, "ke": round(statistics.median(v), 2), "n": len(v)}
                # 修正俄歇参数 α' = KE(俄歇) + BE(主能级)；主能级取记录最多的轨道（如 Cu 用 2p3/2）
                if orbitals:
                    main = max(orbitals.values(), key=lambda o: o["n_records"])
                    for s_ in main["states"]:
                        if s_["formula"] == f_:
                            row["alpha"] = round(row["ke"] + s_["median"], 2)
                            row["alpha_line"] = main["line"]
                rows.append(row)
            auger = {"line": top, "rows": sorted(rows, key=lambda r: r["ke"])}
        out[el] = {"element": el, "source": "NIST XPS Database (SRD 20), https://srdata.nist.gov/xps/",
                   "records": len(R), "orbitals": orbitals, "auger": auger}
    return out


def import_file(path, dest):
    recs, dbl = _read_tables(path)
    data = build(recs, dbl)
    os.makedirs(dest, exist_ok=True)
    for el, d in data.items():
        with open(os.path.join(dest, "%s.json" % el), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
    return data


def load(element, user_dir=None):
    """先找用户导入的，再找随程序发布的。"""
    for d in [os.path.join(user_dir, "reference") if user_dir else None, BUNDLED]:
        if d and os.path.exists(os.path.join(d, "%s.json" % element)):
            with open(os.path.join(d, "%s.json" % element), encoding="utf-8") as f:
                return json.load(f)
    return None


def available(user_dir=None):
    els = set()
    for d in [os.path.join(user_dir, "reference") if user_dir else None, BUNDLED]:
        if d and os.path.isdir(d):
            els |= {f[:-5] for f in os.listdir(d) if f.endswith(".json")}
    return sorted(els)


def for_orbital(orbital, user_dir=None):
    """orbital 如 'Cu2p' → 该轨道的推荐数据（含元素级俄歇信息）。"""
    m = re.match(r"^([A-Z][a-z]?)(\d[spdf])$", orbital or "")
    if not m:
        return None
    ref = load(m.group(1), user_dir)
    if not ref:
        return None
    o = ref["orbitals"].get(orbital)
    if not o:
        return {"element": m.group(1), "orbital": orbital, "missing": True, "have": sorted(ref["orbitals"])}
    return dict(o, element=m.group(1), orbital=orbital, auger=ref.get("auger"), source=ref["source"], curated=bool(ref.get("curated")))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        d = import_file(p, BUNDLED)
        for el, v in d.items():
            print(el, v["records"], "records;", ", ".join("%s: %d states" % (k, len(o["states"])) for k, o in v["orbitals"].items()))
