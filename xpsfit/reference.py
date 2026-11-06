# -*- coding: utf-8 -*-
"""推荐峰位参考库：由 NIST XPS Database (SRD 20) 导出的记录生成。

数据来源：NIST X-ray Photoelectron Spectroscopy Database, SRD 20, https://srdata.nist.gov/xps/
NIST 记录的荷电参考各不相同（污染碳、导体、内标等），因此这里给出的是文献分布（中位数、四分位范围、条数），
不是通用的拟合常数。

生成方式：
  - 程序内“导入 NIST 数据”（zip 或 nist_xps_all_records.csv）→ 保存到 ~/XPSAnalyser/reference/（旧版本为 ~/XPSFitStudio/）<元素>.json
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
CURATED = os.path.join(BUNDLED, "curated")    # 文献整理的轨道（如 Fe2p），按轨道覆盖 NIST 数据
MIN_N = 3            # 至少几条记录才作为“推荐化学态”列出
SMALL_SPLIT = 1.0    # 分裂小于此值（eV）的 p 轨道（Al2p、Si2p、Mg2p、P2p），NIST 多数记录未分辨 3/2、1/2，未分辨的记录也计入

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


def effective_formula(r, el):
    """多组分体系（如 Al2O3/Al，金属上的氧化膜）：NIST 的 xps_formula 用 * 标出测的是哪一个组分（Al*2O3/Al → Al2O3，
    Al2O3/Al* → Al），按该组分归类；写成 metal state 的归为单质。其余多组分体系保持原样（不列为推荐化学态）。"""
    f = (r.get("formula") or "?").strip()
    if "/" not in f:
        return f
    xf = (r.get("xps_formula") or "").strip()
    if "*" in xf:
        for comp in xf.split("/"):
            if "*" in comp:
                c = comp.replace("*", "").strip()
                if re.match(r"^(α-|β-|γ-)?%s(?![a-z])" % re.escape(el), c):
                    return c
    if xf.lower() == "metal state":
        return el
    return f


def state_kind(el, formula):
    """按化学式粗分：metal（单质）/ oxide / hydroxide / nitride / carbide / oxynitride / other（混合物、盐、有机物等）。"""
    f = re.sub(r"^(α|β|γ|δ|a|b|g)-", "", (formula or "").replace(" ", ""))
    if f == el:
        return "metal"
    n = r"\d*(\.\d+)?"
    if re.fullmatch(r"%s%s(O%s|Ox)" % (el, n, n), f):
        return "oxide"
    if re.fullmatch(r"%s%s(\(OH\)%s|OOH|O%sH%s|O%s\(OH\)%s)(·?n?H2O)?" % (el, n, n, n, n, n, n), f):
        return "hydroxide"
    if re.fullmatch(r"%s%sN%s" % (el, n, n), f):
        return "nitride"
    if re.fullmatch(r"%s%sC%s" % (el, n, n), f):
        return "carbide"
    if re.fullmatch(r"%s%s(O%sN%s|N%sO%s)" % (el, n, n, n, n, n), f):
        return "oxynitride"
    # 简单无机盐：碳酸盐、卤化物、磷酸盐、硫酸盐、硝酸盐（可含 H、OH、结晶水）——只用于碱金属 / 碱土金属（见 autofit.SALT_ELEMENTS）
    g = re.sub(r"·?n?\d*H2O$", "", f)
    rest = set(re.findall(r"[A-Z][a-z]?", g)) - {el, "H", "O"}
    if el in g and rest and len(rest) == 1 and rest <= {"C", "F", "Cl", "Br", "I", "P", "S", "N"} \
            and re.fullmatch(r"[A-Za-z0-9().·]+", g) and not re.search(r"[a-z]{2,}|\(-|-\)", g):
        if rest <= {"F", "Cl", "Br", "I"} or "O" in g:
            return "salt"
    return "other"


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


def _split_values(dbl, el, key):
    return [x for x in (_num(d.get("separation_eV")) for d in dbl if d.get("element") == el and
                        (d.get("doublet", "").endswith("DS-" + key) or d.get("doublet", "").startswith(key))) if x]


def build(recs, dbl):
    """按元素生成参考数据 {元素: dict}。"""
    out = {}
    for el in sorted({r["element"] for r in recs if r.get("element")}):
        R = [r for r in recs if r.get("element") == el]
        be_rows = [r for r in R if _num(r.get("binding_energy_eV")) is not None and str(r.get("is_auger", "")).lower() != "true"]
        orbitals = {}
        # 每个轨道（如 Al2p）的主线记录：分辨的主线（2p3/2），小分裂轨道再加上未分辨的（2p）
        by_orb = {}
        for r in be_rows:
            m = re.match(r"^(\d)([spdf])(\d/2)?$", r["line"].strip())
            if not m:
                continue
            n_, l_, j_ = m.groups()
            by_orb.setdefault((n_, l_), {"main": [], "unres": []})
            if l_ == "s" or j_ == MAIN_LINE[l_][0]:
                by_orb[(n_, l_)]["main"].append(r)
            elif j_ is None:
                by_orb[(n_, l_)]["unres"].append(r)
        for (n_, l_), g in sorted(by_orb.items()):
            sv = _split_values(dbl, el, n_ + l_) if l_ != "s" else []
            small = l_ != "s" and (sv and statistics.median(sv) < SMALL_SPLIT)
            use = g["main"] + (g["unres"] if small or not g["main"] else [])
            if not use:
                continue
            line = (n_ + l_ + MAIN_LINE[l_][0]) if l_ != "s" else n_ + l_
            if g["unres"] and use is not g["main"] and (small or not g["main"]):
                line = n_ + l_ if not g["main"] else "%s（含未分辨的 %s）" % (line, n_ + l_)
            orb = "%s%s%s" % (el, n_, l_)
            groups = {}
            for r in use:
                groups.setdefault(effective_formula(r, el), []).append(r)
            states = []
            for formula, rs in groups.items():
                v = [_num(r["binding_energy_eV"]) for r in rs]
                if len(v) < MIN_N or not _simple(formula):
                    continue
                q1, q3 = _quart(v)
                states.append({
                    "formula": formula, "kind": state_kind(el, formula),
                    "name": (rs[0].get("chemical_state_name") or "")[:60],
                    "median": round(statistics.median(v), 2), "q1": round(q1, 2), "q3": round(q3, 2),
                    "min": min(v), "max": max(v), "n": len(v),
                    "refs": [r.get("record_url") for r in rs[:3] if r.get("record_url")],
                })
            states.sort(key=lambda s: s["median"])
            # 双峰间距：直接 DS 记录 + 成对计算，取中位数
            split = None
            if sv:
                split = {"median": round(statistics.median(sv), 3), "n": len(sv), "min": min(sv), "max": max(sv)}
            # 卫星峰
            sat = []
            satg = {}
            main_line = (n_ + l_ + MAIN_LINE[l_][0]) if l_ != "s" else n_ + l_
            for r in be_rows:
                if r["line"].startswith(main_line) and "sat" in r["line"]:
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


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load(element, user_dir=None):
    """用户导入的与随程序发布的 NIST 数据：取记录更多的一份（旧版本导入的少量数据不会挡住新发布的完整数据）；
    再用文献整理的轨道（curated/，如 Fe2p）覆盖。"""
    name = "%s.json" % element
    user = _read(os.path.join(user_dir, "reference", name)) if user_dir else None
    bund = _read(os.path.join(BUNDLED, name))
    ref = user if user and (not bund or user.get("records", 0) >= bund.get("records", 0)) else bund
    cur = _read(os.path.join(CURATED, name))
    if cur:
        if not ref:
            return cur
        ref = dict(ref, orbitals=dict(ref["orbitals"]))
        for orb, o in cur["orbitals"].items():
            ref["orbitals"][orb] = dict(o, curated=True, curated_source=cur.get("source"))
    return ref


def available(user_dir=None):
    els = set()
    for d in [os.path.join(user_dir, "reference") if user_dir else None, BUNDLED, CURATED]:
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
    # 化学态类别按当前规则重算（数据文件里存的是生成时的类别，规则更新后旧文件也能用）
    o = dict(o, states=[dict(t, kind=state_kind(m.group(1), t.get("formula"))) for t in o.get("states", [])])
    return dict(o, element=m.group(1), orbital=orbital, auger=ref.get("auger"),
                source=o.get("curated_source") or ref["source"], curated=bool(o.get("curated") or ref.get("curated")))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        d = import_file(p, BUNDLED)
        for el, v in d.items():
            print(el, v["records"], "records;", ", ".join("%s: %d states" % (k, len(o["states"])) for k, o in v["orbitals"].items()))
