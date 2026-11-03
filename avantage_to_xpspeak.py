#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
avantage_to_xpspeak.py
把 Thermo Avantage 的 XPS 谱（原生 .VGD，或导出的 .xlsx/.xls/.csv/.txt） 转成 XPSPEAK 4.1 可导入的两列 ASCII 文件。

用法:
    python avantage_to_xpspeak.py                      # 弹出文件选择框（可多选）
    python avantage_to_xpspeak.py a.xlsx b.csv dir\    # 命令行传入文件 / 文件夹
    python avantage_to_xpspeak.py --hv 1253.6 --ascending data\

输出: 原文件旁的 xpspeak_txt\ 子文件夹，每个谱区一个 <原文件名>_<谱区>.txt，外加 conversion_log.csv
原始强度不做任何处理；原始文件只读不改。
"""
import argparse
import csv
import os
import re
import struct
import sys
from collections import OrderedDict

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    print("缺少 pandas，请先运行:  pip install pandas openpyxl xlrd")
    input("按回车退出...") if sys.stdin and sys.stdin.isatty() else None
    sys.exit(1)

DEFAULT_HV = 1486.6          # Al Kα
MIN_POINTS = 5
OUT_DIRNAME = "xpspeak_txt"
LOG_NAME = "conversion_log.csv"
SUPPORTED = (".xlsx", ".xlsm", ".xls", ".csv", ".txt", ".vgd")
# 常见 X 射线源能量，用于从 .VGD 元数据里认出激发能
KNOWN_SOURCES = (1486.6, 1253.6, 2984.3, 5414.7, 9251.7)

# ---------------------------------------------------------------- 列名识别
RE_BE = re.compile(r"binding\s*energy|^\s*b\.?\s*e\.?\s*($|[\(\[/:])|结合能", re.I)
RE_KE = re.compile(r"kinetic\s*energy|^\s*k\.?\s*e\.?\s*($|[\(\[/:])|动能", re.I)
RE_EXCLUDE = re.compile(r"back\s*g|bkg|backgnd|envelope|residual|\bfit|peak|fwhm|area|"
                        r"position|at\.?\s*%|sensitivity|energy", re.I)
RE_SCAN = re.compile(r"\b(scan|sweep|frame)\s*#?\s*\d+", re.I)
RE_SUM = re.compile(r"\bsum\b|total|summed", re.I)
RE_AVG = re.compile(r"average|\bavg\b|\bmean\b", re.I)
RE_INT = re.compile(r"counts?|\bcps\b|c/s|intensity|强度|计数", re.I)


def to_float(v):
    """单元格 -> float；非数值返回 None。"""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return None if f != f else f  # NaN
    s = str(v).strip().replace("−", "-")
    if not s:
        return None
    if "," in s and "." not in s:      # 欧式小数逗号
        s = s.replace(",", ".")
    try:
        f = float(s)
    except ValueError:
        return None
    return None if f != f else f


def cell_text(v):
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return str(v).strip()


def classify_intensity(name):
    """返回 'sum' / 'main' / 'avg' / 'scan' / None(不是强度列)。"""
    if not name:
        return None
    if RE_SCAN.search(name):
        return "scan"
    if RE_EXCLUDE.search(name):
        return None
    if RE_SUM.search(name):
        return "sum"
    if RE_AVG.search(name):
        return "avg"
    if RE_INT.search(name):
        return "main"
    return None


# ---------------------------------------------------------------- 读文件 -> 表格网格
def read_text_grid(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    text = None
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    else:
        for enc in ("utf-8-sig", "gbk", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
    lines = text.splitlines()
    sample = "\n".join(lines[:200])
    if "\t" in sample:
        splitter = lambda ln: ln.split("\t")
    elif sample.count(";") > sample.count(","):
        splitter = lambda ln: ln.split(";")
    elif "," in sample and path.lower().endswith(".csv"):
        splitter = lambda ln: next(csv.reader([ln]))
    elif "," in sample and re.search(r"\d,\s*-?\d", sample) and not re.search(r"\d,\d+\s+\d", sample):
        splitter = lambda ln: next(csv.reader([ln]))
    else:
        splitter = lambda ln: ln.split()
    return [[c.strip() for c in splitter(ln)] for ln in lines]


def read_tables(path):
    """返回 [(表名, grid)]，grid 为二维 list。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt"):
        return [(os.path.splitext(os.path.basename(path))[0], read_text_grid(path))]
    if ext == ".xls":
        try:
            import xlrd  # noqa: F401
        except ImportError:
            raise RuntimeError("读取 .xls 需要 xlrd：pip install xlrd")
        engine = "xlrd"
    else:
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            raise RuntimeError("读取 .xlsx 需要 openpyxl：pip install openpyxl")
        engine = "openpyxl"
    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object, engine=engine)
    out = []
    for name, df in sheets.items():
        grid = df.where(pd.notna(df), None).values.tolist()
        out.append((str(name), grid))
    return out


# ---------------------------------------------------------------- Avantage 原生 .VGD
def _vgd_streams(path):
    try:
        from xlrd import compdoc
    except ImportError:
        raise RuntimeError("读取 .VGD 需要 xlrd：pip install xlrd")
    with open(path, "rb") as fh:
        raw = fh.read()
    if not raw.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        raise RuntimeError("不是 OLE 复合文档，无法按 .VGD 读取")
    cd = compdoc.CompDoc(raw, logfile=open(os.devnull, "w"))
    names = [e.name for e in cd.dirlist if e.etype == 2]
    get = lambda n: cd.get_named_stream(n) if n in names else None
    return get, names


def _propset(data):
    """解析 OLE 属性集第一节，返回 {id: (类型, 值)}；只取用到的类型。"""
    out = {}
    if not data or len(data) < 48:
        return out
    off = struct.unpack_from("<I", data, 44)[0]
    n = struct.unpack_from("<I", data, off + 4)[0]
    for k in range(min(n, 2000)):
        pid, po = struct.unpack_from("<II", data, off + 8 + 8 * k)
        q = off + po
        if q + 8 > len(data):
            continue
        t = struct.unpack_from("<H", data, q)[0]
        try:
            if t == 4:
                out[pid] = (t, struct.unpack_from("<f", data, q + 4)[0])
            elif t == 5:
                out[pid] = (t, struct.unpack_from("<d", data, q + 4)[0])
            elif t == 0x1E:
                ln = struct.unpack_from("<I", data, q + 4)[0]
                out[pid] = (t, data[q + 8:q + 8 + ln].split(b"\0")[0].decode("mbcs" if os.name == "nt" else "latin-1", "replace"))
            elif t == 0x1F:
                ln = struct.unpack_from("<I", data, q + 4)[0]
                out[pid] = (t, data[q + 8:q + 8 + 2 * ln].decode("utf-16le", "replace").split("\0")[0])
        except (struct.error, UnicodeDecodeError, LookupError):
            continue
    return out


def read_vgd(path, hv_override):
    """返回 (region, [block])。"""
    get, names = _vgd_streams(path)
    data, axes = get("VGData"), get("VGSpaceAxes")
    if not data or not axes:
        raise RuntimeError("缺少 VGData / VGSpaceAxes 数据流")
    ny = len(data) // 8
    # VGSpaceAxes 中找 (uint32 点数, double 起点, double 步长)
    npts = start = step = None
    for i in range(0, len(axes) - 19):
        n = struct.unpack_from("<I", axes, i)[0]
        if n >= MIN_POINTS and ny % n == 0:
            a, b = struct.unpack_from("<dd", axes, i + 4)
            if -1e5 < a < 1e5 and 0 < abs(b) < 100:
                npts, start, step = n, a, b
                break
    if npts is None:
        raise RuntimeError("无法解析能量轴 (VGSpaceAxes)")
    vals = struct.unpack("<%dd" % ny, data[:ny * 8])
    warns = []
    region = ""
    si = _propset(get("\x05SummaryInformation"))
    if 2 in si and si[2][0] in (0x1E, 0x1F):
        region = si[2][1]
    region = clean_region(region or os.path.splitext(os.path.basename(path))[0])

    if hv_override is not None:
        hv, src = hv_override, "命令行 --hv"
    else:
        hv, src = None, ""
        for n in names:
            if n.startswith("\x05"):
                for t, v in _propset(get(n)).values():
                    if t in (4, 5) and any(abs(v - s) < 0.5 for s in KNOWN_SOURCES):
                        hv, src = round(v, 2), "文件元数据"
                        break
            if hv:
                break
        if hv is None:
            hv, src = DEFAULT_HV, "默认值（文件里未找到）"
    warns.append("VGD 能量轴为动能，已按 BE = %s - KE 换算（hν 来自%s）" % (fmt(hv, 4), src))

    nspec = ny // npts
    blocks = []
    for k in range(nspec):
        E = [hv - (start + j * step) for j in range(npts)]
        I = list(vals[k * npts:(k + 1) * npts])
        w = list(warns)
        if nspec > 1:
            w.append("文件含 %d 条谱（如深度剖析各层），这是第 %d 条" % (nspec, k + 1))
        blocks.append({"energy": E, "intensity": I, "warnings": w,
                       "region": region if nspec == 1 else "%s_%d" % (region, k + 1),
                       "source_cols": "VGSpaceAxes(KE %s+%s*i) | VGData" % (fmt(start, 4), fmt(step, 4))})
    return region, blocks


# ---------------------------------------------------------------- 在网格中找谱区
def find_blocks(grid, table_name, hv):
    """返回 list of dict(region, energy, intensity, warnings)。"""
    blocks = []
    nrows = len(grid)
    width = max((len(r) for r in grid), default=0)
    get = lambda r, c: grid[r][c] if c < len(grid[r]) else None

    for r in range(nrows):
        texts = [cell_text(get(r, c)) for c in range(width)]
        ecols = []
        for c, t in enumerate(texts):
            if t and to_float(t) is None:
                if RE_BE.search(t):
                    ecols.append((c, "BE"))
                elif RE_KE.search(t):
                    ecols.append((c, "KE"))
        if not ecols:
            continue
        # 合并：BE 与 KE 相邻（中间没有强度列）时只用 BE
        groups = []
        for c, kind in ecols:
            if groups and all(classify_intensity(texts[k]) is None for k in range(groups[-1][-1][0] + 1, c)):
                groups[-1].append((c, kind))
            else:
                groups.append([(c, kind)])
        for gi, g in enumerate(groups):
            be = [x for x in g if x[1] == "BE"]
            ecol, ekind = be[0] if be else g[0]
            start = g[-1][0] + 1
            end = groups[gi + 1][0][0] if gi + 1 < len(groups) else width
            # 强度列在这一组之后，直到下一组能量列
            cand = OrderedDict()
            for c in range(start, end):
                k = classify_intensity(texts[c])
                if k:
                    cand.setdefault(k, []).append(c)
            if not cand:
                continue
            blk = extract_block(grid, r, ecol, ekind, cand, texts, hv)
            if blk is None:
                continue
            blk["region"] = guess_region(grid, r, g[0][0], end, table_name)
            blk["hdr_row"] = r
            blocks.append(blk)

    if not blocks:
        blk = headerless_block(grid, table_name)
        if blk:
            blocks.append(blk)
    return blocks


def extract_block(grid, hdr_row, ecol, ekind, cand, texts, hv):
    warnings = []
    if "sum" in cand:
        icols, mode = cand["sum"][:1], "sum"
    elif "main" in cand:
        icols, mode = cand["main"][:1], "main"
    elif "avg" in cand:
        icols, mode = cand["avg"][:1], "avg"
    else:
        icols, mode = cand["scan"], "scans"
    get = lambda r, c: grid[r][c] if c < len(grid[r]) else None

    E, I = [], []
    started = False
    for r in range(hdr_row + 1, len(grid)):
        e = to_float(get(r, ecol))
        vals = [to_float(get(r, c)) for c in icols]
        if e is None or any(v is None for v in vals):
            if started and e is None and all(cell_text(get(r, c)) == "" for c in [ecol] + icols):
                # 空行：允许继续，但若后面不是数据就会自然结束
                continue
            if started and e is None:
                break   # 遇到下一段文字（另一个谱区的表头）停止
            continue
        started = True
        E.append(e)
        I.append(sum(vals))
    if len(E) < MIN_POINTS:
        return None
    if mode == "scans":
        if len(icols) > 1:
            warnings.append("未找到总和/平均列，已对 %d 条扫描列求和 (%s)"
                            % (len(icols), ", ".join(texts[c] for c in icols)))
        else:
            warnings.append("只有单条扫描列 (%s)，直接使用" % texts[icols[0]])
    else:
        if "scan" in cand:
            warnings.append("存在 %d 条单独扫描列，已使用 '%s' 列" % (len(cand["scan"]), texts[icols[0]]))
    if ekind == "KE":
        E = [hv - e for e in E]
        warnings.append("只有动能列 '%s'，已按 BE = %.4g - KE 换算" % (texts[ecol], hv))
    return {"energy": E, "intensity": I, "warnings": warnings,
            "source_cols": "%s | %s" % (texts[ecol], " + ".join(texts[c] for c in icols))}


def guess_region(grid, hdr_row, c0, c1, table_name):
    """在表头上方找谱区名（如 'C1s Scan'），找不到用 sheet 名。"""
    get = lambda r, c: grid[r][c] if c < len(grid[r]) else None
    lo = max(-1, hdr_row - 15)
    # 优先：形如 C1s / Ti2p / O 1s / Survey 的名字
    for r in range(hdr_row - 1, lo, -1):
        if to_float(get(r, c0)) is not None:
            break  # 碰到上一个谱区的数据，停止
        for c in range(c0, min(c1, c0 + 3)):
            t = cell_text(get(r, c))
            if re.match(r"^(survey|wide|[A-Z][a-z]?\s*\d[spdf](\d/2)?)", t, re.I) and len(t) <= 40:
                return clean_region(t)
    for r in range(hdr_row - 1, lo, -1):
        if to_float(get(r, c0)) is not None:
            break  # 碰到上一个谱区的数据，停止
        for c in range(c0, min(c1, c0 + 3)):
            t = cell_text(get(r, c))
            if t and to_float(t) is None and not re.search(r"[:=]", t) and len(t) <= 40 \
                    and not RE_BE.search(t) and not RE_KE.search(t):
                return clean_region(t)
    return clean_region(table_name)


def clean_region(name):
    name = re.sub(r"\s*\b(scan|region|spectrum)\s*$", "", str(name).strip(), flags=re.I)
    name = re.sub(r"\s+", "", name)
    return name or "region"


def headerless_block(grid, table_name):
    """没有列名时：取第一段连续的两列数值。"""
    E, I = [], []
    for row in grid:
        nums = [to_float(v) for v in row if cell_text(v) != ""]
        if len(nums) >= 2 and nums[0] is not None and nums[1] is not None:
            E.append(nums[0]); I.append(nums[1])
        elif E:
            break
    if len(E) < MIN_POINTS:
        return None
    return {"energy": E, "intensity": I, "region": clean_region(table_name), "hdr_row": -1,
            "source_cols": "col1 | col2",
            "warnings": ["未找到列名，假定第1列为结合能 (eV)、第2列为强度，请核对"]}


# ---------------------------------------------------------------- 排序 / 去重 / 写出
def finalize(E, I, ascending):
    pts = sorted(zip(E, I), key=lambda p: p[0], reverse=not ascending)
    out, seen, dup = [], set(), 0
    for e, i in pts:
        key = round(e, 6)
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        out.append((e, i))
    return out, dup


def fmt(x, nd):
    s = ("%." + str(nd) + "f") % x
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def write_xpspeak(path, pts):
    with open(path, "w", encoding="ascii", newline="") as f:
        f.write("".join("%s\t%s\r\n" % (fmt(e, 4), fmt(i, 6)) for e, i in pts))


def safe_name(s):
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f\s]+', "_", s).strip("._ ")
    return s or "region"


def step_of(pts):
    if len(pts) < 2:
        return ""
    d = sorted(abs(pts[k + 1][0] - pts[k][0]) for k in range(len(pts) - 1))
    return fmt(d[len(d) // 2], 4)


LOG_FIELDS = ["output_file", "source_file", "sheet", "region", "columns_used", "points",
              "BE_high", "BE_low", "step_eV", "intensity_max", "status", "warnings"]


def convert_file(path, hv, ascending, used_names=None):
    """返回 log 行列表（不抛异常）。hv=None 表示未指定（VGD 用文件内的值）。"""
    rows = []
    base = os.path.splitext(os.path.basename(path))[0]
    outdir = os.path.join(os.path.dirname(os.path.abspath(path)), OUT_DIRNAME)
    is_vgd = path.lower().endswith(".vgd")
    try:
        if is_vgd:
            region, vblocks = read_vgd(path, hv)
            tables = [(region, None)]
            # VGD 文件名本身就是谱区名，用上级文件夹（样品/位置名）作前缀
            base = os.path.basename(os.path.dirname(os.path.abspath(path))) or base
        else:
            tables = read_tables(path)
    except Exception as ex:
        return [dict(source_file=path, status="ERROR", warnings="无法读取: %s" % ex)]
    if hv is None:
        hv = DEFAULT_HV
    used = used_names.setdefault(outdir.lower(), set()) if used_names is not None else set()
    for tname, grid in tables:
        try:
            blocks = vblocks if is_vgd else find_blocks(grid, tname, hv)
        except Exception as ex:
            rows.append(dict(source_file=path, sheet=tname, status="ERROR", warnings="解析失败: %s" % ex))
            continue
        if not blocks:
            if len(tables) == 1:
                rows.append(dict(source_file=path, sheet=tname, status="SKIPPED",
                                 warnings="未识别到能量/强度数据"))
            else:
                rows.append(dict(source_file=path, sheet=tname, status="SKIPPED",
                                 warnings="此 sheet 未识别到谱数据（可能是标题/峰表）"))
            continue
        for b in blocks:
            pts, dup = finalize(b["energy"], b["intensity"], ascending)
            warns = list(b["warnings"])
            if dup:
                warns.append("去掉 %d 个重复能量点" % dup)
            if any(i < 0 for _, i in pts):
                warns.append("存在负强度值（保持原样）")
            name = "%s_%s" % (safe_name(base), safe_name(b["region"]))
            cand, k = name, 2
            while cand.lower() in used:
                cand = "%s_%d" % (name, k); k += 1
            used.add(cand.lower())
            os.makedirs(outdir, exist_ok=True)
            out = os.path.join(outdir, cand + ".txt")
            write_xpspeak(out, pts)
            es = [e for e, _ in pts]
            rows.append(dict(output_file=out, source_file=path, sheet=tname, region=b["region"],
                             columns_used=b["source_cols"], points=len(pts),
                             BE_high=fmt(max(es), 4), BE_low=fmt(min(es), 4), step_eV=step_of(pts),
                             intensity_max=fmt(max(i for _, i in pts), 6), status="OK",
                             warnings="; ".join(warns)))
    return rows


def collect_inputs(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, fs in os.walk(p):
                dirs[:] = [d for d in dirs if d != OUT_DIRNAME]
                for f in sorted(fs):
                    if f.lower().endswith(SUPPORTED) and not f.startswith("~$") and f != LOG_NAME:
                        files.append(os.path.join(root, f))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print("找不到: %s" % p)
    return files


def write_logs(all_rows, inputs):
    by_dir = OrderedDict()
    for row in all_rows:
        d = os.path.join(os.path.dirname(os.path.abspath(row["source_file"])), OUT_DIRNAME)
        by_dir.setdefault(d, []).append(row)
    logs = []
    for d, rows in by_dir.items():
        os.makedirs(d, exist_ok=True)
        lp = os.path.join(d, LOG_NAME)
        with open(lp, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in LOG_FIELDS})
        logs.append(lp)
    return logs


def run(paths, hv=None, ascending=False, quiet=False):
    files = collect_inputs(paths)
    all_rows = []
    used_names = {}
    for f in files:
        rows = convert_file(f, hv, ascending, used_names)
        for r in rows:
            if not quiet:
                tag = r.get("status")
                msg = os.path.basename(r.get("output_file", "")) or r.get("sheet", "")
                print("[%s] %s  %s  %s" % (tag, os.path.basename(f), msg, r.get("warnings", "")))
        all_rows.extend(rows)
    logs = write_logs(all_rows, files)
    return all_rows, logs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Thermo Avantage 导出 -> XPSPEAK 4.1 ASCII")
    ap.add_argument("paths", nargs="*", help="文件或文件夹；留空则弹出选择框")
    ap.add_argument("--hv", type=float, default=None,
                    help="激发能 eV。表格文件默认 1486.6 (Al Kα)；.VGD 默认用文件里记录的值")
    ap.add_argument("--ascending", action="store_true", help="结合能从低到高排序（默认从高到低）")
    if sys.stdout is None:          # 打包成无控制台 exe 时
        sys.stdout = sys.stderr = open(os.devnull, "w")
    a = ap.parse_args(argv)

    frozen = getattr(sys, "frozen", False)
    gui = not a.paths or frozen     # exe：拖文件到图标上也弹结果窗口
    paths = a.paths
    if gui:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            paths = a.paths or list(filedialog.askopenfilenames(
                title="选择 Avantage 导出文件（可多选）",
                filetypes=[("Avantage", "*.vgd *.VGD *.xlsx *.xls *.xlsm *.csv *.txt"), ("All", "*.*")]))
        except Exception as ex:
            print("无法打开文件选择框 (%s)。请在命令行传入文件路径。" % ex)
            return 1
        if not paths:
            print("未选择文件。")
            return 0

    if frozen and not a.paths:
        from tkinter import messagebox
        if not messagebox.askyesno("hν", "结合能按 BE = hν − KE 换算。\n\n"
                                   "是：用文件记录的 hν（K-Alpha+ 为 1486.68 eV）\n"
                                   "否：用 1486.6 eV"):
            a.hv = 1486.6
    try:
        rows, logs = run(paths, a.hv, a.ascending)
    except Exception as ex:
        if gui:
            from tkinter import messagebox
            messagebox.showerror("avantage_to_xpspeak", "出错：%s" % ex)
        raise
    ok = sum(1 for r in rows if r.get("status") == "OK")
    bad = sum(1 for r in rows if r.get("status") != "OK")
    summary = "完成：输出 %d 个谱区文件，%d 项跳过/出错。\n日志：\n%s" % (ok, bad, "\n".join(logs))
    print(summary)
    if gui:
        try:
            from tkinter import messagebox
            messagebox.showinfo("avantage_to_xpspeak", summary)
            if ok and os.name == "nt":
                os.startfile(os.path.dirname(logs[0]))   # 打开输出文件夹
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
