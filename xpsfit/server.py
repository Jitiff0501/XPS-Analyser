# -*- coding: utf-8 -*-
"""XPS Fit Studio 后端：本地 HTTP 服务 + JSON API；界面在 web/index.html。
运行：python -m xpsfit.server   （有 pywebview 时开桌面窗口，否则用浏览器打开）"""
import base64
import copy
import json
import os
import re
import socket
import sys
import tempfile
import threading
import traceback
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import numpy as np

HERE = getattr(sys, "_MEIPASS", None)
if HERE:
    WEB = os.path.join(HERE, "xpsfit", "web")
else:
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
sys.path.insert(0, HERE)

from xpsfit import core  # noqa: E402
from xpsfit import reference  # noqa: E402
import avantage_to_xpspeak as conv  # noqa: E402

APP = "XPS Fit Studio"
REF_C1S = 284.8
WINDOW = None   # pywebview 窗口（有的话）


def user_dir():
    d = os.path.join(os.path.expanduser("~"), "XPSFitStudio")
    os.makedirs(d, exist_ok=True)
    return d


def _f(v):
    return [float(a) for a in np.asarray(v, float)]


class State:
    def __init__(self):
        self.specs = []
        self.lock = threading.Lock()
        self.sf = core.load_sf(os.path.join(user_dir(), "sensitivity_factors.json")) or core.load_sf()
        self.templates = core.builtin_templates()
        self.templates.update(self._user_templates())

    # ---------------------------------------------------------- 工具
    def _user_templates(self):
        try:
            with open(os.path.join(user_dir(), "templates.json"), encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def S(self, i):
        return self.specs[int(i)]

    def out_dir(self, s):
        src = os.path.dirname(s.source) if s.source and "xpsfit_up" not in s.source else ""
        d = os.path.join(src, "xpsfit_output") if src else os.path.join(user_dir(), "output")
        os.makedirs(d, exist_ok=True)
        return d

    def load_path(self, path, display_name=None):
        name = display_name or os.path.basename(path)
        if path.lower().endswith(".json"):
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            return [core.Spectrum.from_dict(s) for s in d.get("spectra", [])]
        folder = os.path.basename(os.path.dirname(os.path.abspath(path)))
        out = []
        if path.lower().endswith(".vgd"):
            _, blocks = conv.read_vgd(path, None)
            for b in blocks:
                hv = 1486.68
                for w in b["warnings"]:
                    if "BE =" in w:
                        try: hv = float(w.split("BE =")[1].split("-")[0])
                        except ValueError: pass
                out.append(core.Spectrum(b["region"], b["energy"], b["intensity"], path, hv))
            return out
        base = os.path.splitext(name)[0]
        for tname, grid in conv.read_tables(path):
            for b in conv.find_blocks(grid, tname, conv.DEFAULT_HV):
                reg = b["region"]
                if core.guess_orbital(reg):
                    nm = reg
                elif core.guess_orbital(base):
                    nm = conv.clean_region(base)
                else:
                    nm = base if reg == conv.clean_region(tname) else "%s %s" % (base, reg)
                out.append(core.Spectrum(nm, b["energy"], b["intensity"], path))
        return out

    # ---------------------------------------------------------- 视图数据
    def summary(self):
        return [{"name": s.name, "element": s.element, "fitted": bool(s.last_fit), "npeaks": len(s.peaks),
                 "sample": s.sample or "(未分组)"} for s in self.specs]

    def view(self, i):
        s = self.S(i)
        for p, v in zip(s.peaks, core.resolve(s.peaks)):   # 同步绑定参数，显示一致
            for k in core.PARAMS:
                p[k]["value"] = float(v[k])
        x, y, bg, comps, env = s.compute()
        peaks = []
        for p, pct in zip(s.peaks, core.area_percents(s.peaks)):
            q = copy.deepcopy(p)
            q["pct"] = pct
            q.setdefault("role", "main")
            q["linked"] = sorted(core.linked_params(p))
            peaks.append(q)
        return {"i": int(i), "name": s.name, "element": s.element, "sample": s.sample or "(未分组)", "source": s.source, "hv": s.hv,
                "shift": s.shift, "bg": s.bg, "be": _f(s.be), "y": _f(s.y),
                "x": _f(x), "bgline": _f(bg), "comps": [_f(bg + c) for c in comps],
                "env": _f(bg + env) if s.peaks else [], "resid": _f(y - bg - env) if s.peaks else [],
                "peaks": peaks, "fit": s.last_fit, "region": [float(x.min()), float(x.max())],
                "window": [s.window[0] + s.shift, s.window[1] + s.shift]}


ST = State()


# ============================================================== API
def api(method, a):
    st = ST
    if method == "state":
        return {"specs": st.summary(), "templates": list(st.templates), "native": WINDOW is not None}
    if method == "pick_files":
        if WINDOW is None:
            return {"fallback": True}
        import webview
        paths = WINDOW.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=True, file_types=(
            "XPS 数据 (*.vgd;*.VGD;*.txt;*.csv;*.xlsx;*.xls;*.json)", "所有文件 (*.*)")) or []
        return open_paths(list(paths))
    if method == "pick_folder":
        if WINDOW is None:
            return {"fallback": True}
        import webview
        d = WINDOW.create_file_dialog(webview.FOLDER_DIALOG)
        if not d:
            return {"specs": st.summary(), "added": 0, "errors": []}
        return open_paths(folder_files(d[0] if isinstance(d, (list, tuple)) else d))
    if method == "upload":
        tmp = tempfile.mkdtemp(prefix="xpsfit_up")
        paths = []
        rels = [f.get("rel") or f["name"] for f in a["files"]]
        if any("/" in r for r in rels):     # 选择了文件夹：只要 .vgd（若有）
            if any(r.lower().endswith(".vgd") for r in rels):
                a["files"] = [f for f, r in zip(a["files"], rels) if r.lower().endswith(".vgd")]
        a["files"].sort(key=lambda f: (f.get("rel") or f["name"]).lower())
        for f in a["files"]:
            rel = (f.get("rel") or f["name"]).replace("\\", "/")
            parts = [re.sub(r'[<>:"|?*\x00-\x1f]', "_", x).strip() or "_" for x in rel.split("/") if x not in ("", ".", "..")]
            p = os.path.join(tmp, *parts)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(base64.b64decode(f["b64"]))
            paths.append(p)
        return open_paths(paths)
    if method == "view":
        return st.view(a["i"])
    if method == "reference":
        return reference_for(st.S(a["i"]))
    if method == "import_reference":
        paths = []
        if a.get("files"):
            tmp = tempfile.mkdtemp(prefix="xpsfit_ref")
            for f in a["files"]:
                pth = os.path.join(tmp, os.path.basename(f["name"]))
                with open(pth, "wb") as fh:
                    fh.write(base64.b64decode(f["b64"]))
                paths.append(pth)
        elif WINDOW is not None:
            import webview
            paths = list(WINDOW.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=True, file_types=(
                "NIST 导出 (*.zip;*.csv)", "所有文件 (*.*)")) or [])
        else:
            return {"fallback": True}
        done, errors = [], []
        for pth in paths:
            try:
                d = reference.import_file(pth, os.path.join(user_dir(), "reference"))
                done += ["%s（%d 条）" % (el, v["records"]) for el, v in d.items()]
            except Exception as e:
                errors.append("%s：%s" % (os.path.basename(pth), e))
        return {"imported": done, "errors": errors, "available": reference.available(user_dir())}
    s = st.S(a["i"]) if "i" in a and a["i"] is not None and st.specs else None

    if method == "rename":
        old = s.element
        s.name = a["name"]; s.element = core.guess_orbital(a["name"]) or s.element
        if s.element != old:
            s.window = core.default_window(s.element, s.be_raw)
    elif method == "set_element":
        s.element = a["element"]
    elif method == "remove":
        del st.specs[int(a["i"])]
        return {"specs": st.summary()}
    elif method == "set_shift":
        shift_spec(s, float(a["shift"]) - s.shift)
    elif method == "calib_c1s":
        p = s.peaks[int(a["j"])]
        d = REF_C1S - p["center"]["value"]
        same = [t for t in st.specs if t.sample == s.sample]
        for t in same:
            shift_spec(t, d)
        return dict(st.view(a["i"]), msg="已平移 %+.3f eV（%d 条谱）" % (d, len(same)))
    elif method == "set_bg":
        b = a["bg"]
        s.bg.update({"type": b["type"], "lo": b.get("lo"), "hi": b.get("hi"), "navg": max(1, int(b.get("navg", 3)))})
        for k in ("ylo", "yhi"):
            if k in b:
                s.bg[k] = None if b[k] is None else float(b[k])
        if "points" in b:
            s.bg["points"] = sorted([float(px), float(py)] for px, py in (b["points"] or []))
        if "interp" in b:
            s.bg["interp"] = b["interp"]
    elif method == "add_peak":
        add_peak(s, a)
    elif method == "update_peak":
        j, new = int(a["j"]), a["peak"]
        old = s.peaks[j]["name"]
        clean = {"name": new["name"], "link": new.get("link") or {},
                 "role": new.get("role", "main"), "shape": new.get("shape", "GL")}
        core.ensure_peak(new)
        if s.peaks[j].get("pair"):
            clean["pair"] = True
        for k in core.PARAMS:
            clean[k] = {f: new[k][f] for f in ("value", "fixed", "min", "max")}
        s.peaks[j] = clean
        if clean["name"] != old:
            for q in s.peaks:
                if (q.get("link") or {}).get("ref") == old:
                    q["link"]["ref"] = clean["name"]
    elif method == "delete_peak":
        p = s.peaks.pop(int(a["j"]))
        # 双峰副峰随主峰一起删除；卫星峰等解除绑定
        s.peaks = [q for q in s.peaks if not ((q.get("link") or {}).get("ref") == p["name"] and q.get("pair"))]
        for q in s.peaks:
            if (q.get("link") or {}).get("ref") == p["name"]:
                q["link"] = {}
    elif method == "preview_figure":
        import io
        buf = io.BytesIO()
        figure(s, buf, dict(a.get("opts") or {}, fmt="png"), dpi=110)
        return {"png": base64.b64encode(buf.getvalue()).decode()}
    elif method == "equal_param":
        # 多选绑定：members 中的峰与参考峰 j 的 FWHM（或 L%）相等；其余原先绑到参考峰的解除
        ref = s.peaks[int(a["j"])]
        param = a.get("param", "fwhm")
        key = "fwhm_ratio" if param == "fwhm" else "lor_same"
        members = {int(m) for m in a.get("members", [])} - {int(a["j"])}
        n = 0
        for k, q in enumerate(s.peaks):
            if q is ref or q.get("pair"):
                continue
            ln = dict(q.get("link") or {})
            if k in members:
                if ln.get("ref") not in (None, ref["name"]) and any(ln.get(x) is not None for x in core.LINK_KEYS + ("lor_same",)):
                    # 已绑定到别的峰：保留原绑定项，但改参考会破坏它们，因此跳过
                    continue
                ln["ref"] = ref["name"]
                ln[key] = 1.0 if key == "fwhm_ratio" else True
                ln.pop("fwhm_ratio_fit", None)
                q[param]["fixed"] = False
                n += 1
            elif ln.get("ref") == ref["name"] and ln.get(key) is not None:
                ln.pop(key, None); ln.pop(key + "_fit", None)
            q["link"] = ln
        nm = "FWHM" if param == "fwhm" else "L%"
        return dict(st.view(a["i"]), specs=st.summary(), msg="%d 个峰的 %s 与 “%s” 相等" % (n, nm, ref["name"]))
    elif method == "clear_peaks":
        s.peaks, s.last_fit = [], None
    elif method == "apply_template":
        core.apply_template(s, st.templates[a["name"]])
    elif method == "save_template":
        user = st._user_templates(); user[a["name"]] = copy.deepcopy(s.peaks)
        with open(os.path.join(user_dir(), "templates.json"), "w", encoding="utf-8") as f:
            json.dump(user, f, ensure_ascii=False, indent=1)
        st.templates[a["name"]] = user[a["name"]]
        return dict(st.view(a["i"]), templates=list(st.templates), msg="模板已保存")
    elif method == "fit":
        s.do_fit()
    elif method == "fit_same":
        n, bad = 0, []
        for t in st.specs:
            if t is s or t.element != s.element:
                continue
            t.peaks, t.bg = copy.deepcopy(s.peaks), dict(s.bg)
            try:
                t.do_fit(); n += 1
            except Exception as e:
                bad.append("%s: %s" % (t.name, e))
        return dict(st.view(a["i"]), specs=st.summary(), msg="已拟合 %d 条同类谱" % n + ("；失败 %d" % len(bad) if bad else ""))
    elif method == "quant":
        for k, v in (a.get("sf") or {}).items():
            st.sf[k] = float(v)
        if a.get("sf"):
            with open(os.path.join(user_dir(), "sensitivity_factors.json"), "w", encoding="utf-8") as f:
                json.dump({"factors": st.sf}, f, ensure_ascii=False, indent=1)
        inc = a.get("include")
        res = core.quantify_samples(st.specs, st.sf, a.get("use_fit", True),
                                    0.6 if a.get("ecf", True) else 0.0, inc)
        used = sorted({t.element for t in st.specs if t.element})
        return {"samples": res, "sf": st.sf, "orbitals": used}
    elif method.startswith("export_"):
        return {"path": export(method[7:], a), "msg": "已导出"}
    elif method == "save_project":
        p = os.path.join(st.out_dir(st.specs[0]), a.get("name") or "project.xpsfit.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"app": APP, "version": 1, "spectra": [t.to_dict() for t in st.specs]}, f, ensure_ascii=False)
        return {"path": p, "msg": "项目已保存"}
    elif method == "open_folder":
        p = a["path"]
        p = p if os.path.isdir(p) else os.path.dirname(p)
        if os.name == "nt":
            os.startfile(p)
        return {"ok": True}
    else:
        raise ValueError("未知操作 %s" % method)
    return dict(st.view(a["i"]), specs=st.summary())


def reference_for(s):
    """当前谱的推荐峰位：NIST 统计 + 标出与数据中峰位接近的化学态。"""
    ref = reference.for_orbital(s.element, user_dir())
    out = {"orbital": s.element, "available": reference.available(user_dir()), "ref": ref}
    if not ref or ref.get("missing"):
        return out
    # 数据中的峰：扣背景后的平滑曲线上找局部极大
    from scipy.signal import find_peaks
    x, y, bg, _, _ = s.compute()
    sig = core.smooth(y - bg, 7)
    if len(sig) > 10 and np.ptp(sig) > 0:
        k, pr = find_peaks(sig, prominence=0.05 * float(np.ptp(sig)))
        found = [float(x[j]) for j in k]
    else:
        found = []
    out["data_peaks"] = found
    for st_ in ref["states"]:
        st_["near"] = any(abs(st_["median"] - f) <= 0.3 for f in found)
    return out


def folder_files(d):
    """文件夹内（含子文件夹）的数据文件；有 .vgd 时只取 .vgd。"""
    fs = []
    for root, dirs, files in os.walk(d):
        dirs[:] = sorted(x for x in dirs if x not in ("xpsfit_output", conv.OUT_DIRNAME))
        for f in sorted(files):
            if f.lower().endswith((".vgd", ".xlsx", ".xls", ".csv")) and not f.startswith("~$"):
                fs.append(os.path.join(root, f))
    vg = [f for f in fs if f.lower().endswith(".vgd")]
    return vg or fs


def open_paths(paths):
    added, errors = 0, []
    for p in paths:
        try:
            new = ST.load_path(p)
            for t in new:   # 浏览器上传的单个文件：临时目录名没有意义
                if t.sample.startswith("xpsfit_up"):
                    t.sample = "单独打开的文件"
            if not new:
                raise RuntimeError("没有识别到谱数据")
            ST.specs.extend(new); added += len(new)
        except Exception as e:
            errors.append("%s：%s" % (os.path.basename(p), e))
    return {"specs": ST.summary(), "added": added, "errors": errors}


def shift_spec(s, d):
    s.shift += d
    for k in ("lo", "hi"):
        if s.bg.get(k) is not None:
            s.bg[k] += d
    for pt in s.bg.get("points") or []:
        pt[0] += d
    for p in s.peaks:
        p["center"]["value"] += d
        for b in ("min", "max"):
            if p["center"][b] is not None:
                p["center"][b] += d


def curves(s):
    x, y, bg, comps, env = s.compute()
    cols = {"BE_eV": x, "Raw": y, "Background": bg}
    for p, c in zip(s.peaks, comps):
        cols[p["name"]] = bg + c
    if s.peaks:
        cols["Envelope"] = bg + env
        cols["Residual"] = y - bg - env
    return cols


def export(kind, a):
    st = ST
    if kind == "excel":
        import pandas as pd
        p = os.path.join(st.out_dir(st.specs[0]), "fit_results.xlsx")
        rows = []
        for s in st.specs:
            for r in core.peak_table(s):
                r.update(shift_eV=s.shift, background=s.bg["type"], rchi2=s.last_fit["rchi2"] if s.last_fit else None)
                rows.append(r)
        with pd.ExcelWriter(p, engine="openpyxl") as w:
            pd.DataFrame(rows).to_excel(w, sheet_name="Peaks", index=False)
            used = set()
            for s in st.specs:
                nm = conv.safe_name(s.name)[:28]
                k = 2
                while nm.lower() in used:
                    nm = conv.safe_name(s.name)[:26] + "_%d" % k; k += 1
                used.add(nm.lower())
                pd.DataFrame(curves(s)).iloc[::-1].to_excel(w, sheet_name=nm, index=False)
        return p
    s = st.S(a["i"])
    nm = s.name if not s.sample or s.sample.startswith(("单独", "(")) else "%s_%s" % (s.sample, s.name)
    base = os.path.join(st.out_dir(s), conv.safe_name(nm))
    if kind == "csv":
        p = base + "_curves.csv"
        cols = curves(s); keys = list(cols)
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            f.write(",".join(keys) + "\n")
            for i in range(len(cols["BE_eV"]) - 1, -1, -1):
                f.write(",".join("%.6g" % cols[k][i] for k in keys) + "\n")
        return p
    if kind == "txt":
        p = base + ".txt"
        conv.write_xpspeak(p, list(zip(s.be[::-1], s.y[::-1])))
        return p
    if kind == "figure":
        o = a.get("opts") or {}
        fmt = o.get("fmt", "png")
        p = base + "_fit." + fmt
        figure(s, p, o)
        return p
    raise ValueError(kind)


FIG_DEFAULTS = {"fmt": "png", "width_cm": 8.5, "height_cm": 7.0, "dpi": 600, "font": 9,
                "xmin": None, "xmax": None, "xstep": None, "xminor": 1,
                "ynumbers": True, "ylabel": "Intensity (CPS)", "xlabel": "Binding Energy (eV)",
                "legend": True, "legend_loc": "best", "title": True, "background": True, "envelope": True, "raw": "line",
                "fill": True}


def figure(s, path, opts=None, dpi=None):
    """论文图。opts 见 FIG_DEFAULTS：x 范围/主刻度间距/次刻度数、y 轴数字开关等。"""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator, NullLocator
    o = dict(FIG_DEFAULTS); o.update({k: v for k, v in (opts or {}).items() if v is not None and v != ""})
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.size": float(o["font"]),
                                # 西文用 Arial；中文峰名逐字回退到雅黑/黑体等
                                "font.sans-serif": ["Arial", "Helvetica", "Microsoft YaHei", "SimHei",
                                                    "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"],
                                "xtick.direction": "in", "ytick.direction": "in"})
    cols = ["#3B6EA8", "#C4553A", "#4E9A6A", "#8A63B8", "#D08A2E", "#7A5C4F", "#C76B9E", "#3A9AA8"]
    x, y, bg, comps, env = s.compute()
    fig = Figure(figsize=(float(o["width_cm"]) / 2.54, float(o["height_cm"]) / 2.54))
    ax = fig.add_subplot(111)
    if o["raw"] == "line":
        ax.plot(s.be, s.y, "-", color="#222", lw=0.9, label="Raw")
    else:
        ax.plot(s.be, s.y, "o", ms=2.4, mfc="none", mec="#333", mew=0.5, label="Raw")
    if o["background"] and s.bg["type"] != "None":
        ax.plot(x, bg, "-", color="#888", lw=0.8, label="Background")
    for i, (p, c) in enumerate(zip(s.peaks, comps)):
        role = p.get("role", "main")
        col = "#9A9A9A" if role == "unknown" else cols[i % len(cols)]
        ls = "--" if role == "unknown" else (":" if role == "satellite" else "-")
        if o["fill"]:
            ax.fill_between(x, bg, bg + c, color=col, alpha=0.18 if role != "main" else 0.3, lw=0)
        ax.plot(x, bg + c, color=col, lw=0.9, ls=ls, label=p["name"])
    if o["envelope"] and s.peaks:
        ax.plot(x, bg + env, color="#C00000", lw=1.1, label="Envelope")
    xmax = float(o["xmax"]) if o["xmax"] is not None else float(s.be.max())
    xmin = float(o["xmin"]) if o["xmin"] is not None else float(s.be.min())
    ax.set_xlim(max(xmin, xmax), min(xmin, xmax))
    # y 范围按可见 x 区间自动
    m = (s.be >= min(xmin, xmax)) & (s.be <= max(xmin, xmax))
    if m.sum() > 2:
        lo, hi = float(s.y[m].min()), float(s.y[m].max())
        mr = (x >= min(xmin, xmax)) & (x <= max(xmin, xmax))   # 背景/峰也要完整显示
        if mr.any() and (o["background"] or s.peaks):
            lo = min(lo, float(bg[mr].min()))
            if s.peaks and o["envelope"]:
                hi = max(hi, float((bg + env)[mr].max()))
        pad = 0.06 * (hi - lo or 1)
        ax.set_ylim(lo - pad, hi + pad * (3 if o["legend"] and o["legend_loc"] != "outside" else 1.5))
    if o["xstep"]:
        ax.xaxis.set_major_locator(MultipleLocator(float(o["xstep"])))
    n = int(o["xminor"] or 0)
    ax.xaxis.set_minor_locator(AutoMinorLocator(n + 1) if n > 0 else NullLocator())
    if not o["ynumbers"]:
        ax.set_yticks([]); ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax.tick_params(axis="both", which="both", top=False, right=False)
    ax.set_xlabel(o["xlabel"]); ax.set_ylabel(o["ylabel"])
    if o["title"]:
        ax.set_title(s.name, fontsize=float(o["font"]))
    if o["legend"]:
        fs = float(o["font"]) * 0.75
        if o["legend_loc"] == "outside":
            ax.legend(fontsize=fs, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        else:
            ax.legend(fontsize=fs, frameon=False, loc=o["legend_loc"])
    fig.tight_layout()
    fmt = o["fmt"] if o["fmt"] != "tif" else "tiff"
    fig.savefig(path, dpi=dpi or float(o["dpi"]), format=fmt)


def add_peak(s, a):
    """kind: single / doublet / satellite / unknown"""
    x, y, bg, comps, env = s.compute()
    kind = a.get("kind", "single")
    c = float(a.get("center", a.get("x")))
    w = float(a.get("fwhm") or 1.2)
    h = max(float(np.interp(c, x, y - bg - env)), 0.05 * float(np.max(y - bg)), 1.0)
    names = {p["name"] for p in s.peaks}

    def uniq(n):
        b, k = n, 2
        while n in names:
            n = "%s (%d)" % (b, k); k += 1
        return n
    lor = float(a.get("lor") or 30)
    shape = "Asym" if a.get("shape") == "Asym" else "GL"
    n_before = len(s.peaks)
    if kind == "doublet":
        j_lab = {"p": ("3/2", "1/2"), "d": ("5/2", "3/2"), "f": ("7/2", "5/2")}.get(
            (s.element or "p")[-1:], ("3/2", "1/2"))
        if not a.get("split") or float(a["ratio"] or 0) <= 0:
            raise ValueError("双峰需要分裂和面积比")
        base = a.get("name") or s.element or "X"
        k, nm = 2, base
        while ("%s %s" % (nm, j_lab[0])) in names or ("%s %s" % (nm, j_lab[1])) in names:
            nm = "%s (%d)" % (base, k); k += 1
        labels = ("%s %s" % (nm, j_lab[0]), "%s %s" % (nm, j_lab[1]))
        s.peaks.extend(core.doublet(nm, c, float(a["split"]), float(a["ratio"]), w, h * w * 1.064, lor, labels,
                                    fit_split=not a.get("lock_split", True), fit_ratio=not a.get("lock_ratio", True),
                                    shape=shape))
    elif kind == "satellite":
        par = s.peaks[int(a["parent"])]
        dE = float(a.get("dE", c - par["center"]["value"]))
        ratio = float(a["ratio"]) if a.get("bind_area") else None
        p = core.satellite(uniq(a.get("name") or par["name"] + " sat."), par, dE, ratio,
                           bool(a.get("bind_dE", True)), w, lor)
        if ratio is None:
            p["area"]["value"] = h * w * 1.064
        s.peaks.append(p)
    else:
        n = 1
        while "P%d" % n in names: n += 1
        default = "Unknown overlap" if kind == "unknown" else "P%d" % n
        p = core.new_peak(uniq(a.get("name") or default), c, w, h * w * 1.064, lor,
                          role="unknown" if kind == "unknown" else "main", shape=shape)
        if kind == "unknown":   # 未知峰默认限制在添加位置 ±1.5 eV，避免“吃掉”主峰
            p["center"]["min"], p["center"]["max"] = c - 1.5, c + 1.5
        s.peaks.append(p)
    if a.get("cmin") is not None and a.get("cmax") is not None and len(s.peaks) > n_before:
        q = s.peaks[n_before]
        q["center"]["min"], q["center"]["max"] = float(a["cmin"]), float(a["cmax"])
    for p in s.peaks[n_before:]:
        core.bound_center(p)


# ============================================================== HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        name = "index.html" if path in ("/", "") else path.lstrip("/")
        f = os.path.normpath(os.path.join(WEB, name))
        if not f.startswith(os.path.normpath(WEB)) or not os.path.isfile(f):
            return self._send(404, b"not found", "text/plain")
        ctype = {"html": "text/html; charset=utf-8", "js": "application/javascript", "css": "text/css",
                 "svg": "image/svg+xml"}.get(f.rsplit(".", 1)[-1], "application/octet-stream")
        with open(f, "rb") as fh:
            self._send(200, fh.read(), ctype)

    def do_POST(self):
        if not self.path.startswith("/api/"):
            return self._send(404, b"", "text/plain")
        n = int(self.headers.get("Content-Length") or 0)
        try:
            args = json.loads(self.rfile.read(n) or b"{}")
            with ST.lock:
                res = api(self.path[5:], args)
            body, code = json.dumps(res, ensure_ascii=False, allow_nan=False, default=str), 200
        except Exception as e:
            traceback.print_exc()
            body, code = json.dumps({"error": str(e)}, ensure_ascii=False), 400
        self._send(code, body.encode("utf-8"), "application/json; charset=utf-8")


def serve(port=0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d/" % srv.server_address[1]


def smoke(out):
    """CI 自检：不开窗口，验证打包后的网页资源、读取、拟合、导出都可用。"""
    import urllib.request
    srv, url = serve()
    page = urllib.request.urlopen(url).read()
    js = urllib.request.urlopen(url + "plotly.min.js").read()
    assert b"XPS Fit Studio" in page and len(js) > 1000000
    lines = ["page ok", "templates %d" % len(ST.templates), "sf %d" % len(ST.sf),
             "reference %s" % ",".join(reference.available())]
    assert "Cu" in reference.available(), "bundled NIST reference missing"
    vgd = os.environ.get("XPSFIT_SMOKE_VGD")
    if vgd:
        r = open_paths([vgd]); assert r["added"] == 1, r
        api("apply_template", {"i": 0, "name": "Al2p: Al 金属 + Al2O3"})
        v = api("fit", {"i": 0}); assert v["fit"]["success"]
        ST.specs[0].source = os.path.join(os.path.dirname(out), "x.vgd")
        for k in ("figure", "excel"):
            lines.append("export %s %s" % (k, os.path.getsize(api("export_" + k, {"i": 0})["path"])))
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\nOK\n")


def main():
    global WINDOW
    if os.environ.get("XPSFIT_SMOKE"):
        try:
            smoke(os.environ["XPSFIT_SMOKE"])
        except Exception:
            with open(os.environ["XPSFIT_SMOKE"], "w") as f:
                f.write(traceback.format_exc())
        return
    files = [a for a in sys.argv[1:] if os.path.exists(a)]
    if files:
        open_paths(files)
    srv, url = serve()
    try:
        import webview
        WINDOW = webview.create_window(APP, url, width=1440, height=900, min_size=(1100, 700),
                                       background_color="#FAF9F5")
        webview.start()
    except Exception as e:
        print("桌面窗口不可用（%s），改用浏览器：%s" % (e, url))
        import webbrowser, time
        webbrowser.open(url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
