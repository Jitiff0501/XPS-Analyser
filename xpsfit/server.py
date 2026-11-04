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
from xpsfit import autofit  # noqa: E402
from xpsfit import identify  # noqa: E402
from xpsfit import i18n  # noqa: E402
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
        self.stack_annots = {}                    # 堆叠对比图的标注：{轨道: [标注…]}
        self.stack_labels = {}                    # 堆叠对比图样品名的上下位置：{轨道: {样品名: dy}}（相对该谱的基线，单位为谱高）
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
            self.stack_annots.update(d.get("stack_annots") or {})
            self.stack_labels.update(d.get("stack_labels") or {})
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
                "window": [s.window[0] + s.shift, s.window[1] + s.shift],
                "kind": getattr(s, "kind", "region"), "ident": getattr(s, "ident", None),
                "states": _states(s), "interp": _interp(s), "colors": core.peak_colors(s.peaks),
                "relax": autofit.relax_state(s),
                "annots": getattr(s, "annots", []), "stack_annots": self.stack_annots.get(s.element, []),
                "so": {"metal": core.so_rule(s.element, True), "comp": core.so_rule(s.element, False)}
                if core.so_rule(s.element) else None}


ST = State()


def _settings_path():
    return os.path.join(user_dir(), "settings.json")


def load_lang():
    try:
        with open(_settings_path(), encoding="utf-8") as f:
            return json.load(f).get("lang", "zh")
    except Exception:
        return "zh"


ST.lang = load_lang()
def _interp(s):
    try:
        return autofit.interpret(s)
    except Exception:
        return None


_LOC_KEYS = {"interp", "msg", "steps", "warnings", "errors", "error", "imported", "message"}


def localize(obj, key=None):
    """英文界面时翻译返回给前端的消息（报告、提示、警告）。"""
    if ST.lang != "en":
        return obj
    if isinstance(obj, dict):
        return {k: localize(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [localize(v, key) for v in obj]
    if isinstance(obj, str) and key in _LOC_KEYS:
        return i18n.tr(obj)
    return obj


# ============================================================== API
def api(method, a):
    st = ST
    if method == "identify":
        t = st.S(a["i"]); auto_identify(t)
        return dict(st.view(a["i"]), specs=st.summary())
    if method == "i18n":
        return dict(i18n.table(), lang=ST.lang)
    if method == "set_lang":
        ST.lang = "en" if a.get("lang") == "en" else "zh"
        try:
            d = {}
            if os.path.exists(_settings_path()):
                with open(_settings_path(), encoding="utf-8") as f:
                    d = json.load(f)
            d["lang"] = ST.lang
            with open(_settings_path(), "w", encoding="utf-8") as f:
                json.dump(d, f)
        except Exception:
            pass
        return {"lang": ST.lang}
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
        if s.peaks[j].get("color"):
            clean["color"] = s.peaks[j]["color"]
        for k in core.PARAMS:
            clean[k] = {f: new[k][f] for f in ("value", "fixed", "min", "max")}
        s.peaks[j] = clean
        if clean["name"] != old:
            for q in s.peaks:
                if (q.get("link") or {}).get("ref") == old:
                    q["link"]["ref"] = clean["name"]
    elif method == "set_color":
        # 颜色作用于整组：存到主峰上，组内其他峰的自定义颜色清除；color 为空 = 恢复自动配色
        # all=True（堆叠对比）：同一轨道的其他谱中同名的峰组也改成这个颜色
        c = a.get("color")
        name = s.peaks[core.group_root(s.peaks, int(a["j"]))]["name"]
        for t in ([u for u in st.specs if u.element == s.element] if a.get("all") else [s]):
            ks = [k for k, q in enumerate(t.peaks) if q["name"] == name]
            if not ks:
                continue
            j = core.group_root(t.peaks, ks[0])
            for k, q in enumerate(t.peaks):
                if core.group_root(t.peaks, k) == j:
                    q.pop("color", None)
            if c and re.fullmatch(r"#[0-9a-fA-F]{6}", c):
                t.peaks[j]["color"] = c
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
        o = dict(a.get("opts") or {}, fmt="png")
        geo = figure(s, buf, o, dpi=_preview_dpi(o.get("width_cm") or FIG_DEFAULTS["width_cm"], a.get("px")), annots=False)
        return {"png": base64.b64encode(buf.getvalue()).decode(), "geo": geo}
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
        s.do_fit(); _post_fit(s)
    elif method == "autofit":
        targets = [s] if not a.get("all") else [t for t in st.specs if t.sample == s.sample]
        reports = []
        for t in targets:
            if getattr(t, "kind", "") == "survey":
                auto_identify(t)
                el_ = t.ident.get("elements", []) if t.ident else []
                steps = ["全谱：不做峰拟合，识别元素"] + [
                    "%s %s：%s" % (e["el"], "（确认）" if e["confirmed"] else "（待确认：只有部分谱线吻合）",
                                  "、".join("%s %.1f eV" % (n, b) for n, b in e["lines"])) for e in el_]
                reports.append({"spectrum": t.name, "sample": t.sample, "elements": [e["el"] for e in el_], "steps": steps})
                continue
            els = sample_elements(t, a.get("extra"))
            try:
                rep_ = autofit.run(t, els, reference.for_orbital(t.element, user_dir()), oxide=a.get("oxide", "auto"))
                _post_fit(t)
            except Exception as e:
                rep_ = ["出错：%s" % e]
            reports.append({"spectrum": t.name, "sample": t.sample, "elements": els, "steps": rep_})
        if a.get("all"):                          # 整个样品：各元素氧化程度是否符合热力学顺序
            for t_, msg in autofit.sample_consistency([t for t in targets if getattr(t, "kind", "") != "survey"]):
                for rp in reports:
                    if rp["spectrum"] == t_.name and rp["sample"] == t_.sample:
                        rp["steps"].append(msg)
        return dict(st.view(a["i"]), specs=st.summary(), report=reports)
    elif method == "refine":
        autofit.refine(s, float(a.get("window", 0.3))); _post_fit(s)
        return dict(st.view(a["i"]), specs=st.summary(), msg="局部微调完成（峰位 ±%.1f eV）" % float(a.get("window", 0.3)))
    elif method == "multistart":
        before = s.last_fit["rchi2"] if s.last_fit else None
        r = autofit.multistart(s, int(a.get("n", 12))); _post_fit(s)
        msg = "多起点拟合完成：约化 χ² %.4g" % r["rchi2"] + ("（之前 %.4g）" % before if before else "")
        return dict(st.view(a["i"]), specs=st.summary(), msg=msg)
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
    elif method == "relax":
        act_ = a.get("action", "next")
        res = {"next": autofit.relax_next, "undo": autofit.relax_undo, "auto": autofit.relax_auto}[act_](s)
        if s.last_fit:
            _post_fit(s)                          # 不确定度 + 质量警告，与“拟合”按钮一致
        return dict(st.view(a["i"]), specs=st.summary(), msg=res.get("msg"))
    elif method == "set_annots":
        items = clean_annots(a.get("items") or [])
        if a.get("ctx") == "stack":
            st.stack_annots[s.element] = items
        else:
            s.annots = items
        return {"ok": True, "n": len(items)}
    elif method == "preview_stack":
        import io
        buf = io.BytesIO()
        o = dict(a, fmt="png")
        o["dpi"] = _preview_dpi(o.get("width_cm") or STACK_DEFAULTS["width_cm"], a.get("px"))
        geo = export_stack(a["i"], o, out=buf, annots=False)
        return {"png": base64.b64encode(buf.getvalue()).decode(), "geo": geo}
    elif method == "set_stack_label":
        # 堆叠对比图样品名只能上下移动：dy = 标签基线相对该谱基线的高度（单位为谱高）；dy 为空 = 恢复自动位置
        lab = st.stack_labels.setdefault(s.element, {})
        if a.get("dy") is None:
            lab.pop(str(a["key"]), None)
        else:
            lab[str(a["key"])] = min(max(float(a["dy"]), -2.0), 5.0)
        return {"ok": True, "label_dy": lab}
    elif method == "stack":
        return stack_data(a["i"], a)
    elif method == "export_stack":
        p = export_stack(a["i"], a)
        return {"path": p, "msg": "已导出"}
    elif method.startswith("export_"):
        return {"path": export(method[7:], a), "msg": "已导出"}
    elif method == "save_project":
        p = os.path.join(st.out_dir(st.specs[0]), a.get("name") or "project.xpsfit.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"app": APP, "version": 1, "spectra": [t.to_dict() for t in st.specs],
                       "stack_annots": st.stack_annots, "stack_labels": st.stack_labels}, f, ensure_ascii=False)
        return {"path": p, "msg": "项目已保存"}
    elif method == "open_folder":
        p = a["path"]
        p = p if os.path.isdir(p) else os.path.dirname(p)
        open_in_file_manager(p)
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


def open_in_file_manager(p):
    """在系统文件管理器中打开文件夹（Windows 资源管理器 / macOS Finder / Linux）。"""
    import subprocess
    if os.name == "nt":
        os.startfile(p)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", p])
    else:
        subprocess.Popen(["xdg-open", p])


def _states(s):
    try:
        st = core.chemical_states(s)
    except Exception:
        return []
    tot = sum(a for _, a in st) or 1.0
    cols = core.peak_colors(s.peaks)
    col = {}
    for j, p in enumerate(s.peaks):           # 化学态 → 其主峰的颜色
        col.setdefault(core.state_name(s.peaks[core.group_root(s.peaks, j)]["name"]), cols[j])
    return [[n, a / tot, col.get(n, core.UNKNOWN_COLOR)] for n, a in st]


def auto_identify(t):
    """根据谱图识别：全谱 → 元素列表；文件名认不出轨道的高分辨谱 → 识别轨道。"""
    t.ident = None
    if identify.is_survey(t.be_raw):
        t.kind = "survey"
        t.element = ""
        t.bg.update({"type": "SNIP", "lo": None, "hi": None})   # 全谱：SNIP 背景（保留非弹性台阶，不高于数据）
        try:
            t.ident = identify.identify_survey(t.be_raw, t.y, t.hv)
        except Exception:
            t.ident = None
        return
    t.kind = "region"
    cands = identify.identify_region(t.be_raw, t.y)
    t.ident = {"candidates": cands[:4]}
    if not t.element and cands:
        t.element = cands[0][0]
        t.ident["auto"] = True
        if not core.guess_orbital(t.name):
            t.name = "%s（自动识别）" % t.element
        t.window = core.default_window(t.element, t.be_raw)
        if t.window != [float(t.be_raw.min()), float(t.be_raw.max())]:
            t.bg["lo"], t.bg["hi"] = t.window


def sample_elements(s, extra=None):
    """同一样品里有哪些元素（由已打开的谱推断），可附加用户输入的元素。"""
    els = set()
    for t in ST.specs:
        if t.sample == s.sample:
            m = re.match(r"[A-Z][a-z]?", t.element or "")
            if m:
                els.add(m.group(0))
            if getattr(t, "kind", "") == "survey" and t.ident:      # 全谱识别出的元素（已确认的）
                els.update(e["el"] for e in t.ident.get("elements", []) if e.get("confirmed"))
    for e in re.split(r"[,\s，、]+", extra or ""):
        if re.fullmatch(r"[A-Z][a-z]?", e.strip()):
            els.add(e.strip())
    return sorted(els)


def _post_fit(s):
    try:
        autofit.uncertainties(s)
    except Exception:
        pass
    q = autofit.quality(s)
    if s.last_fit is not None:
        s.last_fit["warnings"] = q["warnings"]
        s.last_fit["rel_rms"] = q["rel_rms"]


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
                auto_identify(t)
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
                "legend": False, "legend_loc": "best", "title": True, "labels": "legend", "background": True, "envelope": True, "raw": "line",
                "fill": True}


def _preview_dpi(width_cm, px):
    """预览图的 dpi：让图片宽度约为页面上显示宽度（已乘屏幕像素比），清晰且不过大。"""
    try:
        w_in = max(float(width_cm), 1.0) / 2.54
        return min(max(float(px or 900) / w_in, 50.0), 300.0)
    except (TypeError, ValueError):
        return 110.0


def _geo(fig, ax):
    """坐标轴在图中的位置（图宽高为 pt；ax 为 [x0, y0, x1, y1] 图分数，y 向上）与 x 范围，供页面在预览上叠加标注。"""
    w, h = fig.get_size_inches()
    b = ax.get_position()
    return {"w": float(w) * 72.0, "h": float(h) * 72.0, "ax": [float(b.x0), float(b.y0), float(b.x1), float(b.y1)],
            "xlim": [float(v) for v in ax.get_xlim()]}


def pretty_label(name):
    """峰名转论文写法：Al0 → Al⁰，Co2+ → Co²⁺，2p 3/2 → 2p₃/₂（mathtext）。"""
    t = str(name)
    t = re.sub(r"\b([A-Z][a-z]?)(\d)\+", r"\1$^{\2+}$", t)
    t = re.sub(r"\b([A-Z][a-z]?)0(?=[\s/]|$)", r"\1$^{0}$", t)
    t = re.sub(r"\b([A-Z][a-z]?)\+(?=[\s/]|$)", r"\1$^{+}$", t)
    t = re.sub(r"\s?(\d)/2\b", r"$_{\1/2}$", t)
    t = t.replace("sat.", "sat.").replace("$$", "")
    return t


def _arrow_labels(ax, fig, x, bg, comps, env, peaks, cols, o, ylim, yraw=None):
    """在峰顶画箭头 + 名称。用实际渲染的文字边框检测重叠，重叠就向上错开；返回需要的 y 上限。
    分裂很小（< 1 eV，如 Al2p）的双峰只标一次（合并后的峰顶，名称去掉 3/2）。"""
    import numpy as np
    big = max((float(np.max(c)) for c in comps), default=1.0) or 1.0
    names = {p["name"]: k for k, p in enumerate(peaks)}
    merged = {}                                  # 主峰 -> 合并后的曲线
    small, parts = set(), set()
    for k, p in enumerate(peaks):
        ln = p.get("link") or {}
        R = ln.get("ref")
        if not (p.get("pair") and R in names):
            continue
        if abs(ln.get("dE") or 9) < 1.0:         # 分裂很小的双峰：合并，名称去掉 3/2
            small.add(names[R])
        elif not p["name"].startswith(R + " "):  # 多重分裂分量（名称 = 主线名 + " m"）也并入主线；普通 1/2 分量单独标
            continue
        merged.setdefault(names[R], comps[names[R]].copy())
        merged[names[R]] += comps[k]
        parts.add(k)
    items = []
    for k, (p, c) in enumerate(zip(peaks, comps)):
        if k in parts:
            continue
        if p["name"].endswith("（续）"):           # 同一重叠来源的附属片段不单独标
            continue
        cc = merged.get(k, c)
        if float(np.max(cc)) < 0.03 * big:
            continue                             # 很小的峰不标
        kk = int(np.argmax(cc))
        name = re.sub(r"\s*\d/2\s*$", "", p["name"]) if k in small else p["name"]
        items.append((float(x[kk]), float(bg[kk] + cc[kk]), pretty_label(name), k))
    fs = float(o["font"]) * 0.8
    renderer = fig.canvas.get_renderer()
    envtop = bg + env
    if yraw is not None:                         # 标签也要避开（含噪声的）原始数据
        envtop = np.maximum(envtop, np.asarray(yraw, float))
    y0, y1 = ylim
    top = y1 + 0.42 * (y1 - y0)                  # 预留标签空间
    ax.set_ylim(y0, top)
    from matplotlib.text import Text
    T = ax.transData
    axbb = ax.get_window_extent(renderer)
    placed = []

    def curve_px(xd0, xd1):
        a, b = sorted((xd0, xd1))
        m = (x >= a - 0.15) & (x <= b + 0.15)
        v = float(envtop[m].max()) if m.any() else float(envtop.max())
        return T.transform((x[0], v))[1]

    for xa, ya, lab, k in sorted(items, key=lambda t: -t[1]):
        col = cols[k]
        ann = ax.annotate(lab, xy=(xa, ya), xytext=(xa, ya), textcoords="data", ha="center", va="bottom",
                          fontsize=fs, color=col, annotation_clip=False,
                          arrowprops=dict(arrowstyle="-|>", color=col, lw=0.6, shrinkA=1, shrinkB=1.5, mutation_scale=6))
        apx = T.transform((xa, ya))
        bb0 = Text.get_window_extent(ann, renderer)
        w, h = bb0.width, bb0.height
        best, best_score = None, None
        for lvl in range(0, 60):
            ypx = apx[1] + 10 + lvl * h * 0.55
            if ypx + h > axbb.y1 - 2:
                break
            for dxf in (0, 0.6, -0.6, 1.1, -1.1, 1.7, -1.7):
                cx = apx[0] + dxf * w
                bb = type(bb0).from_bounds(cx - w / 2, ypx, w, h).expanded(1.08, 1.15)
                if bb.x0 < axbb.x0 + 2 or bb.x1 > axbb.x1 - 2:
                    continue
                inv = T.inverted()
                xd0, xd1 = inv.transform((bb.x0, 0))[0], inv.transform((bb.x1, 0))[0]
                clash = sum(1 for q in placed if bb.overlaps(q)) + (1 if bb.y0 < curve_px(xd0, xd1) + 3 else 0)
                score = clash * 1000 + lvl * 3 + abs(dxf) * 4
                if best_score is None or score < best_score:
                    best, best_score = (cx, ypx, bb), score
                if clash == 0:
                    break
            if best_score is not None and best_score < 1000:
                break
        if best is None:
            best = (apx[0], apx[1] + 10, bb0)
        cx, ypx, bb = best
        ann.xyann = tuple(T.inverted().transform((cx, ypx)))
        placed.append(bb)
    return top


# 西文用 Arial（Windows / macOS 自带）；没有 Arial 的系统用度量相同的 Liberation Sans / Arimo；中文逐字回退到雅黑 / 黑体等
FONT_SANS = ["Arial", "Helvetica", "Liberation Sans", "Arimo", "Microsoft YaHei", "SimHei", "PingFang SC",
             "Hiragino Sans GB", "Arial Unicode MS", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
_LATIN = []


def _latin_font():
    """FONT_SANS 中本机第一个可用的西文字体（数学文本 $…$ 也用它，与正文同一字体）。"""
    if not _LATIN:
        from matplotlib import font_manager as fm
        names = {f.name for f in fm.fontManager.ttflist}
        _LATIN.append(next((n for n in FONT_SANS[:4] if n in names), "DejaVu Sans"))
    return _LATIN[0]


def _font_rc(size):
    f = _latin_font()
    return {"font.family": "sans-serif", "font.size": float(size), "font.sans-serif": FONT_SANS,
            "xtick.direction": "in", "ytick.direction": "in",
            "mathtext.fontset": "custom", "mathtext.rm": f, "mathtext.it": f + ":italic", "mathtext.bf": f + ":bold",
            "mathtext.sf": f, "mathtext.default": "regular"}


def _inner_title(ax, text, size):
    """图名放在坐标框内左上角，左对齐；离框边 6 pt（避开朝内的刻度线）。"""
    from matplotlib.transforms import offset_copy
    ax.text(0, 1, text, transform=offset_copy(ax.transAxes, fig=ax.figure, x=6, y=-5, units="points"),
            ha="left", va="top", fontsize=size, zorder=7)


def figure(s, path, opts=None, dpi=None, annots=True):
    """论文图。opts 见 FIG_DEFAULTS：x 范围/主刻度间距/次刻度数、y 轴数字开关等。返回坐标轴位置（_geo）。
    annots=False 时不画标注（导出预览由页面叠加标注，便于直接在预览上编辑）。"""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator, NullLocator
    o = dict(FIG_DEFAULTS); o.update({k: v for k, v in (opts or {}).items() if v is not None and v != ""})
    matplotlib.rcParams.update(_font_rc(o["font"]))
    cols = core.peak_colors(s.peaks)          # 同组同色，含自定义颜色
    x, y, bg, comps, env = s.compute()
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(float(o["width_cm"]) / 2.54, float(o["height_cm"]) / 2.54))
    FigureCanvasAgg(fig)                      # 需要渲染器来测量文字边框（箭头标注防重叠）
    ax = fig.add_subplot(111)
    if o["raw"] == "line":
        ax.plot(s.be, s.y, "-", color="#222", lw=0.9, label="Raw")
    else:
        ax.plot(s.be, s.y, "o", ms=2.4, mfc="none", mec="#333", mew=0.5, label="Raw")
    if o["background"] and s.bg["type"] != "None":
        ax.plot(x, bg, "-", color="#888", lw=0.8, label="Background")
    for i, (p, c) in enumerate(zip(s.peaks, comps)):
        role = p.get("role", "main")
        col = cols[i]
        ls = "--" if role == "unknown" else (":" if role == "satellite" else "-")
        if o["fill"]:
            ax.fill_between(x, bg, bg + c, color=col, alpha=0.18 if role != "main" else 0.3, lw=0)
        ax.plot(x, bg + c, color=col, lw=0.9, ls=ls, label=None if o["labels"] == "arrows" else p["name"])
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
        ax.set_ylim(lo - pad, hi + pad * (3 if (o["legend"] and o["legend_loc"] != "outside") or o["title"] else 1.5))  # 图名在框内左上角，留出空间
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
        _inner_title(ax, s.name, float(o["font"]))
    if o["legend"]:
        fs = float(o["font"]) * 0.75
        if o["legend_loc"] == "outside":
            ax.legend(fontsize=fs, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        else:
            ax.legend(fontsize=fs, frameon=False, loc=o["legend_loc"])
    fig.tight_layout()
    if o["labels"] == "arrows" and s.peaks:
        # 先排版（含标题、轴标签），再放标签；之后不再排版，避免标签与峰错位
        yl = ax.get_ylim()
        top = _arrow_labels(ax, fig, x, bg, comps, env, s.peaks, cols, o, yl, yraw=y)
        ax.set_ylim(yl[0], max(yl[1], top))
    if annots:
        draw_annots(ax, getattr(s, "annots", []))
    fmt = o["fmt"] if o["fmt"] != "tif" else "tiff"
    fig.savefig(path, dpi=dpi or float(o["dpi"]), format=fmt)
    return _geo(fig, ax)


# ------------------------------------------------------------------ 标注（单谱图与堆叠对比图通用）
# 在“导出”对话框的预览图上添加、拖动；导出时画进图里。
# 坐标：x 为 BE（eV），y 为坐标轴高度 %（0 = 坐标轴底部）。尺寸（字号、线宽、虚线段长 / 间隔、箭头大小）都是导出图里的 pt。
# 文字：{type:"text", text, x, y, size, color, bold}；上标写 ^{…}，下标写 _{…}
# 垂直线：{type:"vline", x, len(%), width, dash, gap, color}（底端固定在坐标轴底部；dash=0 为实线）
# 箭头：{type:"arrow", x0, y0（尾）, x1, y1（头）, width, head(filled/open/none), size（箭头长）, hw（箭头宽）, color}
ANN_DEFAULTS = {"text": {"text": "", "x": 0.0, "y": 90.0, "size": 8.0, "color": "#333333", "bold": False},
                "vline": {"x": 0.0, "len": 85.0, "width": 0.7, "dash": 3.0, "gap": 2.0, "color": "#666666"},
                "arrow": {"x0": 0.0, "y0": 80.0, "x1": 0.0, "y1": 60.0, "width": 1.5, "head": "filled", "size": 7.0,
                          "hw": 5.0, "color": "#333333"}}
ARROW_HEADS = ("filled", "open", "none")


def clean_annots(items):
    out = []
    for it in items[:200]:
        t = it.get("type")
        if t not in ANN_DEFAULTS:
            continue
        d = dict(ANN_DEFAULTS[t], type=t)
        for k, v in ANN_DEFAULTS[t].items():
            if k not in it:
                continue
            if isinstance(v, bool):
                d[k] = bool(it[k])
            elif isinstance(v, float):
                try:
                    d[k] = float(it[k])
                except (TypeError, ValueError):
                    pass
            elif k == "color":
                d[k] = it[k] if re.fullmatch(r"#[0-9a-fA-F]{6}", str(it[k])) else v
            elif k == "head":
                d[k] = it[k] if it[k] in ARROW_HEADS else v
            else:
                d[k] = str(it[k])[:200]
        for k, lo, hi in (("y", -20, 120), ("y0", -20, 120), ("y1", -20, 120), ("len", 0, 100), ("width", 0.1, 10),
                          ("dash", 0, 40), ("gap", 0, 40), ("hw", 0.5, 60)):
            if k in d:
                d[k] = min(max(d[k], lo), hi)
        if "size" in d:
            d["size"] = min(max(d["size"], 2.0), 120.0)
        out.append(d)
    return out


# 上下标：字号为正文的 65%，上标基线抬高 0.34 倍字号，下标降低 0.14 倍字号（与 Word / Origin 接近）。
# 文字的锚点是大写字母高度的中点（基线在锚点下方 0.36 倍字号）。页面预览用同样的数值。
SUP_SCALE, SUP_RISE, SUB_DROP, BASE_DROP = 0.65, 0.34, 0.14, 0.36


def ann_segments(t):
    """Co^{2+} 2p_{3/2} → [("Co", 0), ("2+", 1), (" 2p", 0), ("3/2", -1)]（1 上标，-1 下标）。"""
    out, pos = [], 0
    for m in re.finditer(r"(\^|_)\{([^{}]*)\}", str(t)):
        if m.start() > pos:
            out.append((t[pos:m.start()], 0))
        out.append((m.group(2), 1 if m.group(1) == "^" else -1))
        pos = m.end()
    if pos < len(t):
        out.append((t[pos:], 0))
    return [(x, k) for x, k in out if x]


def _rich_text(ax, tr, x, y, text, size, color, bold, ha="center", anchor="mid", x_off=0.0, draw=True):
    """逐段画文字（同一字体），上下标手动排版。ha：center 居中于 x / left 从 x 开始（再右移 x_off pt）；
    anchor：mid 大写字母高度的中点在 y / baseline 基线在 y。返回总宽度（pt）；draw=False 只量宽度。"""
    from matplotlib.transforms import offset_copy
    fig = ax.figure
    r = fig.canvas.get_renderer()
    k = fig.dpi / 72.0
    arts = []
    for seg, kind in ann_segments(text):
        t = ax.text(x, y, seg.replace("$", r"\$"), transform=tr, fontsize=size * (SUP_SCALE if kind else 1.0),
                    color=color, fontweight="bold" if bold else "normal", ha="left", va="baseline", zorder=6, clip_on=False)
        arts.append((t, kind, t.get_window_extent(r).width / k))
    W = sum(w for _, _, w in arts)
    dx = x_off - (W / 2.0 if ha == "center" else 0.0)
    base = -BASE_DROP * size if anchor == "mid" else 0.0
    for t, kind, w in arts:
        if not draw:
            t.remove()
            continue
        dy = base + (SUP_RISE * size if kind == 1 else -SUB_DROP * size if kind == -1 else 0.0)
        t.set_transform(offset_copy(tr, fig=fig, x=dx, y=dy, units="points"))
        dx += w
    return W


def _caret(t):
    """pretty_label 的数学文本写法 → 标注写法：Co$^{2+}$ → Co^{2+}，$_{3/2}$ → _{3/2}。"""
    t = re.sub(r"\$\^\{([^{}]*)\}\$", r"^{\1}", str(t))
    return re.sub(r"\$_\{([^{}]*)\}\$", r"_{\1}", t).replace("$", "")


def _arrow(ax, tr, a):
    """箭头：箭杆 + 实心三角（无描边）/ 开口 / 无。箭头长 size、宽 hw（pt）。
    几何在当前排版下换算到（数据 x，坐标轴 y）坐标，与导出 dpi 无关。"""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon
    fig = ax.figure
    k = fig.dpi / 72.0
    T = np.asarray(tr.transform((a["x0"], a["y0"] / 100.0)))
    H = np.asarray(tr.transform((a["x1"], a["y1"] / 100.0)))
    L = float(np.hypot(*(H - T)))
    if L < 1e-9:
        return
    u = (H - T) / L
    nv = np.array([-u[1], u[0]])
    hl, hw = a["size"] * k, a["hw"] * k / 2.0
    base = H - u * hl
    inv = tr.inverted()
    pts = lambda *P: inv.transform(np.array(P))
    lw = a["width"]
    if a["head"] == "filled":
        end = H - u * min(hl * 0.5, L)                  # 箭杆伸进箭头一半，被实心三角盖住，不留缝
        ax.add_patch(Polygon(pts(H, base + nv * hw, base - nv * hw), closed=True, transform=tr, facecolor=a["color"],
                             edgecolor="none", lw=0, zorder=6, clip_on=False))
    else:
        end = H
        if a["head"] == "open":
            q = pts(base + nv * hw, H, base - nv * hw)
            ax.add_line(Line2D(q[:, 0], q[:, 1], transform=tr, color=a["color"], lw=lw, solid_joinstyle="miter",
                               solid_capstyle="butt", zorder=6, clip_on=False))
    q = pts(T, end)
    ax.add_line(Line2D(q[:, 0], q[:, 1], transform=tr, color=a["color"], lw=lw, solid_capstyle="butt", zorder=6, clip_on=False))


def draw_annots(ax, items):
    """尺寸都是 pt。在排版（tight_layout、y 范围）全部完成后调用。"""
    from matplotlib.transforms import blended_transform_factory
    if not items:
        return
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for a in items:
        if a["type"] == "vline":
            ls = (0, (a["dash"] / a["width"], max(a["gap"], 0.01) / a["width"])) if a["dash"] > 0 else "-"
            ax.axvline(a["x"], ymin=0, ymax=a["len"] / 100.0, color=a["color"], lw=a["width"], linestyle=ls,
                       dash_capstyle="butt", zorder=5)
        elif a["type"] == "arrow":
            _arrow(ax, tr, a)
        elif a.get("text"):
            _rich_text(ax, tr, a["x"], a["y"] / 100.0, a["text"], a["size"], a["color"], a["bold"])


# ------------------------------------------------------------------ 多样品同一轨道堆叠对比
STACK_DEFAULTS = {"mode": "bgsub", "offset": 1.15, "fit": True, "comps": False, "members": None, "states": False, "names": True,
                  "fmt": "png", "width_cm": 8.5, "height_cm": None, "dpi": 600, "font": 9, "xmin": None, "xmax": None}


def stack_data(i, a):
    """当前谱所属轨道在各样品中的谱，按 mode 变换：raw 原始；norm 最小–最大归一化；bgsub 扣背景后按峰高归一化。"""
    o = dict(STACK_DEFAULTS); o.update({k: v for k, v in (a or {}).items() if k in STACK_DEFAULTS and v is not None})
    el = ST.S(i).element
    allk = [k for k, t in enumerate(ST.specs) if t.element == el and getattr(t, "kind", "region") != "survey"]
    sel = set(int(m) for m in o["members"]) if o["members"] is not None else set(allk)
    items = []
    for k in allk:
        if k not in sel:
            continue
        t = ST.specs[k]
        x, y, bg, comps, env = t.compute()
        if o["mode"] == "raw":
            off, sc, yy, base = 0.0, 1.0, y, bg
        elif o["mode"] == "norm":
            off, sc = float(np.min(y)), float(np.ptp(y)) or 1.0
            yy, base = y, bg
        else:
            off, sc = 0.0, float(np.max(y - bg)) or 1.0
            yy, base = y - bg, np.zeros_like(bg)
        f = lambda v: _f((np.asarray(v, float) - off) / sc)
        fitted = bool(t.peaks)
        items.append({"i": k, "sample": t.sample or "", "name": t.name, "key": t.sample or t.name, "x": _f(x), "y": f(yy),
                      "bg": f(base) if o["mode"] != "bgsub" else None,
                      "env": f(base + env) if fitted else None,
                      "comps": [f(base + c) for c in comps] if fitted and o["comps"] else [],
                      "colors": core.peak_colors(t.peaks), "roles": [p.get("role", "main") for p in t.peaks],
                      "states": _states(t) if fitted else []})
    return {"element": el, "mode": o["mode"], "items": items, "label_dy": ST.stack_labels.get(el, {}),
            "all": [{"i": k, "sample": ST.specs[k].sample or "", "name": ST.specs[k].name} for k in allk]}


def _stack_label(it, with_states=True):
    lab = it["sample"] or it["name"]
    if with_states and it["states"]:
        lab += "  " + " · ".join("%s %.0f%%" % (re.sub(r"\s*\d[spdf]$", "", core.state_name(n)), 100 * f)
                                 for n, f, *_ in it["states"] if f >= 0.005)
    return lab


def export_stack(i, a, out=None, annots=True):
    """堆叠对比论文图。out 给定（文件对象）时写入它并返回坐标轴位置（预览用），否则保存到输出目录并返回路径。"""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    o = dict(STACK_DEFAULTS); o.update({k: v for k, v in (a or {}).items() if k in STACK_DEFAULTS and v is not None and v != ""})
    d = stack_data(i, o)
    items = d["items"]
    if not items:
        raise ValueError("没有可堆叠的谱")
    n = len(items)
    matplotlib.rcParams.update(_font_rc(o["font"]))
    hcm = float(o["height_cm"] or (4.0 + 2.3 * n))
    fig = Figure(figsize=(float(o["width_cm"]) / 2.54, hcm / 2.54))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    span = 1.0 if d["mode"] != "raw" else max(max(it["y"]) - min(it["y"]) for it in items)
    step = float(o["offset"]) * span
    xs = [v for it in items for v in it["x"]]
    xmin = float(o["xmin"]) if o["xmin"] is not None else min(xs)
    xmax = float(o["xmax"]) if o["xmax"] is not None else max(xs)
    labels = []
    for j, it in enumerate(items):                 # 第一个样品在最上面
        sh = (n - 1 - j) * step
        x = np.asarray(it["x"])
        if it["comps"]:
            base = np.asarray(it["bg"]) if it["bg"] is not None else np.zeros_like(x)
            for c, col, role in zip(it["comps"], it["colors"], it["roles"]):
                ax.fill_between(x, base + sh, np.asarray(c) + sh, color=col, alpha=0.25 if role == "main" else 0.15, lw=0)
                ax.plot(x, np.asarray(c) + sh, color=col, lw=0.6, ls=":" if role == "satellite" else "-")
        ax.plot(x, np.asarray(it["y"]) + sh, "-", color="#222", lw=0.8)
        if o["fit"] and it["env"] is not None:
            ax.plot(x, np.asarray(it["env"]) + sh, "-", color="#C00000", lw=0.9)
        labels.append((j, sh, x, np.maximum(np.asarray(it["y"]), np.asarray(it["env"]) if it["env"] is not None else -np.inf)))
    ax.set_xlim(xmax, xmin)
    ax.set_xlabel("Binding Energy (eV)")
    ax.set_ylabel("Intensity (a.u.)" if d["mode"] != "raw" else "Intensity (CPS)")
    if d["mode"] != "raw":
        ax.set_yticks([])
    _inner_title(ax, d["element"], float(o["font"]))
    top = (n - 1) * step + 1.25 * span
    ax.set_ylim(ax.get_ylim()[0], max(ax.get_ylim()[1], top))
    fig.tight_layout()
    # 样品名：与框内左上角的图名左对齐（离左框 6 pt）；默认放在这段曲线最高点之上，可在导出预览里上下拖动（stack_labels）
    from matplotlib.transforms import blended_transform_factory
    trl = blended_transform_factory(ax.transAxes, ax.transData)
    k = fig.dpi / 72.0
    fs = float(o["font"]) * 0.8
    over = ST.stack_labels.get(d["element"], {})
    lab_geo = []
    for j, sh, x, yy in (labels if o["names"] else []):
        txt = _caret(pretty_label(_stack_label(items[j], o["states"])))
        W = _rich_text(ax, trl, 0, sh, txt, fs, "black", False, ha="left", anchor="baseline", x_off=6, draw=False)
        x0px = ax.transAxes.transform((0, 0))[0] + 6 * k
        inv = ax.transData.inverted()
        lo_x, hi_x = sorted((inv.transform((x0px, 0))[0], inv.transform((x0px + W * k, 0))[0]))
        m = (x >= lo_x) & (x <= hi_x)
        ymax = float(np.max(yy[m])) if m.any() else 0.0
        auto = max(ymax, 0.0) / span + 0.06
        key = items[j]["key"]
        dy = float(over.get(key, auto))
        if annots:
            _rich_text(ax, trl, 0, sh + dy * span, txt, fs, "black", False, ha="left", anchor="baseline", x_off=6)
        lab_geo.append({"key": key, "text": txt, "y": sh + dy * span, "sh": sh, "span": span, "size": fs, "auto": auto,
                        "moved": key in over})
    if annots:
        draw_annots(ax, ST.stack_annots.get(d["element"], []))
    fmt = o["fmt"] if o["fmt"] != "tif" else "tiff"
    if out is not None:
        fig.savefig(out, dpi=float(o["dpi"]), format=fmt)
        return dict(_geo(fig, ax), ylim=[float(v) for v in ax.get_ylim()], labels=lab_geo)
    s0 = ST.S(i)
    p = os.path.join(ST.out_dir(s0), conv.safe_name("stack_%s_%d-samples" % (d["element"], n)) + "." + o["fmt"])
    fig.savefig(p, dpi=float(o["dpi"]), format=fmt)
    return p


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
        while any(n.startswith(nm + " ") for n in names):
            nm = "%s (%d)" % (base, k); k += 1
        sub = re.search(r"\d[spdf]$", s.element or "")
        sub = sub.group(0) if sub and sub.group(0) not in nm else ""      # 名称里没有能级时补上，如 Cr0 → Cr0 2p3/2
        labels = ("%s %s%s" % (nm, sub, j_lab[0]), "%s %s%s" % (nm, sub, j_lab[1]))
        pk = core.doublet(nm, c, float(a["split"]), float(a["ratio"]), w, h * w * 1.064, lor, labels,
                          fit_split=not a.get("lock_split", True), fit_ratio=not a.get("lock_ratio", True), shape=shape)
        rule = core.so_rule(s.element, shape == "Asym" or core.is_metal_name(nm))
        if rule:                                  # 不锁定时在文献范围内拟合；FWHM 倍率总按文献范围
            ln = pk[1]["link"]
            if rule["lock"]:                      # 分裂远小于峰宽（Al2p、Si2p）：按惯例锁定在常用值
                for k in ("dE_fit", "area_ratio_fit"):
                    ln.pop(k, None)
            lo, hi = rule["dE_rng"]
            if ln.get("dE_fit") and lo - 0.3 <= ln["dE"] <= hi + 0.3:
                ln.update(dE_min=min(lo, ln["dE"]), dE_max=max(hi, ln["dE"]))
            lo, hi = rule["ratio_rng"]
            if ln.get("area_ratio_fit") and hi > lo:
                ln.update(area_ratio_min=min(lo, ln["area_ratio"]), area_ratio_max=max(hi, ln["area_ratio"]))
            lo, hi = rule["fwr_rng"]
            if hi > lo:
                ln.update(fwhm_ratio=(lo + hi) / 2, fwhm_ratio_fit=True, fwhm_ratio_min=lo, fwhm_ratio_max=hi)
                pk[1]["fwhm"]["value"] = w * (lo + hi) / 2
        s.peaks.extend(pk)
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
                res = localize(api(self.path[5:], args))
            body, code = json.dumps(res, ensure_ascii=False, allow_nan=False, default=str), 200
        except Exception as e:
            traceback.print_exc()
            body, code = json.dumps(localize({"error": str(e)}), ensure_ascii=False), 400
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
    tab = api("i18n", {})
    assert i18n.tr("暂无拟合结果") == "No fit yet" and len(tab["patterns"]) > 50
    lines.append("i18n %d phrases %d patterns" % (len(tab["phrases"]), len(tab["patterns"])))
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
