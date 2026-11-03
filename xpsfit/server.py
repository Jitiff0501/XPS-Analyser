# -*- coding: utf-8 -*-
"""XPS Fit Studio 后端：本地 HTTP 服务 + JSON API；界面在 web/index.html。
运行：python -m xpsfit.server   （有 pywebview 时开桌面窗口，否则用浏览器打开）"""
import base64
import copy
import json
import os
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
        src = os.path.dirname(s.source) if s.source and not s.source.startswith(tempfile.gettempdir()) else ""
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
                prefix = "" if folder.startswith("xpsfit_up") else folder + " "
                out.append(core.Spectrum(prefix + b["region"], b["energy"], b["intensity"], path, hv))
            return out
        base = os.path.splitext(name)[0]
        for tname, grid in conv.read_tables(path):
            for b in conv.find_blocks(grid, tname, conv.DEFAULT_HV):
                reg = b["region"]
                nm = base if reg == conv.clean_region(tname) else "%s %s" % (base, reg)
                out.append(core.Spectrum(nm, b["energy"], b["intensity"], path))
        return out

    # ---------------------------------------------------------- 视图数据
    def summary(self):
        return [{"name": s.name, "element": s.element, "fitted": bool(s.last_fit), "npeaks": len(s.peaks)}
                for s in self.specs]

    def view(self, i):
        s = self.S(i)
        for p, v in zip(s.peaks, core.resolve(s.peaks)):   # 同步绑定参数，显示一致
            for k in core.PARAMS:
                p[k]["value"] = float(v[k])
        x, y, bg, comps, env = s.compute()
        tot = sum(p["area"]["value"] for p in s.peaks) or 1.0
        peaks = []
        for p in s.peaks:
            q = copy.deepcopy(p)
            q["pct"] = 100.0 * p["area"]["value"] / tot
            q["linked"] = sorted(core.linked_params(p))
            peaks.append(q)
        return {"i": int(i), "name": s.name, "element": s.element, "source": s.source, "hv": s.hv,
                "shift": s.shift, "bg": s.bg, "be": _f(s.be), "y": _f(s.y),
                "x": _f(x), "bgline": _f(bg), "comps": [_f(bg + c) for c in comps],
                "env": _f(bg + env) if s.peaks else [], "resid": _f(y - bg - env) if s.peaks else [],
                "peaks": peaks, "fit": s.last_fit, "region": [float(x.min()), float(x.max())]}


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
    if method == "upload":
        tmp = tempfile.mkdtemp(prefix="xpsfit_up")
        paths = []
        for f in a["files"]:
            p = os.path.join(tmp, os.path.basename(f["name"]))
            with open(p, "wb") as fh:
                fh.write(base64.b64decode(f["b64"]))
            paths.append(p)
        return open_paths(paths)
    if method == "view":
        return st.view(a["i"])
    s = st.S(a["i"]) if "i" in a and a["i"] is not None and st.specs else None

    if method == "rename":
        s.name = a["name"]; s.element = core.guess_orbital(a["name"]) or s.element
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
        same = [t for t in st.specs if os.path.dirname(t.source) == os.path.dirname(s.source)]
        for t in same:
            shift_spec(t, d)
        return dict(st.view(a["i"]), msg="已平移 %+.3f eV（%d 条谱）" % (d, len(same)))
    elif method == "set_bg":
        b = a["bg"]
        s.bg.update({"type": b["type"], "lo": b.get("lo"), "hi": b.get("hi"), "navg": max(1, int(b.get("navg", 3)))})
    elif method == "add_peak":
        x, y, bg, comps, env = s.compute()
        c = float(a["x"]); w = float(a.get("fwhm", 1.2))
        h = max(float(np.interp(c, x, y - bg - env)), 0.05 * float(np.max(y - bg)))
        n = 1
        while any(p["name"] == "P%d" % n for p in s.peaks): n += 1
        s.peaks.append(core.new_peak(a.get("name") or "P%d" % n, c, w, h * w * 1.064))
    elif method == "add_doublet":
        x, y, bg, _, _ = s.compute()
        c, w = float(a["center"]), float(a["fwhm"])
        h = max(float(np.interp(c, x, y - bg)), 1.0)
        s.peaks.extend(core.doublet(a["name"], c, float(a["split"]), float(a["ratio"]), w, h * w * 1.064))
    elif method == "update_peak":
        j, new = int(a["j"]), a["peak"]
        old = s.peaks[j]["name"]
        clean = {"name": new["name"], "link": new.get("link") or {}}
        for k in core.PARAMS:
            clean[k] = {f: new[k][f] for f in ("value", "fixed", "min", "max")}
        s.peaks[j] = clean
        if clean["name"] != old:
            for q in s.peaks:
                if (q.get("link") or {}).get("ref") == old:
                    q["link"]["ref"] = clean["name"]
    elif method == "delete_peak":
        p = s.peaks.pop(int(a["j"]))
        for q in s.peaks:
            if (q.get("link") or {}).get("ref") == p["name"]:
                q["link"] = {}
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
        inc = a.get("include") or [True] * len(st.specs)
        specs = [t for t, k in zip(st.specs, inc) if k]
        rows = core.quantify(specs, st.sf, a.get("use_fit", True), 0.6 if a.get("ecf", True) else 0.0)
        return {"rows": rows, "sf": st.sf}
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


def open_paths(paths):
    added, errors = 0, []
    for p in paths:
        try:
            new = ST.load_path(p)
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
    base = os.path.join(st.out_dir(s), conv.safe_name(s.name))
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
        fmt = a.get("fmt", "png")
        p = base + "_fit." + fmt
        figure(s, p)
        return p
    raise ValueError(kind)


def figure(s, path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
    cols = ["#3B6EA8", "#C4553A", "#4E9A6A", "#8A63B8", "#D08A2E", "#7A5C4F", "#C76B9E", "#3A9AA8"]
    x, y, bg, comps, env = s.compute()
    fig = Figure(figsize=(4.2, 3.6), dpi=600)
    ax = fig.add_subplot(111)
    ax.plot(s.be, s.y, "o", ms=2.6, mfc="none", mec="#333", mew=0.6, label="Raw")
    if s.bg["type"] != "None":
        ax.plot(x, bg, "-", color="#888", lw=0.9, label="Background")
    for i, (p, c) in enumerate(zip(s.peaks, comps)):
        col = cols[i % len(cols)]
        ax.fill_between(x, bg, bg + c, color=col, alpha=0.3, lw=0)
        ax.plot(x, bg + c, color=col, lw=1.0, label=p["name"])
    if s.peaks:
        ax.plot(x, bg + env, color="#C00000", lw=1.3, label="Envelope")
    ax.set_xlim(s.be.max(), s.be.min())
    ax.set_xlabel("Binding Energy (eV)"); ax.set_ylabel("Intensity (CPS)")
    ax.set_title(s.name, fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=600)


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


def main():
    global WINDOW
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
