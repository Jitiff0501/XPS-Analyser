# -*- coding: utf-8 -*-
import copy, os, unittest
import numpy as np
from xpsfit import core


def synth(seed=0):
    x = np.arange(65, 85.001, 0.1)
    rng = np.random.default_rng(seed)
    truth = core.doublet("Al0 2p", 72.7, 0.44, 0.5, 0.9, 3000, 20) + core.doublet("Al2O3 2p", 74.6, 0.44, 0.5, 1.7, 9000, 20)
    _, env = core.model(x, truth)
    # 真实 Shirley：高 BE 一侧台阶正比于低 BE 侧的峰面积
    cum = np.concatenate([[0], np.cumsum(0.5 * (env[1:] + env[:-1]) * np.diff(x))])
    y = 1800 + 0.08 * cum + env
    return x, rng.poisson(y).astype(float), truth


class Core(unittest.TestCase):
    def test_peak_area_normalised(self):
        x = np.linspace(-50, 50, 200001)
        for lor in (0, 30, 100):
            a = np.trapezoid(core.gl_peak(x, 0, 1.3, 500, lor), x)
            self.assertAlmostEqual(a, 500, delta=500 * (0.02 if lor else 1e-6))

    def test_shirley_endpoints(self):
        x, y, _ = synth()
        bg = core.background(x, y, "Shirley", 5)
        self.assertAlmostEqual(bg[0], y[:5].mean(), places=6)
        self.assertAlmostEqual(bg[-1], y[-5:].mean(), places=6)
        self.assertLess(np.max(-np.diff(bg)), 0.02 * (bg[-1] - bg[0]))  # 噪声下近似单调
        for k in ("Linear", "Tougaard", "None"):
            self.assertEqual(core.background(x, y, k).shape, y.shape)

    def test_fit_recovers_doublets(self):
        x, y, truth = synth()
        s = core.Spectrum("Al2p Scan", x, y)
        self.assertEqual(s.element, "Al2p")
        tpl = core.builtin_templates()["Al2p: Al 金属 + Al2O3"]
        for p in tpl:                       # 故意给偏的初值
            p["center"]["value"] += 0.3
        core.apply_template(s, tpl)
        r = s.do_fit()
        self.assertTrue(r["success"])
        got = {p["name"]: p for p in s.peaks}
        self.assertAlmostEqual(got["Al0 2p 3/2"]["center"]["value"], 72.7, delta=0.05)
        self.assertAlmostEqual(got["Al2O3 2p 3/2"]["center"]["value"], 74.6, delta=0.05)
        # 双峰约束保持
        self.assertAlmostEqual(got["Al2O3 2p 1/2"]["center"]["value"] - got["Al2O3 2p 3/2"]["center"]["value"], 0.44, places=6)
        self.assertAlmostEqual(got["Al2O3 2p 1/2"]["area"]["value"] / got["Al2O3 2p 3/2"]["area"]["value"], 0.5, places=6)
        ratio = got["Al0 2p 3/2"]["area"]["value"] / got["Al2O3 2p 3/2"]["area"]["value"]
        self.assertAlmostEqual(ratio, 3000 / 9000, delta=0.05)
        self.assertLess(r["rchi2"], 2.0)

    def test_fixed_param_not_changed(self):
        x, y, _ = synth()
        s = core.Spectrum("Al2p", x, y)
        core.apply_template(s, core.builtin_templates()["Al2p: 单一 Al2O3"])
        s.peaks[0]["center"]["value"] = 74.0; s.peaks[0]["center"]["fixed"] = True
        s.do_fit()
        self.assertEqual(s.peaks[0]["center"]["value"], 74.0)
        self.assertAlmostEqual(s.peaks[1]["center"]["value"], 74.44, places=6)

    def test_roundtrip_and_quant(self):
        x, y, _ = synth()
        s = core.Spectrum("Al2p Scan", x, y)
        core.apply_template(s, core.builtin_templates()["Al2p: 单一 Al2O3"])
        s.do_fit()
        s2 = core.Spectrum.from_dict(copy.deepcopy(s.to_dict()))
        self.assertEqual(len(s2.peaks), 2)
        o = core.Spectrum("O1s Scan", x + 460, y * 2)
        sf = core.load_sf()
        self.assertEqual(sf["Al2p"], 0.537)
        q = core.quantify([s, o], sf, use_fit=False)
        self.assertAlmostEqual(sum(r["atomic_percent"] for r in q), 100.0, places=6)

    def test_real_vgd(self):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_data", "Al2p Scan.VGD")
        import avantage_to_xpspeak as conv
        _, blocks = conv.read_vgd(p, None)
        s = core.Spectrum("Al2p", blocks[0]["energy"], blocks[0]["intensity"])
        core.apply_template(s, core.builtin_templates()["Al2p: Al 金属 + Al2O3"])
        r = s.do_fit()
        self.assertTrue(r["success"])
        print("\nreal VGD fit:", [(p["name"], round(p["center"]["value"], 2), round(p["fwhm"]["value"], 2), round(p["area"]["value"])) for p in s.peaks], "rchi2=%.2f" % r["rchi2"])

    def test_doublet_free_split(self):
        """双峰间距/比例不锁定时在范围内拟合。真值 0.44/0.5，初值 0.6/0.7。"""
        x, y, _ = synth()
        s = core.Spectrum("Al2p", x, y)
        pk = core.doublet("Al0 2p", 72.7, 0.6, 0.7, 0.9, lor=20, fit_split=True, fit_ratio=True) + \
             core.doublet("Al2O3 2p", 74.6, 0.44, 0.5, 1.7, lor=20)
        pk[1]["link"]["dE_min"], pk[1]["link"]["dE_max"] = 0.3, 0.8
        core.apply_template(s, pk)
        r = s.do_fit()
        ln = s.peaks[1]["link"]
        self.assertTrue(0.3 <= ln["dE"] <= 0.8)
        self.assertAlmostEqual(ln["dE"], 0.44, delta=0.12)
        self.assertNotEqual(ln["area_ratio"], 0.7)
        self.assertAlmostEqual(s.peaks[1]["center"]["value"] - s.peaks[0]["center"]["value"], ln["dE"], places=9)
        self.assertEqual(s.peaks[3]["link"]["dE"], 0.44)   # 锁定的不变

    def test_samples_states_ratio(self):
        x, y, _ = synth()
        specs = []
        for sample in ("S1", "S2"):
            a = core.Spectrum("Al2p Scan", x, y, os.path.join("/data", sample, "Al2p Scan.VGD"))
            core.apply_template(a, core.builtin_templates()["Al2p: Al 金属 + Al2O3"]); a.do_fit()
            o = core.Spectrum("O1s Scan", x + 460, y * (1 if sample == "S1" else 2), os.path.join("/data", sample, "O1s Scan.VGD"))
            o.peaks = [core.new_peak("O lattice", 531, 1.7, 1000)]; o.do_fit()
            specs += [a, o]
        self.assertEqual(specs[0].sample, "S1")
        st = dict(core.chemical_states(specs[0]))
        self.assertEqual(set(st), {"Al0 2p", "Al2O3 2p"})       # 双峰合并成化学态
        res = core.quantify_samples(specs, core.load_sf())
        self.assertEqual([r["sample"] for r in res], ["S1", "S2"])
        for r in res:
            self.assertAlmostEqual(sum(r["elements"].values()), 100, places=6)
            al = sum(v for k, v in r["states"].items() if k.startswith("Al2p"))
            self.assertAlmostEqual(al, r["elements"]["Al2p"], places=6)
        r1, r2 = (r["elements"]["O1s"] / r["elements"]["Al2p"] for r in res)
        self.assertGreater(r2, r1 * 1.5)                          # S2 的 O 信号是 2 倍

    def test_smart_and_manual_endpoints(self):
        """两峰间数据低于端点（仪器本底下倾）：Shirley 会高于数据，Smart 不会。"""
        x = np.arange(927, 963.01, 0.1)
        pk = [core.new_peak("a", 932.4, 1.2, 60000, 30), core.new_peak("b", 952.2, 1.4, 25000, 30)]
        dip = -5000 * np.exp(-((x - 943) / 6) ** 2)
        y = 128000 + core.model(x, pk)[1] + dip + 2000 * (x > 955)
        sh = core.background(x, y, "Shirley")
        sm = core.background(x, y, "Smart")
        self.assertGreater(np.max(sh - y), 1000)
        self.assertLessEqual(np.max(sm - core._smart_ceiling(x, y)), 1e-6)     # 不高于（约 0.8 eV 平滑后的）数据
        m = core.background(x, y, "Shirley", ylo=120000, yhi=125000)
        self.assertAlmostEqual(m[0], 120000); self.assertAlmostEqual(m[-1], 125000)

    def test_points_background(self):
        x = np.arange(927, 963.01, 0.1)
        y = np.full_like(x, 200000.0)
        pts = [[928, 128000], [949, 125500], [938, 125000], [962, 130000]]
        low = core.background(x, np.full_like(x, 1000.0), "Points", points=pts)
        self.assertLessEqual(low.max(), 1000.0 + 1e-6)          # 不高于数据
        bg = core.background(x, y, "Points", points=pts)
        for px, py in pts:
            self.assertAlmostEqual(float(np.interp(px, x, bg)), py, delta=1.0)
        self.assertLessEqual(bg.max(), 130000 + 1e-6)          # PCHIP 不过冲
        self.assertGreaterEqual(bg[(x >= 928) & (x <= 962)].min(), 125000 - 1e-6)
        lin = core.background(x, y, "Points", points=pts, interp="linear")
        self.assertAlmostEqual(float(np.interp(943.5, x, lin)), (125000 + 125500) / 2, delta=1)
        s = core.Spectrum("Cu2p", x, y); s.bg = {"type": "Points", "points": pts, "lo": None, "hi": None, "navg": 3}
        xr = s.be[s.region()]
        self.assertAlmostEqual(xr.min(), 928, delta=0.05); self.assertAlmostEqual(xr.max(), 962, delta=0.05)

    def test_shirley_robust_when_low_end_higher(self):
        """低 BE 端落在相邻峰的上升沿（低端高于高端、中间数据低于背景）：背景必须保持在两端点之间。"""
        x = np.arange(68, 80.01, 0.1)
        y = 16000 + 12000 * np.exp(-((x - 75) / 1.0) ** 2) + 9000 * np.exp(-((x - 66.5) / 1.5) ** 2) - 2500 * np.exp(-((x - 71) / 1.2) ** 2)
        for kind in ("Shirley", "Smart"):
            bg = core.background(x, y, kind)
            lo, hi = sorted((bg[0], bg[-1]))
            self.assertGreaterEqual(bg.min(), min(lo, y.min()) - 1e-6, kind)
            self.assertLessEqual(bg.max(), hi + 1e-6, kind)

    def test_asymmetric_peak(self):
        """不对称峰：面积归一、FWHM 默认 1–3、拟合能回收拖尾参数；双峰副峰跟随峰型。"""
        x = np.linspace(-60, 100, 160001)
        v = dict(center=0, fwhm=1.2, area=500, lor=30, tail=0.5, tau=2.0, shape="Asym")
        self.assertAlmostEqual(np.trapezoid(core.peak_curve(x, v), x), 500, delta=1.0)
        p = core.new_peak("a", 10, 0.5)
        self.assertEqual((p["fwhm"]["min"], p["fwhm"]["max"], p["fwhm"]["value"]), (1.0, 3.0, 1.0))
        self.assertEqual(core.new_peak("s", 10, 6, role="satellite")["fwhm"]["max"], 8.0)
        xe = np.arange(770, 790.01, 0.05)
        truth = [core.new_peak("m", 778.2, 1.1, 40000, 20, shape="Asym", tail=0.4, tau=2.5)]
        y = core.model(xe, truth)[1] + 1.0
        s = core.Spectrum("Co2p", xe, y); s.bg = {"type": "None", "lo": None, "hi": None, "navg": 3}
        s.peaks = [core.new_peak("m", 778.0, 1.5, 1000, 20, shape="Asym", tail=0.1, tau=1.0)]
        r = s.do_fit()
        q = s.peaks[0]
        self.assertAlmostEqual(q["center"]["value"], 778.2, delta=0.02)
        self.assertAlmostEqual(q["tail"]["value"], 0.4, delta=0.05)
        self.assertAlmostEqual(q["tau"]["value"], 2.5, delta=0.3)
        self.assertAlmostEqual(q["area"]["value"], 40000, delta=800)
        d = core.doublet("Co0 2p", 778.2, 15.0, 0.5, 1.1, shape="Asym")
        d[0]["tail"]["value"] = 0.6
        vals = core.resolve(d)
        self.assertEqual(vals[1]["shape"], "Asym"); self.assertEqual(vals[1]["tail"], 0.6)
        old = {"name": "o", "link": {}, "center": {"value": 1, "fixed": False, "min": None, "max": None},
               "fwhm": {"value": 1, "fixed": False, "min": None, "max": None}, "area": {"value": 1, "fixed": False, "min": 0, "max": None},
               "lor": {"value": 30, "fixed": True, "min": 0, "max": 100}}
        core.model(np.linspace(0, 2, 50), [old])          # 旧项目里没有 tail/tau 的峰也能用
        self.assertIn("tail", old)

    def test_reference_cu(self):
        from xpsfit import reference
        r = reference.for_orbital("Cu2p")
        self.assertEqual(r["split"]["median"], 19.8)
        st = {x["formula"]: x for x in r["states"]}
        self.assertAlmostEqual(st["Cu"]["median"], 932.61, places=2)
        self.assertAlmostEqual(st["CuO"]["median"], 933.8, places=2)
        a = {x["formula"]: x for x in r["auger"]["rows"]}
        self.assertAlmostEqual(a["Cu2O"]["alpha"], 1849.2, places=1)    # 修正俄歇参数
        self.assertIsNone(reference.for_orbital("W4f"))                 # 未收录的元素

    def test_reference_nist_v11(self):
        """v1.1 随程序发布 23 种元素的 NIST 数据；小分裂轨道计入未分辨记录；金属上的氧化膜按 * 标记的组分归类；Fe2p 用文献整理的数据。"""
        from xpsfit import reference
        self.assertGreaterEqual(len(reference.available()), 23)
        al = {x["formula"]: x for x in reference.for_orbital("Al2p")["states"]}
        self.assertIn("Al2O3", al)                                      # 来自 Al2O3/Al 记录（xps_formula 标 Al*2O3）
        self.assertEqual(al["Al"]["kind"], "metal")
        self.assertGreater(al["Al"]["n"], 40)                           # 含未分辨的 2p 记录
        ti = {x["formula"]: x for x in reference.for_orbital("Ti2p")["states"]}
        self.assertEqual(ti["TiO2"]["kind"], "oxide")
        self.assertAlmostEqual(ti["TiO2"]["median"], 458.6, places=1)
        fe = reference.for_orbital("Fe2p")
        self.assertTrue(fe["curated"])                                  # 文献整理的 Fe2p 覆盖 NIST
        self.assertFalse(reference.for_orbital("Fe3p")["curated"])

    def test_autofit_classifies_overlap_and_satellite(self):
        from xpsfit import autofit
        x = np.arange(925, 965.01, 0.1)
        cu = core.doublet("Cu", 932.6, 19.8, 0.5, 1.2, 60000, 30)
        cr_auger = [core.new_peak("auger", 957.9, 5.0, 40000, 0)]
        y = 125000 + core.model(x, cu)[1] + core.model(x, cr_auger)[1]
        y = np.random.default_rng(2).poisson(y).astype(float)
        s = core.Spectrum("Cu2p Scan", x, y)
        rep = autofit.run(s, ["Cu", "Cr", "O"])
        names = [p["name"] for p in s.peaks]
        self.assertTrue(any("Cr" in n and p.get("role") == "unknown" for n, p in zip(names, s.peaks)), (names, rep))
        q = autofit.quality(s)
        self.assertLessEqual(q["bg_over"], 1e-6 * y.max())
        # 同一位置，若样品不含 Cr → 不能硬说是 Cr 俄歇
        s2 = core.Spectrum("Cu2p Scan", x, y)
        autofit.run(s2, ["Cu", "O"])
        self.assertFalse(any("Cr" in p["name"] for p in s2.peaks))
        # Co：主峰 +6 eV 的信号归为卫星
        xc = np.arange(772, 808.01, 0.1)
        co = core.doublet("Co2+", 780.6, 15.5, 0.5, 2.5, 50000, 30)
        sat = [core.new_peak("s", 786.5, 5, 30000, 0), core.new_peak("s2", 802.5, 5, 15000, 0)]
        yc = 90000 + core.model(xc, co)[1] + core.model(xc, sat)[1]
        sc = core.Spectrum("Co2p Scan", xc, np.random.default_rng(3).poisson(yc).astype(float))
        autofit.run(sc, ["Co", "O"])
        sats = [p for p in sc.peaks if p.get("role") == "satellite"]
        self.assertTrue(sats and all(p["link"].get("ref", "").startswith("Co") for p in sats))

    def test_refine_multistart_uncertainty(self):
        from xpsfit import autofit
        x, y, _ = synth()
        s = core.Spectrum("Al2p", x, y)
        core.apply_template(s, core.builtin_templates()["Al2p: Al 金属 + Al2O3"])
        r0 = s.do_fit()
        r1 = autofit.refine(s); r2 = autofit.multistart(s, 4)
        self.assertLessEqual(r2["rchi2"], r0["rchi2"] * 1.0001)
        autofit.uncertainties(s)
        self.assertTrue(s.peaks[0]["center"].get("err", 0) > 0)

    def test_identify_region_and_survey(self):
        from xpsfit import identify
        import avantage_to_xpspeak as conv
        _, b = conv.read_vgd(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_data", "Al2p Scan.VGD"), None)
        c = identify.identify_region(b[0]["energy"], b[0]["intensity"])
        self.assertEqual(c[0][0], "Al2p")
        x = np.arange(0, 1350.01, 0.5)
        y = 20000 + 30 * x
        for cc, w, a in [(284.8, 2, 1.5e5), (531, 2, 6e5), (778.2, 2, 8e5), (793.2, 2.4, 4e5), (60, 2, 8e4),
                         (1486.68 - 775, 8, 1.5e5), (1486.68 - 510, 8, 3e5), (23, 3, 4e4)]:
            y = y + core.gl_peak(x, cc, w, a, 30)
        y = np.random.default_rng(0).poisson(y).astype(float)
        r = identify.identify_survey(x, y)
        conf = {e["el"] for e in r["elements"] if e["confirmed"]}
        self.assertEqual(conf, {"C", "O", "Co"})
        self.assertFalse(any(e["el"] in ("Fe", "Ta") for e in r["elements"]))   # Co 俄歇≠Fe2p，O2s≠Ta4f

    def test_peak_colors_group(self):
        from xpsfit import autofit
        pk = autofit.fe_model(True, ("Fe2O3",))
        cols = core.peak_colors(pk)
        self.assertEqual(len(set(cols[:2])), 1)                 # Fe0 两个分量同色
        self.assertEqual(len(set(cols[2:])), 1)                 # Fe2O3 主线 / 多重分裂 / 2p1/2 / 两个卫星同色
        self.assertNotEqual(cols[0], cols[2])
        pk[2]["color"] = "#123456"
        self.assertEqual(set(core.peak_colors(pk)[2:]), {"#123456"})
        s = core.Spectrum("Fe2p", np.arange(700, 740, 0.1), np.ones(400))
        s.peaks = pk
        self.assertEqual([n for n, _ in core.chemical_states(s)], ["Fe0 2p", "Fe2O3 2p"])   # 2p1/2 卫星归到主峰

    def test_active_shirley_recovers_slope(self):
        """本底 = 常数 + 线性斜坡 + Shirley 台阶：主动 Shirley 拟合出的斜坡接近真实值，背景不高于数据。"""
        x = np.arange(700, 740.01, 0.1)
        pk = core.doublet("A", 710.0, 13.6, 0.5, 2.5, 60000, 30)
        _, env = core.model(x, pk)
        cum = np.concatenate([[0], np.cumsum(0.5 * (env[1:] + env[:-1]) * np.diff(x))])
        y = np.random.default_rng(1).poisson(20000 + 60.0 * (x - 700) + 0.15 * cum + env).astype(float)
        s = core.Spectrum("A", x, y)
        s.bg.update({"type": "SlopeShirley", "navg": 5})
        s.peaks = core.doublet("A", 709.5, 13.6, 0.5, 2.0, 50000, 30)
        s.do_fit()
        self.assertAlmostEqual(s.bg["tilt"], 60.0, delta=12.0)
        xx, yy, bg, _, _ = s.compute()
        self.assertLessEqual(float(np.max(bg - core.smooth(yy))), 1e-6)
        self.assertAlmostEqual(s.peaks[0]["area"]["value"] / 60000, 1.0, delta=0.05)

    def test_fe_phase_interpretation(self):
        from xpsfit import autofit
        s = core.Spectrum("Fe2p Scan", np.arange(700, 740, 0.1), np.ones(400))
        s.element = "Fe2p"
        s.peaks = autofit.fe_model(False, ("Fe3O4",))
        d = autofit.fe_phases(s)
        self.assertAlmostEqual(d["x2"], 1 / 3, places=6)
        self.assertIn("Fe₃O₄", autofit.interpret(s))
        s.peaks = autofit.fe_model(False, ("FeO",))
        self.assertAlmostEqual(autofit.fe_phases(s)["x2"], 1.0, places=6)


    def test_spin_orbit_rules(self):
        from xpsfit import autofit
        r = core.so_rule("Co2p", metal=True)
        self.assertEqual(r["dE_rng"], (14.9, 15.1))
        pk = autofit._D("Co2+", 780.6, None, None, 2.6, orb="Co2p")
        ln = pk[1]["link"]
        self.assertEqual((ln["dE_min"], ln["dE_max"]), core.so_rule("Co2p")["dE_rng"])
        self.assertEqual((ln["area_ratio_min"], ln["area_ratio_max"]), (0.45, 0.55))
        al = autofit._D("Al0", 72.6, None, None, 0.8, shape="Asym", orb="Al2p")[1]["link"]
        self.assertFalse(al.get("dE_fit") or al.get("area_ratio_fit"))          # 分裂太小：锁定
        # Fe 氧化物 2p1/2 相对 2p3/2 包络（主线 + 多重分裂线）
        fe = autofit.fe_phase_peaks("Fe2O3")
        v = core.resolve(fe)
        c3 = (v[0]["center"] * v[0]["area"] + v[1]["center"] * v[1]["area"]) / (v[0]["area"] + v[1]["area"])
        self.assertAlmostEqual(v[2]["center"] - c3, core.so_rule("Fe2p")["dE"], places=6)
        self.assertAlmostEqual(v[2]["area"] / (v[0]["area"] + v[1]["area"]), 0.5, places=6)


    def test_relax_steps(self):
        from xpsfit import autofit
        x, y, _ = synth(3)
        s = core.Spectrum("Al2p Scan", x, y)
        autofit.run(s, ["Al", "O"])
        before = copy.deepcopy(s.peaks)
        r = autofit.relax_next(s)
        self.assertEqual(r["level"], 1)
        self.assertIn(r["verdict"], ("keep", "check", "marginal", "worse"))
        self.assertFalse(all(p["lor"]["fixed"] for p in s.peaks if p.get("role", "main") == "main" and not p.get("pair")))
        autofit.relax_undo(s)
        self.assertEqual(s.relax_level, 0)
        self.assertEqual([p["lor"]["fixed"] for p in s.peaks], [p["lor"]["fixed"] for p in before])
        a = autofit.relax_auto(s)
        self.assertTrue(a["auto"])
        self.assertLessEqual(s.relax_level, autofit.AUTO_MAX)          # 自动只走低、中风险的级别
        st = autofit.relax_state(s)
        self.assertEqual(len(st["ladder"]), len(autofit.RELAX_LEVELS))


    def test_oxide_modes(self):
        from xpsfit import autofit
        x, y, _ = synth(4)                     # Al⁰ + Al2O3（氧化物约 75%）
        names = {}
        for mode in ("auto", "both", "metal", "oxide"):
            s = core.Spectrum("Al2p Scan", x, y)
            rep = autofit.run(s, ["Al", "O"], oxide=mode)
            names[mode] = {core.state_name(n) for n, _ in core.chemical_states(s)}
            if mode == "metal":                # 数据明显含氧化态：选“仅金属”时应提醒
                self.assertTrue(any(l.startswith("⚠ 数据显示存在氧化态") for l in rep), rep)
        self.assertEqual(names["metal"], {"Al0 2p"})
        self.assertEqual(names["oxide"], {"Al3+ 2p"})
        self.assertEqual(names["both"], {"Al0 2p", "Al3+ 2p"})
        self.assertEqual(names["auto"], {"Al0 2p", "Al3+ 2p"})


class Api(unittest.TestCase):
    """不开窗口，直接调用后端 API 走一遍完整流程。"""
    def test_workflow(self):
        from xpsfit import server
        server.ST.specs = []
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_data", "Al2p Scan.VGD")
        r = server.open_paths([p])
        self.assertEqual(r["added"], 1, r)
        v = server.api("apply_template", {"i": 0, "name": "Al2p: Al 金属 + Al2O3"})
        self.assertEqual(len(v["peaks"]), 4)
        v = server.api("fit", {"i": 0})
        self.assertTrue(v["fit"]["success"]); self.assertTrue(v["specs"][0]["fitted"])
        v = server.api("add_peak", {"i": 0, "x": 80.0})
        self.assertEqual(len(v["peaks"]), 5)
        v = server.api("set_bg", {"i": 0, "bg": {"type": "Linear", "lo": 68, "hi": 82, "navg": 3}})
        self.assertAlmostEqual(v["region"][0], 68, delta=0.11)
        import tempfile
        d = tempfile.mkdtemp()
        old = server.user_dir
        server.user_dir = lambda: d
        try:
            server.ST.specs[0].source = os.path.join(d, "x.vgd")
            for kind in ("figure", "excel", "csv", "txt"):
                out = server.api("export_" + kind, {"i": 0})["path"]
                self.assertTrue(os.path.getsize(out) > 100, out)
            q = server.api("quant", {"use_fit": True, "ecf": True})
            self.assertAlmostEqual(q["samples"][0]["elements"]["Al2p"], 100.0)
            # 堆叠对比：同一轨道的两条谱
            server.ST.specs.append(copy.deepcopy(server.ST.specs[0]))
            server.ST.specs[1].sample = "B"
            st = server.api("stack", {"i": 0, "mode": "bgsub"})
            self.assertEqual(len(st["items"]), 2)
            self.assertAlmostEqual(max(st["items"][0]["y"]), 1.0, places=6)
            out = server.api("export_stack", {"i": 0, "dpi": 80})["path"]
            self.assertTrue(os.path.getsize(out) > 100, out)
            v = server.api("set_color", {"i": 0, "j": 1, "color": "#112233"})
            self.assertEqual(v["colors"][0], "#112233")                    # 双峰副峰的颜色作用于整组
            # 标注：单谱图与堆叠图各一套，导出时画上，随项目保存
            server.api("set_annots", {"i": 0, "ctx": "spec", "items": [
                {"type": "text", "text": "Al^{3+} 2p_{3/2}", "x": 74.6, "y": 80, "size": 10, "color": "#c00000"},
                {"type": "vline", "x": 72.7, "len": 90, "width": 1, "dash": 4, "gap": 2}]})
            server.api("set_annots", {"i": 0, "ctx": "stack", "items": [{"type": "text", "text": "(a)", "x": 79}]})
            v = server.api("view", {"i": 0})
            self.assertEqual([a["type"] for a in v["annots"]], ["text", "vline"])
            self.assertEqual(v["stack_annots"][0]["text"], "(a)")
            self.assertTrue(os.path.getsize(server.api("export_figure", {"i": 0})["path"]) > 100)
            self.assertTrue(os.path.getsize(server.api("export_stack", {"i": 0, "dpi": 80})["path"]) > 100)
            pj = server.api("save_project", {})["path"]
            server.ST.stack_annots = {}
            back = server.ST.load_path(pj)
            self.assertEqual(back[0].annots[1]["len"], 90.0)
            self.assertEqual(server.ST.stack_annots["Al2p"][0]["text"], "(a)")
            # 默认值与单位：字号 8 pt 不加粗；线 0.7 pt、段长 3、间隔 2；箭头 1.5 pt 实心三角。导出图里就是 pt
            d = server.clean_annots([{"type": "text", "text": "x"}, {"type": "vline", "x": 73}, {"type": "arrow"}])
            self.assertEqual((d[0]["size"], d[0]["bold"]), (8.0, False))
            self.assertEqual((d[1]["width"], d[1]["dash"], d[1]["gap"]), (0.7, 3.0, 2.0))
            self.assertEqual((d[2]["width"], d[2]["head"], d[2]["size"], d[2]["hw"]), (1.5, "filled", 7.0, 5.0))
            self.assertEqual(server.ann_segments("Co^{2+} 2p_{3/2}"), [("Co", 0), ("2+", 1), (" 2p", 0), ("3/2", -1)])
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            fig = Figure(); FigureCanvasAgg(fig); ax = fig.add_subplot(111)
            server.draw_annots(ax, d)
            self.assertEqual(ax.texts[0].get_fontsize(), 8.0)
            self.assertEqual(ax.lines[0].get_linewidth(), 0.7)
            # 导出预览：不画标注（页面叠加），返回坐标轴位置与 x 范围，供页面把标注放到正确的位置
            r = server.api("preview_figure", {"i": 0, "opts": {"xmin": 70, "xmax": 80}, "px": 600})
            g = r["geo"]
            self.assertAlmostEqual(g["w"], 8.5 / 2.54 * 72, places=3)
            self.assertTrue(0 < g["ax"][0] < g["ax"][2] < 1 and 0 < g["ax"][1] < g["ax"][3] < 1, g)
            self.assertEqual([round(v) for v in g["xlim"]], [80, 70])     # 高 BE 在左
            self.assertGreater(len(r["png"]), 1000)
            ps = server.api("preview_stack", {"i": 0, "mode": "bgsub", "px": 600})
            self.assertGreater(len(ps["png"]), 1000)
            # 堆叠图样品名：预览返回每个标签的位置（页面上下拖动）；拖过的位置用于导出并随项目保存
            labs = ps["geo"]["labels"]
            self.assertEqual([l["key"] for l in labs], [server.ST.specs[0].sample, "B"])
            self.assertFalse(labs[0]["moved"])
            server.api("set_stack_label", {"i": 0, "key": "B", "dy": 0.5})
            labs2 = server.api("preview_stack", {"i": 0, "mode": "bgsub", "px": 600})["geo"]["labels"]
            self.assertAlmostEqual(labs2[1]["y"], labs2[1]["sh"] + 0.5 * labs2[1]["span"])
            self.assertTrue(labs2[1]["moved"])
            self.assertEqual(server.api("stack", {"i": 0})["label_dy"], {"B": 0.5})
            self.assertFalse(server.FIG_DEFAULTS["legend"])                  # 导出图默认不带图例
            # 堆叠对比里改颜色：所有样品的同名峰组一起改
            server.api("set_color", {"i": 0, "j": 0, "color": "#445566", "all": True})
            self.assertEqual([t.peaks[0].get("color") for t in server.ST.specs], ["#445566", "#445566"])
        finally:
            server.user_dir = old


class NistModel(unittest.TestCase):
    """v1.1：没有内置模型的轨道按 NIST 数据库建模。"""

    def test_ti2p_metal_and_oxide(self):
        from xpsfit import autofit
        rng = np.random.default_rng(4)
        x = np.arange(450, 470.01, 0.1)
        pk = core.doublet("Ti", 454.0, 6.1, 0.5, 1.0, 0.4, 25, shape="Asym") + core.doublet("TiO2", 458.6, 5.7, 0.5, 1.3, 0.6, 30)
        pk[3]["link"]["fwhm_ratio"] = 1.6
        _, sig = core.model(x, pk)
        y = rng.poisson(2e4 + 3e4 * sig / sig.max()).astype(float)
        s = core.Spectrum("Ti2p Scan", x, y)
        rep = autofit.run(s, ["Ti", "O"])
        self.assertTrue(any("按 NIST 数据库" in t for t in rep), rep)
        kept = {core.state_name(p["name"]) for p in s.peaks if not p.get("pair") and p.get("role", "main") == "main"}
        self.assertIn("Ti0 2p", kept)
        self.assertIn("TiO2 2p", kept)
        st = dict(core.chemical_states(s))
        tot = sum(st.values())
        self.assertAlmostEqual(st["Ti0 2p"] / tot, 0.4, delta=0.06)
        self.assertTrue(autofit.oxide_capable(s))

    def test_plausible_states(self):
        """残差峰只用与样品元素相符的 NIST 化学态命名：样品没有 F 时不会出现 AlF3（v1.1 回归测试）。"""
        from xpsfit import autofit, reference
        s = core.Spectrum("Al2p Scan", np.arange(65, 85.01, 0.1), np.ones(201))
        ref = reference.for_orbital("Al2p")
        f = lambda els: {t["formula"] for t in autofit.plausible_states(ref, s, els)}
        self.assertNotIn("AlF3", f(["Al", "Co", "Cr", "Cu", "C", "O"]))
        self.assertIn("Al2O3", f(["Al", "Co", "Cr", "Cu", "C", "O"]))
        self.assertIn("AlF3", f(["Al", "F", "O"]))
        self.assertEqual({t["kind"] for t in autofit.plausible_states(ref, s, [])} - {"metal", "oxide", "hydroxide"}, set())

    def test_compound_next_to_metal(self):
        """与单质相差 < 0.8 eV 的化合物（MoN 228.4 与 Mo 228.0）：样品没有 N 时不作为候选，有 N 时与单质合并写明。"""
        from xpsfit import autofit
        s = core.Spectrum("Mo3d Scan", np.arange(224, 240.01, 0.1), np.ones(321))
        s._sample_elements = ["Mo", "O"]
        names = [c["name"] for c in autofit.nist_candidates(s)]
        self.assertIn("Mo0", names)
        self.assertFalse(any("MoN" in n for n in names), names)
        s._sample_elements = ["Mo", "N", "O"]
        self.assertIn("Mo0/MoN", [c["name"] for c in autofit.nist_candidates(s)])


class Assessment(unittest.TestCase):
    """检出限（未检出 / 弱 / 充足）、未检出时不拟合峰、定量给上限、总体评估。"""

    def test_detection_quant_and_grade(self):
        from xpsfit import autofit
        rng = np.random.default_rng(0)
        x = np.arange(770, 810.01, 0.1)
        base = 2e4 + 20 * (x - 770)
        _, sig = core.model(x, core.doublet("Co0", 778.2, 14.99, 0.5, 1.2, 1.0, 20))
        sig = sig / sig.max()
        noise = core.Spectrum("Co2p Scan", x, rng.poisson(base).astype(float))
        d = autofit.detection(noise)
        self.assertEqual(d["status"], "nd")
        self.assertGreater(d["upper"], 0)
        autofit.run(noise, ["Co"])
        self.assertEqual(noise.peaks, [])                                 # 未检出：不拟合峰
        self.assertEqual(autofit.assess(noise)["grade"], "nd")
        weak = core.Spectrum("Co2p Scan", x, rng.poisson(base + 350 * sig).astype(float))
        self.assertEqual(autofit.detection(weak)["status"], "weak")
        strong = core.Spectrum("Co2p Scan", x, rng.poisson(base + 2e4 * sig).astype(float))
        self.assertEqual(autofit.detection(strong)["status"], "ok")
        autofit.run(strong, ["Co"])
        strong.last_fit.update(warnings=autofit.quality(strong)["warnings"])
        a = autofit.assess(strong)
        self.assertIn(a["grade"], ("ok", "check"))
        self.assertTrue(any(i["level"] == "good" for i in a["items"]))
        self.assertTrue(all("text" in i for i in autofit.assess(strong, "en")["items"]))
        # 定量：未检出的轨道不计面积，给出上限
        ni = core.Spectrum("Ni2p Scan", np.arange(845, 885.01, 0.1), rng.poisson(np.full(401, 3e4)).astype(float))
        for t in (strong, ni):
            t.sample = "A"
            t.det = autofit.detection(t)
        q = core.quantify_samples([strong, ni], {"Co2p": 3.59, "Ni2p": 4.04})[0]
        self.assertAlmostEqual(q["elements"]["Co2p"], 100.0)
        self.assertIn("Ni2p", q["nd"])
        self.assertTrue(0 < q["nd"]["Ni2p"] < 5)


class DepthProfileAndSalts(unittest.TestCase):
    """v1.2：深度剖析分层、碱土金属盐、噪声峰、堆叠导出选项。"""
    @classmethod
    def setUpClass(cls):
        global autofit, reference, server, conv
        from xpsfit import autofit, reference, server
        import avantage_to_xpspeak as conv

    def _ca(self, centers, areas, seed=0):
        x = np.arange(340, 360.01, 0.1)
        pk = []
        for c, a in zip(centers, areas):
            pk += core.doublet("Ca%.1f" % c, c, 3.55, 0.5, 1.5, a, 30)
        y = 6000 + core.model(x, pk)[1]
        return core.Spectrum("Ca2p Scan", x, np.random.default_rng(seed).poisson(y).astype(float))

    def test_salt_kinds_and_candidates(self):
        self.assertEqual(reference.state_kind("Ca", "CaCO3"), "salt")
        self.assertEqual(reference.state_kind("Ca", "CaCl2"), "salt")
        self.assertEqual(reference.state_kind("K", "KCl"), "salt")
        self.assertEqual(reference.state_kind("Ca", "Bi2Sr2CaCu2Ox"), "other")
        s = self._ca([347.3], [8000])
        s._sample_elements = ["Ca", "C", "Cl"]
        c = autofit.nist_candidates(s)
        names = [k["name"] for k in c]
        self.assertTrue(any("CaCO3" in n for n in names), names)
        self.assertTrue(any("CaCl2" in n for n in names), names)
        for a, b in zip(c, c[1:]):                        # 相邻候选范围不重叠
            self.assertLessEqual(a["hi"], b["lo"] + 1e-9)
        s._sample_elements = ["Ca", "C"]                  # 没有 Cl → 没有 CaCl2
        self.assertFalse(any("CaCl2" in k["name"] for k in autofit.nist_candidates(s)))

    def test_ca2plus_not_metal(self):
        """单一 Ca²⁺ 双峰：不能出现 Ca 金属组分，主峰也不能是“未知峰”。"""
        s = self._ca([347.3], [8000], seed=3)
        autofit.run(s, ["Ca", "C", "Cl", "K", "Mg"])
        mains = [p["name"] for p in s.peaks if not p.get("pair")]
        self.assertFalse(any(n.startswith("Ca0") for n in mains), mains)
        self.assertFalse(any(p.get("role") == "unknown" for p in s.peaks), mains)

    def test_noise_peaks_dropped_and_smart_ceiling(self):
        x = np.arange(194, 206.01, 0.1)
        pk = core.doublet("Cl", 198.9, 1.6, 0.5, 1.3, 1500, 30)
        y = np.random.default_rng(5).poisson(3200 + core.model(x, pk)[1]).astype(float)
        s = core.Spectrum("Cl2p Scan", x, y)
        autofit.run(s, ["Cl", "K", "C"])
        mains = [p for p in s.peaks if not p.get("pair") and p.get("role", "main") == "main"]
        self.assertEqual(len(mains), 1, [p["name"] for p in s.peaks])
        sm = core.background(x, y, "Smart")
        self.assertLessEqual(float(np.max(sm - core._smart_ceiling(x, y))), 1e-6)
        self.assertLess(float(np.std(np.diff(core._smart_ceiling(x, y)))), float(np.std(np.diff(core.smooth(y)))))

    def test_quantify_per_level_and_missing_sf(self):
        specs = []
        for lv, et in enumerate((0.0, 135.19)):
            for orb, c in (("C1s", 284.8), ("Mg1s", 1304.3)):
                x = np.arange(c - 6, c + 6.01, 0.1)
                t = core.Spectrum("%s Scan [%g s]" % (orb, et), x, 1000 + 5000 * np.exp(-((x - c) / 0.7) ** 2))
                t.sample, t.level, t.etch_s = "S1", lv, et
                specs.append(t)
        q = core.quantify_samples(specs, {"C1s": 1.0}, use_fit=False)
        self.assertEqual([g["sample"] for g in q], ["S1 · 0 s", "S1 · 135 s"])
        self.assertTrue(all(g["no_sf"] == ["Mg1s"] and g["duplicates"] == [] for g in q))

    def test_etch_axis(self):
        import struct
        ax = struct.pack("<Idd", 191, 1188.68, 0.1) + b"\0" * 9 + struct.pack("<Idd", 3, 0.0, 135.1905) + struct.pack("<Idd", 3, 0.0, 1.0)
        et = conv._etch_axis(ax, 20, 3)
        self.assertEqual(len(et), 3)
        self.assertAlmostEqual(et[2], 270.381, places=3)
        self.assertIsNone(conv._etch_axis(struct.pack("<Idd", 191, 1188.68, 0.1) + struct.pack("<Idd", 3, 0.0, 1.0), 20, 3))

    def test_stack_export_options(self):
        server.ST.specs = []
        server.ST.stack_names, server.ST.stack_labels = {}, {}
        for lv, et in enumerate((0.0, 135.0, 270.0)):
            x = np.arange(190, 210.01, 0.1)
            t = core.Spectrum("Cl2p Scan [%g s]" % et, x, 3000 + (lv + 1) * 800 * np.exp(-((x - 199) / 0.8) ** 2))
            t.sample, t.level, t.etch_s = "S1", lv, et
            server.ST.specs.append(t)
        d = server.stack_data(0, {})
        self.assertEqual([it["key"] for it in d["items"]], ["S1 · 0 s", "S1 · 135 s", "S1 · 270 s"])
        self.assertAlmostEqual(min(d["items"][0]["x"]), 190.0, places=3)      # 默认画整段数据，不只背景区间
        self.assertGreater(min(server.stack_data(0, {"range": "bg"})["items"][0]["x"]), 193.9)
        server.api("set_stack_name", {"i": 0, "key": "S1 · 135 s", "text": "135 s etched"})
        import io
        geo = server.export_stack(0, {"ymin": -0.5, "ymax": 4.0, "raw": "markers", "ynumbers": True, "dpi": 80}, out=io.BytesIO())
        self.assertEqual(geo["ylim"], [-0.5, 4.0])
        self.assertIn("135 s etched", [g["text"] for g in geo["labels"]])
        server.ST.specs = []


class TiCarbonThreshold(unittest.TestCase):
    """v1.2：C1s sp² 组分与命名、强度门槛、化合物附近不平移模型。"""
    @classmethod
    def setUpClass(cls):
        global autofit
        from xpsfit import autofit

    def test_c1s_sp2_component_not_molecule_name(self):
        x = np.arange(279, 298.01, 0.1)
        pk = [core.new_peak("sp2", 284.3, 1.0, 40000, 30), core.new_peak("sp3", 285.0, 1.3, 40000, 30),
              core.new_peak("C-O", 286.5, 1.4, 4000, 30), core.new_peak("OCO", 288.8, 1.4, 3000, 30)]
        y = np.random.default_rng(4).poisson(3000 + core.model(x, pk)[1]).astype(float)
        s = core.Spectrum("C1s Scan", x, y)
        autofit.run(s, ["C", "O", "Ti"])
        names = [p["name"] for p in s.peaks]
        self.assertIn("C=C sp2", names, names)
        self.assertFalse(any("C2H4" in n or "carbide" in n for n in names), names)

    def test_intensity_threshold(self):
        x = np.arange(342, 356.01, 0.1)
        pk = core.doublet("Ca", 347.3, 3.55, 0.5, 1.5, 6000, 30)
        y = np.random.default_rng(6).poisson(6000 + core.model(x, pk)[1]).astype(float)
        s = core.Spectrum("Ca2p Scan", x, y)
        try:
            autofit.MIN_NET = 0.0
            self.assertNotEqual(autofit.detection(s)["status"], "nd")
            autofit.MIN_NET = 10000.0
            d = autofit.detection(s)
            self.assertEqual(d["status"], "nd")
            self.assertTrue(d["below_min_net"])
            self.assertAlmostEqual(d["upper"], 10000 * 1.6 * 1.0645 * 1.5, places=3)    # 上限按门槛（2p 含 1/2 分量）
            autofit.run(s, ["Ca", "C", "O"])
            self.assertEqual(s.peaks, [])
            self.assertEqual(autofit.assess(s)["grade"], "nd")
        finally:
            autofit.MIN_NET = 0.0

    def test_no_model_shift_when_compound_is_closer(self):
        x = np.arange(450, 468.01, 0.1)
        pk = core.doublet("TiO", 455.1, 5.8, 0.5, 1.6, 30000, 30) + core.doublet("TiO2", 458.8, 5.7, 0.5, 1.2, 20000, 30)
        y = np.random.default_rng(8).poisson(8000 + core.model(x, pk)[1]).astype(float)
        s = core.Spectrum("Ti2p Scan", x, y)
        rep = autofit.run(s, ["Ti", "O", "C"])
        self.assertAlmostEqual(getattr(s, "model_shift", 0.0), 0.0, places=6, msg="\n".join(rep))


if __name__ == "__main__":
    unittest.main()
