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
            self.assertAlmostEqual(q["rows"][0]["atomic_percent"], 100.0)
        finally:
            server.user_dir = old


if __name__ == "__main__":
    unittest.main()
