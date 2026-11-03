# -*- coding: utf-8 -*-
"""用模拟的 Avantage 导出文件测试 avantage_to_xpspeak.py：  python -m unittest -v"""
import csv
import os
import re
import shutil
import tempfile
import unittest

import openpyxl

import avantage_to_xpspeak as conv

LINE = re.compile(r"^-?\d+(\.\d+)?\t-?\d+(\.\d+)?$")


def spectrum(start, stop, step, base=1000.0):
    n = int(round((start - stop) / step)) + 1
    return [(round(start - k * step, 3), base + k * 10.5) for k in range(n)]


META = [["Acquired On:", "2024/05/01 10:00"], ["Source:", "Al K Alpha"], ["Spot size:", "400 um"], []]


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.out = os.path.join(self.d, conv.OUT_DIRNAME)

    def tearDown(self):
        shutil.rmtree(self.d)

    def check_file(self, name, n=None, descending=True):
        p = os.path.join(self.out, name)
        self.assertTrue(os.path.exists(p), "missing %s; have %s" % (name, os.listdir(self.out)))
        raw = open(p, "rb").read()
        raw.decode("ascii")
        lines = raw.decode().split("\r\n")
        self.assertEqual(lines[-1], "")  # 以 CRLF 结尾
        lines = lines[:-1]
        for ln in lines:
            self.assertRegex(ln, LINE)  # 两列纯数字，无表头/空行
        es = [float(l.split("\t")[0]) for l in lines]
        self.assertEqual(len(es), len(set(es)))
        self.assertEqual(es, sorted(es, reverse=descending))
        if n is not None:
            self.assertEqual(len(lines), n)
        return [tuple(map(float, l.split("\t"))) for l in lines]

    def log(self):
        with open(os.path.join(self.out, conv.LOG_NAME), encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))


class Tests(Base):
    def test_multi_sheet_xlsx(self):
        wb = openpyxl.Workbook()
        wb.active.title = "Titles"
        wb.active.append(["Sample1 report"])
        for reg, sp in [("C1s Scan", spectrum(295, 280, 0.1)), ("Ti2p Scan", spectrum(470, 450, 0.1)),
                        ("Survey", spectrum(1350, 0, 1.0))]:
            ws = wb.create_sheet(reg)
            ws.append([reg])
            for m in META:
                ws.append(m)
            ws.append(["Binding Energy (E)", "Counts / s", "C1s A", "Backgnd.", "Envelope"])
            for e, i in sp:
                ws.append([e, i, 1, 2, 3])
        pk = wb.create_sheet("Peak Table")
        pk.append(["Name", "Peak BE", "FWHM eV", "Area (P) CPS.eV"])
        for k in range(6):
            pk.append(["C1s", 284.8, 1.2, 1000])
        src = os.path.join(self.d, "Sample1.xlsx")
        wb.save(src)
        before = open(src, "rb").read()
        conv.run([src], quiet=True)
        self.assertEqual(open(src, "rb").read(), before)
        pts = self.check_file("Sample1_C1s.txt", 151)
        self.assertEqual(pts[0], (295.0, 1000.0))  # 原始强度
        self.check_file("Sample1_Ti2p.txt", 201)
        self.check_file("Sample1_Survey.txt", 1351)
        ok = [r for r in self.log() if r["status"] == "OK"]
        self.assertEqual(len(ok), 3)
        c1s = [r for r in ok if r["region"] == "C1s"][0]
        self.assertEqual((c1s["points"], c1s["BE_high"], c1s["BE_low"], c1s["step_eV"]),
                         ("151", "295", "280", "0.1"))

    def test_one_sheet_multi_groups_and_scans(self):
        wb = openpyxl.Workbook()
        ws = wb.active
        c, o = spectrum(290, 285, 0.5), spectrum(535, 528, 0.5)
        ws.append(["C1s", None, None, None, None, "O1s"])
        ws.append(["Binding Energy (eV)", "Scan 1", "Scan 2", "Scan 3", None,
                   "Binding Energy (eV)", "Counts / s", "Scan 1", "Scan 2"])
        for k in range(max(len(c), len(o))):
            row = [None] * 9
            if k < len(c):
                row[0:4] = [c[k][0], c[k][1], c[k][1], 1]
            if k < len(o):
                row[5:9] = [o[k][0], o[k][1], 0, 0]
            ws.append(row)
        src = os.path.join(self.d, "S2.xlsx")
        wb.save(src)
        conv.run([src], quiet=True)
        pc = self.check_file("S2_C1s.txt", len(c))
        self.assertEqual(pc[0][1], 2 * 1000 + 1)  # 三条扫描求和
        po = self.check_file("S2_O1s.txt", len(o))
        self.assertEqual(po[0][1], 1000)  # 用 Counts / s 列
        w = {r["region"]: r["warnings"] for r in self.log()}
        self.assertIn("求和", w["C1s"])

    def test_csv_kinetic_energy_duplicates_ascending(self):
        src = os.path.join(self.d, "KE run#1.csv")
        with open(src, "w", newline="") as f:
            f.write("Ti2p Scan\nSource:,Al K Alpha\n\nKinetic Energy (eV),Counts\n")
            for ke in [1016.6, 1016.7, 1016.8, 1016.8, 1016.9, 1017.0, 1017.1]:
                f.write("%s,%s\n" % (ke, 500))
            f.write("\nEnd of data\n")
        conv.run([src], hv=1486.6, ascending=True, quiet=True)
        pts = self.check_file("KE_run#1_Ti2p.txt".replace("#", "#"), 6, descending=False)
        self.assertAlmostEqual(pts[-1][0], 470.0, places=4)
        self.assertAlmostEqual(pts[0][0], 469.5, places=4)
        w = self.log()[0]["warnings"]
        self.assertIn("BE =", w)
        self.assertIn("重复", w)

    def test_txt_tab_stacked_regions_and_bad_file(self):
        src = os.path.join(self.d, "sample3.txt")
        with open(src, "w", encoding="utf-16") as f:
            for reg, sp in [("C1s Scan", spectrum(290, 286, 0.2)), ("O1s Scan", spectrum(534, 529, 0.25))]:
                f.write("%s\nAcquired On:\t2024\nBinding Energy (E)\tCounts / s\n" % reg)
                for e, i in sp:
                    f.write("%s\t%s\n" % (e, i))
                f.write("\n")
        bad = os.path.join(self.d, "notes.txt")
        open(bad, "w").write("hello\nworld\n")
        broken = os.path.join(self.d, "broken.xlsx")
        open(broken, "w").write("not a zip")
        conv.run([self.d], quiet=True)  # 文件夹输入
        self.check_file("sample3_C1s.txt", 21)
        self.check_file("sample3_O1s.txt", 21)
        st = {os.path.basename(r["source_file"]): r["status"] for r in self.log()}
        self.assertEqual(st["notes.txt"], "SKIPPED")
        self.assertEqual(st["broken.xlsx"], "ERROR")

    def test_headerless_two_columns(self):
        src = os.path.join(self.d, "plain.txt")
        with open(src, "w") as f:
            for e, i in spectrum(100, 95, 0.5):
                f.write("%s  %s\n" % (e, i))
        conv.run([src], quiet=True)
        self.check_file("plain_plain.txt", 11)
        self.assertIn("未找到列名", self.log()[0]["warnings"])

    def test_vgd_native(self):
        src_real = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_data", "Al2p Scan.VGD")
        if not os.path.exists(src_real):
            self.skipTest("no sample VGD")
        sub = os.path.join(self.d, "Al")
        os.makedirs(sub)
        src = os.path.join(sub, "Al2p Scan.VGD")
        shutil.copy(src_real, src)
        self.out = os.path.join(sub, conv.OUT_DIRNAME)
        conv.run([src], quiet=True)
        pts = self.check_file("Al_Al2p.txt", 201)
        self.assertAlmostEqual(pts[0][0], 85.08, places=4)      # 1486.68 - 1401.6
        self.assertAlmostEqual(pts[-1][0], 65.08, places=4)
        peak = max(pts, key=lambda p: p[1])
        self.assertAlmostEqual(peak[0], 74.68, places=2)
        self.assertAlmostEqual(pts[0][1], 2779.13808, places=4)
        conv.run([src], hv=1486.6, quiet=True)                  # --hv 覆盖
        self.assertAlmostEqual(self.check_file("Al_Al2p.txt")[0][0], 85.0, places=4)
        self.assertIn("命令行", self.log()[0]["warnings"])

    def test_vgd_garbage(self):
        src = os.path.join(self.d, "bad.vgd")
        with open(src, "wb") as f:
            f.write(b"xxxx" * 100)
        conv.run([src], quiet=True)
        self.assertEqual(self.log()[0]["status"], "ERROR")


if __name__ == "__main__":
    unittest.main()
