# -*- coding: utf-8 -*-
"""逐步放宽约束的验证：一键拟合后自动放宽（前 4 级），比较化学态比例误差。  python tools/benchmark_relax.py"""
import sys
import numpy as np
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import benchmark_autofit as B  # noqa: E402
from xpsfit import autofit  # noqa: E402
rng = np.random.default_rng(11)
cases = [("Co2p", B.case_co), ("Cu2p+Cu2+", lambda r: B.case_cu(r, False, True)), ("Cr2p+Cr6+", lambda r: B.case_cr(r, True)),
         ("Cr2p", lambda r: B.case_cr(r, False)), ("Al2p", lambda r: B.case_al(r, False)), ("C1s", lambda r: B.case_c(r, True))]
print("| 场景 | 次数 | 比例误差：默认约束 | 比例误差：自动放宽后 | 平均保留级数 | 保留最多的级别 |")
print("|---|---|---|---|---|---|")
for name, f in cases:
    e0, e1, nk, kept = [], [], [], {}
    for _ in range(5):
        s, bg, els, truth = f(rng)
        autofit.run(s, els)
        pre, fr = truth["frac"]
        e0.append(abs(B.frac_of(s, pre) - fr))
        r = autofit.relax_auto(s)
        e1.append(abs(B.frac_of(s, pre) - fr))
        ks = [st[1] for st in r["steps"] if st[2] == "保留"]
        nk.append(len(ks))
        for k in ks: kept[k] = kept.get(k, 0) + 1
    top = "、".join("%s×%d" % kv for kv in sorted(kept.items(), key=lambda kv: -kv[1])) or "—"
    print("| %s | 5 | %.3f（最大 %.3f） | %.3f（最大 %.3f） | %.1f | %s |" % (name, np.mean(e0), np.max(e0), np.mean(e1), np.max(e1), np.mean(nk), top), flush=True)
