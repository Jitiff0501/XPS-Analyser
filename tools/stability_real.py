# -*- coding: utf-8 -*-
"""真实谱稳定性测试：对每条谱加泊松噪声（重采样）并整体平移 ±0.3 eV，重复一键拟合，看化学态比例的离散度。
    python tools/stability_real.py <包含样品子文件夹的目录> [重复次数=5]"""
import os, sys, copy
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xpsfit import server, core, autofit  # noqa: E402


def main(root, n=5):
    server.ST.specs = []
    server.open_paths(server.folder_files(root))
    rng = np.random.default_rng(0)
    print("| 样品 | 谱 | 背景 | 化学态比例（原始） | 扰动后均值 ± 标准差 | 背景高于数据 |")
    print("|---|---|---|---|---|---|")
    for i, s0 in enumerate(server.ST.specs):
        els = server.sample_elements(s0)
        s = copy.deepcopy(s0); autofit.run(s, els)
        base = {k: v for k, v in states(s).items()}
        runs, over = [], 0
        for _ in range(n):
            t = copy.deepcopy(s0)
            t.y = rng.poisson(np.clip(t.y, 1, None)).astype(float)       # 再加一次泊松噪声
            t.be_raw = t.be_raw + rng.uniform(-0.3, 0.3)                  # 整体能量偏移
            autofit.run(t, els)
            runs.append(states(t))
            q = autofit.quality(t)
            over += q["bg_over"] > 1e-6 * float(np.max(t.y))
        keys = sorted(base, key=lambda k: -base[k])
        txt = "、".join("%s %.0f%%" % (k, 100 * base[k]) for k in keys)
        pert = "、".join("%s %.0f±%.0f%%" % (k, 100 * np.mean([r.get(k, 0) for r in runs]), 100 * np.std([r.get(k, 0) for r in runs])) for k in keys)
        print("| %s | %s | %s | %s | %s | %d/%d |" % (s0.sample, s0.name, s.bg["type"], txt, pert, over, n))


def states(s):
    st = core.chemical_states(s); tot = sum(a for _, a in st) or 1
    return {n: a / tot for n, a in st}


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
