# avantage_to_xpspeak

把 Thermo Avantage 的 XPS 谱——**原生 `.VGD` 文件**，或导出的 .xlsx / .xls / .csv / .txt——转换成 XPSPEAK 4.1 可直接导入的两列 ASCII 文件。

## 1. 安装依赖（只需一次）
安装 Python 3（勾选 “Add Python to PATH”），然后在命令提示符运行：
```
pip install pandas openpyxl xlrd
```

## 2. 运行
**双击**：双击 `avantage_to_xpspeak.py`，在弹出的对话框中多选文件，完成后弹窗显示结果。

**命令行**（可传文件或文件夹，文件夹会递归查找）：
```
python avantage_to_xpspeak.py Sample1.xlsx Sample2.csv D:\XPS\data
python avantage_to_xpspeak.py --hv 1253.6 data\      # Mg Kα；仅在文件只有动能列时用到
python avantage_to_xpspeak.py --ascending data\      # 结合能从低到高（默认从高到低）
```

### 关于 .VGD
- 直接读取 Avantage 采集保存的 `.VGD`（如 `...\Al\Al2p Scan.VGD`），不需要先在 Avantage 里导出。
- 能量轴在 VGD 里存的是动能，按 BE = hν − KE 换算；hν 自动取文件里记录的值（K-Alpha+ 为 1486.68 eV），日志会写明。
  若与 Avantage 显示的 BE 有 0.08 eV 的差别，加 `--hv 1486.6` 重跑即可。
- 输出文件名为 `上级文件夹名_谱区名.txt`（如 `Al_Al2p.txt`），因为 VGD 文件名本身就是谱区名。
- 强度为 VGData 中的原值（Avantage 显示的 Counts/s）。

## 3. 输出
- 原文件旁的 `xpspeak_txt\` 文件夹，每个谱区一个 `原文件名_谱区名.txt`（如 `Sample1_Ti2p.txt`）。
- 内容：两列（结合能 eV ⇥ 强度），无表头、无空行、`.` 作小数点、Windows 换行。
- `conversion_log.csv`：来源文件、sheet、谱区、使用的列、点数、BE 范围、步长、最大强度、警告（KE 换算、扫描求和、重复点、未识别的 sheet/文件等）。
- 强度保持原值（不平滑/归一化/扣背景），原文件不会被修改。

识别规则：列名含 “Binding Energy / BE” 为结合能；只有 “Kinetic Energy / KE” 时按 BE = hν − KE 换算；
强度优先取 Sum/Total 列，其次 “Counts / s / CPS / Intensity” 列，再次 Average 列；都没有时对 “Scan 1, Scan 2…” 求和。
拟合列（Backgnd、Envelope、峰分量、Residual）和峰表 sheet 会被忽略。**每次请看一眼日志里的警告。**

## 4. 在 XPSPEAK 4.1 中导入
1. 打开 XPSPEAK 4.1 → 菜单 **Data → Import (ASCII)**。
2. 选择 `xpspeak_txt` 中的某个 `.txt` 文件，谱图即显示。
3. 之后按常规流程：Background → Add Peak → Optimise。

## 5. 测试
```
python -m unittest -v test_avantage_to_xpspeak.py
```
