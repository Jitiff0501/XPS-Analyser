# XPS Fit Studio + Avantage2XPSPEAK

下载：仓库右侧 **Releases → latest**

| 程序 | 用途 |
|---|---|
| `XPSFitStudio.exe` | XPS 峰拟合（替代 XPSPEAK 4.1）。双击运行，无需安装 Python |
| `Avantage2XPSPEAK.exe` | 把 Avantage `.VGD` / 导出表格转成 XPSPEAK 4.1 的两列 txt |

## XPS Fit Studio 使用
1. **打开数据**：`.VGD`（Avantage 原生）、Avantage 导出的 Excel/CSV、XPSPEAK txt；可拖拽到左侧。
2. **能量校正**：在 C1s 谱选中 C-C 峰 → “选中峰 → C1s 284.8 eV”，同文件夹（同样品）的谱一起平移。
3. **背景**：Shirley / Linear / Tougaard；范围可输入或“选背景范围”后在图上点两下。
4. **加峰**：“＋ 单峰”后在图上点击；“＋ 双峰”添加自旋轨道双峰（副峰的位置差、面积比、FWHM 绑定主峰）；或套用模板（Al2p、C1s、O1s、Ti2p）。双击峰可设固定、上下限、绑定关系。
5. **拟合**，看残差；“用相同设置拟合全部同类谱”可批量处理（如深度剖析）。
6. **导出**：600 dpi PNG / SVG / TIFF 拟合图，Excel（全部峰参数 + 每条谱的曲线），CSV（给 Origin），XPSPEAK txt。保存在原数据旁的 `xpsfit_output\`。
7. **定量**：ALTHERMO1 灵敏度因子，原子% ∝ 面积 / (SF·KE^0.6)；SF 可在窗口内修改。

用户模板、修改过的 SF 保存在 `%USERPROFILE%\XPSFitStudio\`。

峰型为 Gaussian–Lorentzian 混合（GL sum，L% 默认 20–30 固定），拟合用带边界的最小二乘（泊松权重）。

## 开发
```
pip install numpy scipy matplotlib pandas openpyxl xlrd pywebview
python run_xpsfit.py                 # 桌面窗口（无 pywebview 时用浏览器）
python -m unittest test_xpsfit_core test_avantage_to_xpspeak
```
推送到 main 后 GitHub Actions 自动在 Windows 上测试、打包并发布到 Releases。
