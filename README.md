# XPS Fit Studio + Avantage2XPSPEAK

下载：仓库右侧 **Releases → latest**

| 程序 | 用途 |
|---|---|
| `XPSFitStudio.exe` | XPS 峰拟合（替代 XPSPEAK 4.1）。双击运行，无需安装 Python |
| `Avantage2XPSPEAK.exe` | 把 Avantage `.VGD` / 导出表格转成 XPSPEAK 4.1 的两列 txt |

## XPS Fit Studio 使用
1. **打开数据**：`.VGD`（Avantage 原生）、Avantage 导出的 Excel/CSV、XPSPEAK txt；可拖拽到左侧。
2. **能量校正**：在 C1s 谱选中 C-C 峰 → “选中峰 → C1s 284.8 eV”，同文件夹（同样品）的谱一起平移。
3. **背景**：Shirley / Linear / Tougaard；范围可输入，或“选背景范围”后在图上横向拖动框选（或点两下）。
4. **加峰**：“＋ 添加峰”可选 **单峰 / 双峰 / 卫星峰 / 未知重叠峰**；“－ 删除峰”（删除双峰主峰时副峰一起删）。也可套用模板（Al2p、C1s、O1s、Ti2p）。
   - 双峰：间距、面积比可修改，并可选择拟合时**锁定**或在范围内**自由拟合**。
   - 卫星峰：归属某个主峰，ΔE / 面积比可锁定或自由；计入面积%。
   - 未知重叠峰：参与拟合，但**不计入**面积% 和原子% 定量。
   - FWHM：可绑定相等（“FWHM 相等”按钮或编辑里 FWHM 比 = 1），也可勾选“固定”锁定数值。
   - 双击峰可设固定、上下限、绑定关系。
5. **拟合**，看残差；“用相同设置拟合全部同类谱”可批量处理（如深度剖析）。
6. **导出**：拟合图（PNG / TIFF / SVG / PDF，带实时预览；可设 x 范围、主刻度间距、次刻度，y 轴数字和刻度开关，图例位置、尺寸、字号、DPI），Excel（全部峰参数 + 每条谱的曲线），CSV（给 Origin），XPSPEAK txt。保存在原数据旁的 `xpsfit_output\`。
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
