# XPS Fit Studio + Avantage2XPSPEAK

下载：仓库右侧 **Releases → latest**

| 程序 | 用途 |
|---|---|
| `XPSFitStudio.exe` | XPS 峰拟合（替代 XPSPEAK 4.1）。双击运行，无需安装 Python |
| `Avantage2XPSPEAK.exe` | 把 Avantage `.VGD` / 导出表格转成 XPSPEAK 4.1 的两列 txt |

## XPS Fit Studio 使用
1. **打开文件夹**：一次读入样品文件夹（含子文件夹）里所有元素的 `.VGD`，按样品分组；图上方的元素标签（或键盘 ← →）切换元素。也可“打开文件”选单个文件（`.VGD`、Avantage 导出 Excel/CSV、XPSPEAK txt），或拖拽到左侧。
2. **能量校正**：在 C1s 谱选中 C-C 峰（注意：含碳化物的样品，最高峰可能是 ~283 eV 的碳化物，不是 C-C；可用“C1s: 碳化物 + …”模板） → “选中峰 → C1s 284.8 eV”，同文件夹（同样品）的谱一起平移。
3. **背景**：Shirley / Smart（Shirley 但不高于数据）/ **自定义点**（在图上点选背景经过的点，平滑或折线连接，可拖动、右键删除）/ Linear / Tougaard；范围可输入、框选，或直接拖动图上背景两端的方块（左右改范围、上下改端点强度，双击恢复自动）。
4. **加峰**：“＋ 添加峰”可选 **单峰 / 双峰 / 卫星峰 / 未知重叠峰**；“－ 删除峰”（删除双峰主峰时副峰一起删）。也可套用模板（Al2p、C1s、O1s、Ti2p）。
   - 双峰：间距、面积比可修改，并可选择拟合时**锁定**或在范围内**自由拟合**。
   - 卫星峰：归属某个主峰，ΔE / 面积比可锁定或自由；计入面积%。
   - 未知重叠峰：参与拟合，但**不计入**面积% 和原子% 定量。
   - FWHM / L%：“参数相等…”里选参考峰并**多选**要相等的峰（取消勾选即解除）；勾选“固定”可锁定数值。
   - 图上拖动峰顶圆点：左右移位置、上下改高度，按住 Shift 左右拖改 FWHM；框选放大，双击或“⤢ 复位”恢复。
   - 峰型：**对称 GL** 或 **不对称**（高 BE 侧指数拖尾，适合金属 Co、Cr、Cu、Ni、Ti、Al 等），拖尾高度 T 与长度 τ 可拟合/固定；面积为真实面积，可直接定量；双峰副峰跟随主峰峰型。
   - FWHM 默认范围 1–3 eV（卫星峰 1–8 eV），可在编辑中修改。
   - 双击峰可设固定、上下限、绑定关系。
5. **拟合**，看残差；“用相同设置拟合全部同类谱”可批量处理（如深度剖析）。
6. **导出**：拟合图（PNG / TIFF / SVG / PDF，带实时预览；可设 x 范围、主刻度间距、次刻度，y 轴数字和刻度开关，图例位置、尺寸、字号、DPI），Excel（全部峰参数 + 每条谱的曲线），CSV（给 Origin），XPSPEAK txt。保存在原数据旁的 `xpsfit_output\`。
7. **定量与比例**：按样品列出元素原子%、化学态原子%（主峰 + 双峰副峰 + 卫星峰；未知重叠峰不计），可任选两项（如 O/Al、Al2O3/Al0）组成比例表，一键复制到 Excel。ALTHERMO1 灵敏度因子，原子% ∝ 面积 / (SF·KE^0.6)，SF 可修改。

用户模板、修改过的 SF 保存在 `%USERPROFILE%\XPSFitStudio\`。

峰型为 Gaussian–Lorentzian 混合（GL sum，L% 默认 20–30 固定），拟合用带边界的最小二乘（泊松权重）。

## 开发
```
pip install numpy scipy matplotlib pandas openpyxl xlrd pywebview
python run_xpsfit.py                 # 桌面窗口（无 pywebview 时用浏览器）
python -m unittest test_xpsfit_core test_avantage_to_xpspeak
```
推送到 main 后 GitHub Actions 自动在 Windows 上测试、打包并发布到 Releases。
