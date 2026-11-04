# -*- coding: utf-8 -*-
"""中英文切换。程序内部文本为中文；英文界面时：
- 界面：前端把页面上的中文文本按 PHRASES（整句）/ PATTERNS（含数字等变量）/ 词组替换 翻成英文；
- 报告、提示：服务端用同一张表翻译（printf 模板自动转为正则，嵌套消息逐层翻译）。
"""
import re

# ------------------------------------------------------------------ 整句 / 词组
PHRASES = {
    "（风险：低）": " (risk: low)", "（风险：中）": " (risk: medium)", "（风险：高）": " (risk: high)", "（风险：最高）": " (risk: highest)",
    "逐步放宽约束": "Relax constraints step by step", "放宽一级 ▸": "Relax one level ▸", "撤销": "Undo", "自动": "Auto",
    "按风险从低到高，放开下一级约束并重新拟合": "Release the next constraint (lowest risk first) and refit",
    "退回上一级": "Go back one level", "逐级尝试低、中风险的前 4 级：BIC 显著下降、没有新的物理问题才保留": "Try the low/medium-risk levels 1–4 in turn; keep a level only if BIC drops significantly and no new physical problem appears",
    "逐级尝试放宽约束…": "Trying constraint levels…", "放宽下一级约束并重新拟合…": "Relaxing the next constraint and refitting…", "撤销…": "Undoing…",
    "所有级别都已放开过": "All levels have been tried",
    "Lorentzian 比例": "Lorentzian fraction", "背景斜坡": "Background slope", "FWHM 范围": "FWHM range", "金属拖尾": "Metal tail",
    "双峰关系超出文献范围": "Doublet relations beyond literature", "化合物不对称": "Compound asymmetry", "峰位范围": "Peak position range",
    "低": "low", "中": "medium", "高": "high", "最高": "highest", "保留": "kept", "撤回": "reverted",
    "已放开并保留": "released and kept", "已尝试，未保留 / 不适用": "tried, not kept / not applicable", "未放开": "not released",
    "仪器展宽（高斯）与寿命展宽（洛伦兹）的比例因谱仪和化学态而异，放开它物理上最没有争议": "The ratio of instrumental (Gaussian) to lifetime (Lorentzian) broadening varies with spectrometer and chemical state; freeing it is the least controversial",
    "本底在 Shirley 台阶之外常有线性斜坡；随峰拟合，背景仍不高于数据": "The background often has a linear slope besides the Shirley step; it is fitted with the peaks and still never exceeds the data",
    "默认 1–3 eV 适合多数单色 Al Kα 高分辨谱；窄的金属峰或宽的多重分裂峰可能需要更宽的范围": "The default 1–3 eV suits most monochromatic Al Kα high-resolution spectra; narrow metal peaks or broad multiplet peaks may need a wider range",
    "金属的拖尾长度因元素和导电电子而异；过长的拖尾会吃掉相邻氧化态": "Metal tails vary with element and conduction electrons; overly long tails swallow neighbouring oxide states",
    "1/2 分量常受背景、重叠和 Coster–Kronig 效应影响；超出文献范围时更可能是背景或重叠的问题，而不是物理": "The 1/2 component is affected by background, overlaps and Coster–Kronig effects; values beyond the literature range more likely indicate a background or overlap problem than physics",
    "多重分裂或未分辨的成分会让化合物峰看起来不对称；但拖尾很容易把相邻化学态的面积吃掉": "Multiplet splitting or unresolved components can make compound peaks look asymmetric, but tails easily swallow the area of neighbouring states",
    "峰位放开后，化学态可能互相交换位置；只在能量校正可靠时使用": "With free positions, chemical states may swap places; use only when the energy calibration is reliable",
    "⧉ 堆叠对比": "⧉ Stack compare", "把各样品的同一轨道上下错开叠在一起比较": "Stack the same orbital from all samples for comparison",
    "扣背景归一化": "Bkg-subtracted, normalized", "归一化": "Normalized", "原始": "Raw", "间距": "Spacing", "拟合包络": "Fit envelope",
    "分峰": "Components", "化学态比例": "State fractions", "导出堆叠图": "Export stacked figure", "返回单谱": "Back to single spectrum",
    "个样品": "samples", "纵轴": "Y axis", "没有可堆叠的谱": "No spectra to stack",
    "拟合约束（1/2 与 3/2）": "Fit constraints (1/2 vs 3/2)", "金属态": "metal", "化合物": "compound",
    "文献范围（金属态）：": "Literature range (metal): ", "文献范围（化合物）：": "Literature range (compound): ",
    "。不锁定时在此范围内拟合，FWHM 倍率总在此范围内拟合": ". When unlocked it is fitted within this range; the FWHM ratio is always fitted within its range",
    "（分裂太小，锁定）": " (splitting too small, locked)", "常用": "typical", "面积比": "area ratio", "FWHM 倍率": "FWHM ratio",
    "颜色": "Color", "自动": "Auto", "自定义颜色": "Custom color", "整组同色": "whole group shares this color",
    "点击选择颜色（整组同色）": "Click to choose a color (applies to the whole group)",
    "在中段偏离本底": "deviate from the background in the middle", "已限制为不高于数据": "clamped to never exceed the data", "未高于数据": "never above the data",
    "主动 Shirley（台阶 + 线性斜坡，斜坡随峰拟合）": "Active Shirley (step + linear slope fitted with the peaks)",
    "Fe2p：705.8–707.6 eV 无明显金属峰，不加 Fe⁰；氧化物按物相建模——FeO ~709.6（卫星 +6 eV）、Fe₃O₄ ~710.5（几乎无卫星）、Fe₂O₃/FeOOH ~711.0（卫星 +8 eV）；2p3/2 多重分裂包络用两条关联峰近似，2p1/2 用一个宽峰":
        "Fe2p: no clear metal peak at 705.8–707.6 eV, Fe⁰ not added; oxides modelled as phases — FeO ~709.6 (satellite +6 eV), Fe₃O₄ ~710.5 (no distinct satellite), Fe₂O₃/FeOOH ~711.0 (satellite +8 eV); the 2p3/2 multiplet envelope is approximated by two linked lines, 2p1/2 by one broad peak",
    "Fe2p：Fe⁰ 不对称双峰 Δ12.95–13.3；氧化物按物相建模——FeO ~709.6（卫星 +6 eV）、Fe₃O₄ ~710.5（几乎无卫星）、Fe₂O₃/FeOOH ~711.0（卫星 +8 eV）；2p3/2 多重分裂包络用两条关联峰近似，2p1/2 用一个宽峰":
        "Fe2p: Fe⁰ asymmetric doublet Δ12.95–13.3; oxides modelled as phases — FeO ~709.6 (satellite +6 eV), Fe₃O₄ ~710.5 (no distinct satellite), Fe₂O₃/FeOOH ~711.0 (satellite +8 eV); the 2p3/2 multiplet envelope is approximated by two linked lines, 2p1/2 by one broad peak",
    "Fe₂O₃ 与 FeOOH 的 Fe2p 几乎相同，请看 O1s 中 ~531.2 eV 的 OH 组分": "Fe₂O₃ and FeOOH have nearly identical Fe2p — check the OH component at ~531.2 eV in O1s",
    "Fe：只有金属 Fe⁰": "Fe: metallic Fe⁰ only",
    "（或 FeOOH）": " (or FeOOH)", "Fe₂O₃ 或 FeOOH": "Fe₂O₃ or FeOOH",
    "Fe2p: Fe⁰ + Fe₂O₃/FeOOH（卫星 +8 eV）": "Fe2p: Fe⁰ + Fe₂O₃/FeOOH (satellite +8 eV)",
    "Fe2p: Fe⁰ + Fe₃O₄（无明显卫星）": "Fe2p: Fe⁰ + Fe₃O₄ (no distinct satellite)",
    "Fe2p: Fe⁰ + FeO（卫星 +6 eV）": "Fe2p: Fe⁰ + FeO (satellite +6 eV)",
    "文献范围": "literature range", "文献": "lit.", "篇文献": "references",
    "文献汇总（Biesinger 2011；Yamashita & Hayes 2008；Grosvenor 2004；McIntyre & Zetaruk 1977），非 NIST；导入 NIST 的 Fe 数据后自动替换":
        "Literature compilation (Biesinger 2011; Yamashita & Hayes 2008; Grosvenor 2004; McIntyre & Zetaruk 1977), not NIST; replaced automatically when NIST Fe data are imported",
    "不对称峰": "Asymmetric peak", "（分量）": " (partner)",
    "导入 NIST XPS 数据库导出的 zip / csv，用于推荐峰位": "Import a zip / csv exported from the NIST XPS database for reference positions",
    "点“＋ 单峰”后在图上点击，或套用模板": "Click “＋ Add peak” then click on the plot, or apply a template",
    "语言 / Language": "Language / 语言",
    "打开样品文件夹": "Open a sample folder",
    "⚡ 一键拟合（自动选背景、建模、区分重叠/卫星）": "⚡ One-click fit (auto background, model, overlap/satellite detection)",
    "检查报告，按需微调，导出": "Check the report, fine-tune if needed, export",
    "手动调整": "Manual adjustments",
    # 顶栏 / 通用
    "＋ 打开文件夹": "＋ Open folder", "打开文件": "Open files", "保存项目": "Save project", "导出 ▾": "Export ▾",
    "定量 · 原子 %": "Quantify · at.%", "导入 NIST": "Import NIST", "谱": "Spectra",
    "把 .VGD / 导出文件拖到这里": "Drop .VGD / exported files here", "还没有数据": "No data yet",
    "开始一次拟合": "Start a fit", "选择一个样品文件夹，里面所有元素的 .VGD 会一次读入；也可以选多个文件。":
        "Choose a sample folder — all .VGD files inside are loaded at once. You can also pick individual files.",
    "⤢ 复位": "⤢ Reset", "全谱": "Full range", "浏览": "Browse", "点击加峰": "Click to add", "选背景范围": "Pick background",
    "单峰": "Singlet", "双峰": "Doublet", "不对称": "Asymmetric", "原始数据": "Raw data", "背景": "Background",
    "包络": "Envelope", "残差": "Residual", "应用": "Apply", "取消": "Cancel", "保存": "Save", "完成": "Done",
    "添加": "Add", "编辑": "Edit", "清空": "Clear all", "清除": "Clear", "套用": "Apply", "重命名": "Rename",
    "无": "None", "自动": "Auto", "数值": "Value", "固定": "Fixed", "最小": "Min", "最大": "Max", "参数": "Parameter",
    "名称": "Name", "类型": "Type", "峰型": "Line shape", "面积": "Area", "面积 %": "Area %", "轨道": "Orbital",
    "样品": "Sample", "比例": "Ratio", "标题": "Title", "格式": "Format", "内容": "Content", "曲线": "Curve",
    "点选": "Pick", "折线": "Polyline", "未知": "Unknown", "卫星": "Satellite", "条谱": "spectra",
    # 右侧面板
    "拟合": "Fit", "⚡ 一键拟合": "⚡ One-click fit", "⚡ 一键拟合本样品全部谱": "⚡ One-click fit all spectra in sample",
    "局部微调": "Fine-tune", "多起点": "Multi-start", "用相同设置拟合全部同类谱": "Apply to all spectra of this orbital",
    "按当前设置拟合": "Fit with current settings", "峰位限制在当前值 ±0.3 eV 内重新拟合":
        "Refit with peak positions kept within ±0.3 eV", "从 12 组随机起点拟合，取最好结果，避免局部最优":
        "Fit from 12 random starts and keep the best (avoids local minima)",
    "自动选背景、按元素建模、检查残差并区分重叠峰/卫星峰":
        "Auto background, element model, residual check, overlap vs satellite classification",
    "能量校正": "Energy calibration", "平移 ΔBE (eV)": "Shift ΔBE (eV)",
    "选中峰 → C1s 284.8 eV（同样品一起平移）": "Selected peak → C1s 284.8 eV (shift whole sample)",
    "BE 范围": "BE range", "端点平均": "End averaging", "端点强度": "End intensity", "恢复自动": "Reset to auto",
    "图上背景两端的方块可拖动：左右改范围，上下改端点强度。":
        "Drag the square handles at both ends: sideways = range, up/down = end intensity.",
    "平滑（不过冲）": "Smooth (no overshoot)", "吸附到数据曲线": "Snap to data", "编辑点": "Edit points",
    "“编辑点”后在图上点击添加；拖动移动；右键或双击删除。背景范围 = 最左到最右的点。":
        "After “Edit points”, click to add, drag to move, right-click or double-click to delete. Range = leftmost to rightmost point.",
    "Smart（Shirley，不高于数据）": "Smart (Shirley, never above data)", "自定义点（在图上点选）": "Points (picked on plot)",
    "SNIP（全谱）": "SNIP (survey)", "推荐峰位": "Reference positions", "元素识别": "Element ID", "全谱识别": "Survey ID",
    "打开谱后显示": "Shown after a spectrum is opened", "峰": "Peaks", "双击行编辑": "double-click a row to edit",
    "＋ 添加峰": "＋ Add peak", "－ 删除峰": "－ Delete peak", "图上点击加": "Click on plot", "参数相等…": "Equal params…",
    "多选峰，让 FWHM 或 L% 相等": "Make FWHM or L% equal for several peaks", "存为模板": "Save as template",
    "自动（端点平均）": "auto (end average)", "峰位或 FWHM 卡在约束边界": "Position or FWHM at a constraint limit",
    "回到默认分析窗口（也可双击图）": "Back to the default window (or double-click the plot)",
    "显示整个采集范围": "Show the whole acquired range", "图上点击时添加的峰类型": "Peak type added by clicking",
    "点击添加的峰使用不对称峰型（金属态）": "Clicked peaks use the asymmetric line shape (metals)",
    "拖峰顶圆点移动 · 上下改高度 · Shift+拖改宽度 · 双击复位":
        "Drag apex dots to move · up/down = height · Shift+drag = width · double-click to reset",
    "在图上横向拖动框选背景范围，或点击两个端点": "Drag horizontally to select the background range, or click two end points",
    "点击添加背景点 · 拖动移动 · 右键/双击删除 · Esc 完成": "Click to add background points · drag to move · right/double-click to delete · Esc to finish",
    "点击添加单峰，可连续添加；Esc 结束": "Click to add singlets; Esc to finish",
    "数据中 ±0.3 eV 内有峰": "data has a peak within ±0.3 eV", "勾选的化合物在图上显示": "Ticked compounds are marked on the plot",
    "全选": "All", "全不选": "None", "化合物": "Compound", "单峰轨道": "singlet orbital",
    # 添加峰 / 编辑峰
    "添加峰": "Add peak", "一个独立组分": "an independent component", "自旋轨道 p / d / f": "spin–orbit p / d / f",
    "卫星峰": "Satellite", "shake-up 等，归属主峰": "shake-up etc., owned by a main peak", "未知重叠峰": "Unknown overlap",
    "不计入面积% 和定量": "excluded from area % and quantification", "留空自动命名": "leave empty for automatic name",
    "所属主峰": "Parent peak", "位置 BE (eV)": "Position BE (eV)", "主峰 BE (eV)": "Main BE (eV)",
    "卫星峰 BE (eV)": "Satellite BE (eV)", "分裂 ΔE (eV)": "Splitting ΔE (eV)", "面积比 副/主": "Area ratio minor/main",
    "拟合时锁定": "Lock in fit", "相对主峰 ΔE": "ΔE from parent", "固定 ΔE（否则在 ±1.5 eV 内拟合）": "Fix ΔE (else fitted within ±1.5 eV)",
    "面积比 卫星/主": "Area ratio sat/main", "固定面积比（否则自由拟合）": "Fix area ratio (else free)",
    "对称 GL": "Symmetric GL", "不对称（金属态，高 BE 侧拖尾）": "Asymmetric (metal, high-BE tail)",
    "不对称（高 BE 侧拖尾，金属态）": "Asymmetric (high-BE tail, metal)",
    "新峰的位置、FWHM、面积都自由拟合，可在“编辑”中加约束。": "Position, FWHM and area are free; add constraints in “Edit”.",
    "副峰跟随主峰。间距和面积比勾选“锁定”则拟合时保持不变；不勾选则在默认范围内拟合（之后可在“编辑”里改范围）。删除主峰时副峰一起删除。":
        "The minor component follows the main one. Locked split/ratio stay fixed; unlocked ones are fitted within default bounds (editable later). Deleting the main peak deletes its partner.",
    "卫星峰归属所选主峰，计入面积%。ΔE 和面积比可固定，也可自由拟合。":
        "The satellite belongs to the chosen main peak and counts in its area %. ΔE and area ratio can be fixed or free.",
    "用于描述来源不明、但会影响拟合的重叠信号。它参与拟合，但不计入面积% 和原子% 定量。":
        "For overlapping signal of unknown origin: fitted, but excluded from area % and at.% quantification.",
    "编辑峰": "Edit peak", "主峰（计入面积%）": "Main (counts in area %)", "卫星峰（计入面积%）": "Satellite (counts in area %)",
    "未知重叠峰（不计入面积% 与定量）": "Unknown overlap (excluded from area % and quantification)",
    "洛伦兹 L%": "Lorentzian L%", "拖尾高度 T (0–1)": "Tail height T (0–1)", "拖尾长度 τ (eV)": "Tail length τ (eV)",
    "约束：绑定到另一个峰（自旋轨道双峰、等宽…）": "Constraints: link to another peak (spin–orbit doublet, equal width…)",
    "参考峰": "Reference peak", "绑定项": "Linked", "位置差 ΔE (eV)": "Offset ΔE (eV)", "面积比 本/参考": "Area ratio this/ref",
    "FWHM 比（1 = 相等）": "FWHM ratio (1 = equal)", "峰型相同（L%、不对称与拖尾跟随参考峰）": "Same line shape (L%, asymmetry and tail follow the reference)",
    "不锁定时，该项在最小–最大之间拟合；范围留空则用默认（ΔE ±max(0.2, 10%)，比值 ×0.5–1.5）。":
        "Unlocked items are fitted between min and max; empty means default (ΔE ±max(0.2, 10%), ratios ×0.5–1.5).",
    "由绑定关系决定": "set by the link", "（不绑定）": "(no link)",
    "参数相等绑定": "Link equal parameters", "FWHM（半高峰宽）": "FWHM (full width at half maximum)", "L%（洛伦兹比例）": "L% (Lorentzian fraction)",
    "勾选要与参考峰相等的峰（可多选）；取消勾选即解除": "Tick the peaks that should equal the reference (multi-select); untick to unlink",
    "相等后拟合时这些峰一起变化。要把数值锁死：编辑参考峰，勾选 FWHM 的“固定”。双峰副峰自动跟随自己的主峰。":
        "Linked peaks vary together. To lock the value, edit the reference peak and tick FWHM “Fixed”. Doublet partners follow their own main peak.",
    "双峰副峰，跟随": "doublet partner, follows", "已绑定到": "already linked to",
    # 一键拟合报告
    "一键拟合报告": "One-click fit report", "样品中其他元素（可选）": "Other elements in sample (optional)",
    "如 Ni, Fe（survey 里有、但没测高分辨谱的元素）": "e.g. Ni, Fe (seen in the survey but without a region scan)",
    "重新一键拟合": "Re-run one-click fit",
    "用于判断重叠峰：程序会检查这些元素的俄歇峰（按 hν 换算到结合能）和其他能级是否落在当前窗口。":
        "Used to detect overlaps: Auger lines (converted with hν) and other levels of these elements are checked against the window.",
    "正在一键拟合…": "Running one-click fit…", "正在一键拟合本样品全部谱…": "Running one-click fit on all spectra of the sample…",
    # 导出
    "拟合图…（PNG / SVG / TIFF / PDF）": "Fit figure… (PNG / SVG / TIFF / PDF)", "全部峰参数 + 曲线 → Excel": "All peak parameters + curves → Excel",
    "当前谱曲线 CSV（Origin）": "Current curves CSV (Origin)", "当前谱 XPSPEAK txt": "Current spectrum XPSPEAK txt",
    "导出拟合图": "Export fit figure", "X 轴（结合能）": "X axis (binding energy)", "范围 (eV)": "Range (eV)", "主刻度间距": "Major tick step",
    "次刻度": "Minor ticks", "每格 1 条": "1 per interval", "每格 4 条": "4 per interval", "Y 轴（强度）": "Y axis (intensity)",
    "数字与刻度": "Numbers & ticks", "显示（关闭则只留标题，无刻度短线）": "Show (off = title only, no tick marks)",
    "图例": "Legend", "自动位置": "Auto", "左上": "Upper left", "右上": "Upper right", "图外右侧": "Outside right", "图名": "Title",
    "峰填充": "Peak fill", "峰标注": "Peak labels", "箭头指向峰（无峰图例）": "Arrows to peaks (no peak legend)",
    "空心点": "Open circles", "线": "Line", "尺寸与格式": "Size & format", "宽 × 高 (cm)": "Width × height (cm)",
    "字号 / DPI": "Font / DPI", "SVG（矢量）": "SVG (vector)", "PDF（矢量）": "PDF (vector)", "预览生成中…": "Rendering preview…",
    "导出": "Export", "高": "high", "低": "low",
    # 定量
    "定量 · 原子百分比与比例": "Quantification · atomic % and ratios", "用拟合峰面积（否则用扣背景积分）": "Use fitted peak areas (else background-subtracted integral)",
    "能量补偿 KE": "Energy correction KE", "灵敏度因子（ALTHERMO1，可改）": "Sensitivity factors (ALTHERMO1, editable)",
    "元素原子 %（每行一个样品）": "Element at.% (one row per sample)", "化学态原子 %": "Chemical-state at.%", "＋ 加入表格": "＋ Add to table",
    "化学态 ÷ 同元素总量即该态占比": "state ÷ element total = fraction of that state", "复制全部表格": "Copy all tables",
    "元素原子%": "Element at.%", "化学态原子%": "Chemical-state at.%", "（元素）": "(element)", "（有重复轨道，已用第一条）": "(duplicate orbital, first used)",
    # 提示 / 对话框
    "请先打开数据": "Please open data first", "先在峰表里选中一个峰": "Select a peak in the table first",
    "先在峰表里选中 C1s 的 C-C 峰": "Select the C1s C–C peak in the table first", "至少需要两个峰": "At least two peaks are needed",
    "请先添加主峰，卫星峰需要归属一个主峰": "Add a main peak first — a satellite must belong to one",
    "替换当前的峰？": "Replace the current peaks?", "删除全部峰？": "Delete all peaks?", "清除全部背景点？": "Clear all background points?",
    "模板名称": "Template name", "我的模板": "my template", "没有找到可读取的数据文件": "No readable data files found",
    "已复制，可直接粘贴到 Excel": "Copied — paste into Excel", "复制失败": "Copy failed", "拟合完成": "Fit complete",
    "多起点拟合中…": "Multi-start fitting…", "不是数字": "is not a number", "打开文件夹": "Open folder",
    "把选中的峰定到 284.8 eV，并对同一样品的所有谱做同样平移？": "Set the selected peak to 284.8 eV and shift all spectra of this sample?",
    "当前谱不是 p/d/f 轨道，将使用 2p 默认值（5.7 eV / 0.5）；Esc 结束": "Not a p/d/f orbital — using 2p defaults (5.7 eV / 0.5); Esc to finish",
    "已导出": "Exported", "项目已保存": "Project saved", "模板已保存": "Template saved", "已导入：": "Imported: ", "导入失败：": "Import failed: ",
    "出错：": "Error: ", "已读入": "Loaded", "（至少 2 个）": "(at least 2)", "（待确认）": "(tentative)",
    "（自动识别）": "(auto-identified)", "全谱：不做峰拟合，识别元素": "Survey: no peak fitting, element identification",
    "（确认）": "(confirmed)", "（待确认：只有部分谱线吻合）": "(tentative: only some lines match)",
    "峰位": "position", " FWHM ": " FWHM ", "，金属态用不对称峰": ", asymmetric shape for the metal",
    "语言": "Language", "手动调整": "Manual adjustments", "化学态": "Chemical states", "拟合质量": "Fit quality",
    "批量": "Batch", "暂无拟合结果": "No fit yet", "第 1 步": "Step 1", "第 2 步": "Step 2", "第 3 步": "Step 3",
    "打开样品文件夹": "Open a sample folder",
    "检查报告，按需微调，导出": "Check the report, fine-tune if needed, export",
}

# ------------------------------------------------------------------ 含变量的句子：中文 printf 模板 → 英文（{0}{1}… 为各变量，变量本身也会递归翻译）
TEMPLATES = [
    ("%d 个峰的 Lorentzian 比例（L%%）改为在 0–60%% 内拟合（双峰副峰跟随）", "Lorentzian fraction (L%) of {0} peaks now fitted within 0–60% (doublet partners follow)"),
    ("背景 %s → 主动 Shirley（台阶 + 线性斜坡，斜坡随峰拟合，仍不高于数据）", "Background {0} → active Shirley (step + linear slope fitted with the peaks, still never above the data)"),
    ("%d 个峰的 FWHM 范围放宽为原来的 0.7–1.4 倍（双峰的 FWHM 倍率仍按文献范围）", "FWHM range of {0} peaks widened to 0.7–1.4× (doublet FWHM ratios stay within the literature range)"),
    ("%d 个金属态峰的拖尾范围放宽（T ≤ 0.8，τ ≤ 3.5 eV）", "tail range of {0} metal peaks widened (T ≤ 0.8, τ ≤ 3.5 eV)"),
    ("%d 对自旋轨道双峰：分裂范围各放宽 0.1 eV、面积比各放宽 0.05、FWHM 倍率下限 −0.1 / 上限 +0.3（可超出文献范围，超出时会提示）",
     "{0} spin–orbit doublets: splitting range widened by 0.1 eV, area ratio by 0.05, FWHM ratio by −0.1 / +0.3 (may leave the literature range; flagged when it does)"),
    ("%d 个化合物峰允许轻微不对称（T ≤ 0.25）", "{0} compound peaks allowed a slight asymmetry (T ≤ 0.25)"),
    ("%d 个峰的峰位范围两侧各放宽 0.5 eV", "position range of {0} peaks widened by 0.5 eV on each side"),
    ("BIC 下降 %.0f，但化学态比例变化达 %.0f 个百分点 → 结果对这项约束很敏感，请复核后决定",
     "BIC dropped by {0}, but the chemical-state fractions changed by up to {1} percentage points → the result is sensitive to this constraint; review before deciding"),
    ("BIC 下降 %.0f（显著改善），没有新的物理问题 → 建议保留", "BIC dropped by {0} (significant), no new physical problem → keep"),
    ("BIC 下降 %.0f，但出现新的物理问题（见下）→ 请判断后决定是否撤销", "BIC dropped by {0}, but a new physical problem appeared (see below) → decide whether to undo"),
    ("BIC 只下降 %.0f（不显著，多出的参数不值得）→ 建议撤销", "BIC dropped only by {0} (not significant; the extra parameters are not justified) → undo"),
    ("BIC 没有下降（%s）→ 建议撤销", "BIC did not drop ({0}) → undo"),
    ("所有约束都已放开过（共 %d 级）", "All constraint levels have been tried ({0} levels)"),
    ("已撤销第 %d 级（%s）", "Level {0} ({1}) undone"),
    ("自动放宽完成（低、中风险的前 %d 级）：保留 %d 级%s；高风险的级别请手动逐级放开",
     "Auto relaxation done (low/medium-risk levels 1–{0}): kept {1} level(s){2}; release the high-risk levels manually, one at a time"),
    ("，默认约束已经合适", ", the default constraints are already adequate"),
    ("下一级：%s（风险：%s）— %s", "Next level: {0} (risk: {1}) — {2}"),
    ("%s（风险：%s）", "{0} (risk: {1})"),
    ("请先拟合（一键拟合或添加峰）", "Fit first (one-click fit or add peaks)"),
    ("没有可撤销的放宽步骤", "No relaxation step to undo"),
    ("点击添加双峰（%s：分裂、面积比、FWHM 倍率在文献范围内拟合，起点 %s eV / %s）；Esc 结束",
     "Click to add a doublet ({0}: splitting, area ratio and FWHM ratio are fitted within the literature ranges, starting from {1} eV / {2}); Esc to finish"),
    ("%s：%s 到达文献范围边界（约束在起作用，1/2 区域的背景或重叠值得检查）",
     "{0}: {1} reached the literature-range limit (the constraint is active; check the background or overlaps around the 1/2 component)"),
    ("分裂 %s", "splitting {0}"), ("面积比 %s", "area ratio {0}"), ("FWHM 倍率 %s", "FWHM ratio {0}"),
    ("ΔE %s–%s eV（常用 %s）· 面积比 %s–%s · FWHM 倍率 %s–%s（分裂太小，锁定）",
     "ΔE {0}–{1} eV (typical {2}) · area ratio {3}–{4} · FWHM ratio {5}–{6} (splitting too small, locked)"),
    ("ΔE %s–%s eV（常用 %s）· 面积比 %s–%s · FWHM 倍率 %s–%s", "ΔE {0}–{1} eV (typical {2}) · area ratio {3}–{4} · FWHM ratio {5}–{6}"),
    ("金属态 %s", "metal: {0}"), ("化合物 %s", "compound: {0}"),
    ("文献范围（%s）：%s。不锁定时在此范围内拟合，FWHM 倍率总在此范围内拟合",
     "Literature range ({0}): {1}. When unlocked it is fitted within this range; the FWHM ratio is always fitted within its range"),
    ("拟合约束（1/2 与 3/2）：", "Fit constraints (1/2 vs 3/2):"),
    ("%s 的自旋轨道分裂 %.2f eV 不在文献范围 %.2f–%.2f eV", "spin–orbit splitting of {0} is {1} eV, outside the literature range {2}–{3} eV"),
    ("%s 面积比 %.2f 不在文献范围 %.2f–%.2f（背景或重叠可能有问题）", "area ratio of {0} is {1}, outside the literature range {2}–{3} (check background or overlaps)"),
    ("%s 的 FWHM 倍率 %.2f 不在文献范围 %.2f–%.2f", "FWHM ratio of {0} is {1}, outside the literature range {2}–{3}"),
    ("背景：%sFe2p 主峰之间是卫星结构 → 主动 Shirley（Shirley 台阶 + 线性斜坡，斜坡随峰一起拟合，背景不高于数据）",
     "Background: {0}the Fe2p region between the main lines holds satellites → active Shirley (Shirley step + linear slope fitted together with the peaks, never above the data)"),
    ("Shirley 会高于数据 %.0f（%.0fσ）；", "Shirley would exceed the data by {0} ({1}σ); "),
    ("背景：两端平坦区显示本底有线性斜坡 %+.0f /eV（±%.0f），Shirley 会%s → Shirley + 斜坡（%s）",
     "Background: the flat ends show a linear background slope of {0} /eV (±{1}); Shirley would {2} → Shirley + slope ({3})"),
    ("高于数据 %.0f（%.0fσ）", "exceed the data by {0} ({1}σ)"),
    ("背景：Shirley 会高于数据 %.0f（%.0fσ），本底除台阶外还有线性斜坡 → Shirley + 斜坡（斜率取自低 BE 端平坦区；高于数据 %.0f，已限制为不高于数据）",
     "Background: Shirley would exceed the data by {0} ({1}σ); the background has a linear slope besides the step → Shirley + slope (slope from the flat low-BE end; exceeds data by {2}, clamped to never exceed the data)"),
    ("背景：Shirley 会高于数据 %.0f（%.0fσ）；本元素主峰之间是卫星结构，不宜用点背景 → Smart（不高于数据）",
     "Background: Shirley would exceed the data by {0} ({1}σ); the region between the main lines holds satellites, so a point background is unsuitable → Smart (never above data)"),
    ("Fe2p：708.6–713 eV 几乎没有氧化态信号（%.1f%%）→ 只用金属 Fe⁰（不对称双峰 Δ12.95–13.3）",
     "Fe2p: almost no oxide signal at 708.6–713 eV ({0}%) → metallic Fe⁰ only (asymmetric doublet Δ12.95–13.3)"),
    ("Fe 单相模型（BIC，越小越好）：%s → 主物相 %s（Fe²⁺ 占氧化 Fe 的 %.0f%%）",
     "Fe single-phase models (BIC, lower is better): {0} → main phase {1} (Fe²⁺ = {2}% of oxidized Fe)"),
    ("两相模型 %s（%s）：Fe²⁺ 占 %.0f%%。两相模型多出一整套峰，几乎总能拟合得更好，不能据此确认混合 → 默认显示单相；Fe²⁺ 比例报告为 %.0f–%.0f%%",
     "two-phase model {0} ({1}): Fe²⁺ {2}%. A two-phase model adds a whole extra set of peaks and almost always fits better, so this does not prove a mixture → the single phase is shown by default; Fe²⁺ fraction reported as {3}–{4}%"),
    ("若为 %s 混合，Fe²⁺ 约 %.0f%%（报告范围 %.0f–%.0f%%）", "if a {0} mixture, Fe²⁺ ≈ {1}% (reported range {2}–{3}%)"),
    ("解读：%s", "Interpretation: {0}"),
    ("物相 %s", "Phases {0}"),
    ("推算 Fe²⁺:Fe³⁺ = %s（Fe²⁺ 占氧化 Fe 的 %.0f%%） → 与 %s 一致", "estimated Fe²⁺:Fe³⁺ = {0} (Fe²⁺ = {1}% of oxidized Fe) → consistent with {2}"),
    ("推算 Fe²⁺:Fe³⁺ = %s（Fe²⁺ 占氧化 Fe 的 %.0f%%）", "estimated Fe²⁺:Fe³⁺ = {0} (Fe²⁺ = {1}% of oxidized Fe)"),
    ("另有金属 Fe⁰ %.0f%%（占 Fe2p 总面积）", "plus metallic Fe⁰ {0}% (of the total Fe2p area)"),
    ("文献 · %d 种化合物 %s", "Literature · {0} compounds {1}"),
    ("+%d 条提示", "+{0} more notes"),
    ("%s\n双击重命名，右键移除", "{0}\nDouble-click to rename, right-click to remove"),
    ("（样品元素：%s）", "(sample elements: {0})"),
    ("残差峰 %.2f eV（占 %.1f%%）：%s", "Residual peak {0} eV ({1}%): {2}"),
    ("残差峰 %.2f eV（占 %.1f%%）：加入后拟合没有显著改善（BIC），不保留", "Residual peak {0} eV ({1}%): no significant improvement (BIC) — not kept"),
    ("%.2f eV 处的残差峰与 %s（%.1f eV）相差 %.1f eV，且样品含该元素 → 重叠峰（不计入定量）",
     "the residual peak at {0} eV is {3} eV from {1} ({2} eV) and the sample contains that element → overlap (not quantified)"),
    ("%.2f eV 处的残差峰位于 %s 高 BE 侧 +%.1f eV（%s 卫星区间 %.0f–%.0f eV）→ 卫星峰",
     "the residual peak at {0} eV lies +{2} eV above {1} ({3} satellite window {4}–{5} eV) → satellite"),
    ("%.2f eV 接近 NIST 中 %s（%.2f eV）→ 新化学态", "{0} eV is close to NIST {1} ({2} eV) → new chemical state"),
    ("%.2f eV 的残差峰在已识别的 %s 附近（%.1f eV 内），俄歇峰很宽 → 同一重叠来源",
     "the residual peak at {0} eV is within {2} eV of the identified {1}; Auger peaks are broad → same overlap source"),
    ("%.2f eV 的残差峰无法归属（不是同样品其他元素的谱线，也不在卫星区间）→ 作为未知峰，不计入定量，请确认",
     "the residual peak at {0} eV cannot be assigned (no line of another element in the sample, not in a satellite window) → unknown peak, not quantified — please check"),
    ("%.2f eV 落在 %s 的常见范围（%.1f–%.1f eV）→ 新化学态", "{0} eV falls in the usual range of {1} ({2}–{3} eV) → new chemical state"),
    ("%d 个峰的 %s 与 “%s” 相等", "{0} peaks now share {1} with “{2}”"),
    ("%s 俄歇 (KE≈%d)", "{0} Auger (KE≈{1})"), ("overlap: %s 俄歇 (KE≈%d)", "overlap: {0} Auger (KE≈{1})"),
    ("overlap: %s（分量）", "overlap: {0} (partner)"), ("%s（续）", "{0} (cont.)"),
    ("%s 的%s卡在边界（%.2f）", "{1} of {0} is at its limit ({2})"),
    ("%s 面积比 %.2f 偏离理论值 %.2f 超过 15%%（背景或重叠可能有问题）", "{0}: area ratio {1} deviates >15% from the theoretical {2} (check background or overlaps)"),
    ("%s（%d 条）", "{0} ({1} records)"), ("%s（自动识别）", "{0} (auto-identified)"),
    ("Al2p：金属 Al⁰ 用不对称双峰（Δ0.44 锁定）；氧化物 Al³⁺ 双峰 FWHM ≤ 2.0 eV（Thermo）",
     "Al2p: metallic Al⁰ as an asymmetric doublet (Δ0.44 locked); oxide Al³⁺ doublet with FWHM ≤ 2.0 eV (Thermo)"),
    ("C1s：282–283.7 eV 有明显信号（%.0f%%）→ 加入金属碳化物", "C1s: clear signal at 282–283.7 eV ({0}%) → metal carbide added"),
    ("Co2p 模型对比：A（780–787 eV 肩峰全归 Co²⁺）Co²⁺ %.0f%%；B（含金属 Co 的 +3/+5 eV 损失峰）Co²⁺ %.0f%%。两者都与数据一致，报告 Co²⁺ 范围 %.0f–%.0f%%",
     "Co2p model comparison: A (780–787 eV shoulder all Co²⁺) Co²⁺ {0}%; B (with metallic Co +3/+5 eV loss features) Co²⁺ {1}%. Both agree with the data; Co²⁺ range {2}–{3}%"),
    ("Co2p：Co⁰ 不对称双峰 Δ14.9–15.1；Co²⁺ 双峰 Δ15.2–15.7 + 卫星 ~786 / ~802 eV",
     "Co2p: Co⁰ asymmetric doublet Δ14.9–15.1; Co²⁺ doublet Δ15.2–15.7 + satellites ~786 / ~802 eV"),
    ("Cr2p：Cr⁰ 不对称双峰 Δ9.2–9.4；Cr³⁺ 双峰 Δ9.6–9.9（多重分裂以宽峰近似）",
     "Cr2p: Cr⁰ asymmetric doublet Δ9.2–9.4; Cr³⁺ doublet Δ9.6–9.9 (multiplet approximated by a broad peak)"),
    ("Cu2p：940–946 eV 无卫星信号（%.1f%%）→ 不加 Cu²⁺，只用 Cu⁰/Cu⁺", "Cu2p: no satellite signal at 940–946 eV ({0}%) → no Cu²⁺, Cu⁰/Cu⁺ only"),
    ("Cu2p：940–946 eV 有卫星信号（%.1f%%）→ Cu²⁺ 及其卫星峰作为候选，拟合后用 BIC 判断是否保留",
     "Cu2p: satellite signal at 940–946 eV ({0}%) → Cu²⁺ and its satellite as a candidate, kept only if BIC improves"),
    ("O1s：晶格 O / 缺陷·羟基 / C–O·吸附水 三组分", "O1s: lattice O / defect·hydroxyl / C–O·adsorbed water"),
    ("⚠ %s（%s eV）与本谱主峰几乎重合，无法拟合区分，可能使比例偏高；请结合该元素含量判断",
     "⚠ {0} ({1} eV) nearly coincides with a main peak and cannot be separated — the ratio may be overestimated; judge with that element's content"),
    ("⚠ 文件名为 %s，但谱图更像 %s（%s）；如果文件名有误，请重命名后再拟合",
     "⚠ The file name says {0} but the spectrum looks like {1} ({2}); rename and refit if the name is wrong"),
    ("候选 %s：BIC 虽更好，但 %d 个参数卡在约束边界（在填补背景/其他信号，不像真实的 %s）→ 不保留",
     "Candidate {0}: BIC is better but {1} parameters sit at their limits (it is filling background/other signal, not a real {2}) → not kept"),
    ("候选 %s：保留后拟合明显更好（BIC），参数都在合理范围内 → 保留", "Candidate {0}: clearly better fit (BIC) with parameters in range → kept"),
    ("候选 %s：去掉后拟合同样好或更好（BIC），不保留", "Candidate {0}: fit is as good or better without it (BIC) → not kept"),
    ("出错：%s", "Error: {0}"),
    ("初次拟合：约化 χ² %.3g（先固定金属拖尾定位各组分，再放开拖尾）", "Initial fit: reduced χ² {0} (metal tails fixed first to place components, then freed)"),
    ("化学态测试：加入 %s（%.1f–%.1f eV）使拟合明显改善，占 %.1f%% → 新化学态，保留",
     "State test: adding {0} ({1}–{2} eV) clearly improves the fit, {3}% → new chemical state, kept"),
    ("多起点拟合完成：约化 χ² %.4g", "Multi-start fit done: reduced χ² {0}"), ("（之前 %.4g）", " (before {0})"),
    ("多起点拟合：约化 χ² %.3g → %.3g", "Multi-start fit: reduced χ² {0} → {1}"),
    ("完成：约化 χ² %.3g，残差 RMS / 峰高 %.1f%%", "Done: reduced χ² {0}, residual RMS / peak height {1}%"),
    ("局部微调完成（峰位 ±%.1f eV）", "Fine-tune done (positions ±{0} eV)"),
    ("已平移 %+.3f eV（%d 条谱）", "Shifted {0} eV ({1} spectra)"), ("已拟合 %d 条同类谱", "Fitted {0} spectra of the same orbital"),
    ("；失败 %d", "; {0} failed"),
    ("提示：%s（%.1f–%.1f eV）与 %s 相距不到 1.5 eV，此分辨率下无法区分，未单独拟合（可能包含在该组分中）",
     "Note: {0} ({1}–{2} eV) is within 1.5 eV of {3} and cannot be resolved at this resolution — not fitted separately (may be contained in that component)"),
    ("背景在某处高于数据 %.0f（噪声 σ≈%.0f）", "Background exceeds the data by {0} somewhere (noise σ≈{1})"),
    ("背景：Shirley 会高于数据 %.0f（%.0fσ），窗口内可能有其他谱线 → 自定义点背景（%d 个平坦区最低点）",
     "Background: Shirley would exceed the data by {0} ({1}σ), probably another line in the window → point background ({2} flat-region minima)"),
    ("背景：Shirley 会高于数据 %.0f（噪声 σ≈%.0f）→ Smart（不高于数据）", "Background: Shirley would exceed the data by {0} (noise σ≈{1}) → Smart (never above data)"),
    ("背景：Shirley（未高于数据，噪声 σ≈%.0f）", "Background: Shirley (never above data, noise σ≈{0})"),
    ("背景：低 BE 端（%.0f）高于高 BE 端（%.0f），本底反向，Shirley 不适用 → 自定义点背景（%d 个平坦区最低点）",
     "Background: low-BE end ({0}) is higher than high-BE end ({1}) — inverted baseline, Shirley not applicable → point background ({2} flat-region minima)"),
    ("能量：%s 的峰顶在 %.2f eV，与参考值相差 %+.2f eV，模型已整体平移；建议做能量校正（如金属峰或 C1s）",
     "Energy: the {0} apex is at {1} eV, {2} eV from the reference; the model was shifted — energy calibration recommended (metal peak or C1s)"),
    ("轨道：文件名中没有轨道信息，根据谱图识别为 %s（%s）", "Orbital: none in the file name — identified from the spectrum as {0} ({1})"),
    ("通用模型：在数据中找到 %d 个主峰（%s）", "Generic model: {0} main peaks found in the data ({1})"),
    ("预置重叠峰 %s 拟合后面积 <1%%，去掉", "Prior overlap {0} has <1% area after fitting — removed"),
    ("预置重叠峰：样品含 %s，%s（%s eV）落在窗口内 → 预先加入为重叠峰（不计入定量）",
     "Prior overlap: the sample contains {0}; {1} ({2} eV) is in the window → added as an overlap (not quantified)"),
    ("预置重叠峰：样品含 %s，其俄歇峰（KE≈%d → BE≈%.1f eV）落在窗口内 → 预先加入为重叠峰",
     "Prior overlap: the sample contains {0}; its Auger line (KE≈{1} → BE≈{2} eV) is in the window → added as an overlap"),
    ("默认显示模型 A（B 没有明显更好）", "Showing model A (B is not clearly better)"), ("默认显示模型 B（BIC 更好）", "Showing model B (better BIC)"),
    ("%.1f eV 有峰（%s 范围 %.1f–%.1f）", "peak at {0} eV ({1} range {2}–{3})"), ("+%.2f eV 处有自旋轨道分量", "spin–orbit partner at +{0} eV"),
    ("缺少 +%.2f eV 的自旋轨道分量", "no spin–orbit partner at +{0} eV"),
    ("%s （确认）：%s", "{0} (confirmed): {1}"),
    ("%s （待确认：只有部分谱线吻合）：%s", "{0} (tentative: only some lines match): {1}"),
    ("未知操作 %s", "Unknown action {0}"), ("没有找到峰，停止", "No peaks found — stopped"), ("没有识别到谱数据", "No spectrum data recognised"),
    ("双峰需要分裂和面积比", "A doublet needs splitting and area ratio"),
    ("无法读取 %s", "Cannot read {0}"), ("%s：%s", "{0}: {1}"),
    # 前端动态文本（显示在页面上的完整文本）
    ("已读入 %d 条谱", "Loaded {0} spectra"), ("%d 个点", "{0} points"), ("NIST · %d 条 %s", "NIST · {0} records {1}"),
    ("约化 χ² %s   残差 RMS %s%%   自由参数 %s", "reduced χ² {0}   residual RMS {1}%   free parameters {2}"),
    ("约化 χ² %s   自由参数 %s", "reduced χ² {0}   free parameters {1}"),
    ("移除 “%s”？（不会删除原文件）", "Remove “{0}”? (the original file is not deleted)"),
    ("已添加 %s（%s eV，双峰 ΔE %s）", "Added {0} ({1} eV, doublet ΔE {2})"),
    ("已添加 %s（%s eV，双峰 ΔE %s，金属态用不对称峰）", "Added {0} ({1} eV, doublet ΔE {2}, asymmetric shape for the metal)"),
    ("已添加 %s（%s eV）", "Added {0} ({1} eV)"),
    ("暂无 %s 的参考数据。已有：%s。可点顶部“导入 NIST”导入 NIST XPS 数据库导出的文件。",
     "No reference data for {0}. Available: {1}. Use “Import NIST” at the top to import NIST XPS database exports."),
    ("双峰间距 %s eV（%s 条，%s–%s）· 面积比 %s", "doublet split {0} eV ({1} records, {2}–{3}) · area ratio {4}"),
    ("手动（低 BE 端 %s · 高 BE 端 %s）", "manual (low-BE end {0} · high-BE end {1})"),
    ("把当前的峰和背景设置复制到其他所有 %s 谱并拟合？", "Copy the current peaks and background to all other {0} spectra and fit?"),
    ("点击添加双峰（%s：分裂 %s eV、面积比 %s，锁定；可在“编辑”中修改）；Esc 结束",
     "Click to add doublets ({0}: split {1} eV, area ratio {2}, locked; editable in “Edit”); Esc to finish"),
    ("%s · %s　（样品元素：%s）", "{0} · {1}  (sample elements: {2})"),
    ("已根据谱图识别为 %s（%s）；不对的话双击左侧名称重命名", "Identified from the spectrum as {0} ({1}); if wrong, double-click the name on the left to rename"),
    ("已根据谱图识别为 %s（%s）", "Identified from the spectrum as {0} ({1})"),
]


ANY = "(.+?)"
NUM = r"([-+]?\d[\d.]*(?:e[-+]?\d+)?)"


def _compile(zh):
    toks = re.split(r"(%[-+ 0#]*\d*(?:\.\d+)?[sdfgr%])", zh)
    out, n = [], 0
    for t in toks:
        if not t:
            continue
        if re.fullmatch(r"%[-+ 0#]*\d*(?:\.\d+)?[sr]", t):
            out.append(ANY); n += 1
        elif re.fullmatch(r"%[-+ 0#]*\d*(?:\.\d+)?[dfg]", t):
            out.append(NUM); n += 1                       # 数字占位只匹配数字，避免吞掉前面的文字
        elif t == "%%":
            out.append("%")
        else:
            out.append(re.escape(t))
    return "^" + "".join(out) + "$", n


PATTERNS = []
for _zh, _en in TEMPLATES:
    if _zh.startswith("⚠"):                       # tr() 先剥掉 ⚠ 再匹配，模板也去掉
        _zh, _en = _zh.lstrip("⚠ "), _en.lstrip("⚠ ")
    _rx, _n = _compile(_zh)
    PATTERNS.append((_rx, _en))
# 固定文字越多（越具体）的模板越先匹配
PATTERNS.sort(key=lambda t: -len(t[0].replace(ANY, "").replace(NUM, "")))
_COMPILED = [(re.compile(rx), en) for rx, en in PATTERNS]       # 通配不跨行：多行文字逐行翻译
# 兜底的词组替换只用较长的词组（短词会误伤长句，如“轨道”出现在“自旋轨道”里）
_SUBS = sorted((k for k in PHRASES if len(k) >= 4 and re.search(r"[一-鿿]", k)), key=len, reverse=True)
_PUNCT = [("、", ", "), ("；", "; "), ("（", " ("), ("）", ")"), ("：", ": "), ("，", ", "), ("。", ". "), ("　", "  ")]


def _punct(t):
    for a, b in _PUNCT:
        t = t.replace(a, b)
    return re.sub(r"\(\s+", "(", t)


def tr(text, depth=0):
    """中文 → 英文（找不到时尽量替换其中的词组）。"""
    if not text or not re.search(r"[一-鿿]", text) or depth > 4:
        return text
    lead = re.match(r"^\s*", text).group(0)
    tail = re.search(r"\s*$", text).group(0)
    core_ = text.strip()
    if core_.startswith("⚠"):                                  # 警告符号单独处理
        return lead + "⚠ " + tr(core_.lstrip("⚠ "), depth + 1).strip() + tail
    if core_ in PHRASES:
        return lead + PHRASES[core_] + tail
    for rx, en in _COMPILED:
        m = rx.match(core_)
        if m:
            parts = [tr(g.strip(), depth + 1) for g in m.groups()]
            try:
                return lead + _punct(en.format(*parts)) + tail
            except (IndexError, KeyError):
                break
    if "\n" in core_:                                          # 多行：逐行翻译
        return lead + "\n".join(tr(x, depth + 1) for x in core_.split("\n")) + tail
    if "；" in core_:                                           # 多条理由：逐条翻译
        return lead + "; ".join(tr(x, depth + 1) for x in core_.split("；")) + tail
    out = core_
    for k in _SUBS:
        if k in out:
            out = out.replace(k, PHRASES[k])
    return lead + _punct(out) + tail


def table():
    """给前端用：整句表 + 正则模板。"""
    return {"phrases": PHRASES, "patterns": PATTERNS}
