# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

常用信号（冲激、阶跃、指数、正弦、矩形脉冲、高斯等）的傅里叶变换 / 拉普拉斯变换交互式可视化工具。Streamlit 单页面应用。

## 技术栈

- Python 3 + Streamlit
- NumPy (FFT 数值频谱)
- Plotly (交互式矢量图：拖拽、缩放、悬停)

## 代码结构

```
06-transform-visualizer/
├── app.py          # 主应用（单文件）
│   ├─ 字体配置     (复用中文字体检测回退逻辑)
│   ├─ SIGNALS 字典 (所有信号定义，含时域函数/FT解析式/LT极点/ROC)
│   ├─ 辅助函数     (compute_spectrum, plot_pole_zero)
│   └─ UI 布局      (侧边栏 → 3个Tab: 时域/傅里叶/拉普拉斯)
├── 启动.py          # 双击启动脚本
└── CLAUDE.md
```

## 信号定义规范

新增信号时在 `SIGNALS` 字典中添加条目，字段说明：

| 字段 | 说明 |
|------|------|
| `label` | 显示名称（支持中文） |
| `latex` | 时域 LaTeX 表达式 |
| `causal` | 是否为因果信号（决定是否显示 LT） |
| `params` | 可调参数列表：`[(key, label, min, max, default, step), ...]` |
| `time_func` | `lambda t, **params: ndarray` 生成时域采样 |
| `ft_latex` | 傅里叶变换解析表达式 |
| `ft_has_delta` | FT 是否含 δ 冲激（影响数值显示提示） |
| `lt_latex` | 拉普拉斯变换表达式（None 则不显示 LT Tab） |
| `poles` | 默认极点列表 `[(real, imag), ...]` |
| `zeros` | 默认零点列表 `[(real, imag), ...]` |
| `roc` | ROC 描述 LaTeX |
| `note` | 教学说明文字 |

注意：
- FT 数值计算使用 FFT，对含冲激的 FT（如正弦/阶跃）无法精确显示 δ 分量，需在 `ft_has_delta` 标记并在 UI 提示
- 零极点位置需在 Tab 3 中根据 `params` 实时计算（见现有各信号的处理逻辑）
- 非因果信号 (`causal=False`) 的 LT 相关字段置 None，UI 自动显示"不存在"提示

## 关键逻辑

- `compute_spectrum` — 带 `@st.cache_data` 缓存的 FFT 计算，依赖参数序列化字符串
- Tab 3 的零极点图：每个信号的 pole/zero 需根据 `params` 实时重新计算（因参数改变位置）
- 幅度谱显示范围自动截取到能量集中区域，避免过宽的平坦区

## 启动方式

```bash
streamlit run app.py
```
