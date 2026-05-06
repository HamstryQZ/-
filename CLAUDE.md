# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

"信号与系统辅助学习" — 武汉大学电子信息专业「信号与系统」课程的辅助学习工具。目标是开发交互式可视化/演示工具，帮助理解信号与系统的核心概念。

## 技术栈

- **Python 3** + **Jupyter Notebook**：原型验证、公式推导、信号可视化
- 按需引入 **LaTeX**（数学公式）、**NumPy/SciPy**（数值计算）、**Matplotlib**（绘图）、**HTML/JS**（如果做 Web 交互）


## 目录结构约定

```
信号与系统辅助学习/
├── CLAUDE.md                # 本文件
├── notebooks/               # Jupyter notebooks（按主题组织）
│   ├── 01-连续信号/
│   ├── 02-卷积/
│   ├── 03-傅里叶变换/
│   └── ...
├── src/                     # Python 源码模块（如有）
│   └── ...
├── assets/                  # 图片、动画等静态资源
├── scripts/                 # 辅助脚本
└── requirements.txt         # Python 依赖
```

新增主题目录时同步更新本文件。用完不再需要的临时 notebook 及时清理。

## 开发命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Jupyter
jupyter notebook

# 或使用 VS Code 直接打开 .ipynb 文件
```

## 命名规范

- 目录名 / notebook 文件名：`{编号}-{英文描述}`，如 `03-fourier-transform`
- Python 源码：snake_case
- 变量、函数：英文，清晰表意，不缩写
