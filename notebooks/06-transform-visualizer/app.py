"""
常用信号及其傅里叶变换 / 拉普拉斯变换 —— 交互式可视化 (Streamlit 版)
运行方式: streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import matplotlib

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(page_title="信号变换可视化工具", layout="wide")

# ============================================================
# 中文字体配置
# ============================================================
_CN_FONTS = [
    'Microsoft YaHei', 'SimHei', 'STXihei', 'DengXian',
    'Noto Sans CJK SC', 'Noto Sans SC', 'Source Han Sans SC',
    'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
    'PingFang SC', 'STHeiti', 'Apple LiGothic',
]
_AVAILABLE = {f.name for f in fm.fontManager.ttflist}
_CHOSEN_FONT = next((f for f in _CN_FONTS if f in _AVAILABLE), None)

if _CHOSEN_FONT:
    plt.rcParams['font.sans-serif'] = [_CHOSEN_FONT] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['axes.unicode_minus'] = False
else:
    _FONT_CACHE = os.path.join(os.path.dirname(__file__), '.fonts')
    os.makedirs(_FONT_CACHE, exist_ok=True)
    _FONT_PATH = os.path.join(_FONT_CACHE, 'NotoSansSC-Regular.ttf')
    if not os.path.exists(_FONT_PATH):
        import urllib.request
        _url = ("https://github.com/google/fonts/raw/main/ofl/notosanssc/"
                "NotoSansSC%5Bwght%5D.ttf")
        urllib.request.urlretrieve(_url, _FONT_PATH)
    fm.fontManager.addfont(_FONT_PATH)
    plt.rcParams['font.sans-serif'] = ['Noto Sans SC'] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['axes.unicode_minus'] = False

plt.rcParams.update({'figure.dpi': 100, 'font.size': 10, 'axes.titlesize': 12})

# ============================================================
# 信号定义
# ============================================================
# 每个信号包含：
#   label        — 显示名称
#   latex        — 时域表达式 (LaTeX)
#   causal       — 是否为因果信号
#   params       — 参数列表 [(key, label, min, max, default, step), ...]
#   time_func    — 函数 x(t, **params)
#   ft_latex     — 傅里叶变换表达式
#   ft_has_delta — FT 是否含冲激（影响频谱显示方式）
#   lt_latex     — 拉普拉斯变换表达式 (None 表示不存在)
#   poles        — 极点列表 [(real, imag), ...]
#   zeros        — 零点列表 [(real, imag), ...]
#   roc          — ROC 描述文字

SIGNALS = {
    "impulse": {
        "label": "单位冲激 δ(t)",
        "latex": r"$x(t) = \delta(t)$",
        "causal": True,
        "params": [],
        "time_func": lambda t, **kw: (np.abs(t) < 1e-10).astype(float),
        "ft_latex": r"$X(j\omega) = 1$",
        "ft_has_delta": False,
        "lt_latex": r"$X(s) = 1$",
        "poles": [],
        "zeros": [],
        "roc": "全 $s$ 平面",
        "note": "冲激信号包含所有频率分量，且各分量的幅度和相位均相等，是最理想的测试信号。",
    },
    "step": {
        "label": "单位阶跃 u(t)",
        "latex": r"$x(t) = u(t)$",
        "causal": True,
        "params": [],
        "time_func": lambda t, **kw: (t >= 0).astype(float),
        "ft_latex": r"$X(j\omega) = \pi\delta(\omega) + \dfrac{1}{j\omega}$",
        "ft_has_delta": True,
        "lt_latex": r"$X(s) = \dfrac{1}{s}$",
        "poles": [(0, 0)],
        "zeros": [],
        "roc": r"$\mathrm{Re}\{s\} > 0$",
        "note": "阶跃信号在 $t=0$ 处跳变，其频谱以 $1/\omega$ 衰减，包含丰富的低频分量。",
    },
    "exp_decay": {
        "label": "单边指数衰减 e^{-at}u(t)",
        "latex": r"$x(t) = e^{-at}u(t),\quad a>0$",
        "causal": True,
        "params": [("a", "衰减系数 a", 0.2, 5.0, 1.0, 0.1)],
        "time_func": lambda t, a=1.0: np.exp(-a * t) * (t >= 0),
        "ft_latex": r"$X(j\omega) = \dfrac{1}{a + j\omega}$",
        "ft_has_delta": False,
        "lt_latex": r"$X(s) = \dfrac{1}{s + a}$",
        "poles": [(-1, 0)],  # will be scaled by a
        "zeros": [],
        "roc": r"$\mathrm{Re}\{s\} > -a$",
        "note": "指数衰减信号的频谱幅度 $|X(j\omega)| = 1/\sqrt{a^2+\omega^2}$，呈低通特性，$a$ 越大衰减越快，带宽越宽。",
    },
    "exp_ramp": {
        "label": "单边衰减斜坡 t·e^{-at}u(t)",
        "latex": r"$x(t) = t e^{-at}u(t),\quad a>0$",
        "causal": True,
        "params": [("a", "衰减系数 a", 0.2, 5.0, 1.0, 0.1)],
        "time_func": lambda t, a=1.0: t * np.exp(-a * t) * (t >= 0),
        "ft_latex": r"$X(j\omega) = \dfrac{1}{(a + j\omega)^2}$",
        "ft_has_delta": False,
        "lt_latex": r"$X(s) = \dfrac{1}{(s + a)^2}$",
        "poles": [(-1, 0), (-1, 0)],  # 二阶极点
        "zeros": [],
        "roc": r"$\mathrm{Re}\{s\} > -a$",
        "note": "与指数衰减相比，$t e^{-at}$ 在 $t=0$ 处从零开始增长，峰值在 $t=1/a$ 处。",
    },
    "sine": {
        "label": "单边正弦 sin(ω₀t)u(t)",
        "latex": r"$x(t) = \sin(\omega_0 t)u(t)$",
        "causal": True,
        "params": [("f0", "频率 f₀ (Hz)", 0.5, 5.0, 1.0, 0.1)],
        "time_func": lambda t, f0=1.0: np.sin(2 * np.pi * f0 * t) * (t >= 0),
        "ft_latex": r"$X(j\omega) = \dfrac{\omega_0}{\omega_0^2-\omega^2}"
                    r"+ \dfrac{\pi}{2j}[\delta(\omega-\omega_0)-\delta(\omega+\omega_0)]$",
        "ft_has_delta": True,
        "lt_latex": r"$X(s) = \dfrac{\omega_0}{s^2 + \omega_0^2}$",
        "poles": [(0, 1), (0, -1)],  # will be scaled
        "zeros": [(0, 0)],
        "roc": r"$\mathrm{Re}\{s\} > 0$",
        "note": "正弦信号的拉普拉斯变换在 $s=\pm j\omega_0$ 处有一对共轭极点，对应时域的等幅振荡。",
    },
    "cosine": {
        "label": "单边余弦 cos(ω₀t)u(t)",
        "latex": r"$x(t) = \cos(\omega_0 t)u(t)$",
        "causal": True,
        "params": [("f0", "频率 f₀ (Hz)", 0.5, 5.0, 1.0, 0.1)],
        "time_func": lambda t, f0=1.0: np.cos(2 * np.pi * f0 * t) * (t >= 0),
        "ft_latex": r"$X(j\omega) = \dfrac{j\omega}{\omega_0^2-\omega^2}"
                    r"+ \dfrac{\pi}{2}[\delta(\omega-\omega_0)+\delta(\omega+\omega_0)]$",
        "ft_has_delta": True,
        "lt_latex": r"$X(s) = \dfrac{s}{s^2 + \omega_0^2}$",
        "poles": [(0, 1), (0, -1)],  # will be scaled
        "zeros": [(0, 0)],
        "roc": r"$\mathrm{Re}\{s\} > 0$",
        "note": "余弦与正弦的 LT 仅在分子上不同：余弦为 $s$，正弦为 $\omega_0$。",
    },
    "damped_sine": {
        "label": "衰减正弦 e^{-at}sin(ω₀t)u(t)",
        "latex": r"$x(t) = e^{-at}\sin(\omega_0 t)u(t),\quad a>0$",
        "causal": True,
        "params": [("a", "衰减系数 a", 0.2, 3.0, 0.5, 0.1),
                   ("f0", "频率 f₀ (Hz)", 0.5, 5.0, 2.0, 0.1)],
        "time_func": lambda t, a=0.5, f0=2.0: (
            np.exp(-a * t) * np.sin(2 * np.pi * f0 * t) * (t >= 0)),
        "ft_latex": r"$X(j\omega) = \dfrac{\omega_0}{(a+j\omega)^2 + \omega_0^2}$",
        "ft_has_delta": False,
        "lt_latex": r"$X(s) = \dfrac{\omega_0}{(s+a)^2 + \omega_0^2}$",
        "poles": [(-0.5, 2), (-0.5, -2)],
        "zeros": [],
        "roc": r"$\mathrm{Re}\{s\} > -a$",
        "note": "衰减正弦是典型的二阶系统冲激响应，极点位于左半平面时系统稳定。极点实部 $-a$ 决定衰减速度，虚部 $\pm\omega_0$ 决定振荡频率。",
    },
    "rect": {
        "label": "矩形脉冲 rect(t/τ)",
        "latex": r"$x(t) = \mathrm{rect}\!\left(\dfrac{t}{\tau}\right)$",
        "causal": False,
        "params": [("tau", "脉冲宽度 τ", 0.5, 4.0, 2.0, 0.1)],
        "time_func": lambda t, tau=2.0: (np.abs(t) <= tau / 2).astype(float),
        "ft_latex": r"$X(j\omega) = \tau\;\mathrm{sinc}\!\left(\dfrac{\tau\omega}{2}\right)$",
        "ft_has_delta": False,
        "lt_latex": None,
        "poles": [],
        "zeros": [],
        "roc": None,
        "note": "矩形脉冲的傅里叶变换是 sinc 函数。脉冲越窄 ($\tau$ 越小)，频谱越宽，即《时宽-带宽积》为常数。",
    },
    "gaussian": {
        "label": "高斯信号 e^{-(t/τ)²}",
        "latex": r"$x(t) = e^{-(t/\tau)^2}$",
        "causal": False,
        "params": [("tau", "宽度参数 τ", 0.5, 4.0, 1.5, 0.1)],
        "time_func": lambda t, tau=1.5: np.exp(-(t / tau) ** 2),
        "ft_latex": r"$X(j\omega) = \sqrt{\pi}\,\tau\; e^{-(\tau\omega/2)^2}$",
        "ft_has_delta": False,
        "lt_latex": None,
        "poles": [],
        "zeros": [],
        "roc": None,
        "note": "高斯信号的傅里叶变换仍是高斯函数。它是唯一在时域和频域均为同一形式的信号。时域越宽 ($\tau$ 越大)，频域越窄。",
    },
    "sinc": {
        "label": "抽样函数 sinc(t/τ)",
        "latex": r"$x(t) = \mathrm{sinc}\!\left(\dfrac{t}{\tau}\right) = "
                r"\dfrac{\sin(\pi t/\tau)}{\pi t/\tau}$",
        "causal": False,
        "params": [("tau", "宽度参数 τ", 0.5, 3.0, 1.0, 0.1)],
        "time_func": lambda t, tau=1.0: np.sinc(t / tau),
        "ft_latex": r"$X(j\omega) = \tau\;\mathrm{rect}\!\left(\dfrac{\tau\omega}{2\pi}\right)$",
        "ft_has_delta": False,
        "lt_latex": None,
        "poles": [],
        "zeros": [],
        "roc": None,
        "note": "sinc 与 rect 构成一对傅里叶变换对，互为对偶。时域的 sinc 对应频域的矩形窗。",
    },
    "sign": {
        "label": "符号函数 sgn(t)",
        "latex": r"$x(t) = \mathrm{sgn}(t) = \begin{cases}1,&t>0\\-1,&t<0\end{cases}$",
        "causal": False,
        "params": [],
        "time_func": lambda t, **kw: np.sign(t),
        "ft_latex": r"$X(j\omega) = \dfrac{2}{j\omega}$",
        "ft_has_delta": False,
        "lt_latex": None,
        "poles": [],
        "zeros": [],
        "roc": None,
        "note": "符号函数是奇函数，其傅里叶变换为纯虚数。它没有拉普拉斯变换（双边信号，ROC 为空）。",
    },
    "double_exp": {
        "label": "双边指数 e^{-a|t|}",
        "latex": r"$x(t) = e^{-a|t|},\quad a>0$",
        "causal": False,
        "params": [("a", "衰减系数 a", 0.2, 3.0, 1.0, 0.1)],
        "time_func": lambda t, a=1.0: np.exp(-a * np.abs(t)),
        "ft_latex": r"$X(j\omega) = \dfrac{2a}{a^2 + \omega^2}$",
        "ft_has_delta": False,
        "lt_latex": None,
        "poles": [],
        "zeros": [],
        "roc": None,
        "note": "双边指数是偶函数，其傅里叶变换为实偶函数。它没有拉普拉斯变换（ROC 为空）。",
    },
}


def _get_signal(sig_key, params, t):
    """Generate time-domain signal samples."""
    sig = SIGNALS[sig_key]
    return sig["time_func"](t, **params)


@st.cache_data
def compute_spectrum(sig_key, params_str, t, sig_values):
    """Compute FFT-based spectrum."""
    dt = t[1] - t[0]
    N = len(t)
    # Shift signal so t=0 is at index 0 (FFT convention), then FFT, then shift freq to center
    X = np.fft.fft(np.fft.ifftshift(sig_values)) * dt
    freq = np.fft.fftfreq(N, dt)
    X, freq = np.fft.fftshift(X), np.fft.fftshift(freq)
    return freq, X


def plot_pole_zero(ax, poles, zeros, roc_text, s_lim=3.0):
    """Draw pole-zero plot in s-plane with ROC."""
    ax.set_xlim(-s_lim, s_lim)
    ax.set_ylim(-s_lim, s_lim)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title("s 平面零极点图")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Draw jω axis arrow
    ax.annotate("", xy=(0, s_lim * 0.95), xytext=(0, -s_lim * 0.95),
                arrowprops=dict(arrowstyle="->", color='gray', lw=0.8))
    ax.annotate("", xy=(s_lim * 0.95, 0), xytext=(-s_lim * 0.95, 0),
                arrowprops=dict(arrowstyle="->", color='gray', lw=0.8))
    ax.text(s_lim * 0.97, -0.2, "σ", fontsize=11, ha='right')
    ax.text(0.1, s_lim * 0.97, "jω", fontsize=11, va='top')

    # Transform poles/zeros based on param values (handled in caller)
    for (re, im) in poles:
        ax.plot(re, im, 'x', color='red', markersize=10, markeredgewidth=2.5, zorder=5)
    for (re, im) in zeros:
        ax.plot(re, im, 'o', color='blue', markersize=8,
                markeredgewidth=2, fillstyle='none', zorder=5)

    # Legend
    ax.plot([], [], 'x', color='red', label='极点')
    ax.plot([], [], 'o', color='blue', fillstyle='none', label='零点')
    if roc_text:
        ax.plot([], [], ' ', label=f'ROC: {roc_text}')
    ax.legend(fontsize=9, loc='upper right')


# ============================================================
# 辅助：生成带参数键的字符串（用于缓存依赖）
# ============================================================
def _params_str(params_dict):
    return ",".join(f"{k}={v:.4f}" for k, v in sorted(params_dict.items()))


# ============================================================
# 标题
# ============================================================
st.title("信号变换可视化")
st.markdown("选择常用信号，观察其时域波形、傅里叶变换频谱和拉普拉斯变换零极点。")

# ============================================================
# 侧边栏：信号选择与参数控制
# ============================================================
with st.sidebar:
    st.header("信号选择")

    # Group signals by category
    cat_options = {
        "冲激与阶跃": ["impulse", "step"],
        "指数信号": ["exp_decay", "exp_ramp", "double_exp"],
        "正弦与振荡": ["sine", "cosine", "damped_sine"],
        "脉冲与抽样": ["rect", "sinc"],
        "其他": ["gaussian", "sign"],
    }
    flat_keys = []
    for cat, keys in cat_options.items():
        for k in keys:
            flat_keys.append(k)

    # Map key -> category for display
    key_to_cat = {}
    for cat, keys in cat_options.items():
        for k in keys:
            key_to_cat[k] = cat

    sig_key = st.selectbox(
        "信号类型", flat_keys,
        format_func=lambda k: f"[{key_to_cat[k]}] {SIGNALS[k]['label']}",
    )

    sig = SIGNALS[sig_key]
    params = {}
    st.markdown("---")
    st.header("参数调节")

    # Check if dynamic note text references parameters (step doesn't use params)
    # We update the note label reference dynamically

    param_info = {}
    for p in sig["params"]:
        key, label, min_v, max_v, default, step = p
        if isinstance(default, int):
            val = st.slider(label, min_v, max_v, default, step)
        else:
            val = st.slider(label, min_v, max_v, default, step, format="%.2f")
        params[key] = val
        param_info[key] = val

    # Time range control
    st.markdown("---")
    st.caption("显示设置")
    t_max = st.slider("时间范围 (±T)", 2.0, 12.0, 6.0, 0.5)

    st.markdown("---")
    st.markdown("**操作说明**")
    st.caption(
        "切换信号类型或调节参数，时域波形和频谱会同步更新。\n\n"
        "傅里叶变换展示幅度谱和相位谱；拉普拉斯变换展示零极点图和收敛域（因果信号适用）。"
    )


# ============================================================
# 生成时域信号 & 计算频谱
# ============================================================

t_resolution = 2000
t = np.linspace(-t_max, t_max, t_resolution, endpoint=False)
sig_values = _get_signal(sig_key, params, t)
freq, X_spec = compute_spectrum(sig_key, _params_str(params), t, sig_values)

# Positive frequency range for display
pos_mask = freq >= 0
f_pos = freq[pos_mask]
X_pos = X_spec[pos_mask]

# ============================================================
# Tab 布局
# ============================================================
tab_names = ["时域波形", "傅里叶变换", "拉普拉斯变换"]
has_lt = sig["lt_latex"] is not None
if not has_lt:
    tab_names[2] = "拉普拉斯变换 (不存在)"

tabs = st.tabs(tab_names)


# ============================================================
# Tab 1: 时域波形
# ============================================================
with tabs[0]:
    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(t, sig_values, color='steelblue', linewidth=2)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("x(t)")
        ax.set_title("时域波形")
        ax.grid(True, alpha=0.3)
        # Auto y-limit with padding
        ymin, ymax = np.min(sig_values), np.max(sig_values)
        ypad = max((ymax - ymin) * 0.15, 0.1)
        if ymin == ymax == 0:
            ymin, ymax = -0.1, 1.1
        ax.set_ylim(ymin - ypad, ymax + ypad)
        st.pyplot(fig)

    with col_right:
        st.markdown("**时域表达式**")
        st.markdown(sig["latex"])

        st.markdown("**信号参数**")
        if params:
            for k, v in params.items():
                st.markdown(f"- ${k} = {v}$")
        else:
            st.markdown("（无参数）")

        st.markdown("**说明**")
        st.markdown(sig["note"])


# ============================================================
# Tab 2: 傅里叶变换
# ============================================================
with tabs[1]:
    st.markdown("**傅里叶变换表达式**")
    st.markdown(sig["ft_latex"])

    if sig["ft_has_delta"]:
        st.info(
            "该信号的傅里叶变换含有冲激（$\delta$ 函数），"
            "数值 FFT 无法精确显示冲激分量。下方频谱仅展示 FFT 数值结果，"
            "冲激部分请参考解析表达式。"
        )

    col_left, col_right = st.columns([1, 1])

    # Reasonable frequency range for display
    valid_idx = np.searchsorted(f_pos, 10) + 1
    fmax_plot = f_pos[min(valid_idx + 100, len(f_pos) - 1)]
    fmax_show = min(fmax_plot, f_pos[-1])

    with col_left:
        fig, ax = plt.subplots(figsize=(9, 3.5))
        mag = np.abs(X_pos)
        ax.plot(f_pos, mag, color='crimson', linewidth=1.8)
        ax.set_xlim(0, fmax_show)
        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel("|X(f)|")
        ax.set_title("幅度频谱")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col_right:
        fig, ax = plt.subplots(figsize=(9, 3.5))
        phase = np.angle(X_pos)
        ax.plot(f_pos, phase, color='darkgreen', linewidth=1.8)
        ax.set_xlim(0, fmax_show)
        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel("∠X(f) (rad)")
        ax.set_title("相位频谱")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with st.expander("📖 详细说明"):
        st.markdown(r"""
        **如何理解频谱？**
        - **幅度谱** $|X(f)|$ 表示各频率分量的幅度大小
        - **相位谱** $\angle X(f)$ 表示各频率分量的相位偏移

        **重要性质**
        - 实信号的幅度谱为偶函数，相位谱为奇函数
        - 频谱的宽窄与时域的宽窄成反比（时宽-带宽积）
        - 冲激响应 $h(t)$ 的傅里叶变换即为系统的频率响应 $H(j\omega)$
        """)

        # Show parameter effect summary
        st.markdown("**参数对频谱的影响**")
        if sig_key == "exp_decay":
            st.markdown(
                f"- $a = {params.get('a', 1.0)}$：$a$ 越大，指数衰减越快，"
                f"频谱越宽（高频分量越多）"
            )
        elif sig_key == "rect":
            st.markdown(
                f"- $\\tau = {params.get('tau', 2.0)}$：$\\tau$ 越大，脉冲越宽，"
                f"sinc 主瓣越窄（带宽越小）"
            )
        elif sig_key == "gaussian":
            st.markdown(
                f"- $\\tau = {params.get('tau', 1.5)}$：$\\tau$ 越大，时域波形越宽，"
                f"频域高斯函数越窄"
            )
        elif sig_key == "damped_sine":
            st.markdown(
                f"- $a = {params.get('a', 0.5)}$ 控制共振峰宽度（$a$ 越小峰越尖锐），"
                f"$f_0 = {params.get('f0', 2.0)}$ Hz 控制共振峰位置"
            )


# ============================================================
# Tab 3: 拉普拉斯变换
# ============================================================
with tabs[2]:
    if not has_lt:
        st.warning(
            "该信号是 **双边信号**（非因果），拉普拉斯变换不存在（ROC 为空）。\n\n"
            "拉普拉斯变换要求信号在 $t<0$ 时为零（因果）或具有合适的收敛域。"
        )
    else:
        st.markdown("**拉普拉斯变换表达式**")
        st.markdown(sig["lt_latex"])

        col_left, col_right = st.columns([1, 1])

        with col_left:
            fig, ax = plt.subplots(figsize=(7, 6.5))

            # Compute actual pole/zero positions based on params
            real_poles = []
            real_zeros = []
            s_lim = 4.0

            if sig_key in ("exp_decay",):
                a = params.get("a", 1.0)
                real_poles = [(-a, 0)]
            elif sig_key in ("exp_ramp",):
                a = params.get("a", 1.0)
                real_poles = [(-a, 0), (-a, 0)]
            elif sig_key in ("sine", "cosine"):
                w0 = 2 * np.pi * params.get("f0", 1.0)
                real_poles = [(0, w0), (0, -w0)]
                real_zeros = [(0, 0)]
            elif sig_key == "damped_sine":
                a = params.get("a", 0.5)
                w0 = 2 * np.pi * params.get("f0", 2.0)
                real_poles = [(-a, w0), (-a, -w0)]
            elif sig_key == "step":
                real_poles = [(0, 0)]
            elif sig_key == "impulse":
                real_poles = []
            else:
                # Try to use default
                re, im = zip(*sig["poles"]) if sig["poles"] else ([], [])
                real_poles = list(zip(re, im))

            s_lim = max(max((abs(p[0]) for p in real_poles), default=0) + 2,
                        max((abs(p[1]) for p in real_poles), default=0) + 2,
                        3.0)

            plot_pole_zero(ax, real_poles, real_zeros,
                          sig["roc"], s_lim)

            st.pyplot(fig)

        with col_right:
            st.markdown("**收敛域 (ROC)**")
            st.markdown(sig["roc"])

            st.markdown("**零极点信息**")
            if real_poles:
                pole_strs = [f"$s = {p[0]:.2f} + j({p[1]:.2f})$"
                           for p in real_poles]
                st.markdown("极点：")
                for ps in pole_strs:
                    st.markdown(f"- {ps}")
            else:
                st.markdown("- 无极点（整函数）")

            if real_zeros:
                zero_strs = [f"$s = {z[0]:.2f} + j({z[1]:.2f})$"
                           for z in real_zeros]
                st.markdown("零点：")
                for zs in zero_strs:
                    st.markdown(f"- {zs}")
            else:
                st.markdown("- 无零点")

            st.markdown("**稳定性分析**")
            if any(p[0] > 0 for p in real_poles):
                st.error("系统不稳定：存在右半平面极点")
            elif any(p[0] == 0 for p in real_poles) and real_poles:
                st.warning("临界稳定：极点在虚轴上（无阻尼振荡）")
            elif real_poles:
                st.success("系统稳定：所有极点位于左半平面")
            else:
                st.info("无极点，系统稳定")


# ============================================================
# 页脚
# ============================================================
st.markdown("---")
st.caption("武汉大学 电子信息学院 · 信号与系统 · 傅里叶变换与拉普拉斯变换交互式可视化")
