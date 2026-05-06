"""
无失真传输条件 —— 交互式实验 (Streamlit 版)
运行方式: streamlit run 无失真传输_交互.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.font_manager as fm

# ============================================================
# 页面基本配置
# ============================================================

st.set_page_config(page_title="无失真传输条件 - 交互式实验", layout="wide")

# 中文字体配置
_CN_FONTS = ['Microsoft YaHei', 'SimHei', 'STXihei']
_AVAILABLE = {f.name for f in fm.fontManager.ttflist}
_CHOSEN_FONT = next((f for f in _CN_FONTS if f in _AVAILABLE), None)
if _CHOSEN_FONT:
    plt.rcParams['font.sans-serif'] = [_CHOSEN_FONT] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['axes.unicode_minus'] = False

plt.rcParams.update({'figure.dpi': 100, 'font.size': 10, 'axes.titlesize': 12})


# ============================================================
# 核心信号处理函数
# ============================================================

@st.cache_data
def make_time_axis(duration=2.0, fs=500):
    """生成时间轴和频率轴"""
    N = int(duration * fs)
    t = np.linspace(0, duration, N, endpoint=False)
    f = np.fft.fftfreq(N, 1 / fs)
    return t, f, fs


def apply_freq_response(x, t, H):
    """在频域施加系统频率响应 H(f)"""
    fs = 1 / (t[1] - t[0])
    f = np.fft.fftfreq(len(t), 1 / fs)
    X = np.fft.fft(x)
    return np.real(np.fft.ifft(X * H(f)))


def ideal_output(x, t, K, td):
    """理想无失真输出: y = K·x(t - td)"""
    delay = int(td / (t[1] - t[0]))
    if delay >= len(t):
        return np.zeros_like(x)
    y = np.roll(x, delay)
    y[:delay] = 0
    return K * y


# ============================================================
# 标题
# ============================================================

st.title("🔬 无失真传输 —— 交互式实验")
st.markdown("""
调整左侧参数，观察系统的幅频特性和相频特性如何影响信号传输。
""")

# ============================================================
# 侧边栏：参数控制
# ============================================================

with st.sidebar:
    st.header("⚙️ 信号参数")

    signal_type = st.selectbox("信号类型", ["多频正弦波", "矩形脉冲", "方波"])

    if signal_type == "多频正弦波":
        st.caption("各频率分量的幅度与频率")
        cols = st.columns(3)
        with cols[0]:
            a1 = st.slider("A₁", 0.0, 1.5, 1.0, 0.1)
            f1 = st.slider("f₁ (Hz)", 1, 40, 2)
        with cols[1]:
            a2 = st.slider("A₂", 0.0, 1.5, 0.6, 0.1)
            f2 = st.slider("f₂ (Hz)", 1, 40, 8)
        with cols[2]:
            a3 = st.slider("A₃", 0.0, 1.5, 0.3, 0.1)
            f3 = st.slider("f₃ (Hz)", 1, 40, 20)

    elif signal_type == "矩形脉冲":
        pulse_width = st.slider("脉冲宽度 (s)", 0.05, 0.8, 0.15, 0.05)
        pulse_start = st.slider("起始位置 (s)", 0.0, 1.0, 0.2, 0.05)

    else:  # 方波
        f_sq = st.slider("方波频率 (Hz)", 0.5, 5.0, 1.0, 0.5)
        duty = st.slider("占空比", 0.1, 0.9, 0.5, 0.1)

    st.markdown("---")
    st.header("⚙️ 系统参数")

    K = st.slider("增益 K", 0.2, 2.0, 1.0, 0.1)
    td = st.slider("时延 td (s)", 0.0, 0.5, 0.15, 0.01)

    st.markdown("---")
    st.header("⚙️ 失真控制")

    st.caption("幅频特性")
    mag_type = st.radio("幅频", ["平坦", "低通"], horizontal=True, label_visibility="collapsed")
    if mag_type == "低通":
        fc = st.slider("低通截止频率 fc (Hz)", 2, 40, 8)

    st.caption("相频特性")
    phase_type = st.radio("相频", ["线性", "二次", "跳变"], horizontal=True, label_visibility="collapsed")

    if phase_type == "二次":
        beta = st.slider("非线性强度 β", 0.00005, 0.01, 0.002, 0.00005, format="%.5f")
    elif phase_type == "跳变":
        jf = st.slider("跳变频率 (Hz)", 3, 25, 8)
        jp = st.slider("跳变幅度 (度)", 0, 180, 90)

    st.markdown("---")
    st.caption("提示：将幅频设为「平坦」、相频设为「线性」即为无失真条件。分别改变其中一个，观察波形变化。")


# ============================================================
# 信号生成
# ============================================================

t, f_fft, fs = make_time_axis(2.0, 500)
pos = f_fft >= 0

if signal_type == "多频正弦波":
    x = (a1 * np.sin(2 * np.pi * f1 * t) +
         a2 * np.sin(2 * np.pi * f2 * t) +
         a3 * np.sin(2 * np.pi * f3 * t))
    signal_desc = f"${a1}\\sin(2\\pi{f1}t) + {a2}\\sin(2\\pi{f2}t) + {a3}\\sin(2\\pi{f3}t)$"

elif signal_type == "矩形脉冲":
    x = np.where((t >= pulse_start) & (t <= pulse_start + pulse_width), 1.0, 0.0)
    signal_desc = f"矩形脉冲（宽度 {pulse_width}s）"

else:
    x = signal.square(2 * np.pi * f_sq * t, duty=duty)
    signal_desc = f"方波（{f_sq}Hz，占空比 {duty}）"


# ============================================================
# 系统频率响应构造
# ============================================================

def build_system(mag_type, phase_type):
    """返回频率响应函数 H(f)"""
    def H(freq):
        # 幅频
        if mag_type == "平坦":
            mag = np.ones_like(freq)
        else:
            mag = 1.0 / (1.0 + (freq / fc) ** 2)

        # 相频
        if phase_type == "线性":
            phi = -2 * np.pi * freq * td
        elif phase_type == "二次":
            phi = -2 * np.pi * freq * td + beta * (2 * np.pi * freq) ** 2
        else:  # 跳变
            phi = -2 * np.pi * freq * td
            phi += np.where(np.abs(freq) > jf, np.deg2rad(jp), 0.0)

        return mag * np.exp(1j * phi)
    return H


H = build_system(mag_type, phase_type)
y = apply_freq_response(x, t, H)
y_ideal = ideal_output(x, t, K, td)


# ============================================================
# 绘图区域
# ============================================================

tab1, tab2, tab3 = st.tabs(["① 输入信号", "② 系统频率响应", "③ 输出对比"])


# --- 第一栏：输入信号 ---
with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(t, x, color='steelblue', linewidth=1.5)
        ax.set_xlim(0, 1.5)
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("幅度")
        ax.set_title("时域波形 x(t)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col_right:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        X = np.fft.fft(x)
        markerline, _, _ = ax.stem(f_fft[pos][:200], np.abs(X[pos][:200]),
                basefmt=' ', markerfmt='o', linefmt='steelblue')
        plt.setp(markerline, markersize=3)
        ax.set_xlim(0, 40)
        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel("|X(f)|")
        ax.set_title("幅度频谱")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with st.expander("📖 说明"):
        st.markdown(f"""
        **当前信号**：{signal_desc}

        信号的频谱决定了它包含哪些频率分量。观察频谱中的每根谱线——它们经过系统时会被不同程度地改变幅度和相位。
        如果系统不满足无失真条件，这些谱线的相对关系就会被破坏，导致输出波形畸变。
        """)


# --- 第二栏：系统频率响应 ---
with tab2:
    col_left, col_right = st.columns([1, 1])

    f_pos = f_fft[f_fft >= 0]
    H_vals = H(f_pos)
    H_mag = np.abs(H_vals)
    H_phase = np.unwrap(np.angle(H_vals))

    with col_left:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(f_pos, H_mag, color='crimson', linewidth=2.5)
        ax.axhline(1, color='gray', linestyle=':', alpha=0.4, label='|H|=1（无失真）')
        ax.set_xlim(0, 40)
        ax.set_ylim(-0.05, 1.25)
        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel("|H(f)|")
        ax.set_title("幅频响应")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col_right:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        phase_deg = np.rad2deg(H_phase)
        ax.plot(f_pos, phase_deg, color='darkgreen', linewidth=2.5)
        # 标注理想线性相位
        ideal_phase = -2 * np.pi * td * f_pos
        ax.plot(f_pos, np.rad2deg(np.unwrap(ideal_phase)), color='gray',
                linestyle=':', alpha=0.6, label='线性相位（无失真）')
        ax.set_xlim(0, 40)
        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel("相位 (度)")
        ax.set_title("相频响应")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # 群延迟图
    with st.expander("📈 查看群延迟"):
        fig, ax = plt.subplots(figsize=(12, 3))
        gd = -np.diff(H_phase) / (2 * np.pi * (f_pos[1] - f_pos[0]))
        ax.plot(f_pos[:-1], gd, color='purple', linewidth=2)
        ax.axhline(td, color='gray', linestyle=':', alpha=0.5, label=f'τg = td = {td}s（无失真）')
        ax.set_xlim(0, 40)
        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel("群延迟 τg (s)")
        ax.set_title("群延迟 — 不同频率分量的时间延迟")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.markdown(r"""
        **群延迟** $\tau_g(f) = -\frac{1}{2\pi}\frac{d\theta(f)}{df}$ 衡量不同频率分量经过系统后的时间延迟。
        无失真条件要求群延迟为常数（等于 td），即所有频率分量的延迟时间相同。
        """)


# --- 第三栏：输出对比 ---
with tab3:
    fig, ax = plt.subplots(figsize=(14, 5))

    # 归一化对比（忽略增益差异，聚焦波形形状）
    yn = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
    xi = x / np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else x
    yi = y_ideal / np.max(np.abs(y_ideal)) if np.max(np.abs(y_ideal)) > 0 else y_ideal

    ax.plot(t, xi, color='gray', linewidth=1.2, alpha=0.5, label="输入信号（归一化）")
    ax.plot(t, yn, color='crimson', linewidth=2, label="实际输出")
    ax.plot(t, yi, color='steelblue', linewidth=1.5, linestyle='--', alpha=0.7,
            label="理想无失真输出")

    ax.set_xlim(0, 1.5)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("幅度（归一化）")
    ax.set_title("输出信号对比")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    # 结论分析
    is_distortionless = (mag_type == "平坦" and phase_type == "线性")

    if is_distortionless:
        st.success("""
        **✅ 当前系统满足无失真传输条件**

        幅频响应平坦（所有频率分量等幅放大/衰减）+ 相频响应线性（所有频率分量等时延迟）。
        输出波形与输入完全一致，仅幅度缩放 K 倍、时间延迟 td。
        """)
    else:
        issues = []
        if mag_type != "平坦":
            issues.append(f"幅频不平坦（低通截止频率 {fc}Hz，高频分量被衰减）")
        if phase_type != "线性":
            if phase_type == "二次":
                issues.append(f"相频非线性（二次相位，β={beta}，不同频率的时延不同）")
            else:
                issues.append(f"相频非线性（{jf}Hz 处相位跳变 {jp}°，该频率附近的波形被破坏）")

        st.warning(f"""
        **⚠️ 当前系统不满足无失真传输条件**

        {'；'.join(issues)}。

        观察输出波形与理想无失真输出的差异——这就是失真。
        """)

    with st.expander("📖 为什么会失真？"):
        st.markdown(r"""
        | 失真类型 | 原因 | 表现 |
        |---|---|---|
        | **幅度失真** | 幅频响应不平坦，各频率分量衰减不同 | 波形形状改变，如方波变圆滑（高频被衰减） |
        | **相位失真** | 相频响应非线性，各频率分量时延不同 | 波形"散开"，脉冲变宽、出现振铃 |

        **关键理解**：信号可以分解为不同频率的正弦波之和（傅里叶级数/变换）。
        系统对每个频率分量施加一个复数增益 H(f)=|H(f)|·e^{jθ(f)}。
        当 |H(f)| 不是常数或 θ(f) 不是线性函数时，各分量的相对关系被破坏，合成后的波形就会畸变。
        """)


# ============================================================
# 页脚
# ============================================================

st.markdown("---")
st.caption("武汉大学 电子信息学院 · 信号与系统 · 第五章 傅里叶变换应用于通信系统")
