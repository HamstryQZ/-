"""双击此文件启动无失真传输交互实验"""
import subprocess, sys, os, webbrowser, threading, time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
app_path = os.path.join(os.getcwd(), "app.py")

# 先打印中文提示再启动
print("=" * 50)
print("  无失真传输条件 —— 交互式实验")
print("=" * 50)
print()
print("  ⏳ 正在启动，请稍候...")
print("  启动后浏览器将自动打开")
print()
print("  提示：如果浏览器没有自动打开")
print("  请手动访问 http://localhost:8501")
print()
print("  关闭此窗口即可退出程序")
print("=" * 50)

def open_browser():
    time.sleep(4)
    webbrowser.open("http://localhost:8501")

threading.Thread(target=open_browser, daemon=True).start()

# 启动 Streamlit（屏蔽英文横幅，保留错误信息）
with open(os.devnull, 'w', encoding='utf-8') as devnull:
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path],
                   stdout=devnull)
