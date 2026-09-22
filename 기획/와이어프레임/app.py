"""genkei 와이어프레임 — 화면 흐름만 보는 창. 계산은 없다 (숫자는 예시)."""
import os
import sys

import webview

HERE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    webview.create_window("genkei — 와이어프레임", os.path.join(HERE, "index.html"),
                          width=1440, height=900, min_size=(1000, 650), background_color="#F2F2F7")
    webview.start(http_server=True)
