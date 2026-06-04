"""
画面キャプチャモジュール (オプションモジュール)

Tkinterを利用した画面領域選択とオーバーレイ表示。
Feature Registry パターンで動的ロードされる（main.py から importlib で検索）。
"""

import tkinter as tk
import multiprocessing


def show_transparent_overlay(bbox, stop_event):
    """
    別プロセスで実行され、指定されたbboxの位置に赤枠（中身は透過・クリック透過）を表示し続ける。
    Windows環境で透過クリックを実現するための設定を含む。
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    root = tk.Tk()
    root.overrideredirect(True) # タイトルバーなし
    root.attributes("-topmost", True) # 最前面
    # 背景色を特定の色にして、その色を完全に透過する（Windows用）
    transparent_color = "black"
    root.configure(bg=transparent_color)
    root.attributes("-transparentcolor", transparent_color)

    # ウィンドウの位置とサイズを設定
    root.geometry(f"{w}x{h}+{x1}+{y1}")

    # 枠線の描画（Canvasを使用）
    canvas = tk.Canvas(root, width=w, height=h, bg=transparent_color, highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    # 枠線の太さを考慮して内側に描画（線の太さ分だけ内側にずらす）
    line_width = 3
    canvas.create_rectangle(
        line_width/2, line_width/2,
        w - line_width/2, h - line_width/2,
        outline="red", width=line_width, fill=transparent_color
    )

    def check_stop():
        if stop_event.is_set():
            root.quit()
        else:
            root.after(200, check_stop)

    root.after(200, check_stop)
    root.mainloop()
    root.destroy()


def get_inner_bbox(bbox, border_width=3):
    """キャプチャ領域を取得する際、赤枠自体が写り込まないように少し内側に狭めた座標を返す"""
    x1, y1, x2, y2 = bbox
    return (x1 + border_width, y1 + border_width, x2 - border_width, y2 - border_width)


class OverlayManager:
    """オーバーレイ表示のプロセスとイベントを管理するクラス"""
    def __init__(self):
        self.overlay_process = None
        self.overlay_stop_event = multiprocessing.Event()

    def update_overlay(self, bbox):
        if self.overlay_process and self.overlay_process.is_alive():
            self.overlay_stop_event.set()
            self.overlay_process.join()

        if bbox:
            self.overlay_stop_event.clear()
            self.overlay_process = multiprocessing.Process(target=show_transparent_overlay, args=(bbox, self.overlay_stop_event))
            self.overlay_process.start()

    def stop(self):
        if self.overlay_process and self.overlay_process.is_alive():
            self.overlay_stop_event.set()
            self.overlay_process.join()


def select_screen_area():
    """Tkinterを使用して画面全体を覆う半透明ウィンドウを表示し、リアルタイムキャプチャする領域をドラッグで選択する"""
    print("\n--- リアルタイム画面キャプチャ ---")
    print("全画面が半透明になります。マウスをドラッグして、キャプチャしたい範囲を選択してください。")
    print("※選択をキャンセルするにはEscキーを押してください。")

    root = tk.Tk()
    root.attributes("-alpha", 0.3)
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(background='black')
    root.config(cursor="cross")

    canvas = tk.Canvas(root, cursor="cross", bg="black")
    canvas.pack(fill="both", expand=True)

    rect = None
    start_x = None
    start_y = None
    selection = None

    def on_button_press(event):
        nonlocal start_x, start_y, rect
        start_x = event.x
        start_y = event.y
        rect = canvas.create_rectangle(start_x, start_y, 1, 1, outline='red', width=3, fill="white")

    def on_move_press(event):
        nonlocal rect
        cur_x, cur_y = (event.x, event.y)
        if rect:
            canvas.coords(rect, start_x, start_y, cur_x, cur_y)

    def on_button_release(event):
        nonlocal selection
        end_x, end_y = (event.x, event.y)
        x1 = min(start_x, end_x)
        y1 = min(start_y, end_y)
        x2 = max(start_x, end_x)
        y2 = max(start_y, end_y)

        # 領域が小さすぎる場合はキャンセル扱い
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            selection = (x1, y1, x2, y2)
        root.quit()

    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_move_press)
    canvas.bind("<ButtonRelease-1>", on_button_release)
    root.bind("<Escape>", lambda e: root.quit())

    root.mainloop()
    root.destroy()
    return selection
