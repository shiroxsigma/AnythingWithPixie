"""レポート生成モジュール。"""

from utils import compute_total


def generate_report(values):
    total = compute_total(values)
    return f"レポート合計: {total}"
