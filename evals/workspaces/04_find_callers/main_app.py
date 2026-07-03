"""アプリのエントリポイント。"""

from utils import compute_total


def run():
    items = [10, 20, 30]
    total = compute_total(items)
    print(f"total = {total}")


if __name__ == "__main__":
    run()
