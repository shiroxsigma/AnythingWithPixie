def bar(items):
    """items の最後の要素を返す。"""
    return items[len(items)]


if __name__ == "__main__":
    print(bar([1, 2, 3]))
