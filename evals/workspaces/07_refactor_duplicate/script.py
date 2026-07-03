def price_with_tax_jp(price):
    """日本の消費税(10%)込みの価格を計算する。"""
    tax_rate = 0.10
    taxed = price * (1 + tax_rate)
    return round(taxed)


def price_with_tax_us(price):
    """米国の消費税相当(仮に8%)込みの価格を計算する。"""
    tax_rate = 0.08
    taxed = price * (1 + tax_rate)
    return round(taxed)


if __name__ == "__main__":
    print(price_with_tax_jp(1000))
    print(price_with_tax_us(1000))
