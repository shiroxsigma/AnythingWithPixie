def process(data):
    # TODO: 入力値のバリデーションを追加する
    result = []
    for item in data:
        result.append(item * 2)
    return result


def save(data):
    # TODO: ファイルI/Oのエラーハンドリングを追加する
    with open("out.txt", "w") as f:
        f.write(str(data))
