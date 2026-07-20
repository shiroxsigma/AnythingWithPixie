"""run_python（Python サンドボックス実行 + input() 自動入力）のテスト。

_execute_run_python とその構成要素（_count_input_calls / _looks_like_prompt /
ドライバループ）を、スクリプト化モック LLM と実 Python サブプロセスで検証する。
test_verify_loop.py のパターン（_FixMockLLM / _outfn / _Ctx）を踏襲。
"""


import config
from subagent import _count_input_calls, _execute_run_python, _looks_like_prompt

# =====================================================
# ヘルパ
# =====================================================

class _ScriptedInputLLM:
    """create_chat_completion が順に用意した入力値を1回ずつ返すモック（LM Studio 互換 generator）。
    呼ばれるたびに次の値へ進む（入力生成の順序性を検証）。"""

    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def create_chat_completion(self, messages, *, max_tokens, temperature, stream, **kw):
        self.calls += 1
        v = self.values.pop(0) if self.values else ""

        def _gen():
            yield {
                "choices": [{
                    "delta": {"content": v},
                    "finish_reason": "stop",
                }]
            }

        return _gen()


class _ExplodingLLM:
    """create_chat_completion が常に例外を投げるモック（入力生成失敗テスト用）。"""

    def create_chat_completion(self, *a, **kw):
        raise RuntimeError("LLM boom")


class _Ctx:
    """_execute_run_python 用の最小 context（llm のみ必要）。"""

    def __init__(self, llm):
        self.llm = llm
        self.supports_tool_role = False


def _outfn():
    """output_fn 互換（end/flush 等を受け付ける）バッファ。"""
    buf = []

    def _fn(text="", end="", flush=True, **kw):
        buf.append(text)

    _fn.buf = buf
    return _fn


def _run(code, llm, **extra):
    """_execute_run_python を実行し (result, llm, out_fn) を返す便利ラッパ。"""
    out = _outfn()
    tool_args = {"code": code, **extra}
    result = _execute_run_python(_Ctx(llm), tool_args, out)
    return result, llm, out


# =====================================================
# pure 関数の単体テスト
# =====================================================

def test_count_input_calls():
    assert _count_input_calls("x = input()") == 1
    assert _count_input_calls('input("a")\ninput("b")') == 2
    assert _count_input_calls('print("hi")') == 0
    assert _count_input_calls("def f():\n    return input()") == 1
    # 構文エラー時は 0（実行時に SyntaxError を出させる）
    assert _count_input_calls("this is not valid python {{{") == 0


def test_looks_like_prompt():
    assert _looks_like_prompt("名前: ")
    assert _looks_like_prompt("入力> ")
    assert _looks_like_prompt("よろしいですか? ")
    assert _looks_like_prompt("数値：")
    assert _looks_like_prompt("q？")
    assert not _looks_like_prompt("hello world")
    assert not _looks_like_prompt("")
    assert not _looks_like_prompt("行末に改行\n")
    assert not _looks_like_prompt("plain text without prompt")


# =====================================================
# ドライバループの統合テスト（実プロセス）
# =====================================================

def test_single_input_prompt_detected(monkeypatch):
    """プロンプト付き input() を検出し LLM 生成値を送信する。"""
    monkeypatch.setattr(config, "RUNPY_PROMPT_FALSEPOS_GRACE_SEC", 0.05)
    code = 'n = input("名前: ")\nprint(f"やあ{n}")'
    result, llm, out = _run(code, _ScriptedInputLLM(["Alice"]))
    assert "やあAlice" in result
    assert llm.calls == 1
    assert any("[自動入力: Alice]" in t for t in out.buf)


def test_multi_input_sequential(monkeypatch):
    """複数の input() を順番に処理する。"""
    monkeypatch.setattr(config, "RUNPY_PROMPT_FALSEPOS_GRACE_SEC", 0.05)
    code = (
        'name = input("名前: ")\n'
        'age = input("年齢: ")\n'
        'print(f"{name}は{age}歳")'
    )
    result, llm, _ = _run(code, _ScriptedInputLLM(["Bob", "25"]))
    assert "Bobは25歳" in result
    assert llm.calls == 2


def test_bare_input_idle_timeout(monkeypatch):
    """引数なし input() を idle タイムアウトで検出する。"""
    monkeypatch.setattr(config, "RUNPY_IDLE_TIMEOUT_SEC", 1.0)
    monkeypatch.setattr(config, "RUNPY_PROMPT_FALSEPOS_GRACE_SEC", 0.05)
    code = "x = input()\nprint(x * 2)"
    result, llm, _ = _run(code, _ScriptedInputLLM(["hi"]))
    assert "hihi" in result
    assert llm.calls == 1


def test_stdin_seed_takes_priority(monkeypatch):
    """stdin_seed があれば LLM 生成より優先され、LLM は呼ばれない。"""
    monkeypatch.setattr(config, "RUNPY_PROMPT_FALSEPOS_GRACE_SEC", 0.05)
    code = 'x = input("何か: ")\nprint(x)'
    llm = _ScriptedInputLLM(["UNUSED"])
    result, llm, _ = _run(code, llm, stdin_seed="seedval")
    assert "seedval" in result
    assert llm.calls == 0


def test_max_inputs_cap(monkeypatch):
    """max_inputs 到達で実行を停止する。"""
    monkeypatch.setattr(config, "RUNPY_PROMPT_FALSEPOS_GRACE_SEC", 0.05)
    monkeypatch.setattr(config, "RUNPY_IDLE_TIMEOUT_SEC", 0.3)
    code = 'while True:\n    input("x: ")'
    result, llm, _ = _run(code, _ScriptedInputLLM(["a", "b", "c", "d"]),
                          max_inputs=3, timeout=20)
    assert "max_inputs(3)到達" in result
    assert llm.calls == 3


def test_total_timeout(monkeypatch):
    """総タイムアウトで実行を停止する。"""
    monkeypatch.setattr(config, "RUNPY_IDLE_TIMEOUT_SEC", 0.5)
    code = "while True:\n    input()"
    # while True で input() を無限に呼ぶため、十分な値を用意（2s の間に4回程度しか送らない）
    result, _, _ = _run(code, _ScriptedInputLLM(["x"] * 20), timeout=2)
    assert "総タイムアウト" in result


def test_no_input_calls_skips_generation():
    """input() を含まないコードは LLM 入力生成を呼ばない。"""
    code = 'print("hello")'
    result, llm, _ = _run(code, _ScriptedInputLLM(["UNUSED"]))
    assert "hello" in result
    assert llm.calls == 0


def test_multiline_output_truncated_to_first_line(monkeypatch):
    """LLM が複数行を返しても最初の1行だけ送信する。"""
    monkeypatch.setattr(config, "RUNPY_PROMPT_FALSEPOS_GRACE_SEC", 0.05)
    code = 'x = input("v: ")\nprint(repr(x))'
    result, _, _ = _run(code, _ScriptedInputLLM(["line1\nline2"]))
    # 最初の1行 "line1" だけが input に渡る（改行は空白化）
    assert "line1" in result
    assert "line2" not in result


def test_process_crash_observation_safe():
    """子プロセスが例外で終了しても observation は安全（外側例外にならない）。"""
    code = 'raise RuntimeError("boom")'
    result, _, _ = _run(code, _ScriptedInputLLM([]))
    assert "RuntimeError" in result or "boom" in result
    # 外側の try/except に巻き込まれず、実行結果として返る
    assert not result.startswith("Error: run_python の実行に失敗")


def test_empty_code_error():
    """空コードは即エラー。"""
    result, _, _ = _run("", _ScriptedInputLLM([]))
    assert "必須" in result


def test_generate_failure_breaks(monkeypatch):
    """入力生成が例外で失敗したら安全中断する。"""
    monkeypatch.setattr(config, "RUNPY_PROMPT_FALSEPOS_GRACE_SEC", 0.05)
    code = 'x = input("何か: ")\nprint(x)'
    result, _, _ = _run(code, _ExplodingLLM())
    assert "入力生成失敗" in result


def test_prompt_false_positive_recovery(monkeypatch):
    """末尾が ':' でも通常出力なら（直後に続きが出れば）入力送信しない。"""
    monkeypatch.setattr(config, "RUNPY_PROMPT_FALSEPOS_GRACE_SEC", 1.0)
    code = 'import time\nprint("dict:")\ntime.sleep(0.01)\nprint("done")'
    result, llm, _ = _run(code, _ScriptedInputLLM(["SHOULD_NOT_SEND"]))
    assert "done" in result
    assert llm.calls == 0
