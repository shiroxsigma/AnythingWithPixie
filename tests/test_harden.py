"""バグ修正 (#5,#2,#8,#3,#6,#1) の堅牢性テスト。

#4 (eos_token) は llama_cpp + 実GGUF が必要なため CI 対象外。手動スモークで確認。
"""

import os
import sys
import time
import urllib.error
import urllib.request


def test_llama_cpp_optional_flag_is_bool():
    """#5: llama_cpp はオプショナル依存。_has_llama_cpp は環境に関わらず常に bool。

    CI は llama_cpp 未インストールで動くことが前提 → この属性が存在し bool であること。
    """
    import llm_client
    assert isinstance(llm_client._has_llama_cpp, bool)


def test_lmstudio_urlopen_error_yields_error_chunk(monkeypatch):
    """#2: urlopen がタイムアウト/接続エラーでも generator の yield 形状を維持する。"""
    import llm_client

    def fake_urlopen(req, timeout=None):
        # timeout が渡されること（引数が受け付けられている）も暗黙に確認
        raise urllib.error.URLError("simulated timeout")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    # __init__ を回避してネットワーク呼び出し (_fetch_n_ctx) を避ける
    backend = llm_client.LMStudioBackend.__new__(llm_client.LMStudioBackend)
    backend.base_url = "http://localhost:1234/v1"
    backend.api_key = "x"
    backend.model = "m"
    backend._n_ctx = 4096
    backend.overall_timeout = 180.0
    backend.read_idle_timeout = 30.0

    chunks = list(backend.create_chat_completion([], max_tokens=10))
    assert chunks, "エラー時も空でないチャンクを返すべき"
    assert "choices" in chunks[0]


def test_run_async_test_closes_log_handle(tmp_path, monkeypatch):
    """#8: 成功パスで log_f を close する。

    Windows ではハンドルが開いたままだとファイル削除に失敗する (PermissionError) ので、
    削除成功 = close 済み の確実な検証になる。
    """
    import tools

    class FakeProc:
        pid = 99999

    # 実プロセス起動 (CREATE_NEW_CONSOLE) を回避
    monkeypatch.setattr(tools.subprocess, "Popen", lambda *a, **k: FakeProc())

    log = str(tmp_path / "async.log")
    result = tools.run_async_test("anything", log_file=log)
    assert "99999" in result
    # log_f が close 済みでなければ Windows で PermissionError
    os.remove(log)


def test_lmstudio_no_longer_fakes_tokenize():
    """#3: LM Studio は偽の tokenize を持たず、正直な estimate_token_count を持つ。"""
    import llm_client
    backend = llm_client.LMStudioBackend.__new__(llm_client.LMStudioBackend)
    assert not hasattr(backend, "tokenize"), "偽の tokenize は削除済みであるべき"
    assert hasattr(backend, "estimate_token_count")
    count = backend.estimate_token_count("hello world test")
    assert count == len("hello world test") // 3


def test_mcp_stop_terminates_within_timeout():
    """#6: stop() は terminate→wait→kill で確実にプロセスを終了させる。"""
    import mcp_client
    client = mcp_client.LightweightMCPClient(
        [sys.executable, "-c", "import time; time.sleep(60)"]
    )
    try:
        start = time.monotonic()
        client.stop()
        elapsed = time.monotonic() - start
        assert elapsed < 12, "wait(5) + kill(5) + 余裕 以内に終了すべき"
        assert client.process.poll() is not None, "プロセスは終了済みであるべき"
    finally:
        if client.process.poll() is None:
            client.process.kill()


def test_search_and_replace_exact_match(tmp_path):
    """#1: 完全一致 → Success で置換される。"""
    from tools import execute_builtin_tool
    f = tmp_path / "target.txt"
    f.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    result = execute_builtin_tool("search_and_replace", {
        "path": str(f),
        "search_block": "beta",
        "replace_block": "BETA",
    })
    assert "Success" in result
    assert f.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"


def test_search_and_replace_not_found_returns_error(tmp_path):
    """#1: 未発見 → ヒント付き Error（デッドコード削除後も挙動不变）。"""
    from tools import execute_builtin_tool
    f = tmp_path / "target.txt"
    f.write_text("alpha\nbeta\n", encoding="utf-8")
    result = execute_builtin_tool("search_and_replace", {
        "path": str(f),
        "search_block": "totally nonexistent content here",
        "replace_block": "x",
    })
    assert "Error" in result


def test_read_file_large_py_suppressed(tmp_path):
    """read_file は .py 大ファイル(500行超)の全文を抑制し、構造+先頭を返す。"""
    from tools import read_file
    f = tmp_path / "big.py"
    f.write_text(("def foo():\n    pass\n" * 300) + "def bar():\n    return 1\n", encoding="utf-8")
    result = read_file(str(f))
    assert "全文読込を省略" in result
    assert "## 構造" in result


def test_read_file_small_py_not_suppressed(tmp_path):
    """小さい .py ファイルは全文読込される（抑制なし）。"""
    from tools import read_file
    f = tmp_path / "small.py"
    f.write_text("def foo():\n    pass\n", encoding="utf-8")
    result = read_file(str(f))
    assert "全文読込を省略" not in result


def test_read_file_range_not_suppressed(tmp_path):
    """範囲指定(start_line/end_line)は抑制されず範囲を返す。"""
    from tools import read_file
    f = tmp_path / "big.py"
    f.write_text("x = 1\n" * 600, encoding="utf-8")
    result = read_file(str(f), start_line="100", end_line="110")
    assert "全文読込を省略" not in result
    assert "100行目" in result
