"""#4 セッション別 workspace の安全網。

- paths の workspace ContextVar が get_project_root/get_project_data_path に反映されること。
- 2スレッドで別 workspace を束縛しても交差しないこと（ContextVar 分離）。
- engine のディスパッチ正規化: 相対パス→絶対化、絶対パス不変、未束縛時 no-op。
- create_engine が os.chdir しない（プロセス cwd 不変）こと。
- execute_parallel が workspace を並列ワーカーへ伝播すること。
"""
import json
import os
import threading

import paths


def test_get_project_root_prefers_workspace_var():
    assert paths.get_workspace() is None  # 既定 未束縛
    token = paths.bind_workspace(os.path.abspath("."))
    try:
        ws = paths.get_workspace()
        assert ws == os.path.abspath(".")
        assert paths.get_project_root() == ws
        assert paths.get_project_data_path(".pixie_notes/x.json") == os.path.join(ws, ".pixie_notes/x.json")
    finally:
        paths.reset_workspace(token)
    assert paths.get_workspace() is None  # 復元


def test_workspace_var_isolation_across_threads():
    results = {}
    barrier = threading.Barrier(2)

    def worker(name, ws):
        paths.bind_workspace(ws)
        barrier.wait()  # 両方が bind してから読む（global なら混線する）
        results[name] = paths.get_workspace()

    a = os.path.abspath("A")
    b = os.path.abspath("B")
    t1 = threading.Thread(target=worker, args=("A", a))
    t2 = threading.Thread(target=worker, args=("B", b))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert results["A"] == a and results["B"] == b


def _make_call(name, args):
    return {"function": {"name": name, "arguments": json.dumps(args)}}


def test_normalize_relative_paths_to_workspace():
    import engine
    ws = os.path.abspath("some_ws")
    token = paths.bind_workspace(ws)
    try:
        calls = [_make_call("write_file", {"path": "src/foo.py", "content": "x"}),
                 _make_call("move_file", {"src": "a.txt", "dst": "b.txt"}),
                 _make_call("run_command", {"command": "ls", "working_directory": "sub"})]
        engine._normalize_tool_call_paths(calls)
        a0 = json.loads(calls[0]["function"]["arguments"])
        a1 = json.loads(calls[1]["function"]["arguments"])
        a2 = json.loads(calls[2]["function"]["arguments"])
        assert a0["path"] == os.path.normpath(os.path.join(ws, "src/foo.py"))
        assert a0["content"] == "x"  # 非パスキーは不変
        assert a1["src"] == os.path.normpath(os.path.join(ws, "a.txt"))
        assert a1["dst"] == os.path.normpath(os.path.join(ws, "b.txt"))
        assert a2["working_directory"] == os.path.normpath(os.path.join(ws, "sub"))
        assert a2["command"] == "ls"
    finally:
        paths.reset_workspace(token)


def test_normalize_absolute_unchanged_and_unbound_noop():
    import engine
    absp = os.path.abspath("already/abs.py")
    # 未束縛: 完全 no-op
    calls = [_make_call("read_file", {"path": "rel.py"})]
    engine._normalize_tool_call_paths(calls)
    assert json.loads(calls[0]["function"]["arguments"])["path"] == "rel.py"
    # 束縛下でも絶対パスは不変
    token = paths.bind_workspace(os.path.abspath("ws"))
    try:
        calls = [_make_call("read_file", {"path": absp})]
        engine._normalize_tool_call_paths(calls)
        assert json.loads(calls[0]["function"]["arguments"])["path"] == absp
    finally:
        paths.reset_workspace(token)


def test_create_engine_does_not_chdir(tmp_path):
    import pixie_core
    cwd0 = os.getcwd()
    eng = pixie_core.create_engine({"base_url": "http://127.0.0.1:1/v1", "model": "m"}, str(tmp_path))
    assert os.getcwd() == cwd0, "create_engine がプロセス cwd を変更した（chdir 廃止の回帰）"
    assert eng.workspace == str(tmp_path.resolve())
    assert paths.get_workspace() is None, "create_engine 後に束縛が呼び出しスレッドへ漏れている"


def test_execute_parallel_propagates_workspace():
    import engine
    ws = os.path.abspath("ws_parallel")
    token = paths.bind_workspace(ws)
    try:
        seen = engine.execute_parallel(
            [_make_call("read_file", {"path": "x"})],
            executor_fn=lambda name, args: paths.get_workspace() or "NONE",
        )
        assert seen and seen[0][1] == ws, "並列ワーカーに workspace が伝播していない"
    finally:
        paths.reset_workspace(token)
