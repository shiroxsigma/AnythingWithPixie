"""
AnythingPixie — パス解決モジュール

ソース実行とPyInstaller exe化の両方で
ファイルパスを正しく解決するためのユーティリティ。

依存: なし（標準ライブラリのみ）
"""

import contextvars
import hashlib
import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    """PyInstallerでexe化されているかを判定する。"""
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')


def get_app_root() -> str:
    """アプリケーションのルートディレクトリ（絶対パス）を返す。

    - ソース実行時: src/ の親ディレクトリ（プロジェクトルート）
    - PyInstaller exe時: exeの配置ディレクトリ
    """
    if is_frozen():
        return os.path.dirname(sys.executable)
    # このファイルは src/pixie_core/paths.py。AWP ルートは3つ上（pixie_core → src → ルート）。
    # 物理パッケージ化で src/paths.py から1階層深くなったため parent を1つ増やしている。
    return str(Path(__file__).resolve().parent.parent.parent)


def get_data_path(relative_path: str) -> str:
    """データファイル（モデル、設定、キャッシュ等）の絶対パスを返す。

    常にアプリケーションルートからの相対パスとして解決する。
    生成されるファイル（CONTEXT_SUMMARY.md, .pixie_notes/ 等）にも使用する。
    """
    return os.path.join(get_app_root(), relative_path)


# =====================================================
# プロジェクトデータルート（作業対象フォルダ基準）
# =====================================================
# get_app_root() は AnythingPixie 自身のインストール先（config.json/models/rg.exe 用）。
# 一方、エージェントの永続状態（ホワイトボード・コアメモリ・.pixie_notes 等）は
# 「作業対象プロジェクト」ごとに分離したい。起動時に set_project_root() で確定し、
# 以降 get_project_data_path() で解決する。未設定時は現在の作業ディレクトリを使う。

_project_root: str | None = None

# =====================================================
# セッション別ワークスペース（マルチセッション: cwd 非依存）
# =====================================================
# 1プロセスで複数セッション（会話）が別々の作業フォルダを扱えるよう、ワークスペースルートを
# ContextVar で持つ。os.chdir（プロセス全体）に依存せず、ターンを実行するスレッド／その並列
# ツール実行（copy_context 伝播先）で自セッションの値が見える一方、別セッションと干渉しない。
#
# 2つのアクセサに使い分ける（今は同一値だが、将来 .pixie_notes をコードツリー外へ出す等の
# 分離余地を残すため名前を分ける）:
#   - get_workspace()     … ユーザーのファイル/シェル操作の相対パス解決の基準（未束縛時 None）
#   - get_project_root()  … 永続ファイル(.pixie_notes 等)の基準（未束縛時は従来 global/cwd へ）
#
# CLI（単一セッション）は従来どおり起動時に set_project_root()+os.chdir() を使い、ContextVar は
# 未束縛のまま → 全アクセサが従来値を返す（完全な後方互換）。
_workspace_var: contextvars.ContextVar = contextvars.ContextVar("pixie_workspace", default=None)
_workspace_buffers_var: contextvars.ContextVar = contextvars.ContextVar(
    "pixie_workspace_buffers", default=None
)
_working_files_var: contextvars.ContextVar = contextvars.ContextVar(
    "pixie_working_files", default=None
)


def bind_workspace(path: str):
    """現在の実行コンテキストにワークスペースルート（絶対パス）を束縛し、token を返す。

    埋め込み時に create_engine（AgentState 構築前）と各 run_turn（ターンスレッド内）で呼ぶ。
    """
    return _workspace_var.set(str(Path(path).resolve()))


def reset_workspace(token) -> None:
    """bind_workspace が返した token で束縛を元に戻す（create_engine の一時束縛の後始末用）。"""
    _workspace_var.reset(token)


def get_workspace() -> str | None:
    """現在のセッションのワークスペースルート（絶対パス）。未束縛なら None（＝cwd 基準の従来動作）。"""
    return _workspace_var.get()


def bind_workspace_buffers(buffers: dict[str, dict] | None):
    """現在のターンでディスクより優先するエディタバッファを束縛する。

    キーは解決済み絶対パス、値は最低限 ``{"content": str}`` を持つ。Engine が
    セッションごとのスナップショットを検証・正規化してから渡すため、ツール側は
    この ContextVar を読むだけで並行セッション間の未保存内容を分離できる。
    """
    return _workspace_buffers_var.set(buffers or {})


def reset_workspace_buffers(token) -> None:
    """bind_workspace_buffers() の束縛を元に戻す。"""
    _workspace_buffers_var.reset(token)


def create_working_file_context(root: str | Path, files: list[str]) -> dict:
    """最新版だけを保持する、版付き作業ファイルコンテキストを構築する。"""
    workspace = Path(root).resolve()
    entries = {}
    for raw in files:
        target = (workspace / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        try:
            rel = target.relative_to(workspace).as_posix()
        except ValueError as exc:
            raise ValueError(f"workspace 外の作業ファイルは登録できません: {raw}") from exc
        content = target.read_text(encoding="utf-8")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        entries[str(target)] = {
            "path": rel, "content": content, "content_hash": digest,
            "disk_hash": digest, "revision": 1, "working_file": True,
            "injected_revision": 0, "injected_chars": 0, "reread_count": 0,
        }
    return {"root": str(workspace), "entries": entries, "last_injection_chars": 0}


def bind_working_file_context(context: dict | None):
    return _working_files_var.set(context)


def reset_working_file_context(token) -> None:
    _working_files_var.reset(token)


def _refresh_working_entry(key: str, entry: dict) -> dict:
    """外部変更があれば最新版へ置換し、旧版は保持しない。"""
    try:
        content = Path(key).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return entry
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if digest != entry.get("disk_hash"):
        entry.update({
            "content": content, "content_hash": digest, "disk_hash": digest,
            "revision": int(entry.get("revision", 0)) + 1,
        })
    return entry


def get_working_file_entry(path: str | Path) -> dict | None:
    context = _working_files_var.get() or {}
    try:
        key = str(Path(path).resolve())
    except (OSError, RuntimeError, ValueError):
        return None
    entry = (context.get("entries") or {}).get(key)
    return _refresh_working_entry(key, entry) if entry is not None else None


def build_working_file_injection(max_chars: int = 16000, per_file_chars: int = 6000) -> str:
    """LLM送信時だけ使う最新版Workset。会話履歴には保存しない。"""
    context = _working_files_var.get() or {}
    entries = context.get("entries") or {}
    if not entries:
        return ""
    # 今回のsuffixに実際に載ったファイルだけを「提供済み」と扱う。前ターンには載ったが、
    # 他ファイルの増大などで今回のmax_charsから外れたentryの古い掲載状態を残さない。
    for entry in entries.values():
        entry["injected_revision"] = 0
        entry["injected_chars"] = 0
    parts = [
        "【作業ファイル（最新版・履歴外）】\n"
        "以下は編集対象として提供済みの正本です。再度 read_file/list_directory せず、"
        "この内容から編集してください。各編集後、このブロックは最新版へ置換されます。"
    ]
    used = len(parts[0])
    for key, entry in entries.items():
        buffered = (_workspace_buffers_var.get() or {}).get(key)
        if buffered is not None and buffered.get("content") != entry.get("content"):
            content = buffered["content"]
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            entry.update({"content": content, "content_hash": digest,
                          "revision": int(entry.get("revision", 0)) + 1})
        elif buffered is None:
            entry = _refresh_working_entry(key, entry)
        content = entry["content"]
        clipped = content[:per_file_chars]
        suffix = "\n…（以降省略。必要部分だけread_fileで取得）" if len(content) > len(clipped) else ""
        block = (
            f"\n--- {entry['path']} (revision={entry['revision']}, "
            f"sha256={entry['content_hash'][:12]}) ---\n```\n{clipped}{suffix}\n```"
        )
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
        entry["injected_revision"] = entry["revision"]
        entry["injected_chars"] = len(clipped)
    rendered = "".join(parts)
    context["last_injection_chars"] = len(rendered)
    return rendered


def working_file_read_notice(path: str | Path) -> str | None:
    """全文が最新版Worksetに掲載済みなら、重複全文読込を短い通知へ置換する。

    行範囲指定や、注入上限で切り詰められたファイルは対象外。呼び出し側はその場合
    通常の読込を続ける。これは禁止ではなく、モデルが明示範囲を再取得できるsoft guardである。
    """
    entry = get_working_file_entry(path)
    if entry is None:
        return None
    if entry.get("injected_revision") != entry.get("revision"):
        return None
    content = entry.get("content", "")
    if int(entry.get("injected_chars", 0)) < len(content):
        return None
    entry["reread_count"] = int(entry.get("reread_count", 0)) + 1
    return (
        f"[Workset] {entry['path']} は revision={entry['revision']} の全文を"
        "直近コンテキストに提供済みです。掲載済みの最新版を使用してください。"
        "特定箇所の再確認が必要な場合だけ start_line/end_line を指定してください。"
    )


def get_working_file_metrics() -> dict:
    """現在のWorksetについて比較評価に使う軽量な集計値を返す。"""
    context = _working_files_var.get() or {}
    entries = context.get("entries") or {}
    return {
        "file_count": len(entries),
        "injection_chars": int(context.get("last_injection_chars", 0)),
        "reread_count": sum(int(entry.get("reread_count", 0)) for entry in entries.values()),
    }


def get_workspace_buffer(path: str | Path) -> dict | None:
    """path に対応する未保存バッファを返す。無ければ None。"""
    buffers = _workspace_buffers_var.get() or {}
    try:
        key = str(Path(path).resolve())
    except (OSError, RuntimeError, ValueError):
        return None
    return buffers.get(key) or get_working_file_entry(key)


def update_workspace_buffer(path: str | Path, content: str) -> None:
    """登録済みバッファの内容をツール適用後の版へ進める。

    未登録パスは追加しない。ディスクだけを編集した通常ターンへ仮想バッファを新設すると、
    次の外部変更を隠してしまうためである。
    """
    buffers = _workspace_buffers_var.get() or {}
    try:
        key = str(Path(path).resolve())
    except (OSError, RuntimeError, ValueError):
        return
    if key in buffers:
        buffers[key] = {**buffers[key], "content": content}
    context = _working_files_var.get() or {}
    entry = (context.get("entries") or {}).get(key)
    if entry is not None:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        entry.update({
            "content": content, "content_hash": digest, "disk_hash": digest,
            "revision": int(entry.get("revision", 0)) + 1,
        })


def set_project_root(path: str) -> str:
    """作業対象プロジェクトのルート（絶対パス）を確定する。CLI が起動時に1回だけ呼ぶ。

    以降 get_project_data_path() が返すパスの基準になる。解決済みの絶対パスを返す。
    """
    global _project_root
    _project_root = str(Path(path).resolve())
    return _project_root


def get_project_root() -> str:
    """作業対象プロジェクトのルート（絶対パス）を返す。

    優先順位: セッション ContextVar > CLI の global > os.getcwd()。
    ContextVar 未束縛（CLI・テスト）時は従来どおり global/cwd を返す（後方互換）。
    """
    return _workspace_var.get() or _project_root or os.getcwd()


def get_project_data_path(relative_path: str) -> str:
    """プロジェクト単位で分離すべき生成ファイル（ホワイトボード・コアメモリ・
    .pixie_notes/ 配下・履歴・debug 等）の絶対パスを返す。

    常に作業対象プロジェクトのルート（get_project_root()）からの相対で解決する。
    アプリ共通のリソース（config.json/models/rg.exe）は従来通り get_data_path を使うこと。
    """
    return os.path.join(get_project_root(), relative_path)


def get_bundled_path(relative_path: str) -> str:
    """バンドルリソース（rg.exe等）の絶対パスを返す。

    - PyInstaller exe時: sys._MEIPASS（一時展開ディレクトリ）を優先、なければexe同梱ディレクトリ
    - ソース実行時: プロジェクトルートからの相対パス
    """
    if is_frozen():
        bundled = os.path.join(sys._MEIPASS, relative_path)
        if os.path.exists(bundled):
            return bundled
        return os.path.join(get_app_root(), relative_path)
    return os.path.join(get_app_root(), relative_path)


def resolve_venv_python(file_path: str) -> str | None:
    """編集対象ファイルを含むプロジェクトの仮想環境(.venv / venv)の Python を返す。

    file_path の親ディレクトリから上方に .venv / venv を探索し、見つかれば
    プラットフォーム別のインタープリタ絶対パスを返す:
      - Windows: {venv}/Scripts/python.exe
      - Unix:    {venv}/bin/python
    実在確認して返す。見つからなければ None（呼出側で sys.executable にフォールバック）。

    注意: get_app_root() は AnythingPixie 自身のルートであり、編集対象プロジェクトとは
    限らないため、編集ファイル起点で上方探索する。
    """
    try:
        start = Path(file_path).resolve()
    except Exception:
        return None

    candidates = [start, *start.parents]
    home = Path.home().resolve()
    for d in candidates:
        # ユーザー共通の ~/.venv は対象ファイルのプロジェクト環境ではない。
        # home より内側のプロジェクトにある venv は、ここへ到達する前に検出される。
        if d == home or d.parent == d:
            break
        for venv_name in (".venv", "venv"):
            venv_dir = d / venv_name
            if not venv_dir.is_dir():
                continue
            if os.name == "nt":
                exe = venv_dir / "Scripts" / "python.exe"
            else:
                exe = venv_dir / "bin" / "python"
            if exe.exists():
                return str(exe)
    return None
