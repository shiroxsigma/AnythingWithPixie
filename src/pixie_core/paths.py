"""
AnythingPixie — パス解決モジュール

ソース実行とPyInstaller exe化の両方で
ファイルパスを正しく解決するためのユーティリティ。

依存: なし（標準ライブラリのみ）
"""

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


def set_project_root(path: str) -> str:
    """作業対象プロジェクトのルート（絶対パス）を確定する。起動時に1回だけ呼ぶ。

    以降 get_project_data_path() が返すパスの基準になる。解決済みの絶対パスを返す。
    """
    global _project_root
    _project_root = str(Path(path).resolve())
    return _project_root


def get_project_root() -> str:
    """作業対象プロジェクトのルート（絶対パス）を返す。

    set_project_root() 未呼出時は現在の作業ディレクトリ（os.getcwd()）にフォールバックする。
    """
    return _project_root or os.getcwd()


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
    for d in candidates:
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
