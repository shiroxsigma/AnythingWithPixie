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
    return str(Path(__file__).resolve().parent.parent)


def get_data_path(relative_path: str) -> str:
    """データファイル（モデル、設定、キャッシュ等）の絶対パスを返す。

    常にアプリケーションルートからの相対パスとして解決する。
    生成されるファイル（CONTEXT_SUMMARY.md, .pixie_notes/ 等）にも使用する。
    """
    return os.path.join(get_app_root(), relative_path)


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
