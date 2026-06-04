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
