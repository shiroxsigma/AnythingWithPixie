"""pytest 共通設定。

pyproject.toml の `pythonpath = ["src"]` と併用し、フラット import
(from config import ...) をテスト実行時にも解決するための sys.path 操作。
"""

import gc
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest


@pytest.fixture(autouse=True)
def _gc_after_test():
    """各テスト後に gc.collect() で未参照のファイルハンドルを確実に閉じる。

    pyproject.toml の filterwarnings=["error::ResourceWarning"] により、ファイル
    ハンドルリークは hard-fail する。LLM mock の generator や Windows の subprocess
    パイプ破棄タイミングで未参照ハンドルの GC が遅れ、リファクタ無関係の flaky
    failure を引き起こすのを防ぐ（Bug #8 関連）。
    """
    yield
    gc.collect()


def _has_llama_cpp() -> bool:
    """llama_cpp がインストールされているか。CI は未インストールで動く前提。"""
    try:
        import llama_cpp  # noqa: F401
        return True
    except Exception:
        return False


llama_cpp_required = pytest.mark.skipif(
    not _has_llama_cpp(), reason="llama_cpp not installed"
)


def _has_prompt_toolkit() -> bool:
    """prompt_toolkit がインストールされているか。CI は未インストールで動く前提。"""
    try:
        import prompt_toolkit  # noqa: F401
        return True
    except Exception:
        return False


prompt_toolkit_required = pytest.mark.skipif(
    not _has_prompt_toolkit(), reason="prompt_toolkit not installed"
)
