"""Shared runtime context for CLI and embedded engines."""

class AppContext:
    """Holds shared state and capabilities for the entire application."""
    def __init__(self):
        self.llm = None               # LLMBackend (LlamaCppBackend or LMStudioBackend)
        self.delegate_llm = None      # 委譲サブエージェント用バックエンド（別サーバー）。None時は self.llm にフォールバック
        self.llm_model_name = ""       # LM Studio のモデル名（表示用）
        self.use_vision = False
        # delegate_llm がVision対応（mmproj/VLM）かどうかの明示フラグ。LM Studio /
        # llama-server のOpenAI互換APIだけでは接続先モデルがVision対応かを安全に
        # 事前判定できないため、config.json の delegate_server.vision キーを信頼できる
        # 情報源として扱う（_load_delegate_server 参照）。
        self.delegate_vision: bool = False
        self.is_qwen35 = False
        self.is_lfm25 = False  # [LFM専用] 不要時: この行 + 各 # [LFM専用] 行を削除

        self.use_capture = False
        self.capture_bbox = None

        self.overlay_manager = None

        # UI callback functions
        self.update_overlay_func = None
        self.get_inner_bbox_func = None
        self.select_screen_area_func = None

        # Phase management
        self.phase = "EXECUTING"

        # Model compatibility flag: when True, role="tool" is sent as-is.
        # When False, converted to role="user" (for LM Studio + non-FC models).
        self.supports_tool_role: bool = False

        # 深度思考の強制フラグ（/deep コマンドでトグル）。True時は段階的判定をスキップし常に deep。
        self.force_deep: bool = False

        # /code モード（ワンショット）: 次の1ターンをコード専門モードで実行。
        # 固定 CODE_TOOL_SET・強制 deep・コードワークフロープロンプト・ガードレール緩和。
        # 次ターン冒頭でリセットされる（/trace と同じ one-shot 系）。
        self.code_mode: bool = False
        self.code_target: str = ""

        # /review モード: 破壊的ファイル編集の直後に読み取り専用レビューアを起動し、
        # 判定を observation に付加する（observe-only・編集は実行される）。
        # 状態は context のみに置き、ローカル変数ミラーは作らない。
        self.review_mode: bool = False

        # /verify モード: ファイル編集後に「実際に実行して」検証し、エラーがあれば自動で
        # 編集し直すループ（verify → fix → re-verify）。/review（LLM判定・observe-only）とは
        # 独立。検証は py_compile/ruff/pytest で、.venv の Python を優先使用。
        self.verify_mode: bool = False

        # ツールパック機構: 有効化されているパック名。
        # 既定は空集合 = 従来通り全コアツールのみ（詳細設計 docs/design/toolpacks.md）。
        # config.json の "toolpacks" キー、または CLI の /pack コマンドで追加される。
        # ターン中には変化しない（/pack はユーザー入力処理＝ターン境界でのみ実行）ため
        # prefix cache は保護される。
        self.active_packs: set = set()

        # 軌跡ロギング（SFT/DPO 教師データ産出基盤・src/trajectory.py）。
        # setup_application() で TrajectoryLogger インスタンスを設定する。
        # None のまま（未設定）でも engine.py 側は getattr(context, "trajectory", None) で
        # 安全にガードするため、本体の動作には影響しない。
        self.trajectory = None
