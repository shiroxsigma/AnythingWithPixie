"""
AnythingPixie — 設定定数モジュール

全モジュールで共有する定数を一元管理する。他のプロジェクトモジュールには依存しない。
（paths.py のみ、パス解決のためにインポートする）
"""

from paths import get_data_path

# =====================================================
# コンテキストウィンドウ
# =====================================================

#: llama.cpp に渡す n_ctx（コンテキストウィンドウの最大長）
N_CTX: int = 16384 * 2

#: LLMの1回の生成で出力できる最大トークン数
MAX_TOKENS: int = 4096  * 2

#: safe_max 計算時の安全バッファ（システムプロンプトや特殊トークンの余地）
CONTEXT_BUFFER: int = 500

#: 入力コンテキストの最低保証トークン数
MIN_CONTEXT_TOKENS: int = 1000

#: コンテキスト超過判定のデフォルト閾値（トークン数）
DEFAULT_TRIM_THRESHOLD: int = 16000

#: ツール結果1件あたりの文字数上限の【フォールバック値】（動的未設定時に使用）。
#: 実運用では engine がコンテキスト使用率から逆算した動的上限（_dynamic_tool_cap:
#: 使用率<40%→16000, <65%→12000, 65%以上→6000）が優先される。
#: read_file は行番号付きで返すため、切詰め時にも
#: 「続きは start_line=N で再取得」を正確に案内でき、重復読込を防ぐ。
TOOL_RESULT_MAX_CHARS: int = 12000

#: search_and_replace のファジーマッチ適用閾値（difflib SequenceMatcher.ratio()）。
#: 厳格モード: 行内のインデント差・表記揺れ・タイポを吸収するが、行数違い/略記は対象外。
#: 高め(0.85)に設定し誤適用リスクを抑える（一致行のみ置換・スパン拡張なし）。
FUZZY_MATCH_THRESHOLD: float = 0.85

# =====================================================
# ツール分類
# =====================================================

#: 並列実行可能な読み取り専用ツール（I/Oバウンド、副作用なし）
READONLY_TOOLS: frozenset[str] = frozenset({
    "get_cwd", "get_file_dir", "list_directory", "read_file",
    "grep_search", "get_code_outline", "analyze_file",
    "query_whiteboard", "inspect_tool", "view_tree",
    "research_code_paths",
})

#: 直列実行必須の破壊的操作ツール（状態変更・ファイル書き換え・外部プロセス起動）
DESTRUCTIVE_TOOLS: frozenset[str] = frozenset({
    "write_file", "append_to_file", "write_sections", "make_directory", "move_file",
    "delete_file", "run_command", "replace_lines", "search_and_replace",
    "run_async_test", "kill_process",
    "update_core_memory", "update_state", "set_goal",
    "view_image", "gather_project_info",
})

#: 1回の返答で並列実行できる最大ツール数
MAX_PARALLEL_TOOLS: int = 5

#: 常時推奨ツール（ユーザー入力に関わらず常にフルスキーマを表示）
ALWAYS_RECOMMEND: frozenset[str] = frozenset({
    "update_state", "get_cwd", "list_directory", "read_file",
    "grep_search", "run_command", "search_and_replace",
    "run_async_test", "poll_process",
    "get_code_outline", "read_symbol",
})

#: /code モードで固定提供するツールセット（JITスコアリング score_tools をバイパス）。
#: 構造把握(map_codebase/get_code_outline/research_code_paths) → シンボル単位読込
#: (read_symbol) → 編集(search_and_replace 等) のコードワークフローを支える。
CODE_TOOL_SET: frozenset[str] = frozenset({
    "map_codebase", "analyze_file", "get_code_outline", "research_code_paths",
    "read_symbol", "read_file", "grep_search",
    "search_and_replace", "replace_lines", "write_file", "write_sections",
    "detect_dead_code", "gather_project_info", "get_file_stats",
    "view_tree", "get_cwd", "list_directory", "update_state",
})

# =====================================================
# Temperature設定
# =====================================================

TEMPERATURE_MAIN: float = 0.7
TEMPERATURE_SUBQUERY: float = 0.2
TEMPERATURE_LOOP_THRESHOLD: int = 15

# =====================================================
# コンテキスト予算配分
# =====================================================

CONTEXT_CHECKPOINT_THRESHOLD: float = 0.80

# =====================================================
# 思考深度（Progressive Deepening）
# =====================================================

#: deep思考モードでの <think> フェーズ最大継続秒数。
#: これを超えると推論を打ち切り、結論生成へ移行する（無限長考の安全装置）。
DEEP_THINK_BUDGET_SEC: int = 90

# =====================================================
# モデルディレクトリ
# =====================================================

MODEL_DIR: str = get_data_path("models")

# =====================================================
# ホワイトボード
# =====================================================

WHITEBOARD_PATH: str = get_data_path("CONTEXT_SUMMARY.md")
WHITEBOARD_DETAIL_SEPARATOR: str = "\n---\n<!-- DETAIL_SECTION -->\n"

WHITEBOARD_SYSTEM_PROMPT: str = """あなたは優秀なアシスタントです。AIエージェントの作業履歴を「ホワイトボード」形式で整理し直す役割を担います。

既存のホワイトボード内容と、新たに切り捨てられた会話ログが渡されます。
両方の情報を統合して、以下の厳密なフォーマットで新しいホワイトボードを出力してください。
Thinkingの内容や、ユーザーの入力内容は除外してください。
中国語は絶対に使用しないでください。

【上部セクション: コンテキスト注入用（簡潔に）】
```
## 要約
（現在の状況を3行以内で要約。スクラッチパッド(CORE_MEMORY)とは別の客観的な視点で記述）

## タスク経緯
- 実行済みの主要アクションと結果（箇条書き、重複は統合）

## 判明した事実・制約
- ファイル構造、バグ、設定値、環境情報など客観的な事実のみ

## 現在の焦点と次のステップ
- 今何に取り組んでいるか、次に何をすべきか
```

【下部セクション: 詳細記録（長くてもOK・grep検索用）】
上部セクションの後に区切り線を置き、その下にこれまでの全ての詳細情報を記録する。
- 各ツール実行の具体的な結果（パス、エラーメッセージ等）
- 読み込んだファイルの概要
- 試行錯誤の経緯
- 具体的なコード断片やコマンド出力
この部分は省略せず、重要な情報を漏らさないようにしてください。

【厳守事項】
- 上部セクションは合計1000文字以内に収める
- 下部セクションは制限なし（grepで検索する前提）
- 日本語で記述する
- 事実のみを記載し、推測や感想は書かない"""
