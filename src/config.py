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

#: 空応答時の再試行回数上限。LM Studio が空応答(contentもtool_callsもなし)を返した際、
#: ガードレール注入で再生成を促す回数。超過で fallback/empty_response 終了。
#: 無限ループ防止のため厳守（run_graph は total_iterations 安全カウンタで二重防御）。
EMPTY_RESPONSE_MAX_RETRY: int = 2

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
    "delegate_research",  # 独立コンテキストの並列安全な調査委譲
})

#: 直列実行必須の破壊的操作ツール（状態変更・ファイル書き換え・外部プロセス起動）
DESTRUCTIVE_TOOLS: frozenset[str] = frozenset({
    "write_file", "append_to_file", "write_sections", "make_directory", "move_file",
    "delete_file", "run_command", "replace_lines", "search_and_replace",
    "run_async_test", "kill_process",
    "update_core_memory", "update_state", "set_goal",
    "view_image", "gather_project_info",
    "run_python",  # 外部プロセス起動 + input()検出時のLLM連呼自動入力 → 直列実行＋半自動承認
})

#: 1回の返答で並列実行できる最大ツール数
MAX_PARALLEL_TOOLS: int = 5

#: 常時推奨ツール（ユーザー入力に関わらず常にフルスキーマを表示）
# ※ update_state は常時推奨から除外（「実行の代わり」に選ばれるのを防ぐ）。
#    状態記録が必要な文脈（進捗/メモ/記録/次にやること）で score_tools が改めて推奨する。
ALWAYS_RECOMMEND: frozenset[str] = frozenset({
    "get_cwd", "list_directory", "read_file",
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
    "delegate_research",  # /code モードで調査委譲を有効化
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


# =====================================================
# 委譲サブエージェント (delegate_research)
# =====================================================

#: サブエージェントの最大調査ステップ数（LLM往復1回=1ステップ）。
#: 1ステップで複数ツールを並列実行しうるが、調査系は通常1-2件/ステップ。
DELEGATE_MAX_STEPS: int = 6

#: サブエージェントの1ターンあたり最大生成トークン数（控えめ。結論は短い）。
DELEGATE_MAX_TOKENS: int = 2048

#: サブエージェント全体の wall-clock 予算（秒）。超過時は途中結論で打ち切り。
#: LM Studio が深思考で停滞する場合の安全装置。
DELEGATE_BUDGET_SEC: int = 120

#: サブエージェント自身のコンテキスト使用率がこの閾値を超えたら
#: 新規ツール呼び出しを打ち切り結論生成へ強制移行する。
DELEGATE_CTX_USAGE_LIMIT: float = 0.60

#: サブエージェント内部で実行したツール結果1件あたりの文字上限。
#: execute_builtin_tool が読むグローバル(_dynamic_max_chars)には依存せず、
#: サブエージェント専用のキャップでメインループの設定を破壊しない。
DELEGATE_TOOL_RESULT_CAP: int = 4000

#: サブエージェントが内部で使用できるツールセット（決定的読み取り専用）。
#: analyze_file は除外 — ネストLLM呼び出しで遅延・予算消費が増すため。
DELEGATE_SUBAGENT_TOOLS: frozenset[str] = frozenset({
    "get_cwd", "get_file_dir", "list_directory", "read_file",
    "grep_search", "get_code_outline", "read_symbol",
    "view_tree", "research_code_paths", "map_codebase",
    "query_whiteboard", "inspect_tool",
})

#: サブエージェント用の固定システムプロンプト（byte-identical を保証し、
#: LM Studio の自動プレフィックスキャッシュにヒットさせる。動的注入禁止）。
DELEGATE_SYSTEM_PROMPT: str = """\
あなたは調査専用サブエージェントです。与えられた質問に対し、提供された読み取り専用ツールだけを使って事実を集め、最後に日本語で簡潔な結論を返してください。

【ルール】
- 使えるのは読み取り専用ツール(grep_search, read_file, get_code_outline, read_symbol, view_tree, list_directory 等)だけ。ファイル書き換え・コマンド実行は禁止。
- 思考は簡潔に。長い <think> ブロックは出さない。
- 質問を復唱しない。経過報告しない。
- 調査が終わったら、ツールを呼ばずに結論だけを出力すること。
- 結論は事実ベース・箇条書き可・300字程度に収める。推測は「(推測)」と明示。
- 日本語のみで回答。中国語は使用しない。"""


# =====================================================
# 編集前レビューサブエージェント (/review)
# =====================================================
# delegate_research の調査用途に対する「批判」用途の設定。
# run_agent_subquery の mode="review" で使用され、破壊的ファイル編集の直後に
# 編集結果を読み取り専用ツールで検証し、判定と指摘を observation に返す。
# 編集自体は実行済み（observe-only）のため、ここは「事後検証」になる。

#: レビューの最大ステップ数。調査より短く抑え、1ターンあたりの遅延を限定する。
#: バッチ内の複数編集は直列にレビューされるため、厳しめに設定。
REVIEW_MAX_STEPS: int = 3

#: レビューの1ターンあたり最大生成トークン数（判定+指摘なので短い）。
REVIEW_MAX_TOKENS: int = 1024

#: レビュー全体の wall-clock 予算（秒）。直列化するバッチ編集の待ちを bound する。
REVIEW_BUDGET_SEC: int = 30

#: レビューアが内部で使用する読み取り専用ツールセット。
#: 調査(delegate)と同一で十分（read_file/grep_search/read_symbol 等を含む）。
REVIEW_SUBAGENT_TOOLS: frozenset[str] = DELEGATE_SUBAGENT_TOOLS

#: レビューア用の固定システムプロンプト。与えられた「編集案（ファイル・変更内容・目標）」を
#: 読み取り専用ツールで検証し、1行の判定行と簡潔な指摘を返す。
REVIEW_SYSTEM_PROMPT: str = """\
あなたはコード編集のレビュー専門サブエージェントです。提示された「ファイル」「変更内容」「目標」をもとに、読み取り専用ツールだけを使って編集の正しさを検証し、判定と指摘を返してください。

【検証の観点（必要なものだけ、ツールで確かめる）】
- 置換対象(search_block 等)がファイルに実在するか。
- 新規コードが参照する関数・変数・シンボル・import が実在するか。
- 構文ミス・論理的破綻・必要な import の欠落・呼び出し規約の不一致がないか。
- 目標と整合しているか。壊す副作用はないか。

【ルール】
- 使えるのは読み取り専用ツール(grep_search, read_file, get_code_outline, read_symbol, view_tree 等)だけ。
- 思考は簡潔に。長い <think> ブロックは出さない。
- 検証が終わったら、ツールを呼ばずに下記フォーマットで出力すること。

【出力フォーマット（厳守・1行目は必ずこの判定行）】
判定: 問題なし
または
判定: 軽微
または
判定: 要確認
または
判定: 重大
続いて、指摘があれば簡潔な箇条書き（問題なければ指摘は省いてよい）。
- 日本語のみ・全体で300字程度に収める。推測は「(推測)」と明示。"""


# =====================================================
# 設計/提案レビュー ＋ /review_loop 反復改善
# =====================================================
# 編集ではなく「設計/提案の final_answer」を批判する用途。機能A(設計モード自動発火)
# と機能B(/review_loop 反復改善)で共用。run_agent_subquery(mode="review") に
# review_system_prompt=REVIEW_DESIGN_SYSTEM_PROMPT を渡して使用する。

#: 設計レビューを自動発火させる最小文字数。これ未満の短い回答はレビューしない（コスト制御）。
REVIEW_DESIGN_MIN_CHARS: int = 500

#: /review_loop のデフォルト往復数（main↔review のセットを何回繰り返すか）。
REVIEW_LOOP_DEFAULT_ROUNDS: int = 3

#: /review_loop の最大往復数（ローカル LLM のコスト/時間バウンド）。
REVIEW_LOOP_MAX_ROUNDS: int = 5

#: /review_loop の main 改善生成（1往復分）の最大トークン数。
REVIEW_LOOP_REVISE_MAX_TOKENS: int = 2048

#: 設計/提案批判用の固定システムプロンプト。編集の正しさではなく「要件カバレッジ・矛盾・
#: リスク・代替案」を批判する。必要なら read_file/grep で仕様書や既存コードを照合する。
REVIEW_DESIGN_SYSTEM_PROMPT: str = """\
あなたは設計/提案のレビュー専門サブエージェントです。提示された「ユーザの要求」「エージェントの設計/提案」「目標」をもとに、批判的にレビューし、判定と指摘を返してください。必要なら読み取り専用ツールで仕様書や既存コードを読み、提案との整合を確かめてください。

【姿勢（最重要）】
- 積極的にリスク・反論・より良い代替案を探せ。整合しているだけでは“問題なし”にするな。
- “問題なし”は、本当に問題を探し尽くして初めて出す判定。基本は懐疑的に。

【レビューの観点】
- ユーザの要求/仕様の要件を過不足なくカバーしているか（漏れ・余計なもの）。
- 技術的な矛盾・実現性の問題・見落としたリスク・依存の罠がないか。
- より良い代替案・トレードオフ・優先順位の誤りを指摘できるか。
- 子供向け/非技術者向け等の文脈があればそれに合っているか。

【ルール】
- 使えるのは読み取り専用ツール(grep_search, read_file, get_code_outline, read_symbol, view_tree 等)だけ。
- 思考は簡潔に。長い <think> ブロックは出さない。
- レビューが終わったら、ツールを呼ばずに下記フォーマットで出力すること。

【出力フォーマット（厳守・1行目は必ずこの判定行）】
判定: 問題なし
または
判定: 軽微
または
判定: 要確認
または
判定: 重大
続いて、指摘があれば簡潔な箇条書き（問題なければ指摘は省いてよい）。
- 日本語のみ・全体で300字程度に収める。推測は「(推測)」と明示。"""


# =====================================================
# 実行ベース検証 + 自動再編集ループ (/verify)
# =====================================================
# /verify トグル: ファイル編集後に「実際に実行して」検証し、エラーがあれば自動で
# 編集し直すループ（verify → fix → re-verify）。検証の根拠は LLM の主観ではなく
# 実行結果（py_compile / ruff / pytest）。/review（LLM判定・observe-only）とは独立。

#: verify→fix ループの最大往復数（ローカル LLM のコスト/無限ループ防止）。
VERIFY_MAX_ROUNDS: int = 3

#: verify ループ全体の wall-clock 予算（秒）。超過時は未解決でも打ち切り。
VERIFY_BUDGET_SEC: int = 120

#: py_compile ゲートのタイムアウト（秒）。副作用なし・安全な第1ゲート。
VERIFY_COMPILE_TIMEOUT_SEC: int = 10

#: テスト(pytest)ゲートのタイムアウト（秒）。副作用あり。
VERIFY_TEST_TIMEOUT_SEC: int = 60

#: ruff ゲートを有効にするか（軽量・デフォルト ON）。構文/未定義名を機械的に検出。
VERIFY_RUFF_GATE: bool = True

#: import 解決ゲートを有効にするか（AST + find_spec・副作用なし・デフォルト ON）。
#: サードパーティモジュールの未インストール/依存欠落を検出する（py_compile/ruff では見逃される）。
VERIFY_IMPORT_GATE: bool = True

#: import 解決ゲートのタイムアウト（秒）。
VERIFY_IMPORT_TIMEOUT_SEC: int = 15

#: pytest ゲートを有効にするか（副作用リスク・デフォルト OFF）。明示的に信頼する時のみ。
VERIFY_TEST_GATE: bool = False

#: 高速ゲート（py_compile + import解決 + ruff）を /verify トグルに関係なく常時実行するか。
#: これらはLLMを使わない決定的チェックでコストがほぼゼロなため、デフォルト True。
#: 検出のみ行い、LLMによる自動再編集（run_verify_fix_loop の修正ループ）は
#: 従来通り verify_mode（/verify トグル）が有効な時のみ実行する。
#: False にすると旧来の挙動（高速ゲートも /verify トグル時のみ実行）に戻る。
VERIFY_FAST_GATE_ALWAYS: bool = True

#: 検証エラー出力・トレースバックの最大文字数（LLM 入力・observation 双方の圧縮用）。
VERIFY_ERROR_MAX_CHARS: int = 1200

#: 修正 edit 生成の最大トークン数。
VERIFY_FIX_MAX_TOKENS: int = 2048

#: 修正編集生成用のシステムプロンプト。検出された実行エラー（トレースバック等）と現在の
#: ファイル内容を渡し、エラーを解消する編集を厳密な JSON 1件で返させる。
VERIFY_FIX_SYSTEM_PROMPT: str = """\
あなたはコード修正専門サブエージェントです。提示された「ファイル」「検出された実行エラー（トレースバック等）」をもとに、エラーを解消する最小限の編集を生成してください。

【ルール】
- デフォルトは search_and_replace を使え。search_block はファイルに実在する文字列（エラー行周辺）にせよ。
- ファイル全体の再編が必要な場合だけ write_file を使え（トークン消費が大きいので最終手段）。
- 推測で書き換えるな。エラーの根因だけを直せ。元のコードスタイル・インデント・言語を踏襲せよ。
- 思考は簡潔に。長い <think> ブロックは出さない。

【出力（厳守・前後に説明文を置かず JSON 1件のみ）】
search_and_replace の場合:
{"tool": "search_and_replace", "args": {"path": "<ファイルパス>", "search_block": "<既存のコード断片>", "replace_block": "<修正後のコード断片>"}}
write_file の場合（全体再編のみ）:
{"tool": "write_file", "args": {"path": "<ファイルパス>", "content": "<ファイル全体>"}}
"""


# =====================================================
# Python サンドボックス実行 (run_python)
# =====================================================
# run_python ツール: Python コードを一時ディレクトリでサンドボックス実行する。
# input() のプロンプトを検出すると LLM が入力を生成して stdin に自動送信し、
# 対話的プログラムを自律継続する（インタラクティブ自動入力）。
# toggle/observe-only ではなく DESTRUCTIVE（半自動承認＋直列実行の対象）。
# 仕組み: python -u で起動 → CPython の input(prompt) は sys.stdout.write+flush するため、
# プロンプト文字列が即座に stdout パイプに現れる。これを監視スレッドで検出する。
# Job Object によるメモリ/CPU リソース分離は未実装（Phase2）。現状は総タイムアウト・
# max_inputs 上限・env サニタイズによる安全網のみ。

#: LLM 自動入力のデフォルト上限回数（無限入力要求・ループの物理的打ち切り）。
RUNPY_MAX_INPUTS: int = 6

#: プロンプト未検出のまま stdout がこの秒数無言なら、引数なし input() の入力待ちとみなす。
RUNPY_IDLE_TIMEOUT_SEC: float = 4.0

#: 実行全体の wall-clock 上限（秒）。
RUNPY_TOTAL_TIMEOUT_SEC: int = 30

#: 1回の入力生成の max_tokens（入力値は1行なので小さく）。
RUNPY_INPUT_MAX_TOKENS: int = 64

#: 入力生成の temperature（低め・決定的）。
RUNPY_INPUT_TEMPERATURE: float = 0.3

#: observation に返す stdout の最大文字数（中央省略で切り詰め）。
RUNPY_OUTPUT_MAX_CHARS: int = 8000

#: ドライバのプロセス終了ポーリング間隔（秒）。
RUNPY_PROBE_INTERVAL_SEC: float = 0.05

#: プロンプト検出後、追加出力を待つ短い猶予（誤検出緩和。出力が来れば通常出力扱い）。
RUNPY_PROMPT_FALSEPOS_GRACE_SEC: float = 0.3

#: プロンプト末尾パターン（input() の入力待ち判定）。: > ？ のいずれか＋末尾空白。
RUNPY_PROMPT_TAIL_RE: str = r"[:>?：？]\s*$"

#: 入力生成用の固定システムプロンプト（byte-identical で LM Studio プレフィックスキャッシュ狙い）。
RUNPY_INPUT_SYSTEM_PROMPT: str = """\
あなたは実行中のPythonプログラムの入力を生成する役割です。実行中のコードとこれまでの入出力履歴、現在のプロンプト文字列が与えられます。そのプロンプトに対してプログラムが期待する入力値を1行で出力してください。

【ルール】
- 説明・思考・引用は一切出さず、入力値の1行だけを出力すること。
- 文字列値は引用符で囲まない（input()が文字列を返すため）。数値はそのまま。
- コードの文脈から妥当な具体値を選ぶ。履歴と矛盾しないこと。
- 不正入力でプログラムをクラッシュさせるような値は避ける。
- 日本語のみ。"""


# =====================================================
# ネイティブツール呼び出しの構造保証 (Grammar / Constrained Decoding)
# =====================================================
# llama-server を --jinja 付きで起動している場合、OpenAI互換の tools/tool_choice が
# ネイティブサポートされ、ツール呼び出し部分にのみ lazy grammar（ツール呼び出し開始
# トークン検出時に発動する制約付きデコーディング）が適用される。これにより <think> 等の
# 自由文は従来通り生成しつつ、tool_calls の JSON だけが構造的に保証される。
# （/props の chat_template_caps.supports_tool_calls / supports_tools で対応状況を確認可能）

#: True の場合、壊れたツール呼び出し（<tool_call> テキスト漏れ・パース失敗）や空応答を
#: 検知した再試行時に tool_choice="required" を明示送信し、上記のネイティブ grammar 保証を
#: 積極的に使う。False にすると常に tool_choice="auto" のみを使う従来動作に戻る。
#: サーバーが --jinja 非対応の場合でも tool_choice="required" は無害（無視されるか通常の
#: プロンプトベースの誘導として働くだけ）なので、フラグを True のままにしても壊れない。
NATIVE_TOOL_GRAMMAR: bool = True


# =====================================================
# 教訓ストア（経験メモリ・自己進化機構）
# =====================================================
# セッションをまたいで失敗経験を教訓として蓄積し、次回以降の関連タスクで
# engine.py の _build_dynamic_suffix() から自動注入する仕組み（src/lessons.py）。
# 収集はターン中の失敗信号（fast gate 検出・ガードレール発火・異常系 exit_reason）を
# state.failure_signals に貯めるだけで LLM を使わない。失敗信号が1件もないターンでは
# reflection（LLM 1回呼出）自体が発生しないため、通常時の追加コストはゼロ。

#: 教訓の収集・reflection・注入のすべてを有効にするか。False で本機能を丸ごと無効化する。
LESSONS_ENABLED: bool = True

#: LessonStore が保持する教訓の最大件数（超過時は hit_count が低く古いものから GC）。
LESSONS_MAX_ITEMS: int = 50

#: 動的suffixに注入する教訓の最大件数（LessonStore.recall の max_results）。
LESSONS_INJECT_MAX: int = 3

#: reflection 呼出の max_tokens。reasoning 系モデルは思考にトークンを消費するため、
#: 小さすぎると JSON 本文に到達する前に length 打ち切りとなり教訓が保存されない
#: （実測: gemma-4-26B で 512 では高確率で空振り、1600 で成功）。
LESSONS_REFLECT_MAX_TOKENS: int = 1024


# =====================================================
# 分岐点限定 lazy best-of-2（shadow_verify.py）
# =====================================================
# 不可逆・高コストな分岐点（破壊的ファイル編集の実行 / final answer の確定）でのみ、
# 候補を安価に検証し、ダメなときだけ最大1回だけ再サンプルする仕組み。
# prefix cache（実測 98.8% ヒット）が効くため、再サンプルのコストは decode のみで安い、
# という前提を活かす。通常パス（候補が最初からクリーン/高スコア）は追加 LLM 呼出ゼロ。

#: 破壊的編集（write_file/search_and_replace/replace_lines/append_to_file）を実行前に
#: シャドウ検証（py_compile+ruff、実ファイル無変更）し、失敗時のみ最大1回再サンプルするか。
BEST_OF_EDIT_ENABLED: bool = True

#: final answer 確定時、_answer_completeness_score が「閾値は超えたがギリギリ」の場合のみ
#: もう1候補を生成し、スコアの高い方を採用する（lazy best-of-2）を有効にするか。
BEST_OF_ANSWER_ENABLED: bool = True

#: 「ギリギリ合格」とみなすスコアマージン（_answer_completeness_score は 0-100 スケール、
#: 閾値は通常50/SYNTHESIZING中30）。score が [閾値, 閾値+MARGIN] の範囲なら再サンプル対象。
#: 15pt はシグナル1つ分（結論語句 or Markdown書式等、+15/+20 相当）の取りこぼしで
#: 閾値付近に落ちるケースを狙う目安。
BEST_OF_ANSWER_MARGIN: float = 15.0

#: 分岐点再サンプル時の温度オフセット（元の温度に加算）。多様性を上げて別解を狙う。
BEST_OF_RESAMPLE_TEMP_DELTA: float = 0.15
