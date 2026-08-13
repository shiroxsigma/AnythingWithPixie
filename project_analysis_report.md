# AnythingWithPixie プロジェクト仕様書

> 本書は `src/` の実装（2026-07 時点）に基づく仕様書である。概念や構想ではなく、コードから読み取れる動作仕様を正確に記述する。未実装の機能は明示的に「未実装」と区別する。

## 概要

AnythingWithPixie（愛称: AnythingPixie / Pixie）は、ローカル LLM をバックエンドとする **ReAct 型自律エージェントフレームワーク** である。ユーザーの自然言語指示を Function Calling ベースの思考ループで処理し、ファイル操作・コード解析・コマンド実行・サブエージェント委譲などを自律的に遂行する CLI アプリケーション。

| 項目 | 値 |
|---|---|
| バックエンド LLM | llama.cpp（GGUF 直接読込）／ LM Studio（OpenAI 互換 API） |
| 言語 / 実行 | Python 3.11+（CI・起動スクリプトは 3.13） |
| 規模 | `src/` 17 モジュール・約 11,713 行 |
| テスト | `tests/` 16 ファイル（ゴールデンテスト 27 ケース含む） |
| コア依存 | **ゼロ**（`dependencies = []`）。LM Studio 版は Python 標準ライブラリのみで動作 |
| オプション依存 | `llama-cpp-python`（GGUF）、`pillow`（画像）、`prompt_toolkit`（リッチ CLI 入力）、`pytest`/`ruff`（開発） |
| 配布形態 | ソース実行（`python src/main.py`）および PyInstaller exe 化対応 |

## 設計思想

本プロジェクトは、ローカル LLM を「受動的な回答者」から「能動的な実行者（エージェント）」へ変革することを目的とする。この方針は **推論（Reasoning）と実行（Acting）の密結合ループ** と **ローカルファースト（データ主権・低レイテンシ）**、そして **コア依存ゼロ（最小構成で動作する拡張可能なエコシステム）** の三本柱で支えられる。

設計上、システムは以下の4層に見立てられる（比喩であり、後述のモジュール群に対応する）。

- **脳（推論）** … `engine.py` / `engine_helpers.py` / `state.py` — ReAct ループと意思決定
- **手（実行）** … `tools.py` / `registry.py` / `code_tool.py` / `code_index.py` — 35 の組込ツール
- **目（知覚）** … `capture.py` / `llm_client.py`（マルチモーダル） — 画面・画像の認識
- **記憶（状態）** … `state.py`（`AgentStateBoard`）+ ホワイトボード（`CONTEXT_SUMMARY.md`） — 作業記憶と要約継承

> **注記**: 従来版に記載されていた「ベクトルデータベースによる RAG 長期記憶」「ナレッジグラフ」等は **本プロジェクトには未実装** である（§10 参照）。現状の「記憶」はセッション内の作業記憶（ステートボード JSON）と要約退避（ホワイトボード Markdown）のみで構成される。

---

## 1. システム構成とモジュール依存関係

### 1.1 ディレクトリ構成

```
AnythingWithPixie/
├── src/                      # アプリケーション本体（17 モジュール）
│   ├── main.py               # エントリポイント・CLI・AppContext
│   ├── cli_input.py          # CLI 入力（prompt_toolkit optional / input() フォールバック）
│   ├── engine.py             # 推論エンジン（ReAct ループ・ガードレール・動的 suffix）
│   ├── engine_helpers.py     # engine/subagent 共用の純粋関数群
│   ├── shadow_verify.py      # 分岐点限定 lazy best-of-2（編集シャドウ静的検証）
│   ├── state.py              # AgentStateBoard / ChatHistory / 静的プロンプト構築
│   ├── tools.py              # 28 のコアツール定義・実行
│   ├── registry.py           # ツールレジストリ（グローバル状態ホルダー）
│   ├── code_tool.py          # AST コード解析系 7 ツール
│   ├── code_index.py         # 純粋 AST インデックスライブラリ
│   ├── lessons.py            # 教訓ストア（経験メモリ・セッション横断）
│   ├── subagent.py           # サブエージェント・検証機能（toggle/observe-only）
│   ├── llm_client.py         # LLM バックエンド（GGUF / LM Studio）
│   ├── mcp_client.py         # MCP（Model Context Protocol）クライアント
│   ├── capture.py            # 画面キャプチャ・オーバーレイ UI
│   ├── config.py             # 設定定数（モジュールレベル定数）
│   └── paths.py              # パス解決（PyInstaller exe 対応）
├── tests/                    # テストスイート（16 ファイル + golden/）
├── .pixie_notes/             # ランタイム状態・キャッシュ・インデックス
├── debug/                    # ターン別デバッグログ（turn_NNN.md）
├── config.json               # LM Studio サーバー構成（実行時設定）
├── config.json.example       # 設定ひな形
├── pyproject.toml            # 依存・ruff/pytest 設定
├── Pipfile                   # （実質未使用、pyproject.toml に統合済み）
├── rg.exe                    # 同梱 ripgrep（Windows 用 grep_search）
├── StartPixie.bat            # 起動スクリプト（.venv の python で src/main.py）
└── .github/workflows/ci.yml  # CI（Windows・ruff + pytest）
```

### 1.2 モジュール依存グラフ（下層 → 上層）

```
paths.py （依存なし・最下層）
   ↑
config.py （paths.py のみ）
state.py （paths.py のみ・config 非依存）
registry.py （依存なし・最下層）
   ↑
code_index.py （標準ライブラリのみ）
llm_client.py （config のみ）
mcp_client.py （標準ライブラリのみ・独立）
capture.py （標準ライブラリのみ・独立）
cli_input.py （prompt_toolkit optional・標準ライブラリのみ・独立）
lessons.py （標準ライブラリのみ・純粋関数集）
   ↑
tools.py （registry, code_tool, config, paths を直接 import）
code_tool.py （paths, registry を直接 import。code_index は遅延 import）
engine_helpers.py （標準ライブラリのみ・純粋関数集）
shadow_verify.py （tools の純粋関数を再利用・config。実ファイル無変更で静的検証）
   ↑
engine.py （config, engine_helpers, state, tools, llm_client, paths, subagent, registry, lessons, shadow_verify）
subagent.py （registry, tools, config, engine_helpers, llm_client。engine には依存しない）
   ↑
main.py （config, llm_client, paths, cli_input を直接。engine/state/tools/capture/registry は遅延 import）
```

**循環 import の解消**: 本プロジェクトは「コアに近い共有純粋関数を `engine_helpers.py` に切り出す」「レジストリを `registry.py` に分離する」「LLM 追加呼出を `subagent.py` に集約する」という3つの抽出により、`engine ↔ subagent`、`tools ↔ code_tool` の循環 import を回避している。一部モジュールは関数内遅延 import で結合度を下げている。

### 1.3 モジュール一覧（一行サマリ）

| モジュール | 行数 | 役割 |
|---|---:|---|
| `engine.py` | 3,088 | 推論エンジン本体。ReAct ループ・ストリーミング・ガードレール・コンテキスト管理・完了判定・動的 suffix 構築 |
| `tools.py` | 2,039 | 28 の組込ツール定義・実行ディスパッチ・ファジーマッチ置換・プロセス管理 |
| `subagent.py` | 1,705 | サブエージェント・`/verify`・`/review`・`/review_loop`（toggle/observe-only 集約） |
| `main.py` | 1,145 | エントリポイント・CLI ループ・`AppContext`・半自動承認 UI・モデル選択 |
| `code_tool.py` | 575 | AST コード解析系 7 ツール（アウトライン・デッドコード・マップ等） |
| `code_index.py` | 551 | 純粋 AST インデックス（関数/クラス/コールグラフ抽出・増分キャッシュ） |
| `config.py` | 493 | 動作パラメータ・ツール分類・サブエージェント予算・教訓/best-of-2/run_python 定数 |
| `state.py` | 413 | `AgentStateBoard`・`ChatHistory`・`AgentState`・静的システムプロンプト構築 |
| `llm_client.py` | 413 | `LlamaCppBackend`／`LMStudioBackend`・マルチモーダル・`reasoning_content`・`last_timings` |
| `engine_helpers.py` | 265 | engine/subagent 共用の純粋関数（トークン推定・ツールパース・反復検知等） |
| `lessons.py` | 232 | 教訓ストア（`LessonStore`・Jaccard 重複統合・recall・GC・セッション横断永続化） |
| `shadow_verify.py` | 182 | 分岐点限定 lazy best-of-2（`shadow_apply`・`shadow_gate`=py_compile+ruff 静的検証） |
| `cli_input.py` | 165 | CLI 入力（`prompt_toolkit` optional・`input()`+`"""` フォールバック・`SLASH_COMMANDS` 補完） |
| `mcp_client.py` | 162 | `LightweightMCPClient`（JSON-RPC over stdio で MCP サーバーと通信） |
| `capture.py` | 136 | 画面領域選択・赤枠オーバーレイ（別プロセス実行） |
| `paths.py` | 83 | dev/exe 自動判別のパス解決（`get_data_path`／`get_bundled_path`） |
| `registry.py` | 66 | `TOOL_REGISTRY`・`register_tool` デコレータ・状態注入のグローバルホルダー |

---

## 2. 推論エンジン（engine.py / engine_helpers.py / state.py）

推論エンジンは本プロジェクトの中核であり、ReAct（Plan→Action→Observe）ループを駆動し、ローカル LLM 特有の問題（不完全思考タグ・反復ループ・ネイティブツール呼出の非構造化・コンテキスト窮迫）に対する多数の安全装置を実装する。

### 2.1 ReAct ループ（Function Calling 方式）

本プロジェクトは **Function Calling 方式** を採用しており、`Thought:`/`Action:`/`Observation:` のようなテキストプロトコルは用いない。思考は `<think>` ブロック内、アクションは `tool_calls` 構造体、観察は `role="tool"` メッセージとして扱う。ただし `supports_tool_role=False` のモデル（LM Studio 等）では、ツール結果を `role:"user"` + `[ツール結果]` プレフィックスに変換して吸収する。

メインループ `run_graph(context, state, ...)` は、`while state.tool_call_count < state.max_tool_calls`（デフォルト上限 100）で以下の3ノードを回す。

| ノード | シグネチャ（要約） | 役割 |
|---|---|---|
| `node_plan` | `(context, state, *, show_thinking, max_tokens, output_fn, system_msg_builder, temp_delta=0.0) -> (content\|None, tool_calls\|None)` | LLM に次アクションを問う Plan。ストリーミング受信・思考フィルタ・反復検知・ガードレール注入・動的 suffix 付与を統合 |
| `node_action` | `(context, state, tool_call, *, output_fn) -> str` | 指定ツールを安全実行する Action。`execute_tool` のラッパ |
| `node_observe` | `(state, tool_name, tool_result, *, output_fn) -> "PLANNING"\|"DONE"` | ツール結果を評価・状態更新する Observe。エラー記録・`max_tool_calls` チェック |

**終了条件**: ツール呼出回数が `max_tool_calls`（100）に到達／全体反復が安全上限（`max_tool_calls + 10`）に到達／ツール呼出なし＋完全性スコア ≥ 閾値で `final_answer`／空応答／ユーザー却下（`interactive_fn`）／継続 8 回到達。

**継続処理**: `finish_reason == "length"` かつ `tool_calls` なしのとき、`state.phase = "NEEDS_CONTINUATION"` とし、末尾にヒントを付与してもう一度 Plan へ戻る（最大 8 回）。

`execute_tool` はツール実行のディスパッチャであり、`view_image`／`analyze_file`／`write_sections`／`delegate_research` をインターセプトし、ファイル編集後には ruff チェック →（`/review` 時）読取専用レビュー →（`/verify` 時）実行ベース検証を付加する。

### 2.2 段階的思考深化（shallow / deep）

`_resolve_thinking_mode(state, user_text, force_deep)` が、タスクの複雑さに応じて思考深度を `shallow`／`deep` で判定する。

- **ヒステリシス**: 一度 `deep` に入ったら `shallow` に戻さない（`state._was_deep`）。単純質問（`is_simple_question`）は強制的に `shallow`。
- **deep モード**: `max_tokens` を倍増、温度下限 0.5（創発維持）、`DEEP_THINK_BUDGET_SEC = 90s` の思考タイムアウト。深化プロンプト「複数仮説を立てて深く推論」は動的 suffix の `deep_hint` として注入（**system プロンプト本体は thinking_mode によらず固定**・prefix cache 保護）。
- **動的温度**: `tool_call_count > 15` で温度を段階低下（0.05/回）。`shallow` の下限は 0.3。

### 2.3 ガードレール群

ローカル LLM の暴走を防ぐため、多層のガードレールを実装する。発火後は `guardrail_cooldown`（1〜2）で連鎖を防止する。

| 機能 | 実装 | 概要 |
|---|---|---|
| ストリーミング反復検知 | `detect_repetitive_content` | 200文字毎に同一行／迷い語／コマンド／セパレータの反復を検知。`deep` は閾値緩和 |
| クロスターン類似度 | `_detect_content_similarity` | Jaccard 類似度（閾値 0.65〜0.75）で重複出力を検知 |
| ループ検知 | `check_loop_detected` | 同一アクションの連続2回を検知。警告3回で強制終了 |
| 行動予告検知 | `_looks_like_action_promise` | 「次に〜します」型の宣言のみで `tool_call` がない場合、強制実行へ誘導 |
| 短回答検知 | `_answer_completeness_score` | 構造語・根拠・Markdown・長さを評価。閾値未満の短回答を再実行へ |
| 不完全ツール呼出検知 | `parse_native_tool_calls` | `<tool_call>`／特殊トークン残留を検知し OpenAI 形式に補完・再試行 |
| 未閉じ思考保護 | `_has_unclosed_thinking` | 未閉じ `<think>` を持つ content を完了判定から除外（生思考の混入防止） |

### 2.4 思考管理（StreamFilter と Feature A）

- **`StreamFilter` クラス**: ストリーミング受信中に `<think>`／`<|channel>`（Gemma）／絵文字形式の思考ブロックを除去するフィルタ。チャンク境界の部分マッチに対応し、未閉じ検知時は `flush()` で「思考プロセスが閉じられなかったため内容を表示します」マーカーを付与する。
- **Feature A（`_add_assistant_with_think`）**: 履歴に追加する assistant メッセージのうち **直近1件のみ `<think>` を保持** し、他は剥がす。これによりターン間の推論積み上げを可能にしつつ、履歴の肥大化とシステムプロンプト先頭の不安定化を防ぐ。不変量は「`<think>` を持つ assistant は直近1件のみ」。
- **`reasoning_content` 統合**: DeepSeek/R1/Gemma 系の専用思考フィールド `delta.reasoning_content` を検出し、`<think>` と同じ思考パイプライン（`/think ON` 表示・`DEEP_THINK_BUDGET_SEC` タイムアウト・`thinking_notes` 抽出）に合流させる。**`chat_history` には一切含めない**（prefix cache 汚染なし）。prefill 完了判定（`_prefill_done`）も content／reasoning_content いずれかの最初のチャンクで確定し、思考時間が Prefill 経過として誤表示されるバグを防ぐ。

### 2.5 コンテキスト・ホワイトボード管理

**ホワイトボード**（`CONTEXT_SUMMARY.md`）は、長時間対話の文脈を要約して継承する物理ファイル。`<!-- DETAIL_SECTION -->` で注入用上部と詳細部を分離し、毎ターン上部 1500文字をシステムプロンプトへ注入する。ハードトリムで切り捨てられたメッセージから、LLM（max_tokens=1500, temp=0.2）で要約を再生成する（`_update_whiteboard`）。

**2段階トリム**（`check_and_trim_context`）:
- **Phase 1（ソフト・70%超）**: `mask_old_observations` で古いツール結果を要約に置換。`read_file` は行番号ヒントを残す。
- **Phase 2（ハード・上限超）**: 古いメッセージを `pop`（assistant+tool ペアで削除）し、ホワイトボードへ退避。

**能動圧縮**: `read_file` 実行時に def/class アウトラインを決定的に抽出し、ステートボードの `file_summaries` に登録（`_compress_tool_result`／`_register_read_outline`）。生テキストがマスクされても構造知識は残る。

**動的ツール結果キャップ**: `_dynamic_tool_cap` がコンテキスト使用率から1件あたりの文字上限を逆算（6000〜16000）し、`registry.set_tool_result_max_chars` で設定。

**チェックポイント**: `check_context_checkpoint` が使用率 80% 超でステート自動保存＋`/reset` 推奨を通知する。

### 2.6 完了判定（完全性スコア）

ツール呼出のないターンで完了判定を行う。優先順位:

1. 継続処理中 → 継続へ
2. 空応答 → `last_substantive_content`（≥50字）があればフォールバック、なければ `empty_response` 終了
3. 単純質問完了許可（`_is_simple_direct_answer_sufficient`）→ cwd/list/read/analyze 系で対応ツール既実行なら即 `final_answer`
4. 反復／類似検知 → 強制ツール実行（クールダウン中・SYNTHESIZING 中は免除）
5. 不完全ツール呼出 → フォーマットエラーとして再試行
6. 行動予告 → 強制実行
7. 短回答 → `_answer_completeness_score` < 閾値（通常 50／SYNTHESIZING 時 30）で再実行
8. 上記いずれにも該当しなければ `final_answer`（`accumulated_content` があれば結合復元）

### 2.7 ツール並列 / 直列実行

`classify_tools` がツール呼出を読取専用（並列）と破壊的（直列）に分類する。

- **並列（`execute_parallel`）**: 読取専用ツールを `ThreadPoolExecutor`（最大 `MAX_PARALLEL_TOOLS = 5`）で実行。元の順序を保持して結果を返す。
- **直列（`execute_serial`）**: 破壊的ツールを1件ずつ実行。
- `analyze_file` は対象サイズ > 8KB のとき動的に直列へ回す（リソース回避）。

### 2.8 state.py（記憶層）

| 構造 | 役割 |
|---|---|
| `AgentStateBoard` | 永続化される「状態」。`goal`／`current_step`／`next_to_do`／`found_knowledge`／`completed_tasks`／`active_errors`／`file_summaries`／`project_structure`／`waiting_for_async` 等を保持。`.pixie_notes/state_board.json` に JSON で永続化（各更新メソッドが即 `_save()`）。設計ルールは「上書きのみ（追記禁止）」「生データ禁止（結論のみ）」。GC 上限: `completed_tasks`=5, `found_knowledge`=15, `active_errors`=5 |
| `ChatHistory` | スライディングウィンドウ（`max_messages=20`）。`trim()` は assistant+tool ペアが分割されないよう安全境界を調整 |
| `AgentState` | ReAct ループ用状態（`phase`, `tool_call_count`, `max_tool_calls=100`, `loop_warn_count`, `continuation_count`, `accumulated_content`, `guardrail_cooldown`, `thinking_notes`, `_was_deep`） |
| `build_system_prompt` | **静的**システムプロンプトの構築。`base_prompt` のみを返す（thinking_mode によらず固定・prefix cache 保護）。動的コンテキスト（ステートボード注入文 max 2500字／ホワイトボード要約 max 1500字／JIT ヒント／budget_hint／deep_hint／定期リマインダー／教訓）は `engine.py` の `_build_dynamic_suffix` で1つにまとめ、`_apply_dynamic_suffix` で **LLM 送信直前の一時コピー末尾** にのみ付与（`chat_history` には書き込まない＝cache 汚染なし）。Lost in the Middle 対策の recency bias は末尾配置でも同等に効く |

### 2.9 engine_helpers.py（共用純粋関数集）

標準ライブラリのみに依存する純粋関数群で、`engine.py` と `subagent.py` の両方から参照される。主なもの: `estimate_tokens`（トークン推定）、`is_simple_question`、`parse_native_tool_calls`／`accumulate_tool_calls`／`safe_parse_args`（ツール呼出パース）、`detect_repetitive_content`（反復検知）、`strip_all_thinking`、`default_output_fn`、定数 `FILE_EDIT_TOOLS`。

---

## 3. ツール群（tools.py / registry.py / code_tool.py / code_index.py）

エージェントが Function Calling で実行する **全35の組込ツール** の定義・登録・ディスパッチを担う。

### 3.1 ツールレジストリ（registry.py）

依存を持たない最下層のグローバル状態ホルダー（クラスなし、モジュールレベル変数と関数のみ）。

- `TOOL_REGISTRY: dict` — ツール名 → `{func, description, schema, category, prompt_desc}` の辞書
- `register_tool(name, description, schema, category="core", prompt_desc=None)` — **デコレータファクトリ**。装飾関数を登録し、関数本体はそのまま返す。`category` は `"core"`（常時スキーマ公開）または `"extended"`（`inspect_tool` で詳細取得）
- `set_state_board(sb)` — engine/main.py から `StateBoard` を注入する口
- `set_tool_result_max_chars(n)` — engine がコンテキスト使用率から逆算した文字上限を設定（並列実行前に1回だけ設定、実行中は読取専用でスレッドセーフ）

> ファジーマッチング機能は registry.py ではなく `tools.py` の `_fuzzy_apply` に実装されている（§3.3）。

### 3.2 ツール一覧（35個）

#### ファイル読み書き・操作（9）

| ツール | 引数 | 説明 |
|---|---|---|
| `get_cwd` | なし | 現在の作業ディレクトリ絶対パス |
| `get_file_dir` | path | 指定ファイルの親ディレクトリ絶対パス |
| `list_directory` | path? | ディレクトリ内一覧（名前+サイズ/DIR） |
| `read_file` | path, start_line?, end_line? | ファイル読込。`.py` で500行超は構造+先頭50行のみ |
| `write_file` | path, content | ファイル書込（上書き）。親ディレクトリ自動作成 |
| `append_to_file` | path, content | ファイル末尾追記（新規作成も可） |
| `move_file` | src, dst | ファイル移動・リネーム |
| `make_directory` | path | ディレクトリ作成（親階層含む） |
| `delete_file` | path | ファイル削除（ディレクトリ不可） |

#### コード編集（3）

| ツール | 引数 | 説明 |
|---|---|---|
| `replace_lines` | path, start_line, end_line, new_content | 1オリジン行範囲を新内容で置換 |
| `search_and_replace` | path, search_block, replace_block | 文字列ブロック一致で安全置換。**3層ファジーマッチ**（§3.3） |
| `write_sections` | path, sections[{heading,instruction}], context? | セクション構造指定で長文生成。本体はスタブ、engine がインターセプト |

#### 検索・差分（2）

| ツール | 引数 | 説明 |
|---|---|---|
| `grep_search` | pattern, path?, is_regex?, file_extensions?, context_lines? | ripgrep（Windows は同梱 rg.exe、非 Win は PATH の rg → grep フォールバック）で高速検索。出力 10000字上限 |
| `diff_files` | old_path, new_path, old_label?, new_label? | 2ファイル間の unified diff を ANSI カラー表示（extended） |

#### コード解析・調査（code_tool.py、7）

| ツール | 引数 | 説明 |
|---|---|---|
| `get_code_outline` | path | Python(AST)／JS-TS(正規)からクラス・関数シグネチャ抽出 |
| `research_code_paths` | keyword | キーワードの定義箇所(点)と使用箇所(線)を正規表現で調査（extended） |
| `gather_project_info` | path, max_files?, extensions? | プロジェクト構造+主要ファイル列挙、キャッシュ初期化（extended） |
| `map_codebase` | path?, force_refresh? | AST 全体インデックス構築・依存・デッドコード候補数（extended） |
| `detect_dead_code` | path?, force_refresh? | 到達性解析でデッドコード候補をファイル別一覧（extended） |
| `read_symbol` | path, symbol, context? | 指定シンボルのソースを行範囲で読込（extended） |
| `get_file_stats` | path?, extensions? | ディレクトリ内ファイルの行数・サイズ一覧 |

#### シェル・プロセス管理（5）

| ツール | 引数 | 説明 |
|---|---|---|
| `run_command` | command | OS シェル（Win=PowerShell／非 Win=bash）で同期実行。**30秒タイムアウト**。出力 4000字で前後分割省略 |
| `run_async_test` | command, log_file? | 別コンソールで非同期実行し PID+ログパスを返す（非ブロッキング） |
| `poll_process` | pid, log_file | プロセス生存確認＋ログ末尾30行 |
| `kill_process` | pid | プロセス強制終了（Win=taskkill /F／非 Win=SIGKILL） |
| `update_core_memory` | content | Core Memory（`CORE_MEMORY.md`）全体上書き |

#### サンドボックス実行（1）

| ツール | 引数 | 説明 |
|---|---|---|
| `run_python` | code, stdin_seed?, max_inputs?, timeout? | Python コードを一時ディレクトリで `python -u` サンドボックス実行。`input()` のプロンプトを検出すると LLM が自動で入力値を生成して stdin に送り継続（インタラクティブ自動入力）。env サニタイズ・総タイムアウト・`max_inputs` 上限付き。本体はスタブ、engine がインターセプト（extended） |

#### エージェント状態・委譲（4）

| ツール | 引数 | 説明 |
|---|---|---|
| `set_goal` | goal | ユーザー目標をステートボードに設定（extended） |
| `update_state` | current_step?, next_to_do?, found_knowledge?, errors? | エージェント状態を一括更新（全引数省略可）。状態記録専用で実行の代わりにならない（extended） |
| `query_whiteboard` | query | ステートボードから情報検索（extended） |
| `delegate_research` | question, file_hints?, focus?, max_steps? | 独立サブエージェントに調査委譲し結論のみ取得。本体はスタブ、engine がインターセプト |

#### ナビゲーション・画像・メタ（4）

| ツール | 引数 | 説明 |
|---|---|---|
| `view_tree` | path, max_depth? | ディレクトリツリー（`.git` 等は除外、4000字上限）（extended） |
| `inspect_tool` | tool_name | 拡張ツールの詳細マニュアル・スキーマ取得 |
| `analyze_file` | path, analysis_prompt? | 長大ファイルを裏で別 LLM に要約させる（スタブ、engine がインターセプト） |
| `view_image` | path, analysis_prompt? | 画像解析（スタブ。テキスト専用モデルでは非サポートを返す）（extended） |

### 3.3 ファジーマッチ置換（search_and_replace）

LLM によるコード編集で「元の文字を一文字違える」「空白を誤る」等で置換が失敗するリスクを最小化するため、`search_and_replace`（tools.py）と補助関数 `_fuzzy_apply`／`_build_search_hint` で **3層のレイヤードマッチ** を実装する。

1. **完全一致**: `content.count(search_block)` で検索。1件なら置換成功、複数件なら「前後行を含めて一意化せよ」エラー。
2. **L2 空白正規化ウィンドウ一致**: 各行を `strip()` した上で、`search_block` 行数幅の窓が正規化状態で完全一致する位置が **一意（1箇所のみ）** なら採用。
3. **L3 difflib ファジー**: 先頭行をアンカー（`SequenceMatcher.ratio() >= 0.70` で候補絞り込み）に、窓全体の `ratio()` を算出。最高スコアが `FUZZY_MATCH_THRESHOLD = 0.85` 以上 **かつ** 2位候補と 0.05 以上離れている、または 2位が閾値未満なら採用。

> レーベンシュタイン距離そのものは実装せず、類似度には Python 標準 `difflib.SequenceMatcher`（最長共通部分列ベース）を使用する。複数候補が同点、または2位も閾値ギリギリの場合は安全のため失敗し、`_build_search_hint` が実在行を最大3件ヒントとして返し、LLM に正確なコピーを促す。

### 3.4 JIT ツールスコアリング

`score_tools(user_input, top_n=5) -> list[str]`（tools.py）が、ユーザー入力に対してツール名直接マッチ（+5.0）・トークンオーバーラップ（+1.0/個）・部分文字列（+0.5/個、日本語 bi-gram 考慮）でスコアリングし、`ALWAYS_RECOMMEND`（10ツール）を常に含めて上位を推薦する。`/code` モードでは固定の `CODE_TOOL_SET`（20ツール）でスコアリングをバイパスする。

### 3.5 コードインテリジェンス（code_index.py / code_tool.py）

`code_index.py` は Python 標準 `ast` **のみ** を用いる純粋関数ライブラリ（第三者依存ゼロ）で、コードベースを AST 解析してインデックスを構築する。

**抽出対象**: 関数・メソッド・クラス定義（位置・行範囲・装飾子・修飾名 `qualname`）、関数間コールグラフ辺（`qualname → callee 名リスト`）、import 文、外部依存モジュール、`__main__` ガードの有無とその中の呼出。対応言語は `code_index`（インデックス・デッドコード検出）が **Python 専用**、`get_code_outline` は JS/TS/JSX/TSX も正規表現で補助対応する。

**増分キャッシュ**: 差分判定は **MD5 ハッシュのみ**（mtime 不使用）。`.pixie_notes/code_index.json` に保存（`.tmp` 経由のアトミック置換）。`schema_version`（=1）／`root` 不一致でキャッシュを無効化する。

**デッドコード検出**（`find_dead_symbols`）: BFS 到達性解析。シードは「`@register_tool` 装飾関数 ＋ `__main__` ガード内呼出 ＋ `main` 関数」。import API を到達に追加し偽陽性を緩和。動的呼出（getattr／文字列 dispatch／コールバック）は AST で追跡不可能なため、文字列出現 ≥2 で「低信頼」に格下げする **候補** 扱い（確定ではない）。

> **フォールトトレランス**: `code_index` は `code_tool`／`tools` から遅延 import され、1ファイルの構文エラーや本モジュールの欠落時もアプリ全体を止めない（`parse_error` に記録し空レコードで継続）。デコードは UTF-8 → cp932 フォールバック。

---

## 4. サブエージェント・検証機能（subagent.py）

`subagent.py` は、メインの ReAct ループとは別に **LLM を追加呼出しする、コストがかかる機能** を1モジュールに集約したものである。モジュール docstring に **「toggle/observe-only 方針」** が明記され、コストがかかるコードを局所化しつつ `engine ↔ subagent` の循環 import を解消する。

### 4.1 設計方針（toggle / observe-only）

- **toggle（明示的な有効化）**: 全機能とも `context.review_mode`／`context.verify_mode` 等のフラグでガードされ、**デフォルトは無効**（`getattr(context, ..., False)`）。ユーザーが明示的にトグルした時のみ発火する。
- **observe-only（観察のみ・編集結果を絶対に壊さない）**: レビュー系機能は判定文字列を返すだけで、メインの `chat_history` を汚染せず、編集結果・最終回答を書き換えない。例外時は `""` を返し、既存結果を維持する（`try/except` 全面ラップ）。
- **コスト制御の具体値**（ローカル LLM のコスト・時間・無限ループ防止）: `REVIEW_DESIGN_MIN_CHARS=500`（設計レビュー自動発火の最小文字数）、`VERIFY_MAX_ROUNDS=3`、`REVIEW_LOOP_MAX_ROUNDS=5`、`DELEGATE_MAX_STEPS=6`。`DELEGATE_SUBAGENT_TOOLS` からは `analyze_file` を除外（ネスト LLM 呼出の遅延・予算消費を避けるため）。

### 4.2 サブエージェント（run_agent_subquery）

中核関数 `run_agent_subquery(llm, *, question, file_hints=None, focus=None, max_steps=None, supports_tool_role=False, mode="research"|"review", ...)` は、**独立したコンテキスト** で動く軽量 ReAct ループのサブエージェントである。

- **独立性**: メインの `state.chat_history` には一切触れず、関数ローカルの `messages` のみで完結。docstring に「Claude Code の Task ツールに相当する『コンテキストを汚さない委譲』」と明記。
- **制約**: 読取専用ツールのみ許可（`DELEGATE_SUBAGENT_TOOLS` 外は拒否）。ステップ数（調査=6／レビュー=3）・トークン・予算（調査 120s／レビュー 30s）すべて控えめ。GGUF のネイティブ `<tool_call>` テキスト呼出のフォールバック付き。
- **キャッシュ最適化**: ツール順序を固定（`sorted`）し、LM Studio のプレフィックスキャッシュが毎回ヒットするよう配慮。
- **終了条件**: ツール呼出なしで content が得られれば即 return。反復検知・空ストール2回連続・予算超過で break。上限到達時はツール無効の最終サマライズターンで結論を生成。

### 4.3 委譲調査（delegate_research）

`_execute_delegate_research` が `delegate_research` ツールをインターセプトし、`run_agent_subquery(mode="research")` で独立サブエージェントを起動する。

- **サーバーラウンドロビン**: `delegate_server`（第2サーバー）設定時、`_delegate_server_counter`（`threading.Lock` 保護）で偶数→サブ、奇数→メインに分散し、読取専用ツールの並列実行時の負荷を分散する。
- 結果は `[委譲調査の結論]` ヘッダ付きで observation に返し、メインエージェントが生ツール出力と区別可能。

### 4.4 /review（編集レビュー・設計レビュー）

「設計者＝メインエージェント」「レビューア＝読取専用サブエージェント（`mode="review"`）」の構造。明示的な Designer クラスは存在しない。

- **編集レビュー（`_run_edit_review`）**: 破壊的ファイル編集の **直後** に起動（observe-only・編集は既に実行済み）。自明な編集（`append_to_file` <80字、`replace_lines`/`search_and_replace` <20字）はスキップし LLM 呼出を回避。reviewer に「実際に `read_file` で確かめよ」と指示し、判定（400字で切り詰め）を `[レビュー結果]` ブロックとして返す。
- **設計レビュー（`_run_design_review`）**: `final_answer` が設計／提案らしい場合のみ自動発火（`_is_design_proposal`: 500字以上・単純質問除外・design_markers 語彙 or code_mode）。システムプロンプトで「積極的にリスク・反論・代替案を探せ。整合しているだけでは『問題なし』にするな」と **批判的姿勢** を強制する。

### 4.5 /review_loop（往復改善）

`run_review_loop` が、直前の回答を main（設計者: `_one_shot_revise`）↔ reviewer で最大 `REVIEW_LOOP_MAX_ROUNDS=5` 往復（デフォルト3）させて改善する。`_verdict_is_clean`（`"問題なし" in verdict`）で早期収束する。`/review` トグルとは独立の明示起動。

### 4.6 /verify（実行ベース検証＋自動再編集）

`run_verify_fix_loop` が `.py` ファイルのみを対象に、実行ベースの検証と自動修正を行う（`verify_mode` フラグで有効化）。

1. **Python 選択**: `_resolve_verify_python` が編集対象プロジェクトの `.venv` を優先、なければ `sys.executable`。
2. **段階的検証ゲート**（`_run_execution_verification`・short-circuit）:
   - ゲート1: `py_compile`（常時・安全・構文・timeout 10s）
   - ゲート2: import 解決（AST＋`find_spec`・副作用なし・timeout 15s・デフォルト ON）
   - ゲート3: ruff（`.venv` の Python 使用・timeout 10s・デフォルト ON）
   - ゲート4: pytest（副作用あり・timeout 60s・**デフォルト OFF** `VERIFY_TEST_GATE=False`）
   - 最初の失敗ゲートのエラーを返して残りはスキップ。全通過で成功。
3. **エラー検出時**: `_read_file_for_verify` で現在内容を読み → `_generate_fix_edit` がエラー解消編集を JSON 生成（デフォルト `search_and_replace`、全体再編のみ `write_file`）→ `_apply_fix_edit` が **`.pixie_notes/backups/*.bak` バックアップ付き** で適用（`execute_builtin_tool` 直接呼びで verify フックの再帰を防止）→ 再検証。
4. **終了条件**: 検証クリア／最大 `VERIFY_MAX_ROUNDS=3` 往復／wall-clock `VERIFY_BUDGET_SEC=120` 超過／修正生成・適用失敗／例外発生。未収束時は最終エラースニペット（600字）を返す。

> `/verify` の根拠は **機械的な実行結果** であり、`/review`（LLM の主観判定・observe-only）とは明確に区別される。

### 4.7 分岐点限定 lazy best-of-2（shadow_verify.py）

不可逆・高コストな分岐点（破壊的ファイル編集／final answer）でのみ候補を検証し、ダメなときだけ最大1回再サンプルする。**通常パス（候補が最初からクリーン／高スコア）は追加 LLM 呼出ゼロ（lazy）**。prefix cache が高ヒット率（98% 前後）に乗っているため、再サンプルは decode のみで済む。

- **編集シャドウ検証**（`_verify_and_maybe_resample_edits`）: `SHADOW_EDIT_TOOLS`（`write_file`／`search_and_replace`／`replace_lines`／`append_to_file`）の呼出時、`shadow_apply` が **実ファイルを一切変更せず** に「適用後内容」を計算し（`tools.py` から抽出した純粋関数を再利用）、`shadow_gate` が一時ファイルで `py_compile`＋`ruff check --select E,F` を走らせる（import 解決は一時パスでの誤検知を避け **意図的に除外**）。クリーンならゼロコストで return。失敗時のみ一時フィードバックメッセージを履歴に載せ（`_pop_message_by_identity` で事後除去・永続化汚染防止）、`temp_delta=+0.15`・`tool_choice="required"` で1回だけ再サンプル。
- **final answer 再サンプル**（`_maybe_resample_final_answer`）: `_answer_completeness_score` が `[閾値, 閾値+MARGIN]` のギリギリ帯に落ちた場合のみ、もう1本生成してスコア比較し高い方を採用。明らかにクリーン／明らかに不十分な場合は再サンプルしない。
- **`/verify` との区別**: `/verify`（§4.6）は **実ファイルに書いた後** の実行ベース検証（ゲートに import・pytest 含む）。本機能は **書く前** の静的検証（import なし）であり、実ファイルを触らない。

---

## 5. LLM・通信・知覚層（llm_client.py / mcp_client.py / capture.py）

### 5.1 LLM クライアント（llm_client.py）

2種類のバックエンドを同一インターフェース（`create_chat_completion(messages, *, stream=True, tools=None, tool_choice="auto")`）で抽象化する。

| バックエンド | 通信方式 | 特徴 |
|---|---|---|
| `LMStudioBackend` | LM Studio の OpenAI 互換 API（`urllib` で HTTP・外部ライブラリ非依存） | SSE ストリーミング（`data:` 行 → `[DONE]`）。`/v1/models` から `n_ctx` を動的取得（フォールバック 32768）。HTTP 500 時に2秒後に1回リトライ。応答チャンクの `timings`（`cache_n`/`prompt_n`）を `last_timings` に保存し engine が cache ヒット率を常時表示 |
| `LlamaCppBackend` | `llama-cpp-python`（GGUF 直接読込） | `n_threads=4`、`n_gpu_layers`（GPU 有効時 -1）。`tokenize()` による正確なトークンカウントが可能 |

- **マルチモーダル**: `initialize_backend` が mmproj_path の有無で Vision モデルを判定。Qwen3.5-VL（`Qwen35ChatHandler`）と Qwen3-VL（`Qwen3VLChatHandler`）をサポート（モデルパスに `qwen3.5` が含まれるかで切替）。
- **チャットテンプレート**: `_apply_chat_template_from_metadata` が GGUF メタデータの `tokenizer.chat_template` から Jinja2ChatFormatter を構築し適用。
- **`SuppressStderr`**: llama.cpp の C レベルログコールバックを上書きして stderr ノイズを抑制。

### 5.2 MCP クライアント（mcp_client.py）

`LightweightMCPClient` が、MCP（Model Context Protocol）サーバーを `subprocess.Popen` でサブプロセス起動し、**JSON-RPC 2.0 over stdio**（プロトコル `2024-11-05`）で通信する。

- **通信スレッド**: stdout 監視（`_read_loop`）と stderr 監視（`_read_stderr_loop`・デッドロック回避＋ログ収集）の2スレッドを daemon で起動。`response_queues`（dict[int, Queue]）でリクエストID別に応答を振り分け。
- **主要メソッド**: `initialize()`（ハンドシェイク・capabilities 申告）、`get_tool_list()`（`tools/list`）、`call_tool(name, arguments, timeout=30)`（`tools/call`・content 配列の text を結合して返す）。
- **終了処理**（`stop()`）: terminate → wait(5s) → kill の段階的エスカレーション。タイムアウトはデフォルト30秒。

### 5.3 画面キャプチャ・オーバーレイ（capture.py）

オプションモジュール（Feature Registry パターンで動的ロード、`--no-capture` で無効化）。

- **`select_screen_area()`**: 半透明（alpha=0.3）フルスクリーンウィンドウでドラッグによる領域選択。Esc でキャンセル、10px 未満はキャンセル扱い。
- **`show_transparent_overlay(bbox, stop_event)`**: 指定 bbox に赤枠（中身透過・クリック透過、Windows の `-transparentcolor`）を表示。`stop_event` が set されるまで 200ms 毎にポーリング。
- **マルチプロセス分離**: `multiprocessing.Process` でオーバーレイ表示を **別プロセス** で実行（`main.py` の `multiprocessing.freeze_support()` と連動）。`OverlayManager` がプロセスとイベントを管理。
- **`get_inner_bbox`**: 赤枠自体がキャプチャに写り込まないよう内側に狭めた座標を返す。

---

## 6. CLI・設定・基盤（main.py / config.py / paths.py）

### 6.1 起動フロー

`main()` → `parse_args()`（`--no-capture`／`--no-gui` フラグ）→ `setup_application(args)` → `run_cli_chat(context)`。

`setup_application` の初期化順序: (1) Windows コンソール設定（UTF-8・ANSI 色）→ (2) `select_model(MODEL_DIR)` → (3) `initialize_backend()` → (4) delegate_server 読込（任意）→ (5) capture モジュール動的ロード → (6) コマンド一覧表示。終了時は `finally` で `overlay_manager.stop()` と `llm.close()` を実行。

> **注記**: `--no-gui` フラグは存在するが、実質的に CLI ループ（`run_cli_chat`）のみが提供される。GUI 実行パスは現在のコードには見当たらない（将来用のスタブ的扱い）。

### 6.2 CLI コマンド一覧

| コマンド | 引数 | 動作 |
|---|---|---|
| `quit`／`exit` | なし | セッション終了 |
| `/think` | なし | `<think>` 思考プロセス表示の ON/OFF トグル |
| `/deep` | なし | 強制深度思考モードのトグル（段階判定をスキップし常に deep） |
| `/review` | なし | 編集前レビューサブエージェントのトグル（observe-only・読取専用で編集を批判） |
| `/verify` | なし | 実行ベース検証＋自動再編集ループのトグル（py_compile/ruff/pytest・`.venv` 優先） |
| `/review_loop` | `[N]`（省略可） | 直前の回答を main↔review で N 往復させて改善（明示起動・デフォルト3） |
| `/step` | なし | 半自動（ステップ実行）／フル自動モードのトグル |
| `/mem` | なし | メモリモード（会話履歴保持）のトグル。OFF 時は履歴クリア |
| `/debug` | `[full]` | デバッグダンプのトグル。`.pixie_notes/debug/turn_NNN.md` に出力 |
| `/reset` | なし | チャット履歴クリア＋コンテキストリセット |
| `/context` | なし | コンテキスト使用量の可視化（トークン数・ロール別内訳・プログレスバー） |
| `/recap` | なし | リアルタイム画面キャプチャの領域選択（capture モジュール利用時のみ） |
| `/code-init` | `[path]`（省略時 cwd） | プロジェクト構造（view_tree + outline）を記憶し、Code mode ON |
| `/code` | `[<target>\|off]` | コード専門モードのトグル。引数ありで target 指定、`off` で解除 |
| `/trace` | `<keyword>` | 指定キーワードの定義点・使用点を `research_code_paths` で調査 |
| `/api` | なし | LM Studio サーバー切り替えメニュー表示 |
| `/delegate-api` | `[off]` | 委譲サブエージェント用サーバー設定。`off` でメインサーバーに復帰 |
| `/poll_async` | `[PID LOG_FILE]` | 非同期プロセスのポーリング。引数省略時はステートボードから取得 |

**その他の入力動作**: `"""` で始まる入力は閉じ `"""` までマルチライン収集。`PLANNING_WAIT_OK` フェーズ中に `ok` 入力で実行フェーズへ移行（`PLANNING.md` を読込）。非同期タスク待機中は `async_timeout` 秒でタイムアウトし自動 `/poll_async`。

### 6.3 半自動承認 UI（_make_interactive_fn）

各ツール実行前に `Yes`／`No`／`Custom Input` メニューを表示（Windows 専用・`msvcrt` 使用）。デフォルト10秒カウントダウン後、自動承認。`search_and_replace` は Claude Code 風の unified diff（行番号付き・ANSI カラー）を表示。破壊的ツールは `📝`、読取専用は `📖` マーカーで区別。

> **注記**: 本プロジェクトに権限管理（permission/allowlist/sandbox）機能は存在しない。ユーザー承認はこの半自動モードのみで、`DESTRUCTIVE_TOOLS`／`READONLY_TOOLS` の分類は並列実行可否の判定とマーカー表示に使われる。

### 6.4 設定

**config.py** はデータクラスではなく **モジュールレベルの定数** として定義する。主要カテゴリ:

- **コンテキスト**: `N_CTX=32768`, `MAX_TOKENS=8192`, `CONTEXT_BUFFER=500`, `DEFAULT_TRIM_THRESHOLD=16000`, `TOOL_RESULT_MAX_CHARS=12000`, `FUZZY_MATCH_THRESHOLD=0.85`
- **ツール分類**: `READONLY_TOOLS`（並列可能）／`DESTRUCTIVE_TOOLS`（直列必須）、`MAX_PARALLEL_TOOLS=5`、`ALWAYS_RECOMMEND`、`CODE_TOOL_SET`
- **温度・思考**: `TEMPERATURE_MAIN=0.7`, `TEMPERATURE_SUBQUERY=0.2`, `DEEP_THINK_BUDGET_SEC=90`, `CONTEXT_CHECKPOINT_THRESHOLD=0.80`
- **サブエージェント**: delegate（`MAX_STEPS=6`・`BUDGET_SEC=120`）、review（`MAX_STEPS=3`・`BUDGET_SEC=30`）、review_loop（`DEFAULT_ROUNDS=3`・`MAX_ROUNDS=5`）、verify（`MAX_ROUNDS=3`・`BUDGET_SEC=120`・`RUFF_GATE=True`・`IMPORT_GATE=True`・`TEST_GATE=False`）
- **教訓ストア**: `LESSONS_ENABLED=True`・`LESSONS_MAX_ITEMS=50`・`LESSONS_INJECT_MAX=3`・`LESSONS_REFLECT_MAX_TOKENS=1024`
- **lazy best-of-2**: `BEST_OF_EDIT_ENABLED=True`・`BEST_OF_ANSWER_ENABLED=True`・`BEST_OF_ANSWER_MARGIN=15.0`・`BEST_OF_RESAMPLE_TEMP_DELTA=0.15`
- **ツール呼出 grammar**: `NATIVE_TOOL_GRAMMAR=True`（llama-server `--jinja` の lazy grammar 活用）
- **run_python サンドボックス**: `RUNPY_MAX_INPUTS`・`RUNPY_IDLE_TIMEOUT_SEC`・`RUNPY_TOTAL_TIMEOUT_SEC`・`RUNPY_INPUT_MAX_TOKENS`・`RUNPY_INPUT_TEMPERATURE`・`RUNPY_OUTPUT_MAX_CHARS` 等

**config.json**（実行時設定・`get_data_path("config.json")`）は **サーバー構成のみ** を持つ:

```json
{
  "servers": [
    {"name": "Main PC", "base_url": "http://192.168.0.200:8080/v1",
     "api_key": "local-dummy-key", "model": "qwen2.5-14b"}
  ],
  "delegate_server": {
    "name": "Sub PC", "base_url": "...", "api_key": "...", "model": "..."
  }
}
```

- `servers[]`: ラウンドロビン対象の LM Studio サーバーリスト。`/api` で実行時切替。
- `delegate_server`: サブエージェント（`delegate_research`／`analyze_file`）専用サーバー。未設定時はメインを使用。`/delegate-api` で実行時設定。

### 6.5 パス解決（paths.py）

dev（ソース実行）と exe（PyInstaller）を自動判別する。

| 関数 | 役割 |
|---|---|
| `is_frozen()` | PyInstaller exe 化判定（`sys.frozen` and `sys._MEIPASS`） |
| `get_app_root()` | ソース時は `src/` の親、exe 時は exe 配置ディレクトリ |
| `get_data_path(relative_path)` | データファイル絶対パス。常に `get_app_root()` 基準（`CONTEXT_SUMMARY.md`, `.pixie_notes/`, `config.json` 等） |
| `get_bundled_path(relative_path)` | バンドルリソース（`rg.exe` 等）。exe 時は `sys._MEIPASS` 優先、ソース時はプロジェクトルート基準 |
| `resolve_venv_python(file_path)` | 編集対象ファイル起点で上方に `.venv`/`venv` を探索。Win: `Scripts/python.exe`、Unix: `bin/python`。`/verify` で使用 |

> `get_data_path`（永続配置の生成・設定ファイル）と `get_bundled_path`（同梱バイナリ・`_MEIPASS` 優先）の使い分けが重要。`main()` で `multiprocessing.freeze_support()` を呼ぶ。

---

## 7. データフロー（1ターンの処理）

ユーザー入力から最終回答・永続化までのフロー。

1. **入力・コンテキスト初期化**: `engine.run_graph` がリクエストを受け取り、`state.AgentStateBoard`（過去の目標・判明事項・進捗）と `ChatHistory` を読込。
2. **プロンプト構築**: `build_system_prompt` が **静的な** システムプロンプト（`base_prompt`＝`generate_behavior_prompt`・thinking_mode によらず固定）を返す。動的コンテキスト（ステートボード／ホワイトボード／JIT ヒント／budget_hint／deep_hint／定期リマインダー／教訓）は `engine.py` の `_build_dynamic_suffix` で1つにまとめ、`_apply_dynamic_suffix` が LLM 送信直前の **直近ユーザーメッセージ末尾への一時コピー** にのみ付与（`chat_history` 不変＝prefix cache 保護）。
3. **推論（Plan）**: LLM が `tools` + `tool_choice="auto"` で次アクションを決定。出力は `<think>`（思考）と `tool_calls`（アクション）に分かれ、`engine` が解析。Function Calling 不可モデルは `parse_native_tool_calls` で補完。
4. **アクション（Action）**: `node_action` → `execute_tool`。読取ツールは最大5並列、破壊ツールは直列。編集後は ruff → `/review` → `/verify` の検証パイプラインが付加される。
5. **観察（Observe）**: `node_observe` が結果を評価・`state` に反映（エラー記録・`read_file` はアウトラインを `file_summaries` へ）。更新された状態を再び LLM へ送る。この **LLM → Engine → Tool → State → LLM** サイクルを目的達成まで繰り返す。
6. **完了判定**: ツール呼出のないターンで、完全性スコア（≥閾値）等により `final_answer` を生成。
7. **永続化**: やり取りの要点は `.pixie_notes/state_board.json`（上書きのみ・結論のみ）へ。コンテキスト逼迫時はホワイトボード（`CONTEXT_SUMMARY.md`）へ要約退避。

> `engine.py`（実行の主体・Orchestrator）と `state.py`（作業記憶・Working Memory）の役割分離により、複雑なマルチステップタスクでも文脈を維持する。

---

## 8. ランタイムデータ（.pixie_notes/ / debug/）

### 8.1 .pixie_notes/（状態・キャッシュ・インデックス）

| ファイル | 形式 | 役割 |
|---|---|---|
| `state_board.json` | JSON | `AgentStateBoard` の永続化。`goal`／`current_step`／`next_to_do`／`found_knowledge`／`completed_tasks`（上限5）／`active_errors`／`file_summaries`／`project_structure`／`waiting_for_async`／`async_*`／`*_at` |
| `analysis_cache.md` | Markdown | `analyze_file` の要約キャッシュ（MD5 ハッシュベース・未変更ファイルの再解析を回避） |
| `code_index.json` | JSON | `code_index.build_index` のキャッシュ。`schema_version=1`, `root`, `files`, `call_graph`, `built_at`, `stats`。各 file entry に `md5`/`symbols`/`imports`/`external_imports`/`parse_error`/`has_main_guard`/`_call_edges` |
| `backups/*.bak` | — | `/verify` のファイル編集前に作成されるバックアップ |
| `lessons.json` | JSON | 教訓ストア（`LessonStore`）。`{lesson, trigger_keywords, source, hit_count, created_at}` のリスト。失敗ターンの反思で蓄積・Jaccard 重複統合・関連タスクで注入 |

### 8.2 debug/（ターンログ）

`turn_NNN.md` 形式で、`/debug` トグル時に出力される。内容: システムプロンプト（文字数・StateBoard/Whiteboard 状態・プレビュー）、メッセージ一覧（role・content プレビュー）、**JIT ツール選択**（各ツールのスコア・送信スキーマ数）、**コンテキスト使用量**（推定 chars/tokens・safe_max・総ウィンドウ）、ステートボード（実行中タスク・未解決エラー）、タイミング（Prefill 秒・Thinking 秒）。

> ※ 入力履歴はプロジェクトルートの `.pixie_history`（`prompt_toolkit` の `FileHistory`・`cli_input.py`）に永続化される（`.pixie_notes/` 配下ではない）。

---

## 9. ビルド・テスト・CI

### 9.1 ビルド・依存（pyproject.toml）

- `requires-python = ">=3.11"`（CI・起動スクリプトは 3.13）。
- `[project] dependencies = []`（**コア依存ゼロ**）。LM Studio バックエンドは Python 標準ライブラリのみで動作。
- `[project.optional-dependencies]`: `llama`（`llama-cpp-python>=0.2.80`・GGUF）、`image`（`pillow>=10.0`）、`ui`（`prompt_toolkit>=3.0`・リッチ CLI 入力）、`dev`（`pytest>=8.0`, `pytest-timeout>=2.3`, `ruff>=0.6`）。
- **フラット import の維持**: src-layout パッケージ化は行わず、`pythonpath = ["src"]`（pyproject）＋ `conftest.py` の sys.path 操作で `from config import ...` を解決。`[project.scripts]` エントリポイントなし（`.bat` で `python src/main.py` 起動）。
- `Pipfile` は実質未使用（pyproject.toml に統合済み）。

### 9.2 ツール設定

- **pytest**: `testpaths=["tests"]`, `pythonpath=["src"]`, `addopts="-ra -q"`, `timeout=30`, `filterwarnings=["error::ResourceWarning"]`（ファイルハンドルリーク Bug #8 を hard-fail 化）。`per-file-ignores`: `tests/* = ["E402"]`。
- **ruff**: `line-length=120`, `target-version="py311"`, `extend-exclude`（`.venv`/`__pycache__`/`debug`/`.pixie_notes`）, `select=[E,F,W,B,UP,I]`（SIM 除外）, `ignore=[E501,E731,B008,E741,B904]`。

### 9.3 CI（.github/workflows/ci.yml）

- **トリガ**: `push`（`main`／`master`）・`pull_request`。
- **OS**: `windows-latest`（単一ジョブ `lint-test`）。
- **ステップ**: checkout → setup-python 3.13 → `pip install ruff pytest pytest-timeout`（**`llama-cpp-python` は入れない**＝全テストが stdlib-only import で通る前提で依存境界を保証）→ `ruff check .` → `pytest`。

### 9.4 テストスイート（tests/）

`conftest.py` が `src` を sys.path に挿入し、`_gc_after_test` autouse fixture で各テスト後に `gc.collect()`。`llama_cpp_required` マーク（未インストール時に skip）を提供。

| テストファイル | 検証内容 |
|---|---|
| `test_streamfilter.py` | `StreamFilter`（`<think>`／Gemma `<\|channel\|>` 除去・チャンク分割・capture_thinking） |
| `test_state.py` | `AgentStateBoard`（永続化・GC 上限・injection_text）・`ChatHistory`（trim）・`build_system_prompt`（静的 system＋動的 suffix） |
| `test_code_index.py` | `build_index`・デッドコード検出・`summarize`・キャッシュ無効化・`get_symbol_range`・コールグラフ辺 |
| `test_harden.py` | 堅牢性バグ修正（#1,#2,#3,#5,#6,#8: bool 正直化・generator 形状・ログハンドル close・`estimate_token_count`・MCP `stop()`・search_and_replace/read_file） |
| `test_lfm_tooluse.py` | LFM2.5 専用 tool use パーサ（JSON／Pythonic `[func(a=1)]`・AST 安全性・コードブロックフォールバック） |
| `test_engine_guardrails.py` | 反復検知・類似度・完全性スコア・思考 strip・行動予告・思考深度判定・ネイティブツール呼出パース・継続結合 |
| `test_tools_pure.py` | JIT スコアリング・OpenAI スキーマ生成・code_index 系3ツール登録確認 |
| `test_code_session.py` | `outline`・`get_code_outline`（.py=AST／JS-TS=regex）・`project_structure` 永続化・`_run_ruff_check`・7ツール登録（回帰ガード） |
| `test_run_graph_blackbox.py` | `run_graph` 制御フロー遷移（scripted mock LLM・LM Studio 互換 generator）: final_answer／max_tool_calls／continuation／empty／user_rejected |
| `test_verify_loop.py` | `/verify` の `run_verify_fix_loop`: `resolve_venv_python`・各ゲート・short-circuit・`_generate_fix_edit`（JSON 抽出・未知ツール拒否）・ループ制御 |
| `test_delegate_research.py` | `run_agent_subquery`: 結論返却・外部状態非汚染・書込ツール拒否・ステップ上限・ネイティブフォールバック・予算タイムアウト・サーバーラウンドロビン |
| `test_run_python.py` | `run_python` サンドボックス（`input()` 自動入力プローブ・env サニタイズ・タイムアウト・出力上限） |
| `test_lessons.py` | `LessonStore`（追加・Jaccard 重複統合・recall・GC）＋ reflection トリガ（失敗ターンのみ発火・JSON schema・`generalizable=false` は保存しない） |
| `test_shadow_verify.py` | `shadow_apply`／`shadow_gate`（py_compile＋ruff・import 除外）＋ `run_graph` 連携の再サンプル制御・一時メッセージ除去 |
| `test_cli_input.py` | `prompt_toolkit` optional 判定・`input()`+`"""` フォールバック・`SLASH_COMMANDS` ⊇ 実装コマンドの回帰ガード |

**ゴールデンテスト**（`test_behavior_prompt_golden.py` + `tests/golden/behavior_prompt/`）: `generate_behavior_prompt`（tools.py）の出力バイト完全不変を保証するスナップショットテスト。9ツールセット × {shallow, deep, code} = **27 ケース** で事前生成ゴールデンファイル（`{name}__{mode}.txt`）と `==` 完全比較（差分時は sha256[:12] 表示）。リファクタ安全性の要。

---

## 10. 制約事項・未実装

従来版の仕様書に記載されていたが **現状未実装** の機能:

- **ベクトルデータベースによる RAG 長期記憶**: 未実装。現状はセッション内の作業記憶（ステートボード JSON）・要約退避（ホワイトボード MD）・セッション横断の簡易教訓ストア（`lessons.json`・`lessons.py`）のみ。
- **ナレッジグラフ型メモリ**: 未実装（§11 の展望）。
- **権限管理（permission／allowlist／sandbox）**: 未実装。ユーザー承認は半自動 UI のみ。
- **専用の Python サンドボックス実行ツール**: `run_python` で実装（§3.2）。`python -u` でサンドボックス実行し、`input()` のプロンプトを検出すると LLM が自動で入力値を生成して stdin に送り、対話的プログラムを自律継続する。ただし Job Object によるメモリ/CPU リソース分離は **未実装（Phase2）** で、現状は総タイムアウト・`max_inputs` 上限・env サニタイズによる安全網のみ。
- **GUI 実行パス**: `--no-gui` フラグは存在するが実質 CLI のみ。

---

## 11. 今後の展望

本プロジェクトは、ローカル環境におけるプライバシー保護と低レイテンシを維持したまま、クラウドベースの高度な AI エージェントに匹敵する柔軟性と拡張性を備えたエコシステムへの成長を目指す。現状の実装を踏まえた現実的な拡張方向:

- **記憶構造の高度化**: 現状のステートボード（上書き式の作業記憶）と教訓ストア（`lessons.json`・セッション横断の経験メモリ）を足場に、本格的な長期記憶層（ベクトル検索やナレッジグラフ）の導入。ホワイトボード要約機構を拡張する形で段階的に実装可能。
- **ツールセットの動的拡張と OS 統合**: MCP クライアント（§5.2）による外部ツール統合を軸に、Hand（ツール層）をさらに拡張。GUI 操作や高度なファイルシステム権限の付与による自律操作範囲の拡大。
- **ツールパック機構（manga/web パック）**: **設計確定・実装前**（`docs/design/toolpacks.md`）。用途別のツール群をパッケージ化して動的有効化する機構。現在は `category`（core/extended）による分類のみ。
- **多エージェント・オーケストレーションの本格化**: 現状は `subagent.py` で toggle/observe-only のサブエージェント（`delegate_research`／`/review`／`/verify`）を集約しているが、専門タスクごとのサブエージェントを動的に生成・制御する分散型システムへの発展。ただしローカル LLM のコスト制約から、段階的・低コストな設計が前提となる。
- **エッジ・軽量化**: `dependencies = []` のコア依存ゼロ設計と `paths.py` の exe 化対応を活かし、SLM との組み合わせによる軽量環境での動作最適化。

---

*本仕様書は `src/` 実装（2026-07 時点）に基づく。実装の更新に合わせて本書も追従させること。*
