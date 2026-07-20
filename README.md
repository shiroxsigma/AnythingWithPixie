# AnythingPixie

> ローカル LLM をバックエンドにした、ReAct 型の自律エージェント CLI。自然言語で指示すると、ファイル操作・コード解析・コマンド実行・サブエージェント委譲などを **Function Calling ベースの思考ループ** で自律的にこなす。

愛称: **Pixie**。LM Studio（OpenAI 互換 API）または GGUF（llama.cpp）をバックエンドに動く。**コア依存ゼロ**（LM Studio 版は Python 標準ライブラリだけで動作）。

---

## ✨ 特徴

- **ReAct ループ** — Plan → Action → Observe を回してタスクを完遂。テキストプロトコルではなく Function Calling 方式。
- **段階的思考深化** — タスクの複雑さに応じて `shallow` / `deep` を自動判定。`/deep` で強制深化。
- **35 の組込ツール** — ファイル操作・コード編集・検索（ripgrep）・シェル・サンドボックス実行・コード解析（AST）etc。
- **コードインテリジェンス** — `ast` のみでコードベースをインデックス化し、コールグラフ抽出・デッドコード検出が可能。
- **サブエージェント・検証** — `delegate_research`（調査委譲）、`/review`（編集レビュー）、`/verify`（実行ベース検証＋自動修正）。
- **分岐点限定 lazy best-of-2** — 破壊的編集と最終回答の分岐点でのみシャドウ検証（`py_compile`/`ruff`）し、失敗時だけ再サンプル。候補が最初からクリーンなら追加コストゼロ。
- **教訓ストア（経験メモリ）** — 失敗ターンを反思して汎化教訓を蒸留し `.pixie_notes/lessons.json` にセッション横断で蓄積。関連タスクで自動注入。
- **多層ガードレール** — 反復検知・ループ検知・不完全ツール呼出の補完など、ローカル LLM の暴走防止。
- **prefix cache 保護・可視化** — システムプロンプトは完全固定（動的コンテキストは直近ユーザーメッセージ末尾への一時付与のみ）。ヒット率（`cache_n`/`prompt_n`）を毎ターン常時可視化。
- **コンテキスト管理** — ホワイトボード（`CONTEXT_SUMMARY.md`）要約継承・2段階トリム・動的ツール結果キャップ。
- **マルチモーダル** — Qwen3/3.5-VL で画面キャプチャ・画像認識（オプション）。
- **MCP クライアント** — JSON-RPC over stdio で外部 MCP サーバーのツールを統合。

> 📖 実装の正確な動作仕様は [`project_analysis_report.md`](./project_analysis_report.md)（開発者向け詳細仕様書）を参照。

---

## 🚀 クイックスタート

### 前提

- **Python 3.11+**（CI・起動スクリプトは 3.13 を想定）
- **Windows** 想定（Linux/macOS でも動くが、画面キャプチャや承認 UI は Windows 向け）
- いずれかのバックエンド:
  - **LM Studio**（おすすめ・最も簡単。標準ライブラリのみで動く）
  - **GGUF モデル**（`llama-cpp-python` が必要）

### A. LM Studio で起動（推奨）

1. LM Studio でモデルをロードし、**Local Server** を起動（例: `http://localhost:1234/v1`）。
2. プロジェクトルートに `config.json` を作成（[設定例](#configjson-の例)）。
3. 起動:
   ```powershell
   .venv\Scripts\python.exe src\main.py
   # または
   StartPixie.bat   # .venv の python で起動
   ```
4. サーバー一覧から接続先を選んでチャット開始。

### B. GGUF モデルで起動

1. `models/` ディレクトリに `.gguf` を配置（Vision を使うなら `mmproj` も同階層に。自動検出）。
2. バックエンド依存をインストール:
   ```powershell
   .venv\Scripts\python.exe -m pip install "llama-cpp-python>=0.2.80"
   ```
3. 起動:
   ```powershell
   .venv\Scripts\python.exe src\main.py
   ```
   ※ Vision（画像認識）を使う場合は `pip install "pillow>=10.0"` も。

### config.json の例

```json
{
  "servers": [
    {
      "name": "Main PC",
      "base_url": "http://localhost:1234/v1",
      "api_key": "lm-studio",
      "model": "qwen2.5-14b"
    }
  ],
  "delegate_server": {
    "name": "Sub PC",
    "base_url": "http://192.168.0.50:1234/v1",
    "api_key": "lm-studio",
    "model": "qwen2.5-14b"
  }
}
```

- `servers[]` — LM Studio サーバーリスト。`/api` で実行時に切り替え可能。`base_url` 以外は省略可（既定値あり）。
- `delegate_server` — サブエージェント（調査委譲・ファイル要約）専用の第2サーバー（任意）。未設定時はメインサーバーを使用。

---

## ⌨️ 使い方

### 入力操作

**リッチ入力（`prompt_toolkit` 導入時・実装 `src/cli_input.py`）** — Claude Code 風:
```powershell
.venv\Scripts\python.exe -m pip install "prompt_toolkit>=3.0"
```
| 操作 | キー |
|---|---|
| 送信 | `Enter` |
| 改行 | `Ctrl+J` / `Esc`→`Enter` / 末尾 `\`+`Enter` |
| スラッシュコマンド補完 | `/` を入力 |
| 履歴呼び出し | `↑` / `↓`（`.pixie_history` に永続化） |

**フォールバック（未導入時）** — `Enter` で送信、複数行は `"""〜"""` ヒアドキュメント。未導入でも自動でこちらに落ちるので依存ゼロ方針は維持されます。

> ※ `Shift+Enter` での改行は `prompt_toolkit` の制約でバインド不可。改行は `Ctrl+J` / `Esc`→`Enter` / 末尾 `\`+`Enter` のいずれかを使用してください。

### スラッシュコマンド

| コマンド | 動作 |
|---|---|
| `quit` / `exit` | セッション終了 |
| `/think` | `<think>` 思考プロセス表示の ON/OFF |
| `/deep` | 強制深度思考（段階判定をスキップし常に deep） |
| `/review` | 編集レビューサブエージェントのトグル（observe-only） |
| `/verify` | 実行ベース検証＋自動再編集のトグル（py_compile/ruff/pytest・`.venv` 優先） |
| `/review_loop [N]` | 直前の回答を main↔review で N 往復させて改善（既定3） |
| `/step` | 半自動（ステップ実行）／フル自動のトグル |
| `/mem` | メモリモード（会話履歴保持）のトグル |
| `/debug [full]` | デバッグダンプのトグル（`.pixie_notes/debug/turn_NNN.md`） |
| `/reset` | チャット履歴クリア＋コンテキストリセット |
| `/context` | コンテキスト使用量の可視化 |
| `/recap` | リアルタイム画面キャプチャの領域選択（capture 利用時） |
| `/code-init [path]` | プロジェクト構造を記憶し Code モード ON |
| `/code [target\|off]` | コード専門モードのトグル |
| `/trace <keyword>` | キーワードの定義点・使用点を調査 |
| `/api` | LM Studio サーバー切り替え |
| `/delegate-api [off]` | 委譲サブエージェント用サーバー設定 |
| `/poll_async [PID LOG]` | 非同期プロセスのポーリング |

### 代表的なモード

- **`/code`** — コード作業に特化したツールセット（20ツール固定）で動く。
- **`/review`** — ファイル編集の直後に読取専用サブエージェントが批判的レビュー（observe-only・編集結果は壊さない）。
- **`/verify`** — `.py` 編集後に py_compile → import → ruff →(任意)pytest のゲートを回し、エラーがあれば自動で再編集（最大3往復）。
- **Planning / Execution フェーズ** — 計画を `PLANNING.md` に書き出し、`ok` で実行へ移行する2段階フロー。

---

## 🧰 組込ツール（35）

主なものを分類（全一覧と詳細スキーマは `inspect_tool` または[仕様書 §3](./project_analysis_report.md#3-ツール群toolspy--registrypy--code_toolpy--code_indexpy)）。

| 分類 | 代表ツール |
|---|---|
| ファイル操作 | `read_file` `write_file` `append_to_file` `move_file` `delete_file` `make_directory` `list_directory` |
| コード編集 | `search_and_replace`（3層ファジーマッチ） `replace_lines` `write_sections` |
| 検索・差分 | `grep_search`（ripgrep） `diff_files` |
| コード解析 | `get_code_outline` `map_codebase` `detect_dead_code` `read_symbol` `research_code_paths` `gather_project_info` |
| シェル・プロセス | `run_command` `run_async_test` `poll_process` `kill_process` |
| サンドボックス | `run_python`（`input()` を検出すると LLM が自動入力を生成して継続） |
| 状態・委譲 | `update_state` `set_goal` `query_whiteboard` `delegate_research` |
| ナビゲーション | `view_tree` `get_file_stats` `inspect_tool` `view_image` `analyze_file` |

破壊的操作（`write_file` / `run_command` / `run_python` 等）は半自動モードで実行前に承認メニューが出る（`/step` で切替）。

---

## 🏗️ アーキテクチャ

4層モデルで捉える（比喩）:

| 層 | モジュール | 役割 |
|---|---|---|
| 脳（推論） | `engine.py` `engine_helpers.py` `state.py` `shadow_verify.py` `lessons.py` | ReAct ループ・意思決定・ガードレール・分岐点検証・経験メモリ |
| 手（実行） | `tools.py` `registry.py` `code_tool.py` `code_index.py` | 35 の組込ツール |
| 目（知覚） | `capture.py` `llm_client.py` | 画面キャプチャ・マルチモーダル |
| 記憶（状態） | `state.py` + `CONTEXT_SUMMARY.md` | 作業記憶・要約継承 |

1ターンの流れ: **入力 → 静的システムプロンプト＋動的 suffix 構築 → Plan(LLM) → Action(ツール並列/直列) → Observe(状態更新) → … → 完了判定 → 永続化**。

> 循環 import は `engine_helpers.py`（共用純粋関数）・`registry.py`（状態ホルダー分離）・`subagent.py`（追加 LLM 呼出の集約）の3抽出で解消している。CLI 入力は `cli_input.py`（`prompt_toolkit` optional・未導入時は `input()` へフォールバック）。

---

## 🛠️ 開発

```powershell
# テスト
.venv\Scripts\python.exe -m pytest

# リント
.venv\Scripts\python.exe -m ruff check .
```

- **フラット import** — `src-layout` パッケージ化はせず、`pythonpath=["src"]` + `conftest.py` の sys.path 操作で `from config import ...` を解決。
- **CI**（`.github/workflows/ci.yml`）— Windows・Python 3.13・`ruff` + `pytest`。`llama-cpp-python` は入れず、全テストが stdlib-only import で通ることを保証。
- **ゴールデンテスト** — `generate_behavior_prompt` の出力不変を 27 ケースでスナップショット検証。

### 依存関係（`pyproject.toml`）

| extras | パッケージ | いつ必要 |
|---|---|---|
| （コア） | なし | — LM Studio 版は依存ゼロ |
| `llama` | `llama-cpp-python>=0.2.80` | GGUF バックエンド使用時 |
| `image` | `pillow>=10.0` | 画面キャプチャ・画像処理 |
| `ui` | `prompt_toolkit>=3.0` | リッチな CLI 入力（未導入時は `input()` にフォールバック） |
| `dev` | `pytest` `pytest-timeout` `ruff` | 開発・テスト時 |

---

## ⚠️ 制限事項

- **RAG 長期記憶・ナレッジグラフ** — 未実装。現状はセッション内の作業記憶（`state_board.json`）・要約退避（ホワイトボード）・セッション横断の簡易教訓ストア（`.pixie_notes/lessons.json`）のみ。
- **権限管理（allowlist/sandbox）** — 未実装。ユーザー承認は半自動 UI のみ。
- **`run_python` のリソース分離** — Job Object によるメモリ/CPU 分離は未実装（総タイムアウト・`max_inputs` 上限・env サニタイズのみ）。
- **GUI** — `--no-gui` フラグはあるが実質 CLI のみ。

---

## 📄 ライセンス

未定（`pyproject.toml` の `license` は TODO）。決定後に記載します。

---

*本 README は `src/` 実装（2026-07 時点）に基づく。実装の正確な動作仕様は [`project_analysis_report.md`](./project_analysis_report.md) を参照。*
