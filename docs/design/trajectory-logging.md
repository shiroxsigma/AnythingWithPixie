# 詳細設計: 軌跡ロギング（SFT/DPO 教師データ産出基盤）

作成: 2026-07-07 / ステータス: 設計確定・実装前

## 0. 目的と役割分担

gemma-4 を教師として LFM2.5（または他の小型モデル）を「このハーネス専用」に SFT/DPO するための教師データを産出する。

**役割分担（確定事項）**:
- **本プロジェクト（AnythingWithPixie）**: 軌跡の記録（JSONL）と、SFT-ready 形式への export ツールの提供まで。**学習は行わない**
- **別プロジェクト（SFT 側）**: export 出力を入力として Unsloth 等で LoRA/SFT/DPO → GGUF 変換 → LM Studio 配置
- **効果測定**: 本プロジェクトの eval スイート（`evals/runner.py --compare`）が適応度関数。学習前後のモデルを同一タスクで比較する

データフロー:
```
[実運用セッション / eval 実行]
  → .pixie_notes/trajectories/*.jsonl   (生ログ・イベント形式)
  → tools/export_sft.py                  (フィルタ + 再構築 + 変換)
  → dataset/sft_*.jsonl / dpo_*.jsonl    (SFT プロジェクトへの受け渡し物)
```

## 1. 設計原則

1. **再現性が最優先**: 「そのターンで LLM に実際に送った messages（動的 suffix 適用後）と、返ってきた生応答」を後から完全再構築できること。SFT の学習ペアは入力の完全性が命
2. **ラベルは決定的信号から自動付与**: fast gate / shadow_gate / ガードレール / parse 成否 / eval PASS。人手アノテーション前提にしない
3. **DPO ペアは既存の再サンプル機構から無料で採れる**: shadow_verify の再サンプル（1本目=rejected / 2本目=chosen、**同一コンテキスト**）と final answer best-of-2 は、そのまま preference pair になる
4. **記録は本体の邪魔をしない**: 同期 append のみ（JSONL 1行 write）。例外は握り潰し、記録失敗でターンを壊さない。`TRAJECTORY_LOG_ENABLED=False` で完全無効化
5. **シンプル優先**: 差分記録や独自圧縮はしない。プレーン JSONL + サイズ上限 GC。ディスクは安く、バグった再構築ロジックは教師データを汚染する

## 2. 記録フォーマット

### 2.1 ファイルレイアウト

```
.pixie_notes/trajectories/
  20260707/
    s_20260707_101530_a3f2.jsonl     # 1セッション = 1ファイル
  20260708/
    ...
```
- セッション ID: `s_<日時>_<ランダム4hex>`。eval 実行時はタスクごとに独立セッション（`s_..._eval_<task_id>`）
- GC: `TRAJECTORY_MAX_MB`（既定 2048）超過時に古い日付ディレクトリから削除。起動時に1回チェック

### 2.2 レコード（JSONL の1行 = 1イベント）

全レコード共通ヘッダ:
```jsonc
{"schema_version": 1, "ts": 1751852130.5, "session": "s_...", "turn": 3, "type": "<イベント型>", ...}
```

**イベント型一覧**（この5種のみ。増やすときは schema_version を上げる）:

#### `session_meta`（セッション開始時に1回）
```jsonc
{
  "type": "session_meta",
  "model": "gemma-4-26B-A4B-it-...q5_k_s",   // context.llm_model_name
  "base_url": "http://localhost:8080/v1",
  "harness_git": "9b7dfb3",                    // git rev-parse --short HEAD（取得失敗時 null）
  "mode": "normal|code|manga",
  "active_packs": ["manga"],
  "sampling_profile": {"temperature": 1.0, "top_k": 64, "top_p": 0.95},
  "n_ctx": 81920,
  "eval_task": "02_fix_off_by_one"             // eval 実行時のみ。実運用は null
}
```

#### `llm_call`（node_plan の LLM 呼び出しごと。教師データの本体）
```jsonc
{
  "type": "llm_call",
  "call_id": "c_007",                  // セッション内通番
  "messages": [...],                    // 実際に送信した messages_for_llm（動的 suffix 適用後・全量）
  "tools": "sha256:ab12...",           // ツール定義は巨大かつセッション内不変なので、初回のみ全量
                                        // （tools_full フィールド）、以降はハッシュ参照
  "tools_full": [...],                  // 初回 llm_call のみ
  "params": {"temperature": 1.0, "max_tokens": 8192, "tool_choice": "auto"},
  "response": {
    "content": "...",                   // 生 content（<think> 含む。strip 前）
    "reasoning_content": "...",         // reasoning_content 経路の思考（無ければ null）
    "tool_calls": [...],                // 構造化 tool_calls（無ければ null）
    "finish_reason": "tool_calls",
    "timings": {"cache_n": 9536, "prompt_n": 1088, "predicted_n": 210}  // 取れた場合
  },
  "purpose": "plan|resample_edit|resample_answer|forced_final|reflection"
}
```
- `purpose` は呼び出し文脈。`resample_*` は best-of-2 の2本目、`forced_final` は max_tool_calls 到達時の強制生成
- **messages は全量記録**（差分記録にしない）。1呼び出し ~40KB、100呼び出しセッションで ~4MB は許容。トリム（check_and_trim_context）が履歴を書き換えても、各呼び出しの送信内容がそのまま残るので再構築不要

#### `tool_result`（ツール実行ごと）
```jsonc
{
  "type": "tool_result",
  "call_id": "c_007", "tool_call_id": "...", "tool_name": "search_and_replace",
  "result_head": "Success: ...",       // 先頭 500 字のみ（全文は llm_call.messages に次ターンで載る）
  "is_error": false,
  "fast_gate": "pass|fail|na",         // .py 編集後の fast gate 結果
  "fast_gate_detail": "[ruff check] ..."  // fail 時のみ・先頭300字
}
```

#### `judgement`（品質信号の発生ごと）
```jsonc
{
  "type": "judgement",
  "call_id": "c_007",
  "kind": "guardrail|shadow_gate|resample_decision|parse_rescue",
  "detail": "loop_guardrail: read_file の同一呼び出し...",
  // resample_decision の場合のみ（DPO ペアの接着剤）:
  "rejected_call": "c_007", "chosen_call": "c_008", "reason": "shadow_gate_failed"
}
```

#### `turn_end`（ターン終了ごと）
```jsonc
{
  "type": "turn_end",
  "exit_reason": "final_answer (ツール実行 5回後)",
  "tool_call_count": 5,
  "failure_signals": ["fast_gate: ..."],   // state.failure_signals のコピー
  "final_answer_head": "...",               // 先頭 500 字
  "eval_passed": true                       // eval 実行時のみ。実運用は null
}
```

### 2.3 プライバシー・機密
- ログはローカル（`.pixie_notes/` は既に gitignore 対象）のみ。リポジトリにコミットしない
- messages にはユーザー入力・ファイル内容が含まれる。export 時に `--redact-paths <pattern>` で除外フィルタを提供（既定はフィルタなし = ローカル利用前提）

## 3. 収集ポイント（実装箇所）

新モジュール `src/trajectory.py` に `TrajectoryLogger` を実装し、以下からフックする。**すべて try/except で保護し、失敗しても本体に影響させない**。

| イベント | フック位置 | 備考 |
|---|---|---|
| session_meta | `run_cli_chat` 開始時 / eval の `_run_task_body` | context から収集 |
| llm_call | `node_plan` の応答確定後（`_dump_debug_context` の直後付近） | messages_for_llm・応答・timings が全部揃う位置。purpose は呼び出し元が引数で渡す（node_plan に `log_purpose: str = "plan"` を追加） |
| tool_result | `run_graph` のツール結果処理ループ（fast gate 判定 `_has_fast_gate_failure` の直後） | 既存の失敗信号収集と同じ場所 |
| judgement | 各ガードレール分岐（failure_signals append の隣）/ `_verify_and_maybe_resample_edits` / `_maybe_resample_final_answer` | resample は rejected/chosen の call_id を紐付け |
| turn_end | `run_graph` の return 直前（reflection の後） | |

- Logger インスタンスは `AppContext.trajectory` に保持（`get_lesson_store()` と同様のパターンでも可）。eval runner は隔離ディレクトリでなく**実プロジェクトの trajectories/ に書く**（教師データ収穫が目的のため。`_isolated_pixie_env` の monkeypatch 対象から trajectory の出力先だけ除外するか、eval 用の出力先を明示指定）
- config: `TRAJECTORY_LOG_ENABLED: bool = True` / `TRAJECTORY_MAX_MB: int = 2048` / `TRAJECTORY_RESULT_HEAD_CHARS: int = 500`

## 4. export ツール（`tools/export_sft.py`、CLI・LLM 不使用）

### 4.1 SFT 抽出（正例）

```
python tools/export_sft.py sft \
  --model-filter gemma \          # 教師モデルの軌跡のみ
  --tier gold \                   # gold: eval_passed=true のセッション全ターン
                                  # silver: 実運用で fail 信号ゼロ + 正常 exit のターン
  --since 20260701 --out dataset/sft_gold.jsonl
```

出力（Unsloth がそのまま読める OpenAI messages 形式 + 学習対象の completion）:
```jsonc
{"messages": [...送信時全量...], "tools": [...], 
 "completion": {"role": "assistant", "content": "...", "tool_calls": [...]},
 "meta": {"session": "...", "call_id": "c_007", "tier": "gold", "teacher": "gemma-4-..."}}
```

**ティア定義（自動ラベル）**:
- **gold**: eval PASS セッションの全 llm_call（タスク正解が決定的に保証されている）
- **silver**: 実運用セッションで、そのターン以降に judgement(guardrail/shadow_gate fail) が無く、fast_gate が fail していない llm_call
- **reject**（DPO の rejected 素材）: guardrail 発火・shadow_gate fail・parse_rescue が紐付く llm_call

### 4.2 DPO 抽出（preference pair）

```
python tools/export_sft.py dpo --out dataset/dpo_pairs.jsonl
```
judgement(kind=resample_decision) の rejected_call / chosen_call を突き合わせ:
```jsonc
{"prompt": [...両者共通の messages...],
 "chosen": {"content": ..., "tool_calls": ...},     // 2本目（gate 通過側）
 "rejected": {"content": ..., "tool_calls": ...},   // 1本目（gate 失敗側）
 "meta": {"reason": "shadow_gate_failed", "session": "..."}}
```
供給源: ①shadow_verify 再サンプル ②final answer best-of-2 ③ガードレール発火→再試行成功（同一ターン内で発火直後の再 llm_call が成功した組）。**同一コンテキスト保証があるのは①②のみ**なので、③は `--include-guardrail-pairs` オプトイン（prompt が1メッセージ分ずれるため）

### 4.3 思考（reasoning）の扱い
- gemma の `reasoning_content` は既定で **completion に含めない**（`--include-reasoning` でオプトイン）。理由: LFM2.5 は自前の思考形式を持つ reasoning-only モデルであり、gemma の思考をそのまま模倣させると形式が崩れるリスク。まず「行動（tool_calls / 最終回答）」の蒸留から始め、reasoning 蒸留は効果を eval で見てから
- `<think>` が content に inline 混入している場合は export 時に strip

### 4.4 検証コマンド
- `python tools/export_sft.py stats`: ティア別件数・モデル別件数・DPO ペア数・推定トークン数を表示（「学習に足りるか」の判断材料。目安: SFT 500〜5000 サンプル、DPO 200〜1000 ペア）

## 5. 教師データ生成モード（収穫の加速）

eval タスクは正解チェッカー付きなので、**タスクを増やして gemma で回すこと自体が教師データ生成**になる:

```
python evals/runner.py --harvest --repeat 3        # 全タスクを3回ずつ実行し PASS 軌跡を収穫
```
- `--harvest`: trajectory 記録を強制 ON にし、PASS したセッションに eval_passed=true を刻む（既存の runner に小改修）
- 温度によるばらつきで同一タスクから複数の正解軌跡が採れる（多様性はデータ増強として有益）
- **eval タスク拡充が今後は教師データ拡充を兼ねる**: 実運用で LFM が失敗したタスク種別を eval タスク化 → gemma で harvest → その弱点の教師データが増える、という補強ループ

## 6. SFT 側プロジェクトへの引き継ぎ仕様（参考・本プロジェクトのスコープ外）

- 入力: `dataset/sft_*.jsonl` / `dataset/dpo_*.jsonl`（上記形式）
- 推奨レシピ: Unsloth + LFM2.5-1.2B（dense、16GB で確実）を delegate 専用機として SFT → 効果があれば 8B-A1B の LoRA を検証（MoE 対応は要確認）。chat template は LFM2.5 の ChatML 準拠に変換（tools は template の tools 引数に渡す）
- 成果物: GGUF（+ 量子化 q8_0）→ LM Studio 配置
- 受け入れ基準: `evals/runner.py --base-url <sub> --model <ft後>` で学習前と `--compare`。**6/11 → 9/11 以上を成功ライン**とする

## 7. 実装フェーズ

| フェーズ | 内容 | 完了条件 |
|---|---|---|
| T1 | trajectory.py + engine/runner フック + GC + config | 単体テスト（イベント形式・GC・例外安全）green / 実セッション1本で全5イベント型が記録される / 記録 OFF 時のオーバーヘッドゼロ |
| T2 | export_sft.py（sft/dpo/stats） | 実ログからの export が Unsloth の期待形式で読めること（形式検証スクリプト同梱）/ DPO ペアが resample 実測1件から正しく組めること |
| T3 | eval --harvest モード + 収穫実行 | gemma で全11タスク×3回の harvest 実行 → stats で gold サンプル数を確認 |

## 8. 却下した代替案（記録）

- **差分記録（イベントソーシング + 再構築）**: 容量は 1/10 になるが、トリム・suffix・履歴書き換えの再現ロジックが必要になり、再構築バグ = 教師データ汚染のリスクが容量メリットに見合わない。全量記録 + GC を採用
- **debug モード（_dump_debug_context）の拡張**: 人間向け Markdown で機械可読性がなく、`/debug` トグル前提。SFT 用は常時・構造化が要件なので別系統とする（debug モードは現状のまま併存）
- **DB（SQLite）**: クエリは export 時の全走査で十分。JSONL の方が SFT ツールチェーンとの親和性が高く、壊れても行単位で切り捨てられる
- **LLM によるターン品質採点（LLM-as-judge）を記録時に行う**: コストが常時発生する。決定的信号のみで gold/silver/reject が付けられる設計にし、judge が必要なら export 後に別途実行
