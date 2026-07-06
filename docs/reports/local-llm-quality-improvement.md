# AnythingWithPixie ローカル LLM 品質改善 — 統合知見レポート

期間: 2026-07-03 〜 2026-07-06 / 対象構成: llama-server (--swa-full --jinja, build b9298) + gemma-4-26b-a4b (MoE 26B/A4B, Q5_K_S, hybrid SWA attention) / RTX 5070 Ti 16GB / Windows 11

## エグゼクティブサマリ

「ローカルモデルでフロンティアモデルに匹敵する」ための取り組みを、**調査 → 実装 → 実測 → 再調査**のループで進めた。核心の結論:

1. **prefix cache は「不可能」ではなかった** — 前提としていた制約（llama.cpp issue #19794）は別系統モデルの問題で、gemma-4 は `--swa-full` + PR #22288 以降のビルドでキャッシュ再利用が機能する。プロンプト設計をキャッシュ協調型に再構築し、**実測 98.8% ヒット / prefill 約19倍高速化**を達成した。
2. **速度改善は精度改善の原資** — 浮いた時間予算を検証・再サンプルに再投資する設計（lazy best-of-2、verify 常時実行）が成立した。
3. **ハーネスがモデル品質を代替する** — コミュニティでも「ハーネス品質でモデルを最大6倍補える」が合意事項。決定的検証・grammar 制約・エラー分類リトライという本プロジェクトの設計方針は、事後のコミュニティ調査でいずれも最重要プラクティスと一致した。
4. **残る最大の伸びしろはサンプリング設定とコンテキスト規律** — gemma-4 公式推奨（temp 1.0/top_p 0.95/top_k 64）から現行設定がズレている。また 26B-A4B の長文脈品質は 128k で MRCR 44% まで落ちるため、実用上限は 32k〜48k に置くべき。

---

## Part 1. 確立した事実（インフラ・モデル特性）

### 1.1 gemma-4 の prefix cache は機能する（当初認識の訂正）

- **誤解**: issue #19794 により hybrid attention では cache reuse 不可 → **真実**: #19794 は Mamba/SSM 系 recurrent hybrid（Qwen3-Next 等）の問題。gemma-4（SWA+full の純粋ハイブリッド）は issue #21468 → **PR #22288（2026年4月末）で修正済み**。
- 条件: `--swa-full` フラグ + 2026年4月末以降のビルド。代償として SWA の VRAM 節約（約1/5）を放棄。
- 実測（build b9298）: 完全一致プロンプトで prompt_n 201→1（**14.6倍**）、末尾のみ変更でも 201→22（**5.4倍**）。
- LM Studio も内部は同じ llama.cpp のため回避策にならない。低レベル制御（--swa-full、slot、grammar）ができる **raw llama-server が優位**。

### 1.2 gemma-4 は思考を reasoning_content で流す

- 思考は `delta.content` ではなく `delta.reasoning_content`（専用フィールド）でストリームされる。inline `<think>` 前提の処理には一切届かない。
- これに気づかないと「prefill が遅い」ように見える（→ Part 2.6 の診断）。

### 1.3 このクラスのモデル・量子化の特性（コミュニティ調査より）

- **Q5_K_S は妥当**: IFEval 実測で Q5_K_S ≈ Q5_K_M ≈ Q6_K（差が出るのは Q8_0 のみで 16GB には載らない）。不安定域は Q2〜Q3（無限ループ）と無較正 Q4（ツール呼び出しの劣化）。imatrix 較正版を選ぶこと。
- **エージェント用途はチャットより先に量子化ダメージが出る**（ACBench 等）。「Q4 はチャットなら平気、エージェントループが量子化ダメージを先に暴く」。
- **KV cache 量子化は q8_0 まで**（K/V 対称 + Flash Attention 必須）。劣化 1% 未満で VRAM 半減。**q4_0 は禁止**（110k 文脈で -36.8%、ツール呼び出し破壊の最有力機構）。
- **repeat_penalty は無効（1.0）のまま**が正解。有効化は JSON/コード構文を壊す（40人以上が無効化で解決した報告）。llama-server の現行デフォルトは無効。
- **長文脈品質**: gemma-4-26b-a4b の MRCR v2 (8-needle, 128k) は **44.1%**（dense 31B は 66.4%）。公称 128k だが実効はその半分以下と見るべき。
- **量子化のせいにする前に疑うべき llama.cpp バグ**: grammar は失敗時に素通しになる（fail-open, #19051）、ツールスキーマ内の `\d \w \s` で GBNF 変換が失敗（#22314）、thinking 有効時に grammar が不活性（#20345）。

### 1.4 見送りと判断した技術

- **speculative decoding**: MoE + hybrid attention では実測で効果なし〜逆効果、VRAM も圧迫。見送り。
- **26B MoE 本体の LoRA/QLoRA**: 16GB では非現実的（Unsloth 自身が MoE 4bit QLoRA 非推奨）。現実的な FT 経路は小型サイドカーモデルの SFT のみ。
- **常時 best-of-N / MCTS / ToT**: 複数回推論が前提でコスト構造に合わない。→ 分岐点限定 lazy 方式に転換（Part 2.5）。

---

## Part 2. 実装済み改善と実測効果

### 2.1 cache-friendly プロンプト再配置（コミット 2a48600）

**問題**: 旧サンドイッチ構造は state_board/ホワイトボードを system メッセージ先頭に注入 → messages[0] が毎ターン変化 → キャッシュ全壊。

**対策**: 「静的 prefix / 動的 suffix」原則に再構築。
- system = base_prompt のみ（セッション内不変）。ツール定義は全量固定・順序固定。
- 動的情報（state_board・ホワイトボード・JIT 推奨ヒント・budget/deep ヒント・リマインダー）はすべて `_build_dynamic_suffix()` に集約し、LLM 送信直前の一時コピーの末尾 user メッセージにのみ追記。履歴には書き込まない。
- JIT ツール選択は「フィルタ」から「推奨ヒントテキスト」に格下げ（Lost in the Middle 対策は末尾配置の recency bias で維持）。

**実測**: turn2 で cache_n 1685/1706（**98.8% ヒット**）、prefill 1366ms → **70.6ms**。旧設計の再現ではヒット率 0%。

**不変条件（今後の全変更が守るべきルール）**: 動的に変わる情報は必ず動的 suffix 側へ。system・ツール一覧・履歴中間を毎ターン変える変更はキャッシュを全壊させる。

### 2.2 grammar によるツール呼び出し構造保証（a80e26c）

- `--jinja` のネイティブ tools 対応（lazy grammar: 思考は自由、ツール呼び出し JSON だけ構造保証）を活用。
- 壊れたツール呼び出し検知からの再試行時に `tool_choice="required"` を1回だけ強制（`state.force_tool_choice`、使用後消費）。
- コミュニティ実証: grammar/guardrail 層は小型モデルの多段エージェント成功率を 53%→99% に引き上げた報告（Forge）、NVIDIA は bash 生成 62.5%→75.2%。**同一重みでバックエンド差が 7% vs 83%** という報告もあり、サービング層の正しさはプロンプト工夫より効く。

### 2.3 verify 高速ゲートの常時実行（a80e26c）

- py_compile → import 解決 → ruff の決定的チェック（LLM 不使用・コストほぼゼロ）を、.py への破壊的編集直後に常時実行。
- LLM を使う自動修正ループと pytest は opt-in のまま（「決定的チェックは常時、LLM コストは opt-in」の境界）。

### 2.4 自己進化基盤: eval スイート + 教訓ストア（8faa49b, 6c76435）

- **evals/**: 実 LLM で run_graph を回す 11 タスク + 12 種の決定的チェッカー + `--compare`。ハーネス変更の効果を数字で比較する適応度関数。
- **教訓ストア**: 失敗信号（fast gate 検出・ガードレール発火・異常終了）があったターンのみ reflection（json_schema 強制）で教訓に蒸留 → `.pixie_notes/lessons.json` に蓄積（Jaccard 重複統合・GC 上限50）→ 関連タスクで動的 suffix から注入。成功ターンは追加コストゼロ。
- 実装知見: reasoning 系モデルは思考でトークンを消費するため reflection の max_tokens 512 では JSON 到達前に打ち切られる → 1024 に（LESSONS_REFLECT_MAX_TOKENS）。

### 2.5 分岐点限定 lazy best-of-2（ba5dda8）

- prefix cache により再サンプルのコストが decode のみになった前提を活用。
- **編集シャドウ検証**: 破壊的編集を実ファイル無変更で適用計算（shadow_apply）→ py_compile+ruff 検証（shadow_gate）→ 失敗時のみ失敗理由を一時フィードバックにして最大1回再サンプル（temp +0.15, tool_choice=required）。
- **final answer**: 完全性スコアが「閾値は超えたがギリギリ」帯のときのみ2本目を生成し比較。
- 通常パス（クリーン/高スコア）は追加 LLM 呼び出しゼロ（lazy 原則）。

### 2.6 診断事例: 「Prefill 22.5秒」の真相（33fca46, 717fda1）

- 報告: cache-friendly 化後も実運用で prefill 22.5 秒 → **診断結果: cache は 89.8% ヒットで健在。実 prefill 3.4 秒 + 思考 18 秒の表示誤帰属**。
- 原因: フェーズ判定が `delta.content` しか見ておらず、reasoning_content ストリーム中ずっと「Prefill」表示。副作用として /think 表示・90秒思考タイムアウト・思考引き継ぎも全部不通だった。
- 修正: reasoning_content をフェーズ判定・表示・タイムアウト・thinking_notes に統合（履歴には含めない）。cache ヒット率の常時可視化（`✅ Prefill: 3.0s (cache 10320/10860 tok, 95%)`）。
- **教訓**: 「遅い」と感じたらまず timings（cache_n/prompt_n）を実測してから原因を判断する。
- 予防修正: thinking_mode による system 切替を廃止し完全固定化（振動時の全キャッシュ崩壊リスクを根絶）。ガードレール注入文をユーザー入力と誤認する判定バグも修正。

### 2.7 機能拡張: ツールパック機構 + manga パック（e2044dc, 335f715, d83117d）

- 用途特化ツールをセッション単位で有効化する「パック」機構（prefix cache 保護と両立）。パック未有効時はコアのみ・登録順を明示列挙し、実装前とツール一覧のバイト列一致を保証。
- manga パック: 「決定的処理はツール（scan/rename/undo、dry_run 既定・manifest 可逆化）・判断は LLM・適用は承認後」の分離。日本語 zip の cp437/cp932 補正。表紙 Vision 委譲（delegate 優先、json_schema 強制、空応答フォールバック）。

---

## Part 3. コミュニティ調査の知見（2026-07-06、Reddit/HN/GitHub/論文）

### 3.1 現行実装が裏付けられたもの

prefix cache 必須・grammar 制約・エラー分類リトライ（盲目リトライは無限ループの筆頭アンチパターン）・簡潔スキーマ・決定的検証によるモデル補完 — いずれもコミュニティ最重要プラクティスと一致。

### 3.2 新たに得た知見（未実装）

1. **サンプリング**: gemma-4 公式推奨は `temperature=1.0, top_p=0.95, top_k=64`（全ユースケース共通、Google がこの温度で RL 調整）。現行は temp=0.7 のみ指定でサーバーデフォルト top_k=40 が暗黙に効いておりズレている。フェーズ別温度（thinking 高温/出力低温）に確立プラクティスはないが、grammar ターンのみ低温（0.2〜0.3）の二相プロファイルは有望。
2. **context rot**: 公称コンテキストの半分以下で運用が最大合意事項。劣化はコンテキスト長だけでなく**自己条件付け**（自分の過去出力のエラーが将来を汚染）で、モデルスケールで解決しない。20+ ツール呼び出しのセッションが危険域。
3. **プロンプト言語**: 小型モデルでは日本語指示のフォーマット遵守が顕著に劣化（MGSM: EN 24.4% vs JA 12.8%）。ベストプラクティスは「システム/ツール指示は英語、ユーザー向け出力は日本語」。reasoning を日本語に強制すると精度が下がるモデルもある。
4. **few-shot**: ツール呼び出しには +21.5% の報告があるが、**冗長な例はモデルが例文を模倣して「呼んだふり」をする逆効果**。入れるなら最小・スキーマ形のみ。
5. **SLM-first / escalate-on-failure**: 失敗シグナル（スキーマ違反・パース失敗）が出たターンだけ大モデルに委譲するのが 2026 年の主流パターン。80〜90% のステップはローカルで足りる。
6. **小型モデルの署名的失敗**: テストが失敗すると実装でなくアサーションを「修正」する（root-cause でなく self-serving fix）。過剰な並列ツール呼び出しで逐次依存を壊す（→ 直列実行 or 依存チェックで対処、リトライでは直らない）。
7. **フロンティアとの残ギャップ**（ハーネスで埋まらない部分）: 長期一貫性・自己較正（知らないことを知る）・根本原因デバッグ。平均性能差は約4ヶ月まで縮小したが、この3点に集中して残る。

---

## Part 4. 未実装の改善ロードマップ（優先度順）

| # | 施策 | 工数 | 検証方法 |
|---|---|---|---|
| 1 | **サンプリング公式準拠** — 通常ターン temp 1.0/top_p 0.95/top_k 64 を明示、grammar ターンは低温 0.2〜0.3 の二相プロファイル。repeat_penalty は無効維持 | 小（config + リクエスト body） | eval で 0.7 vs 1.0 A/B |
| 2 | **コンテキスト実用上限 32k〜48k** — トリム発動基準を実効上限ベースに変更。長セッションの要約リセット促進、max_tool_calls の引き下げ検討 | 中 | eval + 長セッション実測 |
| 3 | **システムプロンプト英語化の A/B** — 行動プロンプト英語版を作成し eval で比較。採用時も日本語出力は維持 | 中（golden 更新含む） | eval A/B |
| 4 | **モデルエスカレーション** — failure_signals 発生ターンのみ delegate/クラウドに委譲（既存 delegate 機構に乗る） | 中 | eval + 実運用 |
| 5 | **最小 few-shot** — 失敗実測のあるツールに限定して1例 | 小 | 教訓ストアの失敗信号推移 |
| 6 | **量子化ファイル確認** — 現 GGUF が imatrix 較正版か確認、必要なら差し替え。QAT Q4_0 との A/B も選択肢 | 小 | eval |
| 7 | **KV q8_0 + FA**（VRAM が必要になったら） — q4_0 は使わない | 小 | perplexity + eval |

長期（基盤は既に仕込み済み）: 軌跡ロギング → 小型サイドカーモデル SFT（16GB で唯一現実的な FT 経路）、web ツールパック（設計済み: docs/design/toolpacks.md §2）。

---

## 付録A. 運用チェックリスト

- llama-server は `--swa-full --jinja` + 2026年4月末以降のビルドで起動する
- 「遅い」と感じたら Prefill 表示の cache % を見る（2ターン目以降 90% 前後が正常）
- ツール呼び出しが壊れたら、量子化を疑う前に: chat template / grammar fail-open / スキーマ内正規表現ショートハンド / コンテキスト切り詰め を確認
- system プロンプト・ツール一覧を毎ターン変える変更は書かない（動的情報は動的 suffix へ）
- 新機能の効果は evals/runner.py --compare で数字で判断する

## 付録B. 主要コミット

| コミット | 内容 |
|---|---|
| f909002 | clean_schema（ツール定義23%軽量化）+ sorted 決定論化 |
| 2a48600 | prefix cache 対応 — 動的コンテキストを動的 suffix へ再配置 |
| a80e26c | grammar ツール呼び出し保証 + verify 高速ゲート常時実行 |
| 8faa49b | ローカル eval スイート + golden fixture 修復 |
| 6c76435 | 教訓ストア（経験メモリ） |
| ba5dda8 | 分岐点限定 lazy best-of-2 |
| 33fca46 | cache ヒット率の常時可視化 |
| 717fda1 | reasoning_content 対応 + system 完全固定化 |
| e2044dc / 335f715 / d83117d | ツールパック設計 / manga パック / 表紙 Vision 委譲 |

## 付録C. 主要出典

- llama.cpp: PR #22288（swa-full cache 修正）/ PR #13194（SWA cache 設計）/ issue #21468, #19794, #19051, #22314, #20345 / tools/server/README.md / docs/function-calling.md
- Gemma 4 model card（公式サンプリング推奨・MRCR 実測）: https://ai.google.dev/gemma/docs/core/model_card_4
- 量子化: arXiv 2601.14277（IFEval の量子化段差）/ ACBench arXiv 2505.19433 / llama.cpp discussion #5962（ブラインドテスト）
- context rot: HN id=44564248 / RULER arXiv 2404.06654 / arXiv 2509.09677（自己条件付け劣化）
- 実践知: blog.alexewerlof.com/p/local-llms-for-agentic-coding（同クラスモデルでの運用・「6倍」）/ NVIDIA grammar-constrained decoding / RouteLLM（カスケード）
- 多言語: arXiv 2505.15229（日本語指示遵守の劣化）
