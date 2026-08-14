# AnythingWithPixie 性能改善計画 — 小型モデルと最新Web情報の効率利用

> 現在の構成ではMangaモードをTalkWithPixieへ移管済み。以下のManga記述は当時の設計判断を残した履歴であり、AWPの現行仕様ではない。

調査日: 2026-08-13  
対象: `feat/pixie-core-facade`（pixie_core API 1.6）

## 結論

最も効果が大きいのは、単純にモデルを大型化することではない。**4B級モデルに一度に見せるツールを6〜8個へ絞り、定型処理をホスト側へ移し、失敗時だけ上位モデルへ委譲する**構成がよい。

推奨する完成形は次の通り。

```text
ユーザー入力
  → 決定的ルーター（normal / code-read / code-edit / web / manga）
  → 4B級メインモデル（選択された6〜8ツールのみ）
  → ホスト側の型検証・実行・高速ゲート
  → 成功: 4B級モデルが短く統合回答
  → 失敗: 1回だけ再試行、なお失敗なら8B〜14Bまたはdelegateへ昇格

Web調査時
  → web_research（検索・上位ページ取得を内部で並列実行）
  → 出典付き「証拠パック」だけを隔離サブエージェントへ渡す
  → メインモデルは証拠パックから見解を作る
```

優先順位は以下である。

1. 計測を整備し、ツールプロファイルを細分化する。
2. 生成上限を用途別に縮め、non-thinkingを既定にする。
3. 検索と取得をまとめた安全な`web`ツールパックを実装する。
4. 4B級モデル3候補を同一evalで比較する。
5. 軌跡が十分に貯まってから、ツール選択と引数生成だけをSFTする。

## 現状評価

### すでに良い点

- 静的prefixと動的suffixを分離し、KV/prompt cacheを壊しにくくしている（[engine.py](../../src/pixie_core/engine.py#L1676)）。
- 読み取り専用ツールを最大5件まで並列化している（[config.py](../../src/pixie_core/config.py#L71)）。
- native tool call、grammar、テキストからの救済パーサーを持つ。
- `py_compile`、import、ruff、任意pytestによる決定的検証がある。
- trajectory、eval、SFT exportがすでにあり、改善サイクルの土台は強い。
- ツールパックがセッション単位で固定されるため、機能追加とprefix cacheを両立できる。

### 主なボトルネック

| 問題 | 現状 | 影響 |
|---|---|---|
| 生成予算が大きすぎる | `MAX_TOKENS=8192`、deepではさらに最大2倍（[config.py](../../src/pixie_core/config.py#L18)） | 小型モデルが冗長化・ループしやすく、最悪待ち時間が長い |
| コンテキストを常時32K確保 | `N_CTX=32768`（[config.py](../../src/pixie_core/config.py#L15)） | KV cacheのメモリ消費が増え、GPU余裕が減る |
| ツール候補が多い | codeモードは19ツール。normalもコア全件を送信し、JITは候補削減ではなくヒントのみ（[engine.py](../../src/pixie_core/engine.py#L2034)） | 4B級では誤選択、引数混同、schema token増加が起きやすい |
| JITが語彙一致中心 | `score_tools()`はツール名・説明の部分一致が中心（[tools.py](../../src/pixie_core/tools.py#L1406)） | 日本語の言い換えや複合意図に弱い |
| ローカル推論設定が固定 | llama-cpp-pythonの`n_threads=4`が固定（[llm_client.py](../../src/pixie_core/llm_client.py#L256)） | CPU/GPU構成ごとの最適化余地を使えていない |
| Web検索が未実装 | `mcp_client.py`は本体ループ未接続。設計済みwebパックは既知URLのMarkdown化が中心 | 「最新情報」を自律的に検索・比較・引用できない |
| 評価指標が不足 | 成否・時間・tool回数・trajectoryはあるが、TTFT、decode tok/s、schema token、無効call率の集計がない | 速度改善と品質悪化を切り分けにくい |

なお、現環境のテストは `434 passed / 5 failed / 1 skipped` だった。失敗は日本語コンソール復号、ruff検出、プロジェクト外`.venv`探索に集中している。性能比較の前にこの5件を環境非依存にし、ベースラインをクリーンにするべきである。

## 1. 小型モデル向けエージェント構造

### 1.1 二段階ツールルーティング

今のJIT推薦を「ヒント」から「有限プロファイル選択」へ進める。

推奨プロファイル例:

| プロファイル | 最大ツール数 | 主なツール |
|---|---:|---|
| `direct` | 0 | 通常回答のみ |
| `code_read` | 6 | tree、grep、outline、symbol、read、project_info |
| `code_edit` | 8 | read、grep、symbol、search_replace、write、command、verify相当 |
| `web` | 4 | web_research、web_open、read、update_state |
| `ops` | 6 | command、async、poll、read、grep、state |
| `manga` | 既存8 | 現行セットを維持 |

ルーターはまず規則ベースにする。現在の`score_tools()`、スラッシュコマンド、ファイル拡張子、語彙辞書を使えば追加LLM呼び出しは不要である。曖昧な場合だけ、小型モデルへ次のような固定enumを返させる。

```json
{"profile":"code_read","confidence":"high"}
```

毎ターン異なるツール集合にするとprefix cacheが崩れるため、**任意の上位N件ではなく、5〜6個の固定プロファイル**から選ぶ。プロファイル内のschema順も固定する。

### 1.2 複合ツールでReAct往復を減らす

小型モデルには細かいプリミティブを何度も選ばせるより、頻出シーケンスを1ツールにした方が安定する。

- `inspect_symbol_context(path, symbol)`: outline + symbol本体 + caller候補
- `edit_and_check(path, search, replace)`: shadow apply + edit + fast gate
- `web_research(query, freshness, domains, max_sources)`: search + 上位取得 + 証拠パック生成
- `project_snapshot(path)`: tree + 設定 + entry point + testコマンド

ただし、破壊的操作の承認境界は維持する。複合ツール内部で編集内容と検証結果を分けて返し、承認UIには最終差分を表示する。

### 1.3 モデルではなくホストへ移す判断

次はLLMに再判断させない。

- schema、型、必須引数、path scopeの検証
- 同一tool callの重複排除
- `py_compile` / ruff / pytestの実行順
- 検索結果の重複URL除去、domain集約、日付順整列
- 引用番号とURLの対応
- 再試行回数と上位モデルへの昇格条件

モデルには「何を調べるか」「どの編集を意図するか」「証拠から何が言えるか」だけを担当させる。

### 1.4 昇格条件を決定的にする

自己申告confidenceだけに依存せず、次のどれかで上位モデルへ昇格する。

- tool callの構文・schema違反が2回
- 同一callまたは同一エラーが2回
- fast gate失敗後の修正が1回で直らない
- 必要ツールが選択プロファイル外
- Webの主要主張に独立した2ソースがない
- evalで「4Bが苦手」と分類されたタスク群

これにより、日常的な読み取り・検索・単純編集は4Bで処理し、難しい設計や多段デバッグだけを8B〜14Bへ送れる。

## 2. 生成量とコンテキストの削減

### 推奨初期値

| 用途 | 現状 | 推奨開始値 |
|---|---:|---:|
| tool action turn | 最大8192 | 384〜768 |
| 通常最終回答 | 最大8192 | 1024〜2048 |
| code patch | 最大8192 | 2048〜4096 |
| delegate | 2048 | 768〜1536 |
| review | 1024 | 512〜1024 |
| deep thinking | 最大90秒 | 明示指定または昇格時のみ、30〜45秒 |
| 通常context | 32768 | 8192または16384 |

`node_plan()`に`purpose`別budgetを導入し、tool callが期待されるターンでは短く打ち切る。Qwen3系はthinking/non-thinkingを切り替えられ、Qwen3-4Bの公式カードも効率的な一般対話にはnon-thinking、複雑な推論にはthinkingという使い分けを示している。[Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B)

32Kは削除せず`long_context`プロファイルとして残す。通常は8K/16K、ユーザーが長文解析を要求した時だけ32Kへ切り替える。長いツール結果は先頭文字数で切るだけでなく、`path / symbol / line / fact / error`の構造化要約へ変換する。

## 3. 小型モデル候補

モデル名だけで決めず、このプロジェクトの11 eval + 追加するtool-selection evalでA/Bする。

| 候補 | 位置づけ | 理由・注意 |
|---|---|---|
| **Qwen3.5-4B** | 第一候補 | 公式カードにtool parser `qwen3_coder`とMTP設定が明記されている。現行コードにもQwen3.5系のtool-call救済処理がある。[公式カード](https://huggingface.co/Qwen/Qwen3.5-4B) |
| **Qwen3-4B** | 安定比較候補 | 4B、native 32K、thinking切替、agent/tool能力を公式に掲げる。日本語を含む多言語用途にも向く。[公式カード](https://huggingface.co/Qwen/Qwen3-4B) |
| **Phi-4-mini-instruct** | 別系列の比較候補 | 3.8Bでfunction calling向けpost-trainingがあり、メモリ・レイテンシ制約用途を想定。ただし専用tool形式と日本語evalは必須。[公式カード](https://huggingface.co/microsoft/Phi-4-mini-instruct) |
| **現行LFM系** | 回帰基準 | 専用パーサーが実装済みなので、変更効果とモデル差を切り分ける基準として残す |

LM Studioでは「モデル側chat template」と「サーバー側parser」の両方がnative tool useに対応している方が一般に良いと説明されている。[LM Studio Tool Use](https://www.lmstudio.ai/docs/developer/openai-compat/tools) したがって、モデル名だけでなく以下を起動時に診断・記録する。

- native/default tool support
- 実際に適用されたchat template hash
- parser名
- GGUF量子化方式
- context、GPU offload、KV設定

量子化はQ4_K_Mを速度・容量基準、Q5_K_Mを品質基準として同じevalで比較する。ツール名・JSON・パスの1文字誤りが成否へ直結するため、perplexityより`exact tool name / argument validity / task success`を優先する。

## 4. 最新Web情報の取得設計

既存の[toolpacks設計](../design/toolpacks.md)にあるURL→Markdown変換は再利用できるが、最新情報取得には「検索」が不足している。次の2層を追加する。

### 4.1 外部公開するツール

小型モデルへ見せるのは原則2個に留める。

```text
web_research(query, freshness="month", domains=[], max_sources=5)
web_open(url, focus="", max_chars=6000)
```

`web_research`内部では以下を決定的に行う。

1. 検索APIを1回呼ぶ。
2. URLを正規化し、同一domain偏重を除く。
3. 上位2〜4ページを並列取得する。
4. HTMLを本文テキストへ変換する。
5. title、URL、公開日、取得日、抜粋、該当箇所をJSONで返す。
6. `.pixie_notes/web_cache/`へ保存し、回答後も検証可能にする。

検索結果そのものを大量にLLMへ渡さない。推奨上限は5件、1件あたり800〜1200文字、合計6000文字程度である。

### 4.2 検索バックエンド

- **既定・ローカル志向: SearXNG** — HTTP APIで`format=json`を取得でき、自己ホスト可能。JSON形式は設定で有効化が必要で、公開instanceでは無効な場合がある。[SearXNG Search API](https://docs.searxng.org/dev/search_api.html)
- **簡単・品質重視: Tavily** — `time_range`、`start_date`、domain制限、cleaned contentをAPIで扱える。API keyと利用料金が必要。[Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search)

プロバイダーinterfaceを共通化し、最初はTavilyまたは既存SearXNGのどちらか1つだけ実装する。Tavily利用時も`include_answer=false`を既定にし、別LLMの要約ではなく原文抜粋をPixieが統合する。これで出典追跡とモデル比較がしやすい。

### 4.3 最新性と引用

証拠レコードを次の形で固定する。

```json
{
  "source_id": "S1",
  "title": "...",
  "url": "https://...",
  "published_at": "2026-08-10",
  "retrieved_at": "2026-08-13T...+09:00",
  "excerpt": "...",
  "primary_source": true
}
```

回答ルール:

- 「最新」「現在」は必ず検索する。
- 日付が重要な主張は公開日と取得日を保持する。
- 重要な結論は原則2ソース、仕様・リリースは公式一次資料を優先する。
- 見解と出典記載の事実を分離する。
- 最終回答には`[タイトル](URL)`を主張の近くへ置く。

### 4.4 Web内容の安全な隔離

Webページは命令ではなく、常に**信頼できないデータ**として扱う。OWASPは、外部ページに埋め込まれた間接prompt injection、過剰権限、tool outputの再注入、SSRFを主要リスクとして挙げている。[Prompt Injection Prevention](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html)、[MCP Security](https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html)

実装上の必須条件:

- Web取得サブエージェントには読み取りツールしか渡さない。
- HTMLではなく抽出済みの`title/body/url/date`だけを渡す。
- `http/https`のみ許可し、localhost、private/link-local IP、file URI、リダイレクト先を拒否する。
- ページ内の「指示」「tool call」「system prompt風テキスト」は実行しない。
- Web由来の文字列をshell、path、別ツール引数へ直接流さない。
- Web調査後のファイル編集・コマンド実行は別ターンで再承認する。
- 検索queryにも秘密情報、ファイル本文、tokenを含めない。

## 5. 推論速度の改善

### 5.1 設定を外へ出す

`LlamaCppBackend`の以下をconfig化し、自動ベンチ可能にする。

- `n_threads` / `n_threads_batch`
- `n_batch` / `n_ubatch`
- GPU offload
- Flash Attention
- KV cache type
- context size

固定`n_threads=4`はマシン依存性が高い。起動時に短いprefill/decodeベンチを行い、候補設定から選ぶ方式が確実である。

### 5.2 speculative decoding

llama.cppはdraft modelだけでなく、追加モデル不要のn-gram系speculative decodingも提供している。[llama.cpp speculative decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)

- 4Bメイン: まずn-gram方式をA/B。draft modelは効果を仮定しない。
- 8B〜14Bの昇格モデル: 同系列0.5B〜3B draftを試す。
- コード、JSON、定型回答は受理率が高くなりやすいが、必ず実測する。

LM Studioも、小さいdraftが十分速く、予測が一致する時だけ高速化し、組み合わせによっては逆に遅くなると説明している。[LM Studio Speculative Decoding](https://lmstudio.ai/docs/app/advanced/speculative-decoding)

### 5.3 prefix cacheを維持する

現行設計は維持する。変更点は次に限定する。

- profileごとのsystem + schemaをbyte-identicalにする。
- 動的state、Web証拠、JITヒントは最後のuser suffixへ置く。
- schemaの説明文を短くし、順序を固定する。
- `cache_n / (cache_n + prompt_n)`をtrajectoryとeval集計へ入れる。

## 6. 評価設計

BFCLはfunction callingを、単純callだけでなく複数・並列・relevance detectionを含む現実的設定で評価する標準的ベンチマークである。[BFCL論文（ICML 2025）](https://proceedings.mlr.press/v267/patil25a.html) 全BFCLを導入する前に、その分類を既存evalへ取り込む。

追加するeval:

- 呼ぶべきでない質問（relevance detection）
- 似た名前のツールから1つを選ぶ
- 必須・optional・enum・boolean・Windows path
- 2つの独立callを並列化
- 依存するcallを直列化
- 壊れたtool resultから回復
- Webの最新情報を2ソースで引用
- 悪意あるWebページを読んでもshell/editしない

集計指標:

| 分類 | 指標 |
|---|---|
| 品質 | task pass率、tool選択正解率、引数schema合格率、最終回答完全性 |
| 効率 | TTFT、prefill秒、decode tok/s、総wall time、LLM call数、tool call数 |
| token | prompt、schema、tool result、completion token数 |
| cache | cache hit率、profile別hit率 |
| 安全 | 不要な破壊的call率、Web injection阻止率、SSRF阻止率 |
| 安定性 | ループ率、救済parser率、昇格率、p50/p95時間 |

最初の目標値（保証値ではなく開発ゲート）:

- 既存task pass率を下げない。
- schema tokenをnormal比60%以上削減。
- invalid tool call率2%未満。
- 平均LLM call数30%削減。
- p50完了時間30%削減。
- Web回答の引用URL対応率100%、主要主張のsource support率90%以上。

## 7. 実装ロードマップ

### P0 — 1〜2日: ベースラインを信頼できる状態にする

1. 現在の5テスト失敗を環境非依存化する。
2. trajectoryへTTFT、completion tokens、decode tok/s、schema chars、profile、parser rescueを記録する。
3. eval reportへp50/p95とモデル・量子化・template hashを出す。

### P1 — 2〜4日: 最も費用対効果の高い削減

1. `purpose`別token budgetを導入する。
2. 5〜6個の固定tool profileを追加する。
3. normal/codeのschema tokenを計測し、説明文を短縮する。
4. `inspect_symbol_context`と`edit_and_check`を追加する。
5. Qwen3.5-4B / Qwen3-4B / Phi-4-miniを同一条件で比較する。

### P2 — 3〜5日: Web最新情報

1. `toolpacks/web.py`を実装する。
2. `SearchProvider` interfaceとSearXNGまたはTavily backendを1つ実装する。
3. 証拠パック、cache、引用ledgerを追加する。
4. SSRF、prompt injection、秘密情報流出のテストを追加する。

### P3 — 1〜2日: 推論ランタイム調整

1. llama.cpp設定をconfigへ出す。
2. 8K / 16K / 32K contextをA/Bする。
3. Q4_K_M / Q5_K_Mを比較する。
4. n-gram speculativeと、昇格モデル向けdraftを比較する。

### P4 — 軌跡収集後: 小型モデルSFT

既存の`export_sft.py`を活用し、全会話ではなく次を教師化する。

- tool profile選択
- tool / no-tool判定
- 正しい引数JSON
- 失敗後の1回修正
- Web証拠からの引用付き短文統合

成功軌跡だけでなく、失敗call→正解callの対を残す。まずLoRA/QLoRAでsidecar/routerを調整し、メイン4Bの全面fine-tuneはその効果を確認してから行う。

## 最終見解

AnythingWithPixieは、すでにcache、grammar、検証、trajectory、toolpackという重要部品を持っている。したがって全面的な作り直しは不要である。

現状の最大の問題は、**小型モデルに大きな生成余地と多数のツールを与え、モデル自身にワークフロー制御まで任せていること**である。4B級を強くする鍵は、モデルの推論量を増やすことではなく、選択空間を狭め、決定的処理をPythonへ戻し、失敗時だけ計算量を増やすことにある。

最初に実装すべき組み合わせは、次の4点である。

1. Qwen3.5-4Bを第一候補にしたモデルA/B。
2. 固定tool profile + action turn 768 token上限。
3. `web_research`による出典付き証拠パック。
4. schema違反・検証失敗時だけの8B〜14B昇格。

この順なら、既存設計を壊さず、速度・成功率・最新情報対応を同時に改善できる。

## 追補: Qwen3.6 / Gemma 4 / LFMで大規模コンテキストを扱う場合

ユーザーのモデル候補を踏まえると、前節の「4Bを常用」という案は次のように修正する。

### 推奨順位

| 優先 | モデル | 推奨用途 | 判断 |
|---:|---|---|---|
| 1 | **Gemma 4 26B A4B** | 128K〜256Kの資料読解、Web証拠統合、通常エージェント | 長文能力と品質のバランスが最も良い。公式MRCR 128Kは44.1%で、LFMより長文性能の根拠が明確 |
| 2 | **Qwen3.6-35B-A3B** | repository規模のcoding、複雑なtool chain、十分なRAMがある構成 | native 262K、最大1.01Mへ拡張可能。ただし16GB VRAM単体ではweightと大規模KVを同時に載せにくく、CPU offload前提。llama.cppでprefix再処理の報告もある |
| 3 | **LFM2.5-8B-A1B** | 高速router、Web検索サブエージェント、tool selection | 128K対応、active 1Bで高速。ただしreasoning-onlyなので長い思考出力が総tokenを消費しやすい。主モデルよりsidecar向き |
| 4 | **Gemma 4 E4B** | 16GB内で128Kを優先する軽量構成 | weight余裕は大きいが、公式MRCR 128Kは25.4%。「入る」と「長文を正しく使える」は別 |

Gemma 4公式仕様では、E2B/E4Bが128K、12B/26B A4B/31Bが256Kで、26B A4Bは総25.2B・active 3.8Bである。128K MRCR v2は31B 66.4%、26B A4B 44.1%、12B 43.4%、E4B 25.4%だった。[Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)

Qwen3.6-35B-A3Bはnative 262,144 tokenで、公式には最低128Kを保つことが推奨され、YaRNで1,010,000 tokenまで拡張できる。ただし公式の262K serving例は8 GPU構成であり、16GB GPUで同じ条件が現実的という意味ではない。[Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)

LFM2.5-8B-A1Bは32Kから128Kへ追加学習で拡張され、BFCL v4 49.73を報告している。日本語tokenizer効率も旧版より6.9%改善した。一方でreasoning-onlyモデルであるため、短いaction turnを大量に回す本プロジェクトでは思考tokenを抑制できるか実測が必要である。[Liquid AI公式ブログ](https://www.liquid.ai/blog/lfm2-5-8b-a1b)

### 16GB VRAMでの現実的な運用

既存レポートのRTX 5070 Ti 16GBを前提とする場合、**256Kを常時確保する設計にはしない**。モデルweight、KV cache、vision encoder、runtime bufferが同じVRAMを奪い合うためである。

推奨プリセット:

| プリセット | context | 用途 |
|---|---:|---|
| `agent-fast` | 16K〜32K | 通常のtool use、編集、短いWeb検索 |
| `project` | 64K | repository map、複数ファイルの設計判断 |
| `research` | 128K | 長文資料・Web証拠の統合 |
| `archive` | 256K | 明示指定時のみ。速度低下とCPU offloadを許容 |

候補ごとの実用構成:

- **Gemma 4 26B A4B Q4系**: 64Kを標準、128Kを研究モード。16GBでは一部CPU offloadまたはKV量子化が必要になる可能性が高い。
- **Qwen3.6-35B-A3B Q3/Q4系**: 128K以上はsystem RAM 64GB以上と高速RAM帯域を前提に試す。tool callingには`qwen3_coder`相当parserを使い、prefix cache hit率を必ず監視する。
- **LFM2.5-8B-A1B Q4系**: 128Kを最も載せやすい候補。router / retriever / delegateとして常駐させ、GemmaまたはQwenへ圧縮証拠を渡す構成がよい。
- **Gemma 4 E4B**: 128KをGPU内へ収めることを優先する場合の候補。ただし長文検索精度は26B A4Bより下がるため、retrievalで関連箇所を先に絞る。

### 大規模contextでも全文投入を既定にしない

context windowは作業メモリの上限であり、検索精度の保証ではない。推奨パイプラインは次の通り。

```text
全資料・repository・Web cache
  → BM25/rg + 350M級retrieverで候補抽出
  → LFM2.5-8B-A1Bまたは小型モデルで重複除去・圧縮
  → 重要箇所、source、line、dateを64K〜128Kへ配置
  → Gemma 4 26B A4BまたはQwen3.6が判断
  → 詳細不足時だけ原文を追加取得
```

これに合わせて`check_and_trim_context()`を、古いmessageの単純削除から次の三層memoryへ変更する。

1. `working`: 直近tool exchangeと編集中symbol（8K〜16K）
2. `evidence`: source付き抽出結果（最大32K〜96K）
3. `archive`: 原文、全ログ、Web cache。必要時だけretrieval

### この3系列を比較するeval条件

同じ最大contextだけで比較せず、32K / 64K / 128Kの各点で測る。

- needle retrievalではなく、関連する5〜10断片を統合する課題
- 20〜50ファイルから変更対象を選ぶ課題
- 10ソースのWeb結果から、最新かつ一次資料を選ぶ課題
- 30 tool turn後も正しいtool schemaとgoalを維持する課題
- cache hit率、prefill秒、decode tok/s、最大VRAM、system RAM、正答率

最終的な推奨構成は、**LFM2.5-8B-A1Bを高速router/retriever、Gemma 4 26B A4Bを128K主モデル、Qwen3.6-35B-A3Bを難しいcodingへの昇格先**とする三段構成である。単一モデルに絞るなら、長文・tool use・16GB環境の妥協点としてGemma 4 26B A4Bを最初に実測する。
