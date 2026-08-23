# AnythingWithPixie 高速化・設計レビュー依頼

## 依頼

AnythingWithPixie（以下 AWP）のローカルLLMコーディングエージェントについて、次に実装すべき改善を判断してください。

単なる賛否ではなく、以下を提示してください。

1. 改善候補の優先順位
2. 各候補の期待効果、品質リスク、実装コスト
3. 採用・不採用・条件付き採用の判断
4. 推奨アーキテクチャと段階的な実装順序
5. 必要な評価方法と停止条件
6. 見落としている代替案

## 環境

- OS: Windows
- Hardware: AMD Strix Halo
- Backend: llama.cpp / HIP
- Model: `gemma-4-26B-A4B-it-ultra-uncensored-heretic.Q5_K_S.gguf`
- Context: 81,920 tokens
- Server slots: 1
- API: OpenAI互換 `/v1/chat/completions`
- 主用途: ローカルでのコード編集、複数ソース横断編集、文書横断編集

## 比較対象

- AnythingWithPixie
- Aider 0.86.2
- Goose 1.47.0

比較は次の2トラックへ分離済みです。

### Provided track

対象ファイルを開始時に提供し、純粋な編集能力を測ります。

- Aider: edit filesとして登録
- AWP: 版付きWorksetとして登録
- Goose: ファイル名と内容をプロンプトへ追加

### Discovery track

依頼文と同一Git repositoryだけを与え、対象ファイルの探索を含めて測ります。

- Aider repo map: 1,024 tokens
- AWP: 既存の検索・コード索引ツール
- Goose: 内蔵探索ツール

両トラックの結果は混ぜません。

## 実測結果

### 単一ファイル境界修正

| Agent | Result | Duration |
|---|---:|---:|
| Aider | PASS | 53.48秒 |
| Goose | PASS | 115.92秒 |
| AWP（改善前） | PASS | 276.99秒 |
| AWP（単純編集高速化後） | PASS | 69.21秒 |

AWPは、編集後のpytest成功を検出して最終回答へ直行することで約75%短縮しました。この早期終了は誤終了を避けるため「編集操作が1回だけ」の場合に限定しています。

### 複数ソース横断改名

| Agent / Track | Result | Duration | Notes |
|---|---:|---:|---|
| Aider / provided | PASS | 27.53秒 | 1応答で2ファイルを編集 |
| AWP / discovery | PASS | 82.64秒 | 逐次探索・編集 |
| AWP / provided（Workset前） | PASS | 178.94秒 | 提供内容を再探索 |
| AWP / provided（版付きWorkset後） | PASS | 147.36秒 | 9ツール→6ツール |
| Goose / discovery相当 | FAIL | 26.32秒 | action上限で不完全編集 |

AWPの版付きWorkset導入後のツール列は次の通りです。

1. `names.py` を編集
2. 旧名をgrep
3. `app.py` を再読
4. `app.py` を編集
5. pytest
6. update_state
7. 最終回答

初回readは削減できましたが、複数ファイルを1つずつ編集するLLM往復が残っています。

### 文書横断・バージョン計算

| Agent | Result | Duration |
|---|---:|---:|
| Aider | PASS | 93.08秒 |
| Goose | PASS | 32.80秒 |
| AWP | FAIL | 104.0秒 |

AWPは `2.7.9` のpatchを1増やす指示に対して `2.8.0` を書き込みました。モデル自身は `2.7.10` が厳密なpatch incrementだと生成中に認識していましたが、誤った値を採用しました。

`2.7.9 → 2.7.10` をシステムプロンプトへ固定する対症療法は、一度試した後に撤回しました。タスク固有知識ではなく、指示からポストコンディションを抽出して結果を検証する汎用方式が必要と考えています。

## 実装済みの改善

### 1. 版付き作業ファイルWorkset

作業ファイルを以下で管理します。

```text
path
revision
content_hash (SHA-256)
disk_hash
latest content
```

性質:

- ファイル全文は通常のチャット履歴へ保存しない
- LLM送信直前の動的suffixへ最新版だけを注入
- AWPによる編集後は旧内容を破棄しrevisionを更新
- 外部変更はdisk hashで検出して最新版へ置換
- 読込後に外部変更された場合は古い版への編集を拒否
- 大規模ファイルは注入量を制限
- `Engine.set_working_files([...])` APIを追加

つまり「会話履歴は追記型、ファイルコンテキストは最新版への置換型」です。

### 2. 決定論的grep件数

`grep_search` がauthoritativeな総件数を返します。以前失敗した件数タスクを3回再実行し、3/3 PASSでした。

### 3. ツール証拠ガード

実ファイルやコマンド結果が必要な依頼で、ツール未実行のまま断定回答することを防止しています。

### 4. 単純編集の高速終了

次の条件を満たす場合、pytest成功後の追加検索を止めます。

- 編集操作がちょうど1回
- テスト系コマンドである
- コマンドが成功している

複数編集では途中終了を避けるため発火しません。

## 現在の主要ボトルネック

ファイルI/Oやprefillではなく、ツール間のLLM decodeと往復回数です。

横断改名でAWPは複数回のLLM生成を行います。一方Aiderは、提供された全ファイルから1回の生成で複数ファイル分のdiffを出します。

AWPのprefillは多くのターンで約0.4～0.9秒です。したがってread_file自体を高速化するより、LLM呼出回数を減らす方が重要です。

## 改善候補

### 候補A: 複数ファイルChangeSetの一括生成

既存のChangeSet基盤には次があります。

- preview
- base hash競合検証
- 複数ファイル一括適用
- journal
- revert

提案フロー:

```text
最新版Workset
  ↓
LLMが複数ファイル分のChangeSetを1回で生成
  ↓
Engineがbase hash・構文・操作を検証
  ↓
一括適用
  ↓
pytest/lint
  ↓
最終回答
```

期待:

- 横断改名を6ツールから2～3往復へ削減
- Aiderのwhole editに近い速度

懸念:

- 小型ローカルモデルが大きなJSON ChangeSetを正しく生成できるか
- 一部ファイルだけ不正な場合の再生成単位
- 大きな複数ファイルを同時に渡すコンテキストコスト
- 文書操作とコード操作で適切な表現が異なる可能性

### 候補B: 汎用ポストコンディション

ユーザー指示から、成果物が満たすべき条件を抽出・保持し、編集後に検証します。

例:

```text
旧名が対象ソースに残っていない
新名の定義と呼出が存在する
指定されたファイルは変更しない
patch = old_patch + 1
テストが成功する
```

懸念:

- LLMに条件抽出をさせると追加往復になる
- 誤った条件を決定論的に強制する危険
- 自然言語条件をどの中間表現へ落とすべきか
- コード、設定、文書でchecker体系が異なる

### 候補C: 複数編集向け決定論的終了

以下をEngine側で追跡します。

- Workset内の対象
- 適用済み変更
- 未処理成果物
- 最後の検証結果
- hash競合・外部変更

全成果物が完了して検証成功なら、追加grep/read/update_stateを禁止して最終回答へ進みます。

懸念:

- 「全成果物」の判定には候補Bが必要になる可能性
- テスト成功だけでは、文書や要求漏れを検知できない

### 候補D: WorksetのAST・意味単位縮約

現在は小ファイル全文、大きなファイルは文字数で切り詰めます。これを次へ変更します。

- 小ファイル: 全文
- 中規模コード: import＋関連シンボル
- 大規模コード: outline＋関連シンボル＋変更周辺
- 文書: 関連見出し＋frontmatter＋リンク/requirement
- 編集完了済み: 最新diff要約のみ

懸念:

- 過度な縮約による依存・副作用の見落とし
- 関連度判定の誤り

### 候補E: update_stateのEngine自動化

現在はモデルが状態更新ツールを呼ぶことがあります。ツール結果から次を自動生成します。

```text
edited files
verification status
errors
remaining targets
```

モデルによるupdate_stateは、長期調査や判断を伴う事実だけに限定します。

期待:

- 1回以上のLLM往復を削減

### 候補F: 用途別生成上限

現在の通常生成上限は最大8,192 tokensです。用途別に分ける案です。

| Purpose | Proposed max tokens |
|---|---:|
| ツール選択 | 512～1,024 |
| ChangeSet生成 | 2,048～4,096 |
| 最終回答 | 512～1,024 |
| 深い設計調査 | 現状維持 |

懸念:

- reasoningを大量に使うモデルでは、低い上限が空応答や途中切れを増やす
- モデル別プロファイルが必要

### 候補G: Aider型repo map

既存のAST・コード索引から、以下を1,024 tokens程度に圧縮して探索開始時に渡します。

```text
file path
defined symbols
imports
callers
tests
related documents
```

期待:

- `grep → read → read` の削減

懸念:

- ローカルGemmaではrepo mapがモデルを混乱させる可能性
- Aiderのdiscoveryトラックは同モデルで900秒タイムアウトした

### 候補H: llama.cpp backendとサーバー設定

現在:

- HIP backend
- context 81,920
- parallel 1
- Flash Attention有効
- KV cache Q8
- ngram-simple speculative decoding

Strix HaloではモデルとbuildによってHIP/Vulkanの優劣が変わるため、同一条件のllama-benchとエージェント実走で比較する案です。

ただし、現状の主因はdecode速度だけでなくLLM往復回数です。

## 制約

- コンテキスト81,920は維持したい
- ローカルLLMを前提とする
- タスク固有のルールをシステムプロンプトへ増やさない
- 古いファイル全文を会話履歴へ蓄積しない
- 外部編集を上書きしない
- 複数ファイル変更は復旧可能であること
- 単一ファイル高速化の正確性を退行させない
- AWP、Aider、Gooseで同じモデル・workspace・checkerを使用する

## 判断してほしい論点

### 論点1

次に実装すべきものは、候補A（ChangeSet一括生成）でよいでしょうか。それとも候補B（ポストコンディション）を先に設計すべきでしょうか。

### 論点2

小型ローカルモデルに複数ファイルChangeSetを生成させる場合、最も堅牢な出力形式は何でしょうか。

- JSON Patch
- 独自JSON operations
- unified diff
- SEARCH/REPLACE blocksの配列
- ファイル全文
- その他

既存のbase hash、preview、journal、revertを活用できることを重視してください。

### 論点3

ポストコンディションは誰が生成・検証すべきでしょうか。

- メインLLM
- 別の小型LLM
- Engineの決定論的抽出
- タスク種別ごとのchecker
- LLM抽出＋Engine検証のハイブリッド

### 論点4

複数編集後の自動終了条件を、要求漏れを増やさずにどう定義すべきでしょうか。

### 論点5

提供済みWorksetがある場合、探索ツールをどの程度制限すべきでしょうか。

- 完全禁止
- 提供ファイル内の編集を先に行い、不足時だけ許可
- 常に許可

### 論点6

速度・成功率・不要編集・コンテキスト消費を同時に評価する、適切なスコアリング方法を提案してください。

## 希望する回答形式

```markdown
# 結論

# 優先順位

| Rank | Candidate | Decision | Expected impact | Risk | Cost |
|---:|---|---|---|---|---|

# 推奨アーキテクチャ

# 段階的実装計画

## Phase 1
## Phase 2
## Phase 3

# 評価計画

# 採用しない案と理由

# 追加で確認すべき事項
```

判断は、AWPをAiderに似せること自体ではなく、ローカルLLM環境で速度・正確性・復旧性を最大化する観点で行ってください。
