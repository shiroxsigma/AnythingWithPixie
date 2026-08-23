# Codex 作業指示 — AnythingWithPixie 精度改善

対象リポジトリ: `D:\Workspace\PixieProject\AnythingWithPixie`
目的: eval で測れる形でエージェントの実精度を上げる。**新機能は足さない。**

---

## 0. 環境の前提（先に読むこと）

- Windows 11 / Python 3.13。実行は `.venv\Scripts\python.exe` を使う。
- **pipenv は入っていない。** `Pipfile` があるが pipenv を使おうとしないこと。`.venv` を直接使う。
- `pytest` には**環境起因の失敗が3件あり、これはベースライン**。修正対象ではない。
  着手前に必ずベースラインを取得し、作業後に**件数が増えていないこと**だけを確認する。
  ```
  .venv\Scripts\python.exe -m pytest -q
  ```
- `.bat` を書く場合は CRLF + ASCII、`.ps1` は BOM付き UTF-8。`.md` / `.py` は UTF-8 (BOMなし)。
- 本番バックエンド: raw llama-server `http://192.168.0.200:8080/v1`
  （gemma-4-26b-a4b, MoE 26B/A4B, Q5_K_S, `--swa-full --jinja`, RTX 5070 Ti 16GB）
  LM Studio (`:1234`) は低レベル制御ができないため**計測には使わない**。

---

## 1. 絶対に壊してはいけない不変条件（最重要）

このプロジェクトの性能は **prefix cache 98.8%ヒット / prefill 約19倍** に依存している。
ここを壊すと他の改善が全部帳消しになる。**変更前に必ず該当コードのコメントを読むこと。**

1. **system メッセージは静的に保つ。**
   `build_system_text`（`src/pixie_core/engine.py:804`）の出力に、ターンごとに変わる情報を
   絶対に入れない。state_board・ホワイトボード要約・JIT推奨ヒント・budget_hint・deep_hint は
   すべて system の**外**。

2. **動的コンテキストは末尾にのみ付与する。**
   `_build_dynamic_suffix`（`engine.py:1691` 付近）で1ブロックにまとめ、
   `messages_for_llm` の**最後の要素**にだけ付ける。設計意図は `engine.py:1674-1688` のコメント参照。

3. **suffix を履歴に書き戻さない。**
   `state.chat_history.messages` には絶対に書き込まない。送信直前の一時コピーにのみ適用する。

4. **ツール定義の集合をターンの途中で変えない。**
   変更はターン境界のみ（`engine.py:2035` 付近および `_api.py:542` の契約）。

5. 会話履歴の**前方**（古い側）を書き換える処理を新規に追加しない。
   既存の `mask_old_observations` / ハードトリムはこの作業では触らない。

---

## 2. タスクA — 死んだ eval タスクの除去（最優先・所要5分）

### 問題
`evals/tasks/11_manga_rename.json` が **8/8 FAIL、ツール呼び出し0回、4.07〜4.10秒でほぼ固定**。
このタスクは `"toolpacks": ["manga"]` / `"task_mode": "manga"` を要求するが、
manga モードはコミット `f27391a`（2026-08-14）で **TalkWithPixie へ移管済み**。
AWP 側には該当 toolpack が存在せず、構造的に永久 FAIL になっている。

### 影響
最新 eval（`evals/results/eval_20260709_004257.md`）の表示スコア 70/88 = 79.5% のうち、
8タスク分がこの dead test。**実スコアは 70/80 = 87.5%。**
常時真っ赤な1タスクがあることで、他の実 fail が見過ごされる状態になっている。

### やること
`evals/tasks/11_manga_rename.json` を AWP の eval スイートから外す。
削除でよいが、TalkWithPixie 側に同等の eval 基盤があるならそちらへ移設する方が望ましい。
`evals/workspaces/11_manga_rename` および `evals/checkers.py` の `manga_zip_cleanup` チェッカーが
他から参照されていないことを確認してから、未使用なら一緒に整理する。

### 受け入れ基準
- `--dry-run` で設定検証が通る:
  `.venv\Scripts\python.exe evals\runner.py --dry-run`
- 残りのタスクが10本になり、manga 由来の参照エラーが出ない。

---

## 3. タスクB — ツール実行0回での最終回答を遮断（本命）

### 問題
最新 eval の**実 fail 10件のうち4件**が「ツールを1回も呼ばずに最終回答を出した」ケース。

| タスク | 内容 | 症状 |
|---|---|---|
| `10_multistep` (hard) | 「まず version.py を調べて…」と明示 | 3件が**0ツール呼び出し**で即答（1.43s / 4.09s / 22.18s） |
| `06_grep_count` (easy) | 出現回数を数える | 1件が0ツール呼び出し |

1.43秒で「hard」タスクに答えているのは、ファイルを読まずに内容を捏造しているということ。

### 原因の所在
`_is_simple_direct_answer_sufficient`（`engine.py:477`）は
`used_get_cwd` / `used_list_dir` / `used_read` で**対応ツールの実行済みを要求しており、正しく機能している**。
すり抜けているのは**一般の final_answer 経路**の方。

該当箇所: `engine.py:3559-3560`
```python
# --- 通常の最終回答 ---
state.exit_reason = f"final_answer (ツール実行 {state.tool_call_count}回後)"
```

### 実装方針
上記 `# --- 通常の最終回答 ---` の**直前**にガードを1つ挿入する。

条件のイメージ:
`state.tool_call_count == 0` かつ「その回答が実行なしには出せない事実主張を含む」場合、
final_answer を確定させず、ツール実行を促すガードレールを注入して ReAct ループを継続する。

既存のガードレール注入パターン（`state.phase = "PLANNING"` / `state.guardrail_cooldown` /
`continue` する分岐が `engine.py:3535` 付近にある）に**揃えること**。新しい仕組みを作らない。

再発防止のため、注入は**1ターンにつき最大1回**とし、
2回目以降は通す（無限ループ防止）。`state` に専用カウンタを1つ足してよい。

### 最重要の注意 — 誤爆させないこと
**0ツール呼び出しが正当なケースが存在する。** 単純に全部ブロックすると退行する。

- `08_ambiguous_default`（**現在 8/8 PASS**）は曖昧な指示に対する聞き返しで、
  ツール0回が正解の可能性が高い。**このタスクを絶対に壊さないこと。**
- 挨拶・雑談・ユーザーへの確認質問・実行拒否も 0回が正当。

したがって判定は「0回だからブロック」ではなく、
**「0回なのに、ファイルパス・行番号・コード片・バージョン番号など、読まなければ書けない具体的事実を主張している」**
という形にする。回答が質問文で終わっている場合は通す、などの除外を必ず入れる。

判定ロジックは `_is_simple_direct_answer_sufficient` と同じファイル内に
`_requires_tool_evidence(user_text, answer, state) -> bool` のような純粋関数として切り出し、
**単体テストを付けること**（`tests/` に追加）。最低限このケースを網羅する:

| 入力 | 期待 |
|---|---|
| 0回 + 「version.py のバージョンは 1.2.3 です」 | ブロック（事実主張） |
| 0回 + 「どのファイルを対象にしますか？」 | 通す（聞き返し） |
| 0回 + 「こんにちは」 | 通す |
| 1回以上 + 任意 | 通す（このガードの対象外） |

### 受け入れ基準
1. 追加した単体テストが通る。
2. 既存 pytest の失敗が**3件（環境起因ベースライン）から増えていない**。
3. eval で以下を満たす（**本番 `:8080` で計測**）:
   - `08_ambiguous_default` が **8/8 PASS を維持**（最重要）
   - `10_multistep` の 0ツール呼び出し fail が**減っている**
   - 全体スコアが タスクA 適用後のベースラインを**下回らない**

   ```
   .venv\Scripts\python.exe evals\runner.py --base-url http://192.168.0.200:8080/v1 ^
       --task 08_ambiguous_default --repeat 8
   .venv\Scripts\python.exe evals\runner.py --base-url http://192.168.0.200:8080/v1 ^
       --task 10_multistep --repeat 8
   ```

---

## 4. タスクC — 本番構成での再ベースライン取得

### 問題
最新 eval は **2026-07-09**。その後 **2026-08-14 に6コミット**入っており、
うち3件が編集経路・コンテキスト構築という最も危険な領域を変更している:

```
3dc5281 機能追加: 文書ChangeSetの編集と整合性検査を追加
3a134df 機能追加: コード・文書索引でWorksetを高度化
b668946 機能追加: ジャーナル付き複数ファイルChangeSetを追加
```

これらは**一度も eval で測られていない**。
さらに最新 eval の Base URL は `:1234`（LM Studio）で、本番の raw llama-server 構成ではない。

### やること
タスクA・B の適用後、**本番 `:8080` でフルスイートを1回**実行して現在地を確定させる。

```
.venv\Scripts\python.exe evals\runner.py --base-url http://192.168.0.200:8080/v1 --repeat 8
```

所要は前回実績から約110分。`--compare` で `eval_20260709_004257.json` と比較した差分も出す。

### 報告に含めること
- タスク別 PASS/FAIL/TIMEOUT 表
- `:1234` 時代との差分（モデルは同じなので、差はほぼ構成とコード変更に由来する）
- 新規に fail しているタスクがあれば、ChangeSet / Workset 由来かどうかの当たり

---

## 5. やらないこと（スコープ外）

- **新機能の追加**。このフェーズは現在地の確定と回帰潰しに限る。
- ChangeSet / Workset のリファクタリング。まず測る。
- `MAX_TOKENS` / トリム閾値 / `mask_old_observations` の変更。別タスクとして分離済み。
- prefix cache 設計（セクション1）への一切の変更。
- サンプリング設定の変更。`config.py:129` の `gemma: {temperature 1.0, top_k 64, top_p 0.95}` は
  gemma-4 公式推奨に一致しており**正しい**。触らない。
- FreeToken 等、推論エンジンの差し替え検討。

---

## 6. 完了報告のフォーマット

1. 各タスクの実施内容と変更ファイル一覧（diff の要約）
2. pytest ベースライン件数（前 / 後）
3. eval スコア（タスクA前 / タスクA後 / タスクB後）を本番 `:8080` で
4. `08_ambiguous_default` が 8/8 を維持している証跡
5. 想定外だった点・判断が必要だった点
