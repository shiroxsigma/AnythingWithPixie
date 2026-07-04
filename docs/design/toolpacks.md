# 詳細設計: ツールパック機構 + manga / web パック

作成: 2026-07-04 / ステータス: 設計確定・実装前

## 0. 背景と設計原則

用途特化機能（漫画 zip リネーム、web→Markdown）を追加するにあたり、既存の設計資産を壊さないことを最優先とする。

| 原則 | 由来 | 本設計での帰結 |
|---|---|---|
| 依存ゼロ（LM Studio 版は stdlib のみ） | CI で保証済み | web2md の重い依存（playwright 等）は subprocess で分離。manga は stdlib + optional Pillow |
| prefix cache 保護（system・ツール定義はセッション内不変） | コミット 2a48600 | パックの有効/無効は**セッション単位**。ターン中の増減は禁止 |
| タスクモードの先例（/code = 固定ツールセット + 専用ポリシー） | 既存実装 | /manga も同じ機構に載せる。新機構を発明しない |
| 破壊的操作の安全機構（dry-run・バックアップ・半自動承認） | shadow_verify / /step | manga_rename は dry_run デフォルト + manifest による可逆化 |
| 大きなツール結果はファイル退避 + 参照 | 設計方針（未実装だった） | web_to_markdown で初適用。`.pixie_notes/web_cache/` |

---

## 1. ツールパック機構（コア変更・最小）

### 1.1 ディレクトリ構造

```
src/toolpacks/
  __init__.py      # load_pack(name) — パックモジュールの import を遅延実行
  manga.py         # @register_tool(..., pack="manga") のツール群
  web.py           # @register_tool(..., pack="web") のツール群
```

### 1.2 registry の拡張（registry.py）

- `register_tool(...)` に省略可能な `pack: str | None = None` を追加。`None`（既定）はコアツール = 従来動作と完全互換。
- `TOOL_REGISTRY` のエントリに `pack` を保持。
- 新関数 `get_active_tool_names(active_packs: set[str]) -> frozenset[str]`:
  コアツール全部 + `pack in active_packs` のツール。

### 1.3 パックのロードと有効化

- **ロード（import）**: `src/toolpacks/__init__.py` の `load_pack(name)` が `importlib.import_module(f"toolpacks.{name}")` を実行。import 時に `@register_tool` が走って登録される。ロードは冪等（2回目は no-op）。
- **有効化の入口は2つ**:
  1. `config.json` の `"toolpacks": ["web"]` — 起動時に `setup_application()` がロード+有効化
  2. CLI コマンド `/pack <name>` / `/pack <name> off` — **有効化はセッション途中でも許すが、切替時に1回のキャッシュミスが発生する旨を出力する**（/code と同じトレードオフ。出力例: `[System] パック 'manga' を有効化しました（次ターンはプロンプト再構築のため prefill が長くなります）`）
- `AppContext.active_packs: set[str]` を追加（既定は config.json 由来）。

### 1.4 node_plan への影響（engine.py、変更は2行程度)

現状:
```python
available_tools = None  # 通常モード = 全コアツール
tools = registry_to_openai_tools(sorted(available_tools) if available_tools else None)
```
変更後:
```python
_active = get_active_tool_names(getattr(context, "active_packs", set()))
available_tools = set(CODE_TOOL_SET) if code_mode else _active
tools = registry_to_openai_tools(sorted(available_tools))
```
- `active_packs` はターン中に変化しない（/pack はユーザー入力処理＝ターン境界でのみ実行）ため、セッション内でツール一覧は不変 → prefix cache 無傷。
- `generate_behavior_prompt(available_tools=...)` にも同じ集合を渡す（既存引数のまま）。
- golden テストへの影響: コアのみ（パックなし）の出力は完全不変であることを既存 27 ケースで担保。パック有効時の出力は新規 fixture を追加。

### 1.5 タスクモードの一般化（tools.py / engine.py / main.py）

- `generate_behavior_prompt(mode=...)` の分岐に `"manga"` を追加（`_MANGA_MODE_POLICY`）。mode の型は既存どおり文字列。
- `/manga <folder>` コマンド（main.py）: `context.task_mode = "manga"` + manga パック有効化 + 対象フォルダを state_board.goal に設定。`/manga off` で解除。
- engine.py 側は `code_mode` と同じ扱いで `mode="manga"` を system_msg_builder に渡す。**モード切替はユーザー主導なのでキャッシュミス1回は許容**（既存の /code と同一の判断）。

---

## 2. web パック詳細設計（toolpacks/web.py）

### 2.1 前提（web2md.py の実仕様、確認済み）

- 場所: `D:\Workspace\web2md\web2md.py`、自前 venv: `D:\Workspace\web2md\.venv\Scripts\python.exe`
- 単一 URL: `python web2md.py <url>` → **カレントディレクトリの `output.md` に固定保存**
- ブラウザセッション: `./.browser_session`（**カレント相対**）— Edge の persistent context
- `--crawl` / `--download-tree` は `--output-dir` 対応。`--select-pages` は **input() の対話選択があるためツール化対象外**
- ログイン必要サイトでは headless→有頭ブラウザを開いて**手動ログインを待つ**（長時間ブロックしうる）
- mhtml ファイルパスも URL 引数として受け付ける

### 2.2 ツール定義

#### `web_to_markdown(url, mode="single", output_dir=None)`

```python
schema = {
  "type": "object",
  "properties": {
    "url": {"type": "string"},                      # http/https の URL、または .mhtml のローカルパス
    "mode": {"type": "string", "enum": ["single", "crawl", "tree"], "default": "single"},
    "output_dir": {"type": "string"},               # crawl/tree 時の保存先（省略時 .pixie_notes/web_cache/<slug>/）
  },
  "required": ["url"],
}
```

- **分類: DESTRUCTIVE_TOOLS**（外部ネットワークアクセス + プロセス起動のため直列実行・半自動承認の対象）
- URL バリデーション: `http://` / `https://` で始まるか、実在する `.mhtml` ファイルパスのみ許可。それ以外は即 Error 返却（LLM の自己修復用にエラー文で正しい形式を案内）

### 2.3 実行仕様

```python
cmd_single = [WEB2MD_PYTHON, "web2md.py", url]
cmd_crawl  = [WEB2MD_PYTHON, "web2md.py", url, "--crawl", "--output-dir", str(out_dir)]
cmd_tree   = [WEB2MD_PYTHON, "web2md.py", url, "--download-tree", "--output-dir", str(out_dir)]
subprocess.run(cmd, cwd=WEB2MD_DIR, timeout=WEB2MD_TIMEOUT_SEC,
               capture_output=True, text=True, encoding="utf-8", errors="replace",
               env={**os.environ, "PYTHONUTF8": "1"})
```

- **cwd は必ず `WEB2MD_DIR`**: `output.md` と `.browser_session` がカレント相対のため。既存ブラウザセッション（ログイン状態）を再利用できる
- `PYTHONUTF8=1` を明示（Windows コンソールエンコーディング対策。run_command の既知知見と同じ）
- タイムアウト: 既定 300 秒（手動ログイン待ちを考慮）。超過時は「ログイン待ちの可能性。ブラウザ画面を確認するか、タイムアウトを伸ばして再実行」を Error 文に含める
- 実行前に `output.md` が存在すれば削除（前回結果の誤読防止）

### 2.4 結果ハンドリング（ファイル退避 + 参照方式）

- **single**: 成功後 `WEB2MD_DIR/output.md` を `.pixie_notes/web_cache/<yyyymmdd_hhmmss>_<slug>.md` へ移動（slug は URL ホスト+パス末尾から生成、Windows 禁止文字除去、最大 60 字）。ツール結果は:
  ```
  Success: <URL> を Markdown 化しました。
  保存先: .pixie_notes/web_cache/20260704_153000_example-com-page.md (全 420 行)
  --- 冒頭抜粋 (30行) ---
  <先頭30行>
  --- 続きは read_file / grep_search で参照してください ---
  ```
- **crawl / tree**: 保存件数・エラー件数・出力ディレクトリのファイル一覧（上位 20 件）を返す。本文は返さない
- web_cache の GC: 上限 `WEB_CACHE_MAX_FILES`（既定 50）。超過時は古い順に削除

### 2.5 config 追加（config.json 側・機微でない既定値は config.py）

```jsonc
// config.json（ユーザー環境依存のためこちら）
"web2md": { "dir": "D:\\Workspace\\web2md", "timeout_sec": 300 }
```
```python
# config.py（既定値・定数）
WEB_CACHE_DIR_NAME: str = ".pixie_notes/web_cache"
WEB_CACHE_MAX_FILES: int = 50
WEB_EXCERPT_LINES: int = 30
```
- `web2md.dir` 未設定または python.exe 不在の場合、ツールは登録されるが実行時に「config.json の web2md.dir を設定してください」と Error を返す（起動は失敗させない）

---

## 3. manga パック詳細設計（toolpacks/manga.py）

### 3.1 ツール一覧（3個 + タスクモード）

| ツール | 分類 | 役割 |
|---|---|---|
| `manga_scan(folder)` | READONLY | フォルダ内 zip の一括調査（漫画判定・表紙抽出・現名） |
| `manga_rename(zip_path, new_title, dry_run=True)` | DESTRUCTIVE | 1 zip のリネーム適用（内部フォルダ名 + zip 名） |
| `manga_undo(zip_path_or_all)` | DESTRUCTIVE | manifest による復元 |

### 3.2 `manga_scan(folder)`

処理（LLM 不使用・完全決定的）:
1. `folder` 直下の `*.zip` を列挙（再帰しない。サブフォルダは対象外と明記）
2. 各 zip について `zipfile.ZipFile` で**展開せずに** namelist を検査:
   - 画像判定: 拡張子 {jpg, jpeg, png, webp, avif, bmp, gif} のエントリ割合 >= 0.8 かつ画像 >= 5 枚 → `is_manga: true`
   - 内部構造の分類（3.3 のリネーム方式決定に使用）:
     - `flat`: ルート直下に直接画像
     - `single_root`: ルートフォルダ 1 個の下に画像（最頻パターン）
     - `nested`: それ以外（複数フォルダ・多階層）→ リネームは zip ファイル名のみ対象とし内部は触らない
   - 表紙抽出: 画像エントリを**自然順ソート**（`001.jpg` < `002.jpg` < `010.jpg`、stdlib で数値部を int 化して比較）した先頭 1 枚を `.pixie_notes/manga_covers/<zip名のslug>.jpg` に抽出。Pillow が import 可能なら長辺 512px に縮小（Vision 入力のトークン節約）、不可なら原寸コピー
3. 返却（JSON 文字列、1 zip あたり 5 行程度に圧縮）:
```json
{
  "folder": "D:/Comics/incoming",
  "zips": [
    {"path": "...(1).zip", "is_manga": true, "structure": "single_root",
     "root_dir": "img_20240101", "images": 192, "cover": ".pixie_notes/manga_covers/xxx.jpg",
     "current_name": "[作者名] タイトル 第01巻 (1).zip"}
  ],
  "skipped": [{"path": "notes.zip", "reason": "画像割合 0.1 のため漫画ではない"}]
}
```
- 上限: 1 回のスキャンは 100 zip まで（超過分は `"truncated": N` で通知。動的ツール結果キャップとの整合）

### 3.3 `manga_rename(zip_path, new_title, dry_run=True)`

- `new_title` サニタイズ: Windows 禁止文字 `\ / : * ? " < > |` を全角置換、前後空白除去、最大 120 字、予約名（CON 等）拒否。**サニタイズ後の名前を必ず結果に含める**（LLM が意図とのズレを確認できるように）
- 処理シーケンス（`structure` により分岐）:
  - `flat` / `nested`: **zip ファイル名のリネームのみ**（`shutil.move`。再圧縮しない = 高速・無劣化）
  - `single_root`: TEMP に展開 → ルートフォルダを `new_title` に改名 → `zipfile.ZipFile(..., ZIP_STORED)` で再圧縮（画像に再圧縮は無意味なので無圧縮格納）→ 元 zip を `.pixie_notes/manga_backup/` に移動 → 新 zip を `<new_title>.zip` として配置
- **dry_run=True（既定）**: 実行せず計画だけ返す:
  ```
  [DRY RUN] 適用内容:
    zip:    [作者] タイトル 第01巻 (1).zip → タイトル 第01巻.zip
    内部:   img_20240101/ → タイトル 第01巻/
    方式:   single_root（展開→再圧縮、無圧縮格納）
  問題なければ dry_run=false で再実行してください。
  ```
- 衝突: リネーム先が存在する場合は Error（` (2)` 等の自動連番はしない — LLM に判断を返す）
- **manifest（可逆性の要）**: `.pixie_notes/manga_manifest.json` に追記:
  `{"ts": ..., "original_zip": "...", "new_zip": "...", "original_root": "...", "backup": "..."}`
- バックアップ GC: `manga_backup/` は上限 `MANGA_BACKUP_MAX`（既定 20 件）で古い順に削除。dry_run の結果にバックアップ保持数を明記

### 3.4 `manga_undo(zip_path | "all_last_batch")`

manifest を逆順に辿り、backup の zip を元の名前で復元。復元したエントリは manifest から除去。

### 3.5 `/manga` タスクモードと `_MANGA_MODE_POLICY`

`MANGA_TOOL_SET`（config.py）: `manga_scan, manga_rename, manga_undo, list_directory, read_file, update_state, view_image`（view_image は Vision 有効時の表紙確認用）

`_MANGA_MODE_POLICY`（tools.py、要旨）:
```
【漫画整理モード — ワークフロー】
1. manga_scan で対象フォルダを一括調査する（1回だけ。zip ごとに繰り返さない）
2. 各 zip の current_name から正式な漫画タイトルを推定する。
   ノイズ（[作者名]・(同人誌)・v01・(1) などの重複マーカー・DL サイト名）を除去し、
   「タイトル 第NN巻」形式に正規化する。確信が持てない場合は表紙画像（cover）を
   view_image で確認する（Vision 無効時は現名のまま skip し、その旨を報告する）
3. 全件の変更案を「現名 → 新名」の一覧表でユーザーに提示し、承認を得る
4. 承認後、manga_rename を dry_run=false で 1 件ずつ直列に適用する
5. 全件完了後、成功/失敗/skip の集計を報告する。失敗した zip は理由と共に列挙する
【禁止事項】承認前の dry_run=false 実行。スキャン済みフォルダの再スキャン。
```
- ステップ 3 の承認は通常の会話として行う（半自動モード /step とは独立に機能する）
- 繰り返し実行は既存 ReAct ループそのまま: 50 zip なら scan 1 + 提案 1〜2 往復 + rename 50 で max_tool_calls=100 内に収まる。それ以上のフォルダは scan の 100 件上限 + 「残り N 件は次バッチで」の運用

### 3.6 表紙 Vision 委譲（フェーズ 3・任意)

- `manga_scan` が返す `cover` パスが受け渡しインターフェース（フェーズ 1 から返しておく）
- Vision 有効時（mmproj 付き起動 or delegate_server に Vision モデル）: `view_image(cover)` → 既存 `run_vision_subquery` 経由でタイトル・作者・巻数を抽出
- プロンプト（Vision サブクエリ用固定文）: 「この漫画の表紙から タイトル/作者/巻数 を JSON で抽出。読み取れない項目は null」+ json_schema 強制（reflection と同じ手法）
- delegate_server の Vision 対応可否は `initialize_backend` の既存判定を流用

---

## 4. テスト・eval 計画

| 対象 | 種別 | 内容 |
|---|---|---|
| registry の pack 対応 | 単体 | pack 指定/未指定の登録、get_active_tool_names の集合演算 |
| manga_scan | 単体（LLM 不要） | fixture でダミー zip 生成（flat/single_root/nested/非漫画）→ 判定・構造分類・表紙抽出・自然順ソート |
| manga_rename | 単体 | dry_run が無変更なこと、3 方式の適用結果、サニタイズ、衝突 Error、manifest 記録 |
| manga_undo | 単体 | rename → undo のラウンドトリップで完全復元 |
| web_to_markdown | 単体 | subprocess を mock（cmd 組み立て・URL バリデーション・退避/抜粋・GC）。実 web2md は手動確認のみ |
| golden | 回帰 | コアのみの出力が不変 + manga モードの fixture 追加 |
| eval | 統合 | 新タスク「ダミー漫画 zip 3 個の一括リネーム」（checker: リネーム後のファイル名一致 + manifest 存在）。/manga モードで実行 |

## 5. config 追加一覧

| 場所 | キー | 既定 | 用途 |
|---|---|---|---|
| config.py | `MANGA_BACKUP_MAX` | 20 | バックアップ GC 上限 |
| config.py | `MANGA_SCAN_MAX_ZIPS` | 100 | 1 スキャンの上限 |
| config.py | `MANGA_TOOL_SET` | frozenset | /manga モードの固定ツールセット |
| config.py | `WEB_CACHE_MAX_FILES` / `WEB_EXCERPT_LINES` | 50 / 30 | web_cache GC / 抜粋行数 |
| config.json | `toolpacks` | `[]` | 起動時有効化パック |
| config.json | `web2md.dir` / `web2md.timeout_sec` | なし / 300 | web2md の場所とタイムアウト |

## 6. 実装フェーズと完了条件

| フェーズ | 内容 | 完了条件 |
|---|---|---|
| P1 | ツールパック機構 + web パック | 単体テスト green / 実 URL 1 件の変換で web_cache 退避・抜粋返却を確認 / golden 不変 / cache ヒット率が非パックセッションで劣化しない |
| P2 | manga パック（zip 名ベース）+ /manga モード | 単体テスト green / ダミー zip での eval タスク PASS / dry_run→承認→適用→undo の一連を実 LLM で確認 |
| P3 | 表紙 Vision 委譲 | delegate Vision 環境が用意でき次第。P1/P2 に依存なし |

## 7. 却下した代替案（記録）

- **web2md の MCP サーバー化**: mcp_client は既存だが、単一クライアント・単一スクリプトに対してプロトコル実装のオーバーヘッドが見合わない。将来他クライアントから共用したくなった時に再検討
- **web2md のライブラリ import**: playwright 等の重依存が本体に入り依存ゼロポリシー違反。却下
- **manga 処理を 1 つの全自動ツールに**（`process_all(folder)`）: 名前判断が LLM の仕事でなくなり誤リネームの検証機会も失われる。「決定的処理はツール・判断は LLM・適用は承認後」の分離を維持
- **zip 内画像の全 Vision 判定**: コスト過大。表紙 1 枚 + ファイル名で十分、精度不足なら将来 2〜3 枚に拡張
