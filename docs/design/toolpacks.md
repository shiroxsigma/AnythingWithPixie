# ツールパック機構

AWPには用途別ツールを遅延登録する汎用基盤だけを残す。パック名は
`src/toolpacks/__init__.py` の `AVAILABLE_PACKS` に登録し、各モジュールのツールは
`register_tool(..., pack="<name>")` で所属を宣言する。現在、同梱パックはない。

## Manga機能の移管

Mangaモードは2026年8月にTalkWithPixieへ移管した。漫画ZIPのスキャン、表紙抽出、
リネーム案のプレビュー、承認後の適用、バックアップ付きundoはTWPのチャットコマンド
`/manga` が担当する。AWPにはManga専用ツール、固定ツールセット、プロンプト、CLIモードを
重複実装しない。

## 新しいパックの追加

1. `src/toolpacks/<name>.py` を作成する。
2. ツールを `register_tool(..., pack="<name>")` で登録する。
3. `AVAILABLE_PACKS` に名前を追加する。
4. 遅延ロード、無効名、ツール可視性をテストする。

パックの有効・無効はターン境界でのみ変更し、プロンプトとツール定義のprefix cacheを
不必要に壊さない。破壊的操作にはdry-run、明示承認、復元可能なjournalを要求する。
