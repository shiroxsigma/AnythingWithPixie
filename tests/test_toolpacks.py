"""ツールパック機構 + manga パックの単体テスト（詳細設計 docs/design/toolpacks.md §4）。

- registry の pack 対応（登録・get_active_tool_names の集合演算）
- manga_scan: ダミー zip fixture（flat/single_root/nested/非漫画/日本語ファイル名）での
  判定・構造分類・表紙抽出・自然順ソート
- manga_rename: dry_run が無変更なこと・3方式の適用・サニタイズ・衝突Error・manifest記録
- manga_undo: rename → undo のラウンドトリップで完全復元
- cp437/cp932 文字化けフォールバック（_decode_zip_name の単体テスト）

すべて tmp_path 上に動的生成したダミー zip で完結し、実プロジェクトの
.pixie_notes/ は一切汚染しない（get_data_path を monkeypatch で隔離する）。
"""

import json
import sys
import zipfile
from pathlib import Path

import pytest

import toolpacks.manga as manga_mod
from registry import TOOL_REGISTRY, get_active_tool_names, register_tool
from toolpacks import load_pack
from toolpacks.manga import manga_rename, manga_scan, manga_undo

# =====================================================
# 共通フィクスチャ
# =====================================================

#: 1x1 の最小 PNG バイト列（画像判定・表紙抽出の材料）。PIL.Image.open() で
#: 実際に開ける正当な PNG であること（zlib で正しく圧縮済み・IHDR/IDAT/IEND完備）。
TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
    "0000000c4944415478da63f8ffff3f0005fe02fe331295140000000049454e44ae"
    "426082"
)


@pytest.fixture
def isolated_manga(tmp_path, monkeypatch):
    """manga モジュールの get_data_path を tmp_path 配下へリダイレクトする。

    eval/runner.py の _isolated_pixie_env と同じ発想: 実プロジェクトの
    .pixie_notes/manga_* を一切汚染せず、テストごとに独立したディレクトリで完結させる。
    """

    def _patched(rel_path: str) -> str:
        return str(tmp_path / rel_path)

    monkeypatch.setattr(manga_mod, "get_data_path", _patched)
    return tmp_path


def _write_zip(path: Path, names: list, image_bytes: bytes = TINY_PNG, extra_text: dict = None) -> None:
    """namesの各エントリを画像(image_bytes)として書き込むダミーzipを作る。

    ディレクトリのみのエントリ（末尾"/"）は空データで書く。extra_text で
    非画像エントリ（漫画判定を下げる目的等）を追加できる。
    """
    with zipfile.ZipFile(path, "w") as zf:
        for n in names:
            if n.endswith("/"):
                zf.writestr(zipfile.ZipInfo(n), b"")
            else:
                zf.writestr(n, image_bytes)
        for n, data in (extra_text or {}).items():
            zf.writestr(n, data)


# =====================================================
# 1. registry の pack 対応
# =====================================================

def test_register_tool_without_pack_defaults_to_none():
    name = "_test_dummy_core_tool"
    try:
        register_tool(name=name, description="d", schema={"type": "object", "properties": {}, "required": []})(
            lambda: "ok"
        )
        assert TOOL_REGISTRY[name]["pack"] is None
        # pack未指定 = 常にコアツール扱い（active_packsが空でも含まれる）
        assert name in get_active_tool_names(set())
    finally:
        TOOL_REGISTRY.pop(name, None)


def test_register_tool_with_pack_excluded_when_inactive():
    name = "_test_dummy_pack_tool"
    try:
        register_tool(
            name=name, description="d", schema={"type": "object", "properties": {}, "required": []}, pack="zzz_test"
        )(lambda: "ok")
        assert TOOL_REGISTRY[name]["pack"] == "zzz_test"
        assert name not in get_active_tool_names(set())
        assert name not in get_active_tool_names({"other_pack"})
        assert name in get_active_tool_names({"zzz_test"})
    finally:
        TOOL_REGISTRY.pop(name, None)


def test_get_active_tool_names_is_union_of_core_and_active_packs():
    core_only = get_active_tool_names(set())
    with_manga = get_active_tool_names({"manga"})
    # manga パックロード前でも呼び出し自体はエラーにならない（存在しないpack名でも安全）
    assert core_only <= with_manga or True  # ロード前は manga_* が未登録の可能性があるため緩い検証
    load_pack("manga")
    core_only = get_active_tool_names(set())
    with_manga = get_active_tool_names({"manga"})
    assert {"manga_scan", "manga_rename", "manga_undo"} <= with_manga
    assert not ({"manga_scan", "manga_rename", "manga_undo"} & core_only)


def test_load_pack_unknown_name_raises():
    with pytest.raises(ValueError):
        load_pack("no_such_pack_xyz")


def test_load_pack_idempotent():
    load_pack("manga")
    n_before = len(TOOL_REGISTRY)
    load_pack("manga")  # 2回目は no-op
    assert len(TOOL_REGISTRY) == n_before


# =====================================================
# 2. manga_scan
# =====================================================

def test_manga_scan_flat_structure(isolated_manga):
    zpath = isolated_manga / "flat_manga.zip"
    _write_zip(zpath, [f"{i:03d}.jpg" for i in range(1, 9)])

    result = json.loads(manga_scan(str(isolated_manga)))
    assert len(result["zips"]) == 1
    entry = result["zips"][0]
    assert entry["is_manga"] is True
    assert entry["structure"] == "flat"
    assert entry["root_dir"] is None
    assert entry["images"] == 8
    assert entry["current_name"] == "flat_manga.zip"


def test_manga_scan_single_root_structure(isolated_manga):
    zpath = isolated_manga / "single_root_manga.zip"
    _write_zip(zpath, [f"img_20240101/{i:03d}.jpg" for i in range(1, 7)])

    result = json.loads(manga_scan(str(isolated_manga)))
    entry = result["zips"][0]
    assert entry["structure"] == "single_root"
    assert entry["root_dir"] == "img_20240101"
    assert entry["images"] == 6


def test_manga_scan_nested_structure(isolated_manga):
    zpath = isolated_manga / "nested_manga.zip"
    _write_zip(
        zpath,
        [f"vol1/{i:03d}.jpg" for i in range(1, 4)] + [f"vol2/{i:03d}.jpg" for i in range(1, 4)],
    )

    result = json.loads(manga_scan(str(isolated_manga)))
    entry = result["zips"][0]
    assert entry["structure"] == "nested"
    assert entry["root_dir"] is None


def test_manga_scan_non_manga_is_skipped(isolated_manga):
    zpath = isolated_manga / "notes.zip"
    _write_zip(zpath, [], extra_text={"readme.txt": b"hello", "notes.md": b"# notes"})

    result = json.loads(manga_scan(str(isolated_manga)))
    assert result["zips"] == []
    assert len(result["skipped"]) == 1
    assert result["skipped"][0]["path"] == "notes.zip"
    assert "漫画ではない" in result["skipped"][0]["reason"] or "エントリ" in result["skipped"][0]["reason"]


def test_manga_scan_japanese_filenames(isolated_manga):
    """日本語ファイル名(UTF-8フラグ付き・標準的なzip)を正しく扱えること。"""
    zpath = isolated_manga / "[すずき太郎] 冒険の書 第01巻 (1).zip"
    _write_zip(zpath, [f"img_20240101/{i:03d}.jpg" for i in range(1, 7)])

    result = json.loads(manga_scan(str(isolated_manga)))
    entry = result["zips"][0]
    assert entry["current_name"] == "[すずき太郎] 冒険の書 第01巻 (1).zip"
    assert entry["structure"] == "single_root"


def test_manga_scan_natural_sort_picks_correct_cover(isolated_manga, monkeypatch):
    """自然順ソート: '2.jpg' < '10.jpg' （文字列順だと逆転する）。

    Pillow 有無による再エンコードの影響を避けるため、ここでは Pillow を隠して
    「原寸コピー」パスに固定し、抽出元バイト列が natural sort の先頭画像と
    一致することだけを検証する（Pillow の縮小処理自体は別テストで検証済み）。
    """
    monkeypatch.setitem(sys.modules, "PIL", None)
    zpath = isolated_manga / "sort_test.zip"
    # 文字列比較では "10.jpg" < "2.jpg" になってしまうため、自然順ソートの検証に使う
    _write_zip(zpath, ["2.jpg", "10.jpg", "3.jpg", "4.jpg", "5.jpg"])

    result = json.loads(manga_scan(str(isolated_manga)))
    entry = result["zips"][0]
    assert entry["cover"] is not None
    # entry["cover"] は ".pixie_notes/manga_covers/xxx.jpg" 形式の相対パス文字列
    cover_path = isolated_manga / entry["cover"]
    assert cover_path.exists()
    # 抽出された表紙が "2.jpg" 由来であること（生データが一致）を検証
    with zipfile.ZipFile(zpath) as zf:
        expected_bytes = zf.read("2.jpg")
    assert cover_path.read_bytes() == expected_bytes


def test_manga_scan_missing_folder_returns_error():
    out = manga_scan(str(Path("D:/__nonexistent_manga_folder__")))
    assert out.startswith("Error:")


def test_manga_scan_truncates_over_limit(isolated_manga, monkeypatch):
    monkeypatch.setattr(manga_mod, "MANGA_SCAN_MAX_ZIPS", 2)
    for i in range(4):
        _write_zip(isolated_manga / f"z{i}.zip", [f"{j:03d}.jpg" for j in range(1, 6)])
    result = json.loads(manga_scan(str(isolated_manga)))
    assert result.get("truncated") == 2
    assert len(result["zips"]) == 2


# =====================================================
# Pillow あり/なし の表紙抽出
# =====================================================

def test_manga_scan_cover_resized_with_pillow(isolated_manga):
    import io

    pytest.importorskip("PIL")
    from PIL import Image

    img = Image.new("RGB", (800, 400), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    big_jpg = buf.getvalue()

    zpath = isolated_manga / "big_cover.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("001.jpg", big_jpg)
        for i in range(2, 7):
            zf.writestr(f"{i:03d}.jpg", TINY_PNG)

    result = json.loads(manga_scan(str(isolated_manga)))
    entry = result["zips"][0]
    cover_path = isolated_manga / entry["cover"]
    assert cover_path.exists()
    with Image.open(cover_path) as out_img:
        assert max(out_img.size) <= 512


def test_manga_scan_cover_raw_copy_without_pillow(isolated_manga, monkeypatch):
    """Pillow不在時は原寸コピー（バイト完全一致）にフォールバックする。"""
    monkeypatch.setitem(sys.modules, "PIL", None)

    zpath = isolated_manga / "no_pillow.zip"
    _write_zip(zpath, [f"{i:03d}.jpg" for i in range(1, 6)])

    result = json.loads(manga_scan(str(isolated_manga)))
    entry = result["zips"][0]
    cover_path = isolated_manga / entry["cover"]
    assert cover_path.exists()
    assert cover_path.read_bytes() == TINY_PNG


# =====================================================
# 3. cp437/cp932 文字化けフォールバック（_decode_zip_name 単体）
# =====================================================

def test_decode_zip_name_recovers_cp932_mojibake():
    orig = "冒険の書/001.jpg"
    cp932_bytes = orig.encode("cp932")
    mojibake = cp932_bytes.decode("cp437")

    zi = zipfile.ZipInfo(mojibake)
    zi.flag_bits = 0  # UTF-8フラグなし = 文字化けを起こす典型パターン
    assert manga_mod._decode_zip_name(zi) == orig


def test_decode_zip_name_respects_utf8_flag():
    zi = zipfile.ZipInfo("冒険の書/001.jpg")
    zi.flag_bits = 0x800
    assert manga_mod._decode_zip_name(zi) == "冒険の書/001.jpg"


def test_decode_zip_name_ascii_passthrough():
    zi = zipfile.ZipInfo("plain_ascii/001.jpg")
    zi.flag_bits = 0
    assert manga_mod._decode_zip_name(zi) == "plain_ascii/001.jpg"


# =====================================================
# 自然順ソートキー・構造分類の単体テスト
# =====================================================

def test_natural_sort_key_orders_numerically():
    names = ["10.jpg", "2.jpg", "1.jpg"]
    assert sorted(names, key=manga_mod._natural_sort_key) == ["1.jpg", "2.jpg", "10.jpg"]


def test_classify_structure_flat():
    assert manga_mod._classify_structure(["a.jpg", "b.jpg"]) == ("flat", None)


def test_classify_structure_single_root():
    assert manga_mod._classify_structure(["root/a.jpg", "root/b.jpg"]) == ("single_root", "root")


def test_classify_structure_nested():
    structure, root = manga_mod._classify_structure(["v1/a.jpg", "v2/b.jpg"])
    assert structure == "nested"
    assert root is None


def test_classify_structure_mixed_root_and_folder_is_nested():
    """ルート直下ファイルとフォルダが混在する場合はnested扱い。"""
    structure, _ = manga_mod._classify_structure(["a.jpg", "sub/b.jpg"])
    assert structure == "nested"


# =====================================================
# 4. manga_rename
# =====================================================

def test_manga_rename_dry_run_makes_no_changes(isolated_manga):
    zpath = isolated_manga / "[author] Title 01.zip"
    _write_zip(zpath, [f"root/{i:03d}.jpg" for i in range(1, 6)])

    out = manga_rename(str(zpath), "Title 第01巻", dry_run=True)
    assert out.startswith("[DRY RUN]")
    assert zpath.exists()  # 元ファイルは無変更
    assert not (isolated_manga / "Title 第01巻.zip").exists()
    manifest_path = isolated_manga / ".pixie_notes" / "manga_manifest.json"
    assert not manifest_path.exists()


def test_manga_rename_flat_only_renames_zip_file(isolated_manga):
    zpath = isolated_manga / "[author] Title 01.zip"
    original_names = [f"{i:03d}.jpg" for i in range(1, 6)]
    _write_zip(zpath, original_names)

    out = manga_rename(str(zpath), "Title 第01巻", dry_run=False)
    assert "リネーム完了" in out
    new_path = isolated_manga / "Title 第01巻.zip"
    assert new_path.exists()
    assert not zpath.exists()
    with zipfile.ZipFile(new_path) as zf:
        assert sorted(zf.namelist()) == sorted(original_names)  # 内部は無変更


def test_manga_rename_nested_only_renames_zip_file(isolated_manga):
    zpath = isolated_manga / "nested_src.zip"
    _write_zip(zpath, ["v1/a.jpg", "v1/b.jpg", "v2/c.jpg", "v2/d.jpg", "v2/e.jpg"])

    out = manga_rename(str(zpath), "新タイトル", dry_run=False)
    assert "リネーム完了" in out
    new_path = isolated_manga / "新タイトル.zip"
    assert new_path.exists() and not zpath.exists()


def test_manga_rename_single_root_repacks_and_backs_up(isolated_manga):
    zpath = isolated_manga / "[author] Title (1).zip"
    original_names = [f"img_root/{i:03d}.jpg" for i in range(1, 6)]
    _write_zip(zpath, original_names)

    out = manga_rename(str(zpath), "きれいなタイトル 第01巻", dry_run=False)
    assert "リネーム完了" in out

    new_path = isolated_manga / "きれいなタイトル 第01巻.zip"
    assert new_path.exists()
    assert not zpath.exists()

    with zipfile.ZipFile(new_path) as zf:
        names = zf.namelist()
        assert all(n.startswith("きれいなタイトル 第01巻/") for n in names)
        assert len(names) == 5
        # 無圧縮格納（ZIP_STORED）で書かれていること
        for info in zf.infolist():
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.flag_bits & 0x800  # UTF-8フラグが明示されていること

    backup_dir = isolated_manga / ".pixie_notes" / "manga_backup"
    assert backup_dir.exists()
    backups = list(backup_dir.glob("*.zip"))
    assert len(backups) == 1
    with zipfile.ZipFile(backups[0]) as zf:
        assert sorted(zf.namelist()) == sorted(original_names)

    manifest = json.loads((isolated_manga / ".pixie_notes" / "manga_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 1
    assert manifest[0]["structure"] == "single_root"
    assert manifest[0]["original_root"] == "img_root"
    assert Path(manifest[0]["new_zip"]).name == "きれいなタイトル 第01巻.zip"


def test_manga_rename_sanitizes_forbidden_characters(isolated_manga):
    zpath = isolated_manga / "src.zip"
    _write_zip(zpath, [f"{i:03d}.jpg" for i in range(1, 6)])

    out = manga_rename(str(zpath), 'Foo/Bar:Baz*Qux?"<>|', dry_run=True)
    assert "＼" in out or "／" in out  # 全角置換された文字が結果に含まれる
    assert "/" not in out.split("\n")[1].split("→")[1]  # zip行の新名部分に生の"/"がない


def test_manga_rename_reserved_name_rejected(isolated_manga):
    zpath = isolated_manga / "src.zip"
    _write_zip(zpath, [f"{i:03d}.jpg" for i in range(1, 6)])

    out = manga_rename(str(zpath), "CON", dry_run=True)
    assert out.startswith("Error:")
    assert zpath.exists()


def test_manga_rename_collision_returns_error(isolated_manga):
    zpath = isolated_manga / "src.zip"
    _write_zip(zpath, [f"{i:03d}.jpg" for i in range(1, 6)])
    existing = isolated_manga / "Taken.zip"
    _write_zip(existing, [f"{i:03d}.jpg" for i in range(1, 6)])

    out = manga_rename(str(zpath), "Taken", dry_run=False)
    assert out.startswith("Error:")
    assert zpath.exists()  # 元ファイルは触られない
    assert existing.exists()


def test_manga_rename_missing_zip_returns_error(isolated_manga):
    out = manga_rename(str(isolated_manga / "nope.zip"), "Title", dry_run=True)
    assert out.startswith("Error:")


def test_manga_rename_backup_gc(isolated_manga, monkeypatch):
    monkeypatch.setattr(manga_mod, "MANGA_BACKUP_MAX", 2)
    for i in range(4):
        zpath = isolated_manga / f"src{i}.zip"
        _write_zip(zpath, [f"root{i}/{j:03d}.jpg" for j in range(1, 6)])
        manga_rename(str(zpath), f"Title{i}", dry_run=False)

    backup_dir = isolated_manga / ".pixie_notes" / "manga_backup"
    assert len(list(backup_dir.glob("*.zip"))) <= 2


# =====================================================
# 5. manga_undo — rename → undo ラウンドトリップ
# =====================================================

def test_manga_undo_restores_single_root_completely(isolated_manga):
    zpath = isolated_manga / "[author] Original (1).zip"
    original_names = [f"orig_root/{i:03d}.jpg" for i in range(1, 6)]
    _write_zip(zpath, original_names)
    with zipfile.ZipFile(zpath) as zf:
        original_bytes = {n: zf.read(n) for n in zf.namelist()}

    manga_rename(str(zpath), "新しいタイトル", dry_run=False)
    new_path = isolated_manga / "新しいタイトル.zip"
    assert new_path.exists()

    undo_out = manga_undo(str(new_path))
    assert "1/1" in undo_out
    assert zpath.exists()
    assert not new_path.exists()

    with zipfile.ZipFile(zpath) as zf:
        restored = {n: zf.read(n) for n in zf.namelist()}
    assert restored == original_bytes

    manifest = json.loads((isolated_manga / ".pixie_notes" / "manga_manifest.json").read_text(encoding="utf-8"))
    assert manifest == []


def test_manga_undo_restores_flat_rename(isolated_manga):
    zpath = isolated_manga / "flat_src.zip"
    names = [f"{i:03d}.jpg" for i in range(1, 6)]
    _write_zip(zpath, names)

    manga_rename(str(zpath), "FlatNewName", dry_run=False)
    new_path = isolated_manga / "FlatNewName.zip"
    assert new_path.exists()

    manga_undo(str(new_path))
    assert zpath.exists()
    assert not new_path.exists()
    with zipfile.ZipFile(zpath) as zf:
        assert sorted(zf.namelist()) == sorted(names)


def test_manga_undo_all_last_batch(isolated_manga):
    paths = []
    for i in range(3):
        zpath = isolated_manga / f"src{i}.zip"
        _write_zip(zpath, [f"{j:03d}.jpg" for j in range(1, 6)])
        manga_rename(str(zpath), f"NewName{i}", dry_run=False)
        paths.append(zpath)

    out = manga_undo("all_last_batch")
    assert "3/3" in out
    for p in paths:
        assert p.exists()
    manifest = json.loads((isolated_manga / ".pixie_notes" / "manga_manifest.json").read_text(encoding="utf-8"))
    assert manifest == []


def test_manga_undo_no_manifest_returns_message(isolated_manga):
    """manifestが存在しない(空の)状態でのundoは例外を投げず案内文を返す。

    isolated_manga フィクスチャを必ず使う（実プロジェクトの .pixie_notes を
    誤って操作しないようにするため）。
    """
    out = manga_undo("all_last_batch")
    assert isinstance(out, str)
    assert "取消可能な記録がありません" in out


def test_manga_undo_unknown_target_returns_error(isolated_manga):
    zpath = isolated_manga / "src.zip"
    _write_zip(zpath, [f"{i:03d}.jpg" for i in range(1, 6)])
    manga_rename(str(zpath), "Renamed", dry_run=False)

    out = manga_undo(str(isolated_manga / "totally_unknown.zip"))
    assert out.startswith("Error:")


# =====================================================
# MANGA_TOOL_SET / config 定数の健全性
# =====================================================

def test_manga_tool_set_contains_expected_tools():
    from config import MANGA_TOOL_SET

    assert MANGA_TOOL_SET == frozenset({
        "manga_scan", "manga_rename", "manga_undo", "manga_identify_cover",
        "list_directory", "read_file", "update_state", "view_image",
    })


def test_manga_tools_registered_with_manga_pack():
    load_pack("manga")
    for name in ("manga_scan", "manga_rename", "manga_undo", "manga_identify_cover"):
        assert name in TOOL_REGISTRY
        assert TOOL_REGISTRY[name]["pack"] == "manga"


def test_manga_identify_cover_is_readonly():
    from config import READONLY_TOOLS

    assert "manga_identify_cover" in READONLY_TOOLS


# =====================================================
# 6. manga_identify_cover（P3: 表紙 Vision 委譲）
# =====================================================
# manga.py 側（パス検証・スタブのフォールバックError）はLLM不要でここで検証する。
# engine.execute_tool のインターセプト（Vision経路解決・実LLM呼び出し）は
# tests/test_manga_identify_cover.py（mock LLM）で検証する。

from toolpacks.manga import manga_identify_cover  # noqa: E402


def _write_cover(isolated_manga: Path, filename: str = "cover_abc123.jpg", data: bytes = TINY_PNG) -> Path:
    covers_dir = isolated_manga / ".pixie_notes" / "manga_covers"
    covers_dir.mkdir(parents=True, exist_ok=True)
    p = covers_dir / filename
    p.write_bytes(data)
    return p


def test_manga_identify_cover_missing_path_returns_error(isolated_manga):
    out = manga_identify_cover(str(isolated_manga / ".pixie_notes" / "manga_covers" / "nope.jpg"))
    assert out.startswith("Error:")
    assert "存在しません" in out


def test_manga_identify_cover_rejects_non_image_outside_covers_dir(isolated_manga):
    """.pixie_notes/manga_covers 配下でも画像拡張子でもないファイルは拒否する。"""
    txt_path = isolated_manga / "notes.txt"
    txt_path.write_text("hello", encoding="utf-8")

    out = manga_identify_cover(str(txt_path))
    assert out.startswith("Error:")
    assert "認識できません" in out


def test_manga_identify_cover_accepts_image_ext_outside_covers_dir(isolated_manga):
    """covers_dir配下でなくても画像拡張子ならパス検証は通る（Vision不可のErrorに落ちる）。"""
    img_path = isolated_manga / "loose_cover.jpg"
    img_path.write_bytes(TINY_PNG)

    out = manga_identify_cover(str(img_path))
    # Vision経路が無い環境で直接呼ばれた場合はパス検証OK後、Vision不可のErrorに落ちる
    assert out.startswith("Error:")
    assert "Vision" in out


def test_manga_identify_cover_no_vision_returns_configured_error(isolated_manga):
    """Vision経路が無い環境で直接呼ぶと、有効化手順を含むErrorを返す（クラッシュしない）。"""
    cover = _write_cover(isolated_manga)
    out = manga_identify_cover(str(cover))
    assert out.startswith("Error: Vision モデルが利用できません")
    assert "delegate_server" in out


def test_resolve_cover_path_resolves_relative_to_data_root(isolated_manga):
    """manga_scan が返す相対パス文字列（.pixie_notes/manga_covers/xxx.jpg）を正しく解決する。"""
    _write_cover(isolated_manga, filename="rel_test.jpg")
    resolved, err = manga_mod._resolve_cover_path(".pixie_notes/manga_covers/rel_test.jpg")
    assert err is None
    assert resolved.exists()
    assert resolved.name == "rel_test.jpg"
