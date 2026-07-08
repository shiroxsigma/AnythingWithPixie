"""manga ツールパック — 漫画zipの一括調査・リネーム・取消(undo)。

詳細設計: docs/design/toolpacks.md §3。LLM は「判断」（現名からの正式タイトル推定）
だけを担い、決定的な処理（zip検査・リネーム適用・可逆化）はすべてこのモジュールの
stdlib のみのコードで行う（Pillow はオプショナル・任意）。

3ツール:
  - manga_scan(folder): フォルダ直下のzipを一括調査（展開しない・決定的）
  - manga_rename(zip_path, new_title, dry_run=True): 1zipのリネーム適用
  - manga_undo(zip_path_or_all): manifest による復元

依存: stdlib (zipfile/shutil/tempfile/json/re) + オプショナル Pillow。
"""

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

from config import MANGA_BACKUP_MAX, MANGA_SCAN_MAX_ZIPS
from paths import get_data_path, get_project_data_path
from registry import register_tool

# =====================================================
# 定数・パス
# =====================================================

#: 漫画判定の対象画像拡張子（namelist 検査・拡張子ベース）。
IMAGE_EXTS: frozenset[str] = frozenset({"jpg", "jpeg", "png", "webp", "avif", "bmp", "gif"})

#: 画像判定の閾値（エントリ中の画像割合）。
_IMAGE_RATIO_THRESHOLD = 0.8
#: 漫画判定の最低画像枚数。
_IMAGE_MIN_COUNT = 5

#: Windows で使用できないファイル名文字 → 全角置換テーブル。
_FORBIDDEN_CHAR_MAP = {
    "\\": "＼", "/": "／", ":": "：", "*": "＊",
    "?": "？", '"': "＂", "<": "＜", ">": "＞", "|": "｜",
}

#: Windows 予約デバイス名（拡張子を除いた基底名で大文字小文字無視の完全一致判定）。
_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

_NEW_TITLE_MAX_LEN = 120


def _manifest_path() -> Path:
    return Path(get_project_data_path(".pixie_notes/manga_manifest.json"))


def _backup_dir() -> Path:
    return Path(get_project_data_path(".pixie_notes/manga_backup"))


def _covers_dir() -> Path:
    return Path(get_project_data_path(".pixie_notes/manga_covers"))


# =====================================================
# zip 内部エントリ名の文字化け対策
# =====================================================

def _decode_zip_name(zi: zipfile.ZipInfo) -> str:
    """ZipInfo のファイル名を、日本語zip特有の文字化けを補正して返す。

    flag_bits の bit 11 (0x800) が立っていれば zipfile 側で既に UTF-8 デコード
    済みなのでそのまま使う。立っていない場合、zipfile は CP437 でデコードする
    ため、日本語ファイル名(多くは CP932 で書かれている)が文字化けする。
    cp437 に再エンコードして cp932 でデコードし直せるなら差し替える。
    """
    name = zi.filename
    if zi.flag_bits & 0x800:
        return name
    try:
        raw = name.encode("cp437")
    except UnicodeEncodeError:
        return name
    try:
        return raw.decode("cp932")
    except UnicodeDecodeError:
        return name


# =====================================================
# 自然順ソート（001.jpg < 002.jpg < 010.jpg）
# =====================================================

_NUM_RE = re.compile(r"(\d+)")


def _natural_sort_key(s: str):
    return [int(part) if part.isdigit() else part.lower() for part in _NUM_RE.split(s)]


# =====================================================
# 内部構造分類（flat / single_root / nested）
# =====================================================

def _classify_structure(names: list) -> tuple:
    """namelist（デコード済み・ディレクトリエントリ含む可）からフォルダ構造を分類する。

    Returns:
        (structure, root_dir): structure は "flat"/"single_root"/"nested"。
        root_dir は single_root の時のみそのフォルダ名、それ以外は None。
    """
    top_levels = set()
    has_root_files = False
    for n in names:
        if not n or n.endswith("/"):
            continue
        parts = n.split("/")
        if len(parts) == 1:
            has_root_files = True
        else:
            top_levels.add(parts[0])

    if not top_levels:
        return "flat", None
    if len(top_levels) == 1 and not has_root_files:
        return "single_root", next(iter(top_levels))
    return "nested", None


# =====================================================
# サニタイズ
# =====================================================

def _sanitize_title(new_title: str) -> tuple:
    """new_title を Windows 安全なファイル名に正規化する。

    Returns:
        (sanitized, error): error が None なら sanitized は使用可能。
        error があれば sanitized は空文字（呼び出し側はエラーを返すこと）。
    """
    title = (new_title or "").strip()
    if not title:
        return "", "new_title が空です"

    for bad, repl in _FORBIDDEN_CHAR_MAP.items():
        title = title.replace(bad, repl)
    title = title.strip()
    if len(title) > _NEW_TITLE_MAX_LEN:
        title = title[:_NEW_TITLE_MAX_LEN].strip()
    if not title:
        return "", "サニタイズ後のタイトルが空になりました"

    base = title.split(".")[0].upper()
    if base in _RESERVED_NAMES:
        return "", f"'{title}' はWindowsの予約デバイス名のため使用できません"

    return title, None


def _slugify(name: str, max_len: int = 40) -> str:
    """パス用の安全なスラグを生成する（衝突回避のため短いハッシュを付与）。"""
    stem = Path(name).stem
    safe = re.sub(r'[\\/:*?"<>|]', "_", stem).strip()
    safe = re.sub(r"\s+", "_", safe)[:max_len] or "cover"
    digest = hashlib.md5(name.encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"{safe}_{digest}"


# =====================================================
# 表紙抽出（Pillow はオプショナル）
# =====================================================

def _extract_cover(zf: zipfile.ZipFile, zi: zipfile.ZipInfo, dest_path: Path) -> None:
    """表紙1枚を dest_path (.jpg固定) へ抽出する。Pillow があれば長辺512pxに縮小。"""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(zi) as src:
        data = src.read()
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            max_side = 512
            if max(w, h) > max_side:
                if w >= h:
                    new_w, new_h = max_side, max(1, round(h * max_side / w))
                else:
                    new_h, new_w = max_side, max(1, round(w * max_side / h))
                resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS
                img = img.resize((new_w, new_h), resample)
            img.save(str(dest_path), format="JPEG", quality=85)
    except ImportError:
        # Pillow 不在: 原寸コピー（拡張子は .jpg 固定のまま・内容は元フォーマット）
        with open(dest_path, "wb") as f:
            f.write(data)


# =====================================================
# manga_scan
# =====================================================

@register_tool(
    name="manga_scan",
    description=(
        "指定フォルダ直下のzipファイルを一括調査し、漫画判定・内部構造分類・表紙抽出・"
        "現在のファイル名をJSONで返します。zipは展開せずnamelistのみ検査する決定的処理"
        "（LLM不使用）。サブフォルダは対象外（再帰しない）。"
    ),
    schema={
        "type": "object",
        "properties": {
            "folder": {"type": "string", "description": "調査対象フォルダの絶対パス（直下のzipのみ、再帰しない）"},
        },
        "required": ["folder"],
    },
    prompt_desc="manga_scan(folder): フォルダ直下のzipを一括調査し、漫画判定・構造・表紙・現名をJSONで返す(1回だけ実行)",
    pack="manga",
)
def manga_scan(folder: str) -> str:
    target = Path(folder)
    if not target.exists():
        return f"Error: フォルダが存在しません ({folder})"
    if not target.is_dir():
        return f"Error: フォルダではありません ({folder})"

    zip_files = sorted(target.glob("*.zip"))
    truncated = None
    if len(zip_files) > MANGA_SCAN_MAX_ZIPS:
        truncated = len(zip_files) - MANGA_SCAN_MAX_ZIPS
        zip_files = zip_files[:MANGA_SCAN_MAX_ZIPS]

    zips_out = []
    skipped = []

    for zip_path in zip_files:
        try:
            zf = zipfile.ZipFile(zip_path)
        except (zipfile.BadZipFile, OSError) as e:
            skipped.append({"path": zip_path.name, "reason": f"zipとして開けません: {e}"})
            continue

        try:
            infolist = zf.infolist()
            name_map = {}
            for zi in infolist:
                if zi.is_dir():
                    continue
                decoded = _decode_zip_name(zi)
                name_map[decoded] = zi

            all_names = list(name_map.keys())
            total_entries = len(all_names)
            image_names = [n for n in all_names if Path(n).suffix.lower().lstrip(".") in IMAGE_EXTS]
            image_count = len(image_names)
            ratio = (image_count / total_entries) if total_entries else 0.0
            is_manga = ratio >= _IMAGE_RATIO_THRESHOLD and image_count >= _IMAGE_MIN_COUNT

            if not is_manga:
                reason = (
                    f"画像割合 {ratio:.2f}（{image_count}/{total_entries}）のため漫画ではない"
                    if total_entries
                    else "zip内にエントリがありません"
                )
                skipped.append({"path": zip_path.name, "reason": reason})
                continue

            structure, root_dir = _classify_structure(all_names)

            sorted_images = sorted(image_names, key=_natural_sort_key)
            cover_rel = ".pixie_notes/manga_covers"
            cover_path = None
            if sorted_images:
                cover_zi = name_map[sorted_images[0]]
                slug = _slugify(zip_path.name)
                cover_dest = _covers_dir() / f"{slug}.jpg"
                try:
                    _extract_cover(zf, cover_zi, cover_dest)
                    cover_path = f"{cover_rel}/{slug}.jpg"
                except Exception:
                    cover_path = None

            zips_out.append({
                "path": str(zip_path),
                "is_manga": True,
                "structure": structure,
                "root_dir": root_dir,
                "images": image_count,
                "cover": cover_path,
                "current_name": zip_path.name,
            })
        finally:
            zf.close()

    result = {
        "folder": str(target.resolve()),
        "zips": zips_out,
        "skipped": skipped,
    }
    if truncated:
        result["truncated"] = truncated

    return json.dumps(result, ensure_ascii=False, indent=2)


# =====================================================
# manga_identify_cover（P3: 表紙 Vision 委譲）
# =====================================================
# 詳細設計 docs/design/toolpacks.md §3.6。実際のVision呼び出しは、このモジュールが
# AppContext（llm/delegate_llm）にアクセスできないため、engine.execute_tool のインター
# セプト（subagent._execute_manga_identify_cover）が担う（view_image / analyze_file /
# write_sections / delegate_research と同じ既存パターン）。このモジュールが持つのは:
#   - 固定プロンプト・JSON Schema（Vision呼び出し側と共有し重複定義を避ける）
#   - cover_path の検証（_resolve_cover_path。インターセプト側・スタブ側の両方で再利用）
#   - スタブ関数 manga_identify_cover 自体（Vision経路が全く無い環境、または
#     ツールを直接呼んだ場合のフォールバック。パス検証のみ行いエラー文を返す）

#: Vision サブクエリに渡す固定プロンプト（表紙1枚 → タイトル/作者/巻数のJSON抽出）。
MANGA_COVER_PROMPT: str = (
    "この漫画の表紙画像から、タイトル・作者名・巻数を読み取ってJSON形式で返してください。"
    "読み取れない、あるいは写っていない項目はnullにしてください。"
    "装飾文字や帯の宣伝文（キャッチコピー等）はタイトルに含めないでください。"
)

#: Vision サブクエリのシステムプロンプト（run_vision_subquery の既定文言を専門家向けに上書き）。
MANGA_COVER_SYSTEM_PROMPT: str = (
    "あなたは漫画の表紙画像からタイトル・作者名・巻数を正確に抽出する専門家です。"
    "推測で埋めず、画像から読み取れない項目は必ずnullにしてください。日本語で回答してください。"
)

#: response_format による JSON Schema 強制（教訓ストア reflection と同じ手法。json_schema/strict）。
MANGA_COVER_RESPONSE_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "manga_cover_identify",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": ["string", "null"]},
                "author": {"type": ["string", "null"]},
                "volume": {"type": ["string", "null"]},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["title", "author", "volume", "confidence"],
        },
        "strict": True,
    },
}

#: Vision不可時の共通エラー文（有効化手順を案内し、LLMがzipファイル名ベースの推定に
#: フォールバックできるようにする）。
_VISION_UNAVAILABLE_MSG = (
    "Error: Vision モデルが利用できません。zip ファイル名から推定してください"
    "（設定方法: メインモデルを mmproj 付きで起動する、または config.json の "
    "delegate_server に \"vision\": true を設定してVision対応サーバーを指定してください）。"
)


def _resolve_cover_path(cover_path: str):
    """cover_path を検証し、実在する画像ファイルの絶対 Path を返す。

    manga_scan が返す cover はプロジェクトルート相対の文字列（例:
    ".pixie_notes/manga_covers/xxx.jpg"）のため、絶対パスでなければ get_project_data_path で
    解決する。存在確認に加え、`.pixie_notes/manga_covers/` 配下 または 画像拡張子の
    いずれかであることを要求する（無関係なファイルを誤って読ませないための最低限の防御）。

    Returns:
        (path, error): 検証OKなら (Path, None)。NGなら (None, "Error: ...")。
    """
    raw = (cover_path or "").strip()
    if not raw:
        return None, "Error: cover_path が空です"

    p = Path(raw)
    if not p.is_absolute():
        p = Path(get_project_data_path(raw))

    if not p.exists():
        return None, f"Error: 画像ファイルが存在しません ({cover_path})"
    if not p.is_file():
        return None, f"Error: ファイルではありません ({cover_path})"

    ext = p.suffix.lower().lstrip(".")
    try:
        under_covers = _covers_dir().resolve() in p.resolve().parents
    except OSError:
        under_covers = False
    if not under_covers and ext not in IMAGE_EXTS:
        return None, (
            "Error: 表紙画像として認識できません"
            f"（.pixie_notes/manga_covers 配下、または画像ファイルを指定してください）: {cover_path}"
        )

    return p, None


@register_tool(
    name="manga_identify_cover",
    description=(
        "漫画の表紙画像をVisionモデルに一度だけ渡し、タイトル・作者・巻数をJSONで抽出します"
        "（{title, author, volume, confidence}）。manga_scanが返したcoverパスを入力に取ります。"
        "Vision対応のLLMが利用できない環境ではErrorを返すので、その場合はzipファイル名から"
        "推定してください。"
    ),
    schema={
        "type": "object",
        "properties": {
            "cover_path": {
                "type": "string",
                "description": "manga_scanが返したcoverパス（.pixie_notes/manga_covers/配下の表紙画像）",
            },
        },
        "required": ["cover_path"],
    },
    prompt_desc=(
        "manga_identify_cover(cover_path): 表紙画像からVisionでタイトル/作者/巻数をJSON抽出"
        "（Vision利用可能時のみ・1枚1回のみ）"
    ),
    pack="manga",
)
def manga_identify_cover(cover_path: str) -> str:
    """engine.execute_tool のインターセプトでVision呼び出しに置き換わるスタブ。

    Vision経路が全く無い環境（メイン・delegate双方ともVision非対応）で到達した場合、
    または本関数がインターセプトを経由せず直接呼ばれた場合（例: 単体テスト）はここが
    実行され、パス検証のみ行った上でVision利用不可のエラーを返す。
    """
    _, err = _resolve_cover_path(cover_path)
    if err:
        return err
    return _VISION_UNAVAILABLE_MSG


# =====================================================
# manga_rename
# =====================================================

def _load_manifest() -> list:
    path = _manifest_path()
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_manifest(entries: list) -> None:
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def _append_manifest(entry: dict) -> None:
    entries = _load_manifest()
    entries.append(entry)
    _save_manifest(entries)


def _gc_backup_dir() -> None:
    """manga_backup/ の保持数が MANGA_BACKUP_MAX を超えたら古い順に削除する。"""
    d = _backup_dir()
    if not d.exists():
        return
    files = sorted((p for p in d.iterdir() if p.is_file()), key=lambda p: p.stat().st_mtime)
    excess = len(files) - MANGA_BACKUP_MAX
    for p in files[:max(0, excess)]:
        try:
            p.unlink()
        except OSError:
            pass


def _repack_single_root(zip_path: Path, root_dir: str, sanitized_title: str, new_zip_path: Path) -> None:
    """single_root 構造の zip を「展開 → ルートフォルダ改名 → 無圧縮再圧縮」する。

    再圧縮後のファイルは new_zip_path.parent 配下の一時ファイルとして書き出し、
    呼び出し側が最後に shutil.move で確定させる（例外時の中途半端な上書きを防ぐ）。
    """
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".zip.tmp", dir=str(new_zip_path.parent))
    os.close(tmp_fd)
    tmp_zip_path = Path(tmp_name)
    try:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            with zipfile.ZipFile(zip_path) as zf:
                for zi in zf.infolist():
                    decoded = _decode_zip_name(zi)
                    if zi.is_dir() or decoded.endswith("/"):
                        (td / decoded).mkdir(parents=True, exist_ok=True)
                        continue
                    dest = td / decoded
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(zi) as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out)

            old_root = td / root_dir
            new_root = td / sanitized_title
            if old_root.exists() and old_root != new_root:
                if new_root.exists():
                    shutil.rmtree(new_root)
                old_root.rename(new_root)
            elif not old_root.exists():
                new_root = td

            with zipfile.ZipFile(tmp_zip_path, "w", zipfile.ZIP_STORED) as zf_out:
                for f in sorted(new_root.rglob("*")):
                    if not f.is_file():
                        continue
                    arcname = str(Path(sanitized_title) / f.relative_to(new_root)).replace("\\", "/")
                    zi_out = zipfile.ZipInfo(arcname, date_time=time.localtime()[:6])
                    zi_out.compress_type = zipfile.ZIP_STORED
                    # 日本語ファイル名の文字化け対策: UTF-8 フラグを明示的に立てる
                    zi_out.flag_bits |= 0x800
                    with open(f, "rb") as fh:
                        zf_out.writestr(zi_out, fh.read())

        shutil.move(str(tmp_zip_path), str(new_zip_path))
    finally:
        if tmp_zip_path.exists():
            try:
                tmp_zip_path.unlink()
            except OSError:
                pass


@register_tool(
    name="manga_rename",
    description=(
        "1件のzipファイルを新タイトルにリネームします。既定はdry_run=true（計画のみ表示・"
        "未実行）。内部構造が single_root の場合は展開→フォルダ名変更→無圧縮再圧縮、"
        "flat/nested の場合はzipファイル名の変更のみです。適用時は "
        ".pixie_notes/manga_manifest.json に記録し、manga_undo で取り消せます。"
    ),
    schema={
        "type": "object",
        "properties": {
            "zip_path": {"type": "string", "description": "対象zipファイルの絶対パス（manga_scanのpathをそのまま使う）"},
            "new_title": {"type": "string", "description": "新しいタイトル（拡張子なし。禁止文字は自動で全角置換される）"},
            "dry_run": {"type": "boolean", "description": "true(既定)なら計画のみ返し実行しない。false で実際に適用する"},
        },
        "required": ["zip_path", "new_title"],
    },
    prompt_desc="manga_rename(zip_path, new_title, dry_run=True): zipを新タイトルにリネーム(既定dry_run・falseで適用・manifest記録)",
    pack="manga",
)
def manga_rename(zip_path: str, new_title: str, dry_run: bool = True) -> str:
    zpath = Path(zip_path)
    if not zpath.exists():
        return f"Error: zipファイルが存在しません ({zip_path})"
    if zpath.suffix.lower() != ".zip":
        return f"Error: zipファイルではありません ({zip_path})"

    sanitized, err = _sanitize_title(new_title)
    if err:
        return f"Error: {err}"

    try:
        with zipfile.ZipFile(zpath) as zf:
            names = [_decode_zip_name(zi) for zi in zf.infolist()]
    except (zipfile.BadZipFile, OSError) as e:
        return f"Error: zipとして開けません: {e}"

    structure, root_dir = _classify_structure(names)
    new_zip_path = zpath.parent / f"{sanitized}.zip"

    if new_zip_path.exists() and new_zip_path.resolve() != zpath.resolve():
        return f"Error: リネーム先が既に存在します ({new_zip_path})。別のタイトルを指定してください。"

    if structure == "single_root":
        method_desc = "single_root（展開→再圧縮、無圧縮格納）"
        internal_change = f"{root_dir}/ → {sanitized}/"
    elif structure == "flat":
        method_desc = "flat（zip名のみ変更・再圧縮なし）"
        internal_change = "変更なし（zip名のみ変更）"
    else:
        method_desc = "nested（zip名のみ変更・内部はネスト構造のため対象外）"
        internal_change = "変更なし（内部はネスト構造のため対象外）"

    if dry_run:
        lines = [
            "[DRY RUN] 適用内容:",
            f"  zip:    {zpath.name} → {new_zip_path.name}",
            f"  内部:   {internal_change}",
            f"  方式:   {method_desc}",
            "問題なければ dry_run=false で再実行してください。",
        ]
        if sanitized != new_title.strip():
            lines.append(f"（サニタイズ後のタイトル: {sanitized}）")
        backup_count = len(list(_backup_dir().glob("*"))) if _backup_dir().exists() else 0
        lines.append(f"バックアップ保持数: {backup_count}/{MANGA_BACKUP_MAX}（構造がsingle_rootの場合のみ生成されます）")
        return "\n".join(lines)

    # ---- 実適用 ----
    try:
        backup_path = None
        if structure == "single_root":
            backup_dir = _backup_dir()
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / zpath.name
            if backup_path.exists():
                backup_path = backup_dir / f"{zpath.stem}_{int(time.time())}{zpath.suffix}"
            # 先に原本を退避してから再圧縮する（new_zip_path が元の zpath と同名になる
            # エッジケースでも、再圧縮の読込元(backup_path)と書込先(new_zip_path)が
            # 衝突しないようにするため）。再圧縮失敗時は退避を元に戻す。
            shutil.move(str(zpath), str(backup_path))
            try:
                _repack_single_root(backup_path, root_dir, sanitized, new_zip_path)
            except Exception:
                shutil.move(str(backup_path), str(zpath))
                raise
            _gc_backup_dir()
        else:
            shutil.move(str(zpath), str(new_zip_path))

        _append_manifest({
            "ts": time.time(),
            "structure": structure,
            "original_zip": str(zpath),
            "new_zip": str(new_zip_path),
            "original_root": root_dir,
            "backup": str(backup_path) if backup_path else None,
        })
    except Exception as e:
        return f"Error: リネーム適用中にエラーが発生しました: {e}"

    lines = [
        "リネーム完了:",
        f"  zip:    {zpath.name} → {new_zip_path.name}",
        f"  内部:   {internal_change}",
        f"  方式:   {method_desc}",
    ]
    if backup_path:
        backup_count = len(list(_backup_dir().glob("*")))
        lines.append(f"  バックアップ: {backup_path}（保持数: {backup_count}/{MANGA_BACKUP_MAX}）")
    if sanitized != new_title.strip():
        lines.append(f"（サニタイズ後のタイトル: {sanitized}）")
    lines.append("取り消す場合は manga_undo を使用してください。")
    return "\n".join(lines)


# =====================================================
# manga_undo
# =====================================================

def _undo_one(entry: dict) -> tuple:
    """1件のmanifestエントリを復元する。Returns (ok, message)。"""
    new_zip = Path(entry["new_zip"])
    original_zip = Path(entry["original_zip"])
    backup = entry.get("backup")

    try:
        if backup:
            backup_path = Path(backup)
            if not backup_path.exists():
                return False, f"バックアップが見つかりません ({backup})。GCで削除された可能性があります。"
            if new_zip.exists():
                new_zip.unlink()
            original_zip.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backup_path), str(original_zip))
        else:
            if not new_zip.exists():
                return False, f"リネーム後のファイルが見つかりません ({new_zip})"
            shutil.move(str(new_zip), str(original_zip))
    except Exception as e:
        return False, f"復元中にエラー: {e}"

    return True, f"{new_zip.name} → {original_zip.name}"


@register_tool(
    name="manga_undo",
    description=(
        "manga_rename で適用した変更を .pixie_notes/manga_manifest.json の記録に基づき"
        "復元します。zip_path_or_all に現在のzipパスを1件指定するか、"
        "'all_last_batch' で記録済み全件を新しい順に取り消します。"
    ),
    schema={
        "type": "object",
        "properties": {
            "zip_path_or_all": {
                "type": "string",
                "description": "復元したいzipの現在のパス、または全件取消の 'all_last_batch'",
            },
        },
        "required": ["zip_path_or_all"],
    },
    prompt_desc="manga_undo(zip_path_or_all): manifestに基づきリネームを復元(1件 or 'all_last_batch'で全件)",
    pack="manga",
)
def manga_undo(zip_path_or_all: str) -> str:
    entries = _load_manifest()
    if not entries:
        return "取消可能な記録がありません（manga_manifest.json が空です）。"

    is_all = zip_path_or_all.strip().lower() == "all_last_batch"
    if is_all:
        targets = list(reversed(entries))
        remaining = []
    else:
        target_resolved = str(Path(zip_path_or_all))
        matches = [
            e for e in entries
            if str(Path(e["new_zip"])) == target_resolved or str(Path(e["original_zip"])) == target_resolved
        ]
        if not matches:
            return f"Error: 記録が見つかりません ({zip_path_or_all})"
        targets = [matches[-1]]
        remaining = [e for e in entries if e is not matches[-1]]

    results = []
    ok_count = 0
    for entry in targets:
        ok, msg = _undo_one(entry)
        results.append(f"{'OK' if ok else 'NG'}: {msg}")
        if ok:
            ok_count += 1

    if is_all:
        # 成功した分だけ manifest から除去（失敗分は記録を残し再試行可能にする）
        failed = [t for t, r in zip(targets, results, strict=True) if r.startswith("NG")]
        _save_manifest(list(reversed(failed)))
    else:
        _save_manifest(remaining)

    header = f"{ok_count}/{len(targets)} 件を復元しました。"
    return header + "\n" + "\n".join(results)
