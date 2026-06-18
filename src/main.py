"""
AnythingPixie -- Entry point + CLI loop + AppContext + model selection

CLI-based main module. Handles model selection, AppContext initialization,
and the interactive CLI chat loop with command support.
"""

import argparse
import difflib
import importlib.util
import json
import multiprocessing
import os
import platform
import sys

from config import (
    CONTEXT_BUFFER,
    DESTRUCTIVE_TOOLS,
    MAX_TOKENS,
    MODEL_DIR,
    N_CTX,
)
from llm_client import initialize_backend
from paths import get_data_path

# =====================================================
# AppContext -- shared application state
# =====================================================

class AppContext:
    """Holds shared state and capabilities for the entire application."""
    def __init__(self):
        self.llm = None               # LLMBackend (LlamaCppBackend or LMStudioBackend)
        self.llm_model_name = ""       # LM Studio のモデル名（表示用）
        self.use_vision = False
        self.is_qwen35 = False

        self.use_capture = False
        self.capture_bbox = None

        self.overlay_manager = None

        # UI callback functions
        self.update_overlay_func = None
        self.get_inner_bbox_func = None
        self.select_screen_area_func = None

        # Phase management
        self.phase = "EXECUTING"

        # Model compatibility flag: when True, role="tool" is sent as-is.
        # When False, converted to role="assistant" (for LM Studio + non-FC models).
        self.supports_tool_role: bool = False

        # 深度思考の強制フラグ（/deep コマンドでトグル）。True時は段階的判定をスキップし常に deep。
        self.force_deep: bool = False


# =====================================================
# Model selection
# =====================================================

def _load_lmstudio_servers(config_path):
    """config.json から LM Studio サーバーリストを読み込む。

    Returns:
        list[dict]: 各サーバー設定（base_url, api_key, model, name を含む）。
                     name が未設定の場合はホスト名を自動生成する。
    """
    if not os.path.exists(config_path):
        return []

    try:
        with open(config_path, encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception as e:
        print(f"[Warning] Failed to read config.json: {e}")
        return []

    servers = []
    for entry in config_data.get("servers", []):
        if "base_url" not in entry:
            continue
        from urllib.parse import urlparse
        parsed = urlparse(entry["base_url"])
        servers.append({
            "name": entry.get("name", parsed.hostname or entry["base_url"]),
            "base_url": entry["base_url"],
            "api_key": entry.get("api_key", "lm-studio"),
            "model": entry.get("model", "local-model"),
        })
    return servers


def select_model(model_dir):
    """List GGUF models in the directory and let the user choose one.

    LM Studio サーバーが config.json に定義されている場合、
    それらも選択肢に含まれる。
    """
    available_models = []  # (display_name, internal_id, config_dict_or_None)

    # Recursively search for GGUF files
    for root, _dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".gguf") and "mmproj" not in file.lower():
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, model_dir)
                available_models.append((rel_path, full_path, None))

    # config.json から LM Studio サーバーリストを読み込み
    config_path = get_data_path("config.json")
    servers = _load_lmstudio_servers(config_path)

    if not available_models and not servers:
        print(f"Error: No GGUF models in {model_dir} and no servers in config.json.")
        sys.exit(1)

    # GGUFモデルがない → サーバー選択メニュー
    if not available_models:
        if len(servers) == 1:
            # サーバーが1つだけなら自動接続
            s = servers[0]
            print(f"No GGUF models found. Connecting to LM Studio: {s['name']} ({s['base_url']})...")
            return "LMSTUDIO", "LMSTUDIO_MMPROJ", s
        # 複数サーバーから選択
        print("\n=== Available LM Studio Servers ===")
        for idx, s in enumerate(servers):
            print(f"[{idx + 1}] {s['name']} ({s['base_url']})")
        while True:
            try:
                choice = input("\nEnter the number of the server: ")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(servers):
                    selected = servers[choice_idx]
                    break
                else:
                    print("Invalid number. Please try again.")
            except ValueError:
                print("Please enter a number.")
        print(f"\n=> Selected: {selected['name']} ({selected['base_url']})\n")
        return "LMSTUDIO", "LMSTUDIO_MMPROJ", selected

    # GGUFモデルがある → モデル＋サーバーを一覧表示
    for s in servers:
        label = f"LM Studio: {s['name']} ({s['base_url']})"
        available_models.append((label, "LMSTUDIO", s))

    print("\n=== Available Models ===")
    for idx, (display_name, _, _) in enumerate(available_models):
        print(f"[{idx + 1}] {display_name}")

    while True:
        try:
            choice = input("\nEnter the number of the model to load: ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_models):
                selected_rel, selected_full, selected_config = available_models[choice_idx]
                break
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Please enter a number.")

    print(f"\n=> Selected: {selected_rel}\n")

    if selected_full == "LMSTUDIO":
        return selected_full, "LMSTUDIO_MMPROJ", selected_config

    # Check for mmproj (Vision projector) in the same directory as the selected model
    model_parent_dir = os.path.dirname(selected_full)
    mmproj_path = ""
    for file in os.listdir(model_parent_dir):
        if file.endswith(".gguf") and "mmproj" in file.lower():
            mmproj_path = os.path.join(model_parent_dir, file)
            print(f"[*] Auto-detected Vision projector: {file}")
            break

    return selected_full, mmproj_path, None


def _input_with_timeout(prompt, timeout=30):
    """指定秒数でタイムアウトするinput関数。タイムアウト時はNoneを返す。"""
    import threading

    result = [None]

    def get_input():
        try:
            result[0] = input(prompt)
        except Exception:
            result[0] = ""

    thread = threading.Thread(target=get_input)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    return result[0]  # Noneならタイムアウト


# =====================================================
# Shared helpers (CLI/GUI common)
# =====================================================

def _build_user_message(context, user_input, output_fn=None):
    """Build a user message (format differs between Vision/non-Vision). Common to CLI/GUI."""
    if context.use_vision:
        user_content = [{"type": "text", "text": user_input}]

        if context.use_capture and context.capture_bbox and context.get_inner_bbox_func:
            try:
                safe_bbox = context.get_inner_bbox_func(context.capture_bbox)
                from tools import grab_screen_and_encode
                img_url = grab_screen_and_encode(safe_bbox)
                if img_url:
                    user_content.append({"type": "image_url", "image_url": {"url": img_url}})
                else:
                    raise Exception("Screen capture via PowerShell failed or returned empty.")
            except Exception as e:
                if output_fn:
                    output_fn(f"\n[Warning] Real-time screen capture failed: {e}\n")
                else:
                    print(f"[Warning] Real-time screen capture failed: {e}")

        return {"role": "user", "content": user_content}
    else:
        return {"role": "user", "content": user_input}


# =====================================================
# Interactive mode callback (semi-auto)
# =====================================================

def _truncate_lines(text: str, max_lines: int = 6) -> str:
    """テキストを指定行数以内に切り詰める。"""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text.rstrip()
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines)}行中{max_lines}行表示)"


# ANSI カラーコード
_CLR_RED = "\033[31m"
_CLR_GREEN = "\033[32m"
_CLR_CYAN = "\033[36m"
_CLR_RESET = "\033[0m"


def _show_file_diff(path: str, search_block: str, replace_block: str, max_context_lines: int = 14):
    """ファイル内の対象箇所を特定し、Claude Code風の差分を表示する。

    1. ファイルを読み込み search_block に最も近い箇所を特定
    2. fromfile/tofile にファイル名を付けて unified_diff を生成
    3. 行番号付きで差分を表示
    """
    if not path or not os.path.isfile(path):
        return False

    try:
        with open(path, encoding="utf-8") as f:
            file_content = f.read()
    except UnicodeDecodeError:
        try:
            with open(path, encoding="cp932") as f:
                file_content = f.read()
        except Exception:
            return False
    except Exception:
        return False

    fname = os.path.basename(path)

    # --- ファイル内で search_block に最も近い位置を特定 ---
    search_lines = search_block.splitlines()
    file_lines = file_content.splitlines()
    match_start = _find_best_match(search_lines, file_lines)

    if match_start is not None:
        # マッチした箇所の前後コンテキストを含めて抽出
        context = 1
        extract_start = max(0, match_start - context)
        extract_end = min(len(file_lines), match_start + len(search_lines) + context)

        before_lines = file_lines[extract_start:extract_end]
        # before の中で search_block に対応する部分を replace_block で置換
        local_search = "\n".join(file_lines[match_start:match_start + len(search_lines)])
        if local_search == search_block:
            after_lines = (
                file_lines[extract_start:match_start]
                + replace_block.splitlines()
                + file_lines[match_start + len(search_lines):extract_end]
            )
        else:
            # インデント等が違う場合は近似マッチ — search_block を replace_block で置換して表示
            after_text = file_content.replace(search_block, replace_block, 1)
            after_all = after_text.splitlines()
            after_lines = after_all[extract_start:extract_end]

        start_line_no = extract_start + 1
    else:
        # マッチ位置が特定できない場合は全文比較
        before_lines = file_lines[:max_context_lines]
        after_text = file_content.replace(search_block, replace_block, 1)
        after_lines = after_text.splitlines()[:max_context_lines]
        start_line_no = 1

    # --- Claude Code風 unified diff を生成 ---
    diff_lines = list(difflib.unified_diff(
        before_lines, after_lines,
        fromfile=f"  {fname} (before)",
        tofile=f"  {fname} (after)",
        n=0, lineterm="",
    ))

    # --- 行番号付きで表示 ---
    print(f"  │  {_CLR_CYAN}Modified{_CLR_RESET}  {fname}")
    before_no = start_line_no
    after_no = start_line_no
    line_idx = 0
    displayed = 0
    for dl in diff_lines:
        if dl.startswith("---") or dl.startswith("+++"):
            continue  # fromfile/tofile 行はスキップ（既にファイル名を表示済み）
        if dl.startswith("@@"):
            print(f"  │  {_CLR_CYAN}{dl}{_CLR_RESET}")
            before_no = start_line_no
            after_no = start_line_no
            line_idx = 0
            continue
        if dl.startswith("-"):
            print(f"  │  {_CLR_RED}-{before_no + line_idx}: {dl[1:]}{_CLR_RESET}")
        elif dl.startswith("+"):
            print(f"  │  {_CLR_GREEN}+{after_no + line_idx}: {dl[1:]}{_CLR_RESET}")
        else:
            print(f"  │   {before_no + line_idx}: {dl}")
            after_no += 1
        if dl.startswith("-") or dl.startswith("+"):
            line_idx += 1
        else:
            before_no += 1
            after_no = before_no
            line_idx += 1
        displayed += 1
        if displayed >= max_context_lines:
            remaining = len(diff_lines) - displayed - 2  # header分を除く
            if remaining > 0:
                print(f"  │  ... ({remaining}行省略)")
            break

    return True


def _find_best_match(search_lines: list[str], file_lines: list[str]) -> int | None:
    """ファイル内で search_lines に最も一致する位置を返す（インデント無視・strip比較）。

    Returns:
        マッチ開始行インデックス、見つからなければ None
    """
    if not search_lines or not file_lines:
        return None

    # 1. 完全一致
    search_text = "\n".join(search_lines)
    file_text = "\n".join(file_lines)
    exact_idx = file_text.find(search_text)
    if exact_idx >= 0:
        lines_before = file_text[:exact_idx].count("\n")
        return lines_before

    # 2. strip比較で行単位マッチ
    stripped_search = [l.strip() for l in search_lines]
    for i in range(len(file_lines) - len(stripped_search) + 1):
        match = True
        for j, sl in enumerate(stripped_search):
            if not sl:
                continue  # 空行はスキップ
            if file_lines[i + j].strip() != sl:
                match = False
                break
        if match:
            return i

    # 3. 先頭行だけの部分一致（フォールバック）
    first_stripped = search_lines[0].strip() if search_lines else ""
    if len(first_stripped) >= 10:
        for i, fl in enumerate(file_lines):
            if first_stripped in fl.strip():
                return i

    return None


def _make_interactive_fn(auto_approve_timeout=10):
    """半自動モード用: ツール実行前にユーザー承認を求めるコールバックを生成する。

    カウントダウン中にキーを押すとタイマーが停止し、
    ← → 矢印キーで [Yes] [No] [Custom Input] を選択できる。
    Custom Input を選ぶとタイムアウトなしのテキスト入力モードになる。

    auto_approve_timeout: 自動承認までの秒数（Noneで無期限待機）。
    """
    import msvcrt
    import time as _time

    def interactive_fn(tool_calls, content):
        # 提案されたツール一覧を表示
        print("\n  ┌─ Proposed Actions ─────────────────┐")
        for i, tc in enumerate(tool_calls, 1):
            func = tc.get("function", {})
            name = func.get("name", "")
            args = json.loads(func.get("arguments", "{}"))
            marker = "📝" if name in DESTRUCTIVE_TOOLS else "📖"

            if name == "search_and_replace":
                # search_and_replace はファイル差分を表示（Claude Code風）
                path = args.get("path", "")
                search_block = args.get("search_block", "")
                replace_block = args.get("replace_block", "")
                fname = os.path.basename(path) if path else "?"
                print(f"  │ {marker} {i}. search_and_replace({fname})")
                shown = _show_file_diff(path, search_block, replace_block)
                if not shown:
                    # ファイルが読めない場合は search_block → replace_block の比較をフォールバック表示
                    diff = list(difflib.unified_diff(
                        search_block.splitlines(keepends=True),
                        replace_block.splitlines(keepends=True),
                        fromfile="search_block", tofile="replace_block", lineterm="",
                    ))
                    for dline in diff[:10]:
                        _d = dline.rstrip()
                        if _d.startswith("-"):
                            print(f"  │  {_CLR_RED}-{_d[1:]}{_CLR_RESET}")
                        elif _d.startswith("+"):
                            print(f"  │  {_CLR_GREEN}+{_d[1:]}{_CLR_RESET}")
                    if len(diff) > 10:
                        print(f"  │  ... ({len(diff) - 10}行省略)")
            elif name in ("write_file", "write_sections", "replace_lines"):
                # 他の書き込み系ツールは引数を整形表示
                path = args.get("path", "")
                fname = os.path.basename(path) if path else "?"
                print(f"  │ {marker} {i}. {name}(path={fname})")
                if "new_content" in args:
                    nc = args["new_content"]
                    preview = _truncate_lines(nc, max_lines=6)
                    print(f"  │   new_content ({len(nc)}文字):")
                    for pl in preview.split("\n"):
                        print(f"  │     {pl}")
            else:
                # 読み取り系ツール: 従来の1行表示（長い値は省略）
                parts = []
                for k, v in args.items():
                    v_str = str(v)
                    if "\n" in v_str or len(v_str) > 50:
                        first_line = v_str.split("\n")[0][:50]
                        parts.append(f"{k}={first_line}... ({len(v_str)}文字)")
                    else:
                        parts.append(f"{k}={v_str}")
                args_str = ", ".join(parts)
                print(f"  │ {marker} {i}. {name}({args_str})")
        print("  └────────────────────────────────────┘")

        options = ["Yes", "No", "Custom Input"]
        selected = 0  # デフォルト: Yes
        _flush = sys.stdout.flush

        def render(countdown=None):
            """メニュー行を同じ行に上書き描画。"""
            parts = []
            for i, opt in enumerate(options):
                pointer = "▸" if i == selected else " "
                parts.append(f"{pointer}[{opt}]")
            line = "  Execute? " + "  ".join(parts)
            if countdown is not None:
                line += f"   ({countdown})"
            sys.stdout.write("\r" + line + "\033[K")
            _flush()

        # --- Phase 1: カウントダウン（auto_approve_timeout 秒間） ---
        if auto_approve_timeout is not None:
            # 前回の Phase 2 で残ったキーバッファをクリア
            while msvcrt.kbhit():
                msvcrt.getwch()

            remaining = auto_approve_timeout
            interrupted = False
            while remaining > 0:
                render(countdown=remaining)
                deadline = _time.monotonic() + 1.0
                while _time.monotonic() < deadline:
                    if msvcrt.kbhit():
                        interrupted = True
                        break
                    _time.sleep(0.05)
                if interrupted:
                    break
                remaining -= 1

            if not interrupted:
                sys.stdout.write("\r  ⏱ [Auto-approved: no input received]\033[K\n")
                _flush()
                return tool_calls, None

        # --- Phase 2: 矢印キー選択（無期限待機） ---
        render()  # カウントダウンなしで再描画
        while True:
            ch = msvcrt.getwch()
            if ch in ('\xe0', '\x00'):
                # 特殊キー（矢印キー）
                ch2 = msvcrt.getwch()
                if ch2 in ('H', 'K'):  # ↑ または ←
                    selected = (selected - 1) % len(options)
                elif ch2 in ('P', 'M'):  # ↓ または →
                    selected = (selected + 1) % len(options)
                render()
            elif ch in ('\r', '\n'):
                break  # 現在の選択を確定
            elif ch.upper() == 'Y':
                selected = 0
                break
            elif ch.upper() == 'N':
                selected = 1
                break
            elif ch == '\t':  # Tabで巡回
                selected = (selected + 1) % len(options)
                render()
            elif ch == '\x03':  # Ctrl+C
                raise KeyboardInterrupt
            else:
                # その他のキー: カスタム入力モードへ遷移
                # 打った文字をバッファに戻して input() で受ける
                try:
                    msvcrt.ungetch(ch.encode('utf-8')[:1])
                except (OSError, ValueError):
                    pass
                sys.stdout.write("\r  Execute? > \033[K")
                _flush()
                custom = input().strip()
                if custom:
                    return [], custom
                render()  # 空入力ならメニューに戻る

        sys.stdout.write("\n")
        _flush()

        if selected == 0:  # Yes
            return tool_calls, None
        elif selected == 1:  # No
            return [], None
        else:  # Custom Input（タイムアウトなし）
            sys.stdout.write("  Execute? > ")
            _flush()
            custom = input().strip()
            return [], custom if custom else None

    return interactive_fn


# =====================================================
# CLI chat loop
# =====================================================

def run_cli_chat(context):
    """Main CLI chat loop with command support."""
    from engine import build_system_text, run_graph
    from state import AgentState
    from tools import set_state_board

    show_thinking = False
    memory_mode = True   # Default: memory mode ON
    semi_auto = True  # 半自動モード（各ツール実行前にユーザー承認を求める）
    force_deep = False  # /deep で強制深度思考（段階的判定をスキップ）

    agent_state = AgentState()

    # Inject state board into tools module
    set_state_board(agent_state.state_board)

    # Avoid UnicodeEncodeError on Windows console for emoji etc.
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    while True:
        try:
            # === 非同期タスク実行中かチェック ===
            is_async_waiting = bool(agent_state.state_board.waiting_for_async)
            input_timeout = agent_state.state_board.async_timeout if is_async_waiting else None

            if getattr(context, 'phase', 'EXECUTING') == "PLANNING_WAIT_OK":
                if is_async_waiting:
                    user_input = _input_with_timeout("If the plan looks good, enter 'ok' (or provide correction instructions) > ", timeout=input_timeout)
                    if user_input is None:
                        # タイムアウト時: 自動でpoll_process実行
                        user_input = f"/poll_async {agent_state.state_board.async_pid} {agent_state.state_board.async_log_file}"
                else:
                    user_input = input("If the plan looks good, enter 'ok' (or provide correction instructions) > ")
                if user_input.strip().lower() == 'ok':
                    context.phase = "EXECUTING"
                    if os.path.exists(get_data_path("PLANNING.md")):
                        with open(get_data_path("PLANNING.md"), encoding="utf-8") as f:
                            plan_content = f.read()
                        user_input = f"Please execute the following plan using the necessary tools.\n\n{plan_content}"
                        print("[System] Switching to execution phase.")
                    else:
                        print("[System] PLANNING.md not found. Switching to normal execution phase.")
                        continue
                else:
                    context.phase = "PLANNING"

            else:
                prompt_prefix = "[Planning] " if getattr(context, 'phase', 'EXECUTING') == "PLANNING" else ""
                if is_async_waiting:
                    user_input = _input_with_timeout(f"{prompt_prefix}You (polling timeout: {input_timeout}s): ", timeout=input_timeout)
                    if user_input is None:
                        # タイムアウト時: 自動でpoll_process実行
                        user_input = f"/poll_async {agent_state.state_board.async_pid} {agent_state.state_board.async_log_file}"
                else:
                    user_input = input(f"{prompt_prefix}You: ")
                    # Multi-line paste: if input starts with """, collect until closing """
                    if user_input.startswith('"""'):
                        lines = [user_input[3:]]  # remove opening """
                        if lines[0].rstrip().endswith('"""') and len(lines[0].rstrip()) > 3:
                            user_input = lines[0].rstrip()[:-3]
                        else:
                            while True:
                                line = input("... ")
                                if line.rstrip().endswith('"""') and len(line.rstrip()) > 3:
                                    lines.append(line.rstrip()[:-3])
                                    break
                                lines.append(line)
                            user_input = "\n".join(lines)

            if user_input.strip().lower() in ['quit', 'exit']:
                break
            if user_input.strip().lower() == '/think':
                show_thinking = not show_thinking
                print(f"[System] Thinking mode is now {'ON' if show_thinking else 'OFF'}.")
                continue
            if user_input.strip().lower() == '/deep':
                force_deep = not force_deep
                print(f"[System] Deep thinking mode is now {'ON (強制的に深度思考)' if force_deep else 'OFF (段階的思考深化に戻る)'}.")
                continue
            if user_input.strip().lower() == '/step':
                semi_auto = not semi_auto
                print(f"[System] Mode: {'Semi-auto (step-by-step)' if semi_auto else 'Full-auto'}")
                continue
            if user_input.strip().lower() == '/mem':
                memory_mode = not memory_mode
                if not memory_mode:
                    agent_state.chat_history.clear()
                print(f"[System] Memory mode is now {'ON' if memory_mode else 'OFF'}.")
                continue
            if user_input.strip().lower() == '/debug' or user_input.strip().lower().startswith('/debug '):
                if not getattr(context, 'debug_mode', False):
                    arg = user_input.strip()[6:].strip().lower()
                    if arg == 'full':
                        context.debug_mode = 'full'
                    else:
                        context.debug_mode = 'summary'
                    context.debug_turn = 0
                    debug_dir = get_data_path("debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    print(f"[System] Debug mode: {context.debug_mode.upper()} → .pixie_notes/debug/turn_NNN.md")
                else:
                    context.debug_mode = False
                    print("[System] Debug mode: OFF")
                continue
            if user_input.strip().lower() == '/reset':
                agent_state.chat_history.clear()
                agent_state.reset_for_new_turn()
                print("[System] Context reset. Chat history cleared.")
                continue

            if user_input.strip().lower() == '/context':
                # --- /context: コンテキスト使用量の可視化 ---
                from engine import _messages_to_text, estimate_tokens, get_total_context
                total_ctx = get_total_context(context.llm)
                safe_max = max(1000, int(total_ctx) - int(MAX_TOKENS) - int(CONTEXT_BUFFER))
                soft_threshold = int(safe_max * 0.70)

                sys_msg = {"role": "system", "content": ""}
                msgs = agent_state.chat_history.get_messages(sys_msg)
                prompt_text = _messages_to_text(msgs)
                token_count = estimate_tokens(context.llm, prompt_text)
                usage_pct = token_count / safe_max if safe_max > 0 else 1.0

                # メッセージ内訳
                role_counts = {"assistant": 0, "user": 0, "tool": 0, "system": 0}
                role_tokens = {"assistant": 0, "user": 0, "tool": 0, "system": 0}
                for m in msgs:
                    role = m.get("role", "")
                    if role in role_counts:
                        role_counts[role] += 1
                        text = m.get("content", "") or ""
                        if isinstance(text, list):
                            text = "\n".join(
                                item.get("text", "") for item in text if isinstance(item, dict)
                            )
                        role_tokens[role] += estimate_tokens(context.llm, text)
                        # tool_calls のトークンも概算
                        tc = m.get("tool_calls")
                        if tc:
                            role_tokens[role] += estimate_tokens(context.llm, json.dumps(tc))
                sys_tokens = role_tokens.get("system", 0)
                chat_tokens = token_count - sys_tokens

                bar_len = 32
                filled = int(bar_len * usage_pct)
                bar = "■" * filled + "□" * (bar_len - filled)

                def pct_bar(pct, width=bar_len):
                    f = int(width * min(pct, 1.0))
                    return "■" * f + "□" * (width - f)

                print("\n  === Context Usage ===")
                print(f"  Model:    {context.llm_model_name}")
                print(f"  n_ctx:   {total_ctx:,} tokens")
                print(f"  safe_max:{safe_max:,} tokens (n_ctx - max_output - buffer)")
                print()
                print(f"  Current: ~{token_count:,} tokens ({usage_pct:.1%})")
                print(f"  Soft trim @ 70%: {soft_threshold:,} tokens")
                print(f"  Hard trim @100%: {safe_max:,} tokens")
                print()
                print(f"  {bar} {usage_pct:.1%} / safe_max")
                print(f"  {pct_bar(1.0, bar_len)} soft@70%")
                print(f"  {pct_bar(1.0, bar_len)} hard@100%")
                print()
                print("  Breakdown:")
                if sys_tokens > 0:
                    print(f"    System prompt:  ~{sys_tokens:,} tokens ({sys_tokens/token_count*100:.1f}%)")
                print(f"    Chat history:   ~{chat_tokens:,} tokens ({chat_tokens/token_count*100:.1f}%)")
                if role_counts["assistant"]:
                    print(f"      [assistant]   {role_counts['assistant']} msgs / ~{role_tokens['assistant']:,} tokens")
                if role_counts["user"]:
                    print(f"      [user]       {role_counts['user']} msgs / ~{role_tokens['user']:,} tokens")
                if role_counts["tool"]:
                    print(f"      [tool]       {role_counts['tool']} msgs / ~{role_tokens['tool']:,} tokens")
                print()
                continue
            if context.select_screen_area_func and user_input.strip().lower() == '/recap':
                print("[System] Starting capture area selection.")
                if context.update_overlay_func:
                    context.update_overlay_func(None)
                if context.select_screen_area_func:
                    new_bbox = context.select_screen_area_func()
                    if new_bbox:
                        context.use_capture = True
                        context.capture_bbox = new_bbox
                        if context.update_overlay_func:
                            context.update_overlay_func(context.capture_bbox)
                        print(f"[*] Capture area set: {context.capture_bbox}")
                    else:
                        if context.use_capture:
                            print("[-] Area was not properly selected. Keeping previous area.")
                            if context.update_overlay_func:
                                context.update_overlay_func(context.capture_bbox)
                        else:
                            print("[-] Please set a capture area.")
                continue

            if user_input.strip().lower().startswith('/trace'):
                keyword = user_input.strip()[6:].strip()
                if keyword:
                    user_input = (
                        f"Use the research_code_paths tool to investigate the definition points and "
                        f"usage points of the keyword '{keyword}'."
                    )
                    print(f"[System] Starting investigation of keyword '{keyword}'.")
                else:
                    print("[System] Please specify a keyword to investigate (e.g. /trace max_tokens)")
                    continue

            if user_input.strip().lower().startswith('/api'):
                config_path = get_data_path("config.json")
                servers = _load_lmstudio_servers(config_path)

                if not servers:
                    print("[System] No LM Studio servers found in config.json.")
                    continue

                print("\n=== Available LM Studio Servers ===")
                for idx, s in enumerate(servers):
                    print(f"[{idx + 1}] {s['name']} ({s['base_url']})")

                while True:
                    choice = input("\nEnter the number of the server to switch to (or 'q' to cancel): ").strip()
                    if choice.lower() == 'q':
                        break

                    try:
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(servers):
                            selected = servers[choice_idx]
                            print(f"[System] Switching to LM Studio server: {selected['name']} ({selected['base_url']})...")
                            from llm_client import LMStudioBackend
                            context.llm = LMStudioBackend(selected['base_url'], selected.get('api_key', 'lm-studio'), selected.get('model', 'local-model'))
                            context.llm_model_name = selected.get('model', 'local-model')
                            print(f"[System] Successfully switched to {selected['name']}.")
                            break
                        else:
                            print("Invalid number. Please try again.")
                    except ValueError:
                        print("Please enter a number.")
                continue

            if user_input.strip().lower().startswith('/poll_async'):
                # 非同期プロセスのポーリング（/poll_async PID LOG_PATH の形式）
                parts = user_input.strip().split()
                if len(parts) >= 3:
                    try:
                        pid = int(parts[1])
                        log_file = " ".join(parts[2:])
                        from tools import execute_builtin_tool
                        poll_result = execute_builtin_tool("poll_process", {"pid": pid, "log_file": log_file})
                        print(f"[Async Poll]\n{poll_result}")
                        continue
                    except (ValueError, IndexError):
                        print("[System] Usage: /poll_async <PID> <LOG_FILE>")
                        continue
                else:
                    # 引数省略時はステートボードから取得
                    if agent_state.state_board.async_pid:
                        pid = agent_state.state_board.async_pid
                        log_file = agent_state.state_board.async_log_file
                        from tools import execute_builtin_tool
                        poll_result = execute_builtin_tool("poll_process", {"pid": pid, "log_file": log_file})
                        print(f"[Async Poll]\n{poll_result}")
                        continue
                    else:
                        print("[System] No async process is currently running.")
                        continue

            if not user_input.strip():
                continue

            # ============================================
            # Integrated inference loop (always uses run_graph)
            # ============================================
            agent_state.reset_for_new_turn()

            # Build user message with vision support
            user_msg = _build_user_message(context, user_input)

            # Apply memory mode
            if memory_mode:
                if agent_state.chat_history.messages and agent_state.chat_history.messages[0].get("role") == "system":
                    agent_state.chat_history.messages = agent_state.chat_history.messages[1:]
                agent_state.chat_history.add(user_msg["role"], user_msg["content"])
            else:
                agent_state.chat_history.clear()
                agent_state.chat_history.add(user_msg["role"], user_msg["content"])

            # Execute State Graph
            context.force_deep = force_deep
            interactive_callback = _make_interactive_fn() if semi_auto else None
            run_graph(
                context=context,
                state=agent_state,
                show_thinking=show_thinking,
                max_tokens=MAX_TOKENS,
                system_msg_builder=build_system_text,
                interactive_fn=interactive_callback,
            )

        except KeyboardInterrupt:
            print("\n\n[System] 処理が中断されました (Ctrl+C)。")
            print("※プログラムを完全に終了する場合は 'quit' または 'exit' と入力してください。")
            continue
        except Exception as e:
            error_msg = str(e)
            if "exceed context window" in error_msg.lower() or "failed completely" in error_msg.lower() or "batch size 1" in error_msg.lower():
                print("\n[Error] AI memory capacity (context size) limit reached!")
                print("Cause: Conversation history or loaded file (tool execution result) is too long.")
                print("Solution: Restart the app for the current task, or use '/mem' mode to temporarily disable memory.")
            else:
                print(f"\nAn error occurred: {e}")


# =====================================================
# Entry point
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(description="AnythingPixie LLM Local Chat")
    parser.add_argument("--no-capture", action="store_true", help="Force-disable screen capture functionality")
    parser.add_argument("--no-gui", action="store_true", help="Force-disable GUI and start in CLI mode")
    return parser.parse_args()


def setup_application(args):
    """Initialize the application: parse args, select model, create AppContext."""
    context = AppContext()

    # Prevent Unicode errors on Windows console
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stdin.reconfigure(encoding='utf-8', errors='replace')

    # Enable ANSI escape codes (colors) on Windows
    if platform.system() == "Windows":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        for handle_id in (-11, -10, -12):  # STD_OUTPUT, STD_INPUT, STD_ERROR
            handle = kernel32.GetStdHandle(handle_id)
            if handle:
                mode = ctypes.c_ulong()
                kernel32.GetConsoleMode(handle, ctypes.byref(mode))
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    sys.stdin.reconfigure(encoding='utf-8', errors='replace')

    # 1. LLM initialization
    MODEL_PATH, MMPROJ_PATH, LMSTUDIO_CONFIG = select_model(MODEL_DIR)

    ans_gpu = 'y'
    if MODEL_PATH != "LMSTUDIO":
        print("\n[GPU Acceleration]")
        ans_gpu = input("Use GPU? (Y/n): ").strip().lower()

    use_vision_flag = None  # Will prompt interactively in initialize_backend

    llm, use_vision, is_qwen35, use_capture_suggestion = initialize_backend(
        model_path=MODEL_PATH,
        mmproj_path=MMPROJ_PATH,
        lmstudio_config=LMSTUDIO_CONFIG,
        n_ctx=N_CTX,
        use_gpu=(ans_gpu != 'n'),
        use_vision_flag=use_vision_flag,
    )

    context.llm = llm
    context.llm_model_name = getattr(llm, 'model', '')
    context.use_vision = use_vision
    context.is_qwen35 = is_qwen35

    # 2. Load optional modules (capture/overlay)
    if not args.no_capture and use_capture_suggestion:
        if importlib.util.find_spec("capture"):
            from capture import OverlayManager, get_inner_bbox, select_screen_area

            context.overlay_manager = OverlayManager()
            context.update_overlay_func = context.overlay_manager.update_overlay
            context.get_inner_bbox_func = get_inner_bbox
            context.select_screen_area_func = select_screen_area

            context.use_capture = False

    print("\n=======================================================")
    print("Enter 'quit' or 'exit' to end the session.")
    print("Enter '/think' to toggle thinking process display.")
    print("Enter '/deep' to toggle forced deep thinking (skip shallow phase).")
    print("Enter '/step' to toggle between semi-auto and full-auto mode.")
    print("Enter '/mem' to toggle memory mode (default: ON).")
    print("Enter '/reset' to clear chat history and reset context.")
    print("Enter '/debug' to toggle debug context dump to file (/debug full for full-text).")
    if context.select_screen_area_func:
        print("Enter '/recap' to enable real-time screen capture by selecting an area.")
    print("Enter '/api' to switch LM Studio server.")
    print('Wrap input in """...""" for multi-line paste.')
    print("=======================================================\n")

    return context


def main():
    args = parse_args()
    context = setup_application(args)

    try:
        run_cli_chat(context)
    finally:
        if context.overlay_manager:
            context.overlay_manager.stop()
        if context.llm and hasattr(context.llm, 'close'):
            context.llm.close()


if __name__ == "__main__":
    # Required for Windows/multiprocessing
    multiprocessing.freeze_support()
    main()
