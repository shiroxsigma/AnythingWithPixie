"""LLM を追加呼び出しする機能を集約するモジュール（toggle/observe-only 方針）。

サブクエリ系（run_agent_subquery / run_vision_subquery / run_text_subquery）など、
メインの ReAct コンテキストとは独立に LLM を呼ぶ「コストがかかるコード」を
1 モジュールに閉じ込める。これにより engine ↔ subagent の循環 import を解消し、
LLM 追加呼び出しの有効化ゲートを局所化する。

依存方向:
- subagent → tools (execute_builtin_tool, registry_to_openai_tools 等): top-level OK
- subagent → engine_helpers (純粋関数・軽量ヘルパ): top-level OK（遅延 import 不要）
- subagent → engine: 依存なし（共有ヘルパは engine_helpers に集約済み）。
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import registry
import tools
from config import (
    DELEGATE_BUDGET_SEC,
    DELEGATE_CTX_USAGE_LIMIT,
    DELEGATE_MAX_STEPS,
    DELEGATE_MAX_TOKENS,
    DELEGATE_SUBAGENT_TOOLS,
    DELEGATE_SYSTEM_PROMPT,
    DELEGATE_TOOL_RESULT_CAP,
    REVIEW_BUDGET_SEC,
    REVIEW_MAX_STEPS,
    REVIEW_MAX_TOKENS,
    REVIEW_SYSTEM_PROMPT,
)
from engine_helpers import (
    FILE_EDIT_TOOLS,
    default_output_fn,
    estimate_tokens,
    is_simple_question,
)
from engine_helpers import (
    accumulate_tool_calls as _accumulate_tool_calls,
)
from engine_helpers import (
    detect_repetitive_content as _detect_repetitive_content,
)
from engine_helpers import (
    parse_native_tool_calls as _parse_native_tool_calls,
)
from engine_helpers import (
    safe_parse_args as _safe_parse_args,
)
from engine_helpers import (
    strip_all_thinking as _strip_all_thinking,
)
from llm_client import SuppressStderr
from paths import get_data_path
from tools import registry_to_openai_tools


def _collect_response(response):
    """LLMのレスポンスからテキストを収集する（dict / generator 両対応）。

    LMStudioLLM は stream=True がデフォルトのため、stream=False を指定しても
    generator が返る場合がある。llama-cpp-python は通常 dict を返す。
    """
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "")
    # generator（ストリーミング）の場合
    content = ""
    for chunk in response:
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            if "content" in delta:
                content += delta["content"]
    return content


def run_vision_subquery(llm, img_data_url: str, prompt: str = None) -> str:
    """画像専用のクリーンな1問1答でVLMを呼び出し、テキスト説明を返す。

    メインのReActコンテキストとは完全に独立した会話で画像を解析し、
    結果をテキストとして返すことで、コンテキスト汚染を防ぐ。

    Args:
        llm: llama-cpp-python の Llama インスタンス
        img_data_url: "data:image/jpeg;base64,..." 形式の画像データ
        prompt: 画像に対する質問（Noneの場合はデフォルトの解析プロンプト）

    Returns:
        VLMが生成した画像の説明テキスト
    """
    if prompt is None:
        prompt = "この画像の内容を詳細にテキスト化して報告してください。テキストが写っている場合は正確に書き起こしてください。"

    vision_messages = [
        {
            "role": "system",
            "content": "あなたは優秀な画像解析エージェントです。与えられた画像を詳細に観察し、何が写っているか、どんなテキストが含まれているかを客観的かつ正確に報告してください。日本語で回答してください。",
        },
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": img_data_url}}, {"type": "text", "text": prompt}],
        },
    ]

    response = llm.create_chat_completion(
        messages=vision_messages,
        max_tokens=1024,
        temperature=0.2,
        stream=False,
    )

    return _collect_response(response)


def run_text_subquery(llm, file_path: str, file_content: str, prompt: str = None) -> str:
    """テキストファイル専用のクリーンなクエリでVLM/LLMを呼び出し、要約や解析結果を返す。

    メインのReActコンテキストを汚染せずに長文ファイルを読み込み、
    結果をテキスト要約として返すことで、コンテキスト枯渇を防ぐ。

    Args:
        llm: llama-cpp-python の Llam インスタンス
        file_path: 解析対象のファイルパス（コンテキスト補足用）
        file_content: ファイルのテキスト内容
        prompt: ファイルに対する質問や要約指示

    Returns:
        LLMが生成した解析・要約テキスト
    """
    if prompt is None or not prompt.strip():
        prompt = "このファイルの内容を要約し、主要なクラス、関数、役割をMarkdown形式で簡潔にリストアップしてください。思考や出力は日本語のみで記載してください"

    text_messages = [
        {
            "role": "system",
            "content": "あなたは優秀なコード解析アシスタントです。与えられたファイルの内容を分析し、ユーザーの指示に従って正確に必要な情報だけを抽出・要約してください。日本語で回答してください。",
        },
        {
            "role": "user",
            "content": f"以下のファイル（{file_path}）の内容を解析してください。\n\n【指示】\n{prompt}\n\n【ファイル内容】\n```\n{file_content}\n```",
        },
    ]

    with SuppressStderr():
        response = llm.create_chat_completion(
            messages=text_messages,
            max_tokens=2048 * 2,
            temperature=0.2,
            stream=False,
        )

    return _collect_response(response)


def run_agent_subquery(llm, *, question, file_hints=None, focus=None, max_steps=None, supports_tool_role=False, mode="research", review_payload=None, review_system_prompt=None) -> str:
    """独立コンテキストで動くサブエージェント（調査 または レビュー）。

    メインの state.chat_history には一切触れず、ローカルの messages 配列で
    軽量 ReAct ループを回す。読み取り専用ツールのみ許可し、結論文字列だけ返す。
    Claude Code の Task ツールに相当する「コンテキストを汚さない委譲」を実現する。

    Args:
        llm: LLM バックエンド(LMStudioBackend / LlamaCppBackend)
        question: 調査・回答すべき質問（review モードではレビュー指示）
        file_hints: 調査開始のヒントとなるパス群(省略可)
        focus: 調査の焦点・制約(省略可)
        max_steps: 最大ステップ数(省略時モード別の既定値)
        supports_tool_role: role="tool" をそのまま送れるか(False なら user に変換)
        mode: "research"(既定・調査) または "review"(レビュー)。review のとき上限を
              REVIEW_* に差し替え、review_payload を初回 user メッセージに前置する。
              システムプロンプトは review_system_prompt 指定時はそれ、未指定時は
              REVIEW_SYSTEM_PROMPT(編集レビュー用)。ReAct 本体・読取専用強制は共通。
        review_payload: review モード専用。検証対象のテキスト（編集案・設計案など）
                        (ファイルパス・変更内容・目標 等)。research では無視。
        review_system_prompt: review モード専用。システムプロンプトを上書きする。
                              設計レビューでは REVIEW_DESIGN_SYSTEM_PROMPT を渡す。
                              未指定時は REVIEW_SYSTEM_PROMPT(編集レビュー用)。

    Returns:
        サブエージェントの結論文字列
    """
    is_review = mode == "review"
    # モード別にシステムプロンプトと上限を選択（ReAct 本体は共通）。
    # REVIEW_SUBAGENT_TOOLS は DELEGATE_SUBAGENT_TOOLS と同一のため再利用。
    # review モードで review_system_prompt の指定があればそれ優先（設計レビュー等）。
    if is_review and review_system_prompt:
        system_prompt = review_system_prompt
    elif is_review:
        system_prompt = REVIEW_SYSTEM_PROMPT
    else:
        system_prompt = DELEGATE_SYSTEM_PROMPT
    default_max_steps = REVIEW_MAX_STEPS if is_review else DELEGATE_MAX_STEPS
    budget_sec = REVIEW_BUDGET_SEC if is_review else DELEGATE_BUDGET_SEC
    gen_max_tokens = REVIEW_MAX_TOKENS if is_review else DELEGATE_MAX_TOKENS
    subagent_tools = DELEGATE_SUBAGENT_TOOLS

    max_steps = max_steps or default_max_steps
    # sorted で順序を固定 → system/tools 部が毎回同一になり LM Studio の
    # プレフィックスキャッシュがヒットする。
    tools_schema = registry_to_openai_tools(sorted(subagent_tools))

    # 固定システムプロンプト(動的注入禁止) + 初回 user メッセージ
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    user_parts = []
    if is_review and review_payload:
        user_parts.append(review_payload)
    user_parts.append(question)
    if file_hints:
        hints = file_hints if isinstance(file_hints, (list, tuple)) else [file_hints]
        hints = [str(h) for h in hints if h]
        if hints:
            user_parts.append("\n【調査ヒント】\n" + "\n".join(f"- {h}" for h in hints))
    if focus:
        user_parts.append(f"\n【焦点】\n{focus}")
    messages.append({"role": "user", "content": "\n".join(p for p in user_parts if p)})

    deadline = time.monotonic() + budget_sec
    last_content = ""
    stall_count = 0

    for _step in range(1, max_steps + 1):
        if time.monotonic() > deadline:
            break  # → 強制サマライズへ

        # サブエージェント自身のコンテキスト逼迫判定
        ctx_total = getattr(llm, "n_ctx", 32768)
        ctx_total = ctx_total() if callable(ctx_total) else ctx_total
        usage = estimate_tokens(llm, json.dumps(messages, ensure_ascii=False)) / max(1, int(ctx_total))
        use_tools = usage < DELEGATE_CTX_USAGE_LIMIT

        try:
            with SuppressStderr():
                gen = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=gen_max_tokens,
                    temperature=0.3,
                    stream=True,
                    tools=tools_schema if use_tools else None,
                    tool_choice="auto" if use_tools else None,
                )
                # LMStudioBackend は常に generator を返す
                chunks = list(gen)
        except Exception as e:
            return f"（サブエージェント: LLM 呼び出し失敗: {e}）"

        content, tool_calls = _accumulate_tool_calls(chunks)
        # GGUF モデルが <tool_call> テキストで呼ぶ場合のフォールバック
        if not tool_calls and content:
            content, tool_calls = _parse_native_tool_calls(content)

        content = _strip_all_thinking(content or "")

        if tool_calls:
            # assistant ターン(思考除去済み)を記録
            messages.append(
                {
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls,
                }
            )
            stall_count = 0
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                args = _safe_parse_args(func)
                if name not in subagent_tools:
                    result = f"Error: サブエージェントは読み取り専用ツールのみ使用可能 ({name})"
                else:
                    try:
                        result = tools.execute_builtin_tool(name, args)
                    except Exception as e:
                        result = f"Error: {name} 実行中の例外: {e}"
                # 独自キャップで切り詰め(グローバル _dynamic_max_chars に依存しない)
                if len(result) > DELEGATE_TOOL_RESULT_CAP:
                    result = result[:DELEGATE_TOOL_RESULT_CAP] + "\n...[サブエージェント内で切り詰め]..."
                if supports_tool_role:
                    messages.append(
                        {
                            "role": "tool",
                            "content": result,
                            "tool_call_id": tc.get("id", ""),
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"[ツール結果]\n{result}",
                        }
                    )
            continue  # 次ステップへ

        # tool_calls なし
        if content and content.strip():
            if _detect_repetitive_content(content):
                break  # 反復 → この content を結論とする
            return content  # 結論確定
        else:
            stall_count += 1
            if stall_count >= 2:
                break

    # ステップ上限/予算超過/ストール → 強制サマライズターン
    messages.append(
        {
            "role": "user",
            "content": "調査ステップ上限に達しました。判明した事実だけを基に、"
            "日本語で簡潔に結論をまとめてください(ツールは使わない)。",
        }
    )
    try:
        with SuppressStderr():
            gen = llm.create_chat_completion(
                messages=messages,
                max_tokens=gen_max_tokens,
                temperature=0.2,
                stream=True,
                tools=None,
                tool_choice=None,
            )
            chunks = list(gen)
    except Exception as e:
        return last_content or f"（サブエージェント: 結論生成失敗: {e}）"

    content, _ = _accumulate_tool_calls(chunks)
    content = _strip_all_thinking(content or "")
    return content or last_content or "（サブエージェント: 有意な結論を得られませんでした）"



# =====================================================
# /review 系（編集レビュー・設計レビュー・往復レビューループ）
# =====================================================
# observe-only: 編集は既に実行済み。これらの関数は判定文字列を返すだけで編集結果を
# 書き換えず、例外時は "" を返して編集結果を絶対に壊さない。review_mode のガードは
# 呼び出し側（engine.execute_tool）で行う。


def _build_review_payload(tool_name: str, tool_args: dict, goal: str, new_blob: str) -> str:
    """reviewer の初回 user メッセージに前置する編集案テキストを組み立てる。"""
    def _snip(text: str, limit: int = 1500) -> str:
        text = (text or "").rstrip()
        if len(text) > limit:
            return text[:limit] + "\n…(省略)"
        return text

    path = str(tool_args.get("path", ""))
    parts = [f"【検証対象ファイル】\n{path or '(不明)'}"]
    if goal:
        parts.append(f"【編集の目標（コンテキスト）】\n{_snip(goal, 400)}")

    if tool_name == "search_and_replace":
        parts.append(
            f"【変更内容（search_and_replace）】\n"
            f"--- 置換対象(既存) ---\n{_snip(tool_args.get('search_block', ''))}\n"
            f"--- 新規 ---\n{_snip(new_blob)}"
        )
    elif tool_name == "replace_lines":
        parts.append(
            f"【変更内容（replace_lines: "
            f"{tool_args.get('start_line', '?')}-{tool_args.get('end_line', '?')}行）】\n"
            f"--- 新規 ---\n{_snip(new_blob)}"
        )
    elif tool_name == "append_to_file":
        parts.append(f"【変更内容（append_to_file: 末尾に追記）】\n{_snip(new_blob)}")
    else:  # write_file
        parts.append(f"【変更内容（write_file: ファイル全体）】\n{_snip(new_blob)}")

    parts.append("※ ファイルは実際に read_file で読み、上記の変更が適用された現在の状態を検証すること。")
    return "\n\n".join(parts)


def _run_edit_review(context, tool_name: str, tool_args: dict, output_fn) -> str:
    """破壊的ファイル編集の直後に読み取り専用レビューアを起動し、判定を observation 用に返す。

    observe-only: 編集は既に実行済み。本関数は判定文字列を返すだけで編集結果を書き換えず、
    例外時は "" を返して編集結果を絶対に壊さない。review_mode のガードは呼び出し側で行う。
    サーバ選択は _execute_delegate_research と同じく _delegate_server_lock/_counter で
    メイン/サブをラウンドロビン（delegate_llm が無ければメインのみ）。
    """
    try:
        sb = registry._state_board
        goal = getattr(sb, "goal", "") if sb else ""

        path = str(tool_args.get("path", ""))
        # ツール別の「新規内容」候補（replace_block / new_content / content）
        new_blob = (tool_args.get("replace_block")
                    or tool_args.get("new_content")
                    or tool_args.get("content")
                    or "")

        # 自明な編集はレビューしない（LLM 呼び出しの遅延回避・決定的ガード）
        if tool_name == "append_to_file" and len(new_blob) < 80:
            return ""
        if tool_name in ("replace_lines", "search_and_replace") and len(new_blob) < 20:
            return ""

        payload = _build_review_payload(tool_name, tool_args, goal, new_blob)

        # サーバ選択: delegate_llm(第2サーバ)があればラウンドロビン、なければ main
        global _delegate_server_counter
        delegate_llm = getattr(context, "delegate_llm", None)
        if delegate_llm is not None:
            with _delegate_server_lock:
                _delegate_server_counter += 1
                use_sub = (_delegate_server_counter % 2 == 0)
            llm_to_use = delegate_llm if use_sub else context.llm
        else:
            llm_to_use = context.llm

        fname = os.path.basename(path) if path else "(path?)"
        output_fn(f"[System] レビュー中: {fname}...\n", end="", flush=True)
        verdict = run_agent_subquery(
            llm_to_use,
            question="この編集案を検証し、指定フォーマットで判定と指摘を返してください。"
                     "実際にファイルを読んで確かめること。",
            mode="review",
            review_payload=payload,
            supports_tool_role=getattr(context, "supports_tool_role", False),
        )
        verdict = (verdict or "").strip()
        if not verdict:
            output_fn("[System] レビュー完了 (判定なし)\n", end="", flush=True)
            return ""
        if len(verdict) > 400:
            verdict = verdict[:400].rstrip() + "…"
        # 判定を observation に載せるだけでなくユーザー端末にも可視化する。
        # 本機能の目的は「レビューアの議論が見える」こと。CLI ではツール結果の
        # 本文が履歴行きで端末に表示されないため、ここで直接 print する。
        verdict_block = f"[レビュー結果]\n{verdict}"
        output_fn(f"[System] レビュー完了\n{verdict_block}\n\n", end="", flush=True)
        return verdict_block
    except Exception:
        # レビューアの失敗が編集結果を欠落させないようにする
        return ""


def _build_design_payload(user_request: str, answer: str, goal: str) -> str:
    """設計レビュー用の初回 user メッセージに前置するテキストを組み立てる。"""
    def _snip(text: str, limit: int = 4000) -> str:
        text = (text or "").rstrip()
        if len(text) > limit:
            return text[:limit] + "\n…(省略)"
        return text

    parts = []
    if user_request:
        parts.append(f"【ユーザの要求】\n{_snip(user_request, 1000)}")
    parts.append(f"【エージェントの設計/提案】\n{_snip(answer)}")
    if goal:
        parts.append(f"【目標（コンテキスト）】\n{_snip(goal, 400)}")
    return "\n\n".join(parts)


def _is_design_proposal(answer: str, user_text: str, code_mode: bool) -> bool:
    """final_answer が「設計/提案」らしく設計レビューの価値があるかを判定する。

    短すぎる回答・単純質問は除外。設計マーカー語を含むか code_mode なら True。
    """
    from config import REVIEW_DESIGN_MIN_CHARS
    if not answer or len(answer) < REVIEW_DESIGN_MIN_CHARS:
        return False
    if user_text and is_simple_question(user_text):
        return False
    if code_mode:
        return True
    design_markers = (
        "設計", "アーキテクチャ", "実装", "提案", "フェーズ", "ロードマップ",
        "構成", "方針", "ステップ", "技術スタック", "モジュール", "要件",
        "仕様", "アプローチ", "構造", "プラン", "スケジュール", "比較",
        "リスク", "トレードオフ", "選定", "ライブラリ",
    )
    return any(m in answer for m in design_markers)


def _run_design_review(context, answer: str, user_text: str, output_fn) -> str:
    """設計/提案の final_answer を読み取り専用レビューアで批判し、判定を返す（observe-only）。

    _run_edit_review と同形（try/except 全面ラップ・サーバラウンドロビン・進捗表示・
    400字切り詰め）。run_agent_subquery の review_system_prompt に
    REVIEW_DESIGN_SYSTEM_PROMPT を渡し、編集検証ではなく設計批判に切り替える。
    """
    try:
        from config import REVIEW_DESIGN_SYSTEM_PROMPT
        sb = registry._state_board
        goal = getattr(sb, "goal", "") if sb else ""

        payload = _build_design_payload(user_text, answer, goal)

        # サーバ選択: delegate_llm(第2サーバ)があればラウンドロビン、なければ main
        global _delegate_server_counter
        delegate_llm = getattr(context, "delegate_llm", None)
        if delegate_llm is not None:
            with _delegate_server_lock:
                _delegate_server_counter += 1
                use_sub = (_delegate_server_counter % 2 == 0)
            llm_to_use = delegate_llm if use_sub else context.llm
        else:
            llm_to_use = context.llm

        output_fn("[System] 設計レビュー中...\n", end="", flush=True)
        verdict = run_agent_subquery(
            llm_to_use,
            question="この設計/提案を批判的にレビューし、指定フォーマットで判定と指摘を返してください。"
                     "必要なら仕様書や既存コードを読んで照合すること。",
            mode="review",
            review_system_prompt=REVIEW_DESIGN_SYSTEM_PROMPT,
            review_payload=payload,
            supports_tool_role=getattr(context, "supports_tool_role", False),
        )
        verdict = (verdict or "").strip()
        if not verdict:
            output_fn("[System] 設計レビュー完了 (判定なし)\n", end="", flush=True)
            return ""
        if len(verdict) > 400:
            verdict = verdict[:400].rstrip() + "…"
        verdict_block = f"[レビュー結果(設計)]\n{verdict}"
        output_fn(f"[System] 設計レビュー完了\n{verdict_block}\n\n", end="", flush=True)
        return verdict_block
    except Exception:
        # 設計レビューの失敗が回答を欠落させないようにする
        return ""


def _verdict_is_clean(verdict: str) -> bool:
    """レビュー判定が「問題なし」（収束）かを判定する。"""
    return bool(verdict) and "問題なし" in verdict


def _collect_subquery_response(response) -> str:
    """LLMのサブクエリレスポンスからテキストを収集する（dict / generator 両対応）。"""
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "")
    # generator（ストリーミング）の場合
    content = ""
    for chunk in response:
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            if "content" in delta:
                content += delta["content"]
    return content


def _one_shot_revise(llm, user_request: str, current: str, verdict: str) -> str:
    """main の改善生成。レビューアの指摘を反映して current を書き直す（ツールなし・1往復分）。"""
    from config import REVIEW_LOOP_REVISE_MAX_TOKENS

    system_msg = (
        "あなたは設計者/実装者です。レビューアの指摘を忠実に反映し、元の意図と要件を保ちつつ、"
        "出力を改善してください。日本語で。思考は簡潔にし、改善版の全文をマークダウンで出力すること。"
    )
    user_msg = (
        f"【元の依頼】\n{user_request}\n\n"
        f"【現在の案】\n{current}\n\n"
        f"【レビューアの指摘】\n{verdict}\n\n"
        f"この指摘を取り込み、より良い案を出力してください。"
    )
    try:
        with SuppressStderr():
            response = llm.create_chat_completion(
                messages=[{"role": "system", "content": system_msg},
                          {"role": "user", "content": user_msg}],
                max_tokens=REVIEW_LOOP_REVISE_MAX_TOKENS,
                temperature=0.4,
                stream=True,
            )
        return _collect_subquery_response(response)
    except Exception as e:
        return f"（改善生成エラー: {e}）"


def run_review_loop(context, state, rounds=None, output_fn=None) -> str:
    """/review_loop: 直前の回答を main↔review で N 往復させて改善する。

    chat_history の最後の assistant メッセージを初期案とし、reviewer が指摘 → main が改善
    を最大 rounds 往復繰り返す（「問題なし」で早期収束）。各往復の review 指摘は output_fn
    で可視化し、最終的な改善案を返す（例外時は直前の案を返し結果を欠落させない）。
    `/review` トグルとは独立（明示起動）。
    """
    from config import (
        REVIEW_DESIGN_SYSTEM_PROMPT,
        REVIEW_LOOP_DEFAULT_ROUNDS,
        REVIEW_LOOP_MAX_ROUNDS,
    )
    output_fn = output_fn or default_output_fn

    # 初期案 = 最後の assistant メッセージ（think 剥離）
    base_output = ""
    for msg in reversed(state.chat_history.messages):
        if msg.get("role") == "assistant":
            base_output = _strip_all_thinking(msg.get("content", "") or "")
            break
    if not base_output.strip():
        output_fn("[System] レビュー対象の直前の回答がありません。\n")
        return ""

    # 元の要求 = 最後の実ユーザ入力（【システム をスキップ）
    user_request = ""
    for msg in reversed(state.chat_history.messages):
        if msg.get("role") != "user":
            continue
        txt = msg.get("content", "")
        txt = txt if isinstance(txt, str) else str(txt)
        if txt.startswith("【システム"):
            continue
        user_request = txt.strip()
        break

    # goal（コンテキスト補強）
    try:
        sb = registry._state_board
        goal = getattr(sb, "goal", "") if sb else ""
    except Exception:
        goal = ""

    if rounds is None:
        rounds = REVIEW_LOOP_DEFAULT_ROUNDS
    try:
        rounds = max(1, min(int(rounds), REVIEW_LOOP_MAX_ROUNDS))
    except (TypeError, ValueError):
        rounds = REVIEW_LOOP_DEFAULT_ROUNDS

    reviewer_llm = getattr(context, "delegate_llm", None) or context.llm
    revise_llm = context.llm
    current = base_output

    output_fn(f"\n[System] === Review Loop 開始 ({rounds}往復) ===\n")
    try:
        for i in range(1, rounds + 1):
            output_fn(f"\n--- Round {i}/{rounds} ---\n")
            # Review（読取専用サブエージェント）
            payload = _build_design_payload(user_request, current, goal)
            verdict = run_agent_subquery(
                reviewer_llm,
                question="この設計/提案を批判的にレビューし、指定フォーマットで判定と指摘を返してください。"
                         "必要なら仕様書や既存コードを読んで照合すること。",
                mode="review",
                review_system_prompt=REVIEW_DESIGN_SYSTEM_PROMPT,
                review_payload=payload,
                supports_tool_role=getattr(context, "supports_tool_role", False),
            )
            verdict = (verdict or "").strip()[:600]
            output_fn(f"[レビュー({i})]\n{verdict}\n")
            if _verdict_is_clean(verdict):
                output_fn("[System] 問題なしのため収束しました。\n")
                break
            # Revise（main が指摘を反映）
            output_fn("[System] main が指摘を反映して改善中...\n", end="", flush=True)
            revised = _one_shot_revise(revise_llm, user_request, current, verdict)
            if not revised or not revised.strip() or revised.startswith("（改善生成エラー"):
                output_fn("[System] 改善生成に失敗しました。直前の案を維持します。\n")
                break
            current = revised
            output_fn(f"[System] 改善完了({i}) ({len(current)}文字)\n")
    except Exception as e:
        output_fn(f"[System] Review Loop 中に例外が発生: {e}。直前の案を返します。\n")

    output_fn("\n[System] === Review Loop 終了 ===\n")
    return current


# =====================================================
# /verify 系（実行ベース検証 + 自動再編集）
# =====================================================
# /review（LLM判定・observe-only）や ruff（違反を付加するだけ）と違い、
# verify は「実際に実行して」エラーを検出し、それを根拠に自動で編集し直す
# クローズドループ（verify → fix → re-verify）。.venv の Python を優先使用。


def _run_ruff_check(file_path: str, python_exe: str = None) -> str:
    """編集後の .py を ruff で検査し、違反出力を返す。

    ruff 未導入・非 .py・タイムアウト・設定エラー時は "" を返す（非致命）。
    エージェントが observation 内で違反を即確認し、次ターンで修正できる。

    python_exe: ruff を起動する Python（省略時 sys.executable＝AnythingPixie 起動環境）。
    /verify では編集対象プロジェクトの .venv の Python を渡す（後方互換: 既存呼出は省略可）。
    """
    if not file_path or not str(file_path).endswith(".py") or not os.path.exists(file_path):
        return ""
    exe = python_exe or sys.executable
    cmd = [exe, "-m", "ruff", "check", "--select", "E,F",
           "--output-format=concise", str(file_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if result.returncode == 0:
        return ""
    if result.returncode >= 2 and not (result.stdout or "").strip():
        return ""  # ruff 自体の設定エラーは黙殺
    out = (result.stdout or "").strip()
    return f"[ruff check --select E,F]\n{out}\n" if out else ""


def _backup_if_file_edit(tool_name: str, tool_args: dict):
    """ファイル編集ツールの実行前に .bak バックアップを作成する。"""
    if tool_name not in FILE_EDIT_TOOLS:
        return
    file_path = tool_args.get("path", "")
    if not file_path or not os.path.exists(file_path):
        return
    try:
        backup_dir = get_data_path(".pixie_notes/backups")
        os.makedirs(backup_dir, exist_ok=True)
        src = os.path.abspath(file_path)
        bak_name = os.path.basename(src) + ".bak"
        shutil.copy2(src, os.path.join(backup_dir, bak_name))
    except Exception:
        pass  # バックアップ失敗で処理を止めない


def _resolve_verify_python(file_path: str) -> str:
    """検証実行に使う Python インタープリタを決定。

    編集対象ファイルを含むプロジェクトの .venv があればそれを優先、
    なければ AnythingPixie 起動の sys.executable にフォールバックする。
    """
    from paths import resolve_venv_python
    return resolve_venv_python(file_path) or sys.executable


def _read_file_for_verify(file_path: str, max_chars: int = None) -> str:
    """検証/修正生成用にファイルを読み込む（エラー耐性・切り詰め付き）。"""
    from config import VERIFY_ERROR_MAX_CHARS
    if max_chars is None:
        max_chars = VERIFY_ERROR_MAX_CHARS
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return ""
    if len(content) > max_chars:
        content = content[:max_chars] + "\n…(省略)"
    return content


def _run_py_compile(file_path: str, python_exe: str) -> str:
    """py_compile で構文を検査し、SyntaxError 等の出力を返す（成功時 ""）。

    副作用なし・安全な第1ゲート。_run_ruff_check と同じ subprocess.run 直接パターン
    （run_command 経由にしない — 30秒固定タイムアウト・PowerShell経由のオーバーヘッド回避）。
    """
    from config import VERIFY_COMPILE_TIMEOUT_SEC, VERIFY_ERROR_MAX_CHARS
    if not file_path or not str(file_path).endswith(".py") or not os.path.exists(file_path):
        return ""
    cmd = [python_exe, "-m", "py_compile", str(file_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace",
                                timeout=VERIFY_COMPILE_TIMEOUT_SEC)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if result.returncode == 0:
        return ""
    err = (result.stderr or "").strip() or (result.stdout or "").strip()
    if not err:
        return f"[py_compile] 失敗 (exit {result.returncode})"
    if len(err) > VERIFY_ERROR_MAX_CHARS:
        err = err[:VERIFY_ERROR_MAX_CHARS] + "\n…(省略)"
    return f"[py_compile]\n{err}"


def _run_import_check(file_path: str, python_exe: str) -> str:
    """AST で import 文を抽出し、python_exe 環境で find_spec して未解決モジュールを検出。

    副作用なし（import を実行しないので GUI/通信は起動しない）。py_compile 通過後
    （構文OK）に走らせる。サードパーティモジュールの未インストール/依存欠落を検出し、
    py_compile+ruff では見逃される実行時 ImportError を事前に捉える。
    未解決モジュールがあればエラー文字列、なければ ""。
    """
    from config import VERIFY_ERROR_MAX_CHARS, VERIFY_IMPORT_TIMEOUT_SEC
    if not file_path or not str(file_path).endswith(".py") or not os.path.exists(file_path):
        return ""
    # subprocess 内で動くスクリプト（対象 Python で find_spec を実行）
    script = (
        "import ast, importlib.util, sys\n"
        "f = sys.argv[1]\n"
        "try:\n"
        "    s = open(f, encoding='utf-8', errors='replace').read()\n"
        "except Exception:\n"
        "    sys.exit(0)\n"
        "try:\n"
        "    t = ast.parse(s)\n"
        "except SyntaxError:\n"
        "    sys.exit(0)\n"  # 構文エラーは py_compile ゲートに任せる
        "missing = []\n"
        "for node in ast.walk(t):\n"
        "    if isinstance(node, ast.Import):\n"
        "        for a in node.names:\n"
        "            try:\n"
        "                if importlib.util.find_spec(a.name) is None:\n"
        "                    missing.append(a.name)\n"
        "            except (ImportError, ValueError):\n"
        "                pass\n"
        "    elif isinstance(node, ast.ImportFrom):\n"
        "        mod = node.module or ''\n"
        "        if mod:\n"
        "            top = mod.split('.')[0]\n"
        "            try:\n"
        "                if importlib.util.find_spec(top) is None and importlib.util.find_spec(mod) is None:\n"
        "                    missing.append(mod)\n"
        "            except (ImportError, ValueError):\n"
        "                pass\n"
        "if missing:\n"
        "    uniq = list(dict.fromkeys(missing))\n"
        "    print('MISSING:' + ','.join(uniq))\n"
    )
    try:
        result = subprocess.run([python_exe, "-c", script, str(file_path)],
                                capture_output=True, text=True,
                                encoding="utf-8", errors="replace",
                                timeout=VERIFY_IMPORT_TIMEOUT_SEC)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    out = (result.stdout or "").strip()
    if out.startswith("MISSING:"):
        mods = out[len("MISSING:"):].strip()
        msg = (f"[import check] 解決不能なモジュール: {mods}"
               f"（{os.path.basename(python_exe)} 環境に未インストールの可能性）")
        if len(msg) > VERIFY_ERROR_MAX_CHARS:
            msg = msg[:VERIFY_ERROR_MAX_CHARS] + "\n…(省略)"
        return msg
    return ""


def _run_verify_pytest(file_path: str, python_exe: str, timeout_sec: int, max_chars: int) -> str:
    """pytest ゲート（副作用あり）。失敗時はトレースバック、成功時は "" を返す。

    pytest 未導入環境（No module named pytest）は "" でゲート無効扱い（誤検知防止）。
    """
    cmd = [python_exe, "-m", "pytest", str(file_path), "-x", "--no-header",
           "-q", "--tb=short", "-p", "no:cacheprovider"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout_sec)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return f"[pytest] 実行エラー/タイムアウト: {e}"
    if result.returncode == 0:
        return ""
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    if "No module named pytest" in combined:
        return ""  # pytest 未導入 → ゲート無効
    out = combined.strip()
    if not out:
        return f"[pytest] 失敗 (exit {result.returncode})"
    if len(out) > max_chars:
        out = out[:max_chars] + "\n…(省略)"
    return f"[pytest]\n{out}"


def _run_fast_gates(file_path: str, python_exe: str) -> str:
    """高速・決定的な検証ゲートのみを順に走らせる（short-circuit）。

    py_compile → import 解決 → ruff の3段のみ。pytest は含まない（副作用あり・
    コストも相対的に高いため常時実行の対象外）。LLM を一切使わないためコストは
    ほぼゼロで、VERIFY_FAST_GATE_ALWAYS が真の間は /verify トグルの状態に関わらず
    破壊的編集の直後に毎回呼ばれる（engine.execute_tool 参照）。
    全ゲート通過 / .py 以外 / 未存在ファイルは ""。
    """
    from config import VERIFY_ERROR_MAX_CHARS, VERIFY_IMPORT_GATE, VERIFY_RUFF_GATE
    if not file_path or not str(file_path).endswith(".py") or not os.path.exists(file_path):
        return ""

    # 1. py_compile ゲート
    err = _run_py_compile(file_path, python_exe)
    if err:
        return err

    # 2. import 解決ゲート（AST + find_spec・副作用なし・サードパーティ依存欠落を検出）
    if VERIFY_IMPORT_GATE:
        imp_err = _run_import_check(file_path, python_exe)
        if imp_err:
            return imp_err

    # 3. ruff ゲート（.venv の Python を使う。既存 _run_ruff_check を python_exe 指定で再利用）
    if VERIFY_RUFF_GATE:
        ruff_err = _run_ruff_check(file_path, python_exe)
        if ruff_err:
            if len(ruff_err) > VERIFY_ERROR_MAX_CHARS:
                ruff_err = ruff_err[:VERIFY_ERROR_MAX_CHARS] + "\n…(省略)"
            return ruff_err

    return ""


def _run_execution_verification(file_path: str, python_exe: str) -> str:
    """段階的検証ゲートを順に走らせ、最初の失敗でそのエラーを返す（short-circuit）。

    全ゲート通過で ""。.py 以外・未存在ファイルは ""（検証スキップ）。
    ゲート順（安価/安全 → 高コスト/副作用）:
      1-3. 高速ゲート（_run_fast_gates: py_compile → import解決 → ruff）
      4. pytest（VERIFY_TEST_GATE 時のみ・副作用あり）
    """
    from config import VERIFY_ERROR_MAX_CHARS, VERIFY_TEST_GATE, VERIFY_TEST_TIMEOUT_SEC
    if not file_path or not str(file_path).endswith(".py") or not os.path.exists(file_path):
        return ""

    fast_err = _run_fast_gates(file_path, python_exe)
    if fast_err:
        return fast_err

    # 4. pytest ゲート（副作用あり・デフォルト OFF）
    if VERIFY_TEST_GATE:
        test_err = _run_verify_pytest(file_path, python_exe,
                                      VERIFY_TEST_TIMEOUT_SEC, VERIFY_ERROR_MAX_CHARS)
        if test_err:
            return test_err

    return ""


def run_fast_gate_check(file_path: str) -> str:
    """VERIFY_FAST_GATE_ALWAYS: /verify トグルに関係なく常時実行する高速検出ゲート。

    py_compile → import解決 → ruff のみ（_run_fast_gates）を、編集対象ファイルを
    含むプロジェクトの .venv があればそれを優先して実行する（run_verify_fix_loop と
    同じ解決規則。_resolve_verify_python）。LLM は一切呼ばないため追加コストはほぼ
    ゼロで、検出のみを行う（自動修正は行わない・それは verify_mode 有効時の
    run_verify_fix_loop の役割）。.py 以外・未存在ファイルは "" を返す（検証対象外）。
    """
    if not file_path or not str(file_path).endswith(".py") or not os.path.exists(file_path):
        return ""
    python_exe = _resolve_verify_python(file_path)
    try:
        return _run_fast_gates(file_path, python_exe)
    except Exception:
        # 常時実行のパスなので、検証機構自体の例外でツール結果を壊さない。
        return ""


def _generate_fix_edit(llm, file_path: str, current_blob: str, error_text: str, goal: str):
    """検出エラーを解消する修正編集を LLM に生成させる（JSON をパースして dict 返却）。

    _one_shot_revise の LLM 呼出構造を踏襲。パース失敗/例外時は None（ループ側で安全スキップ）。
    戻り値: {"tool": "search_and_replace"|"write_file", "args": {...}} または None。
    """
    from config import VERIFY_FIX_MAX_TOKENS, VERIFY_FIX_SYSTEM_PROMPT

    def _snip(text, limit=2000):
        text = (text or "").rstrip()
        return text[:limit] + "\n…(省略)" if len(text) > limit else text

    user_msg = (
        f"【ファイル】\n{file_path}\n\n"
        f"【現在のファイル内容】\n{_snip(current_blob)}\n\n"
        f"【検出された実行エラー】\n{_snip(error_text, 1200)}\n\n"
    )
    if goal:
        user_msg += f"【編集の目標（コンテキスト）】\n{_snip(goal, 400)}\n\n"
    user_msg += "このエラーを解消する編集を、指定フォーマットの JSON 1件だけを出力してください。"

    try:
        with SuppressStderr():
            response = llm.create_chat_completion(
                messages=[{"role": "system", "content": VERIFY_FIX_SYSTEM_PROMPT},
                          {"role": "user", "content": user_msg}],
                max_tokens=VERIFY_FIX_MAX_TOKENS,
                temperature=0.2,
                stream=True,
            )
        raw = _collect_subquery_response(response)
    except Exception:
        return None

    raw = (raw or "").strip()
    # LLM が前後に文を置いた場合に備え、最初の { から最後の } を抽出
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    tool = parsed.get("tool")
    args = parsed.get("args")
    if tool not in ("search_and_replace", "write_file") or not isinstance(args, dict):
        return None
    if not args.get("path"):
        args["path"] = file_path
    return {"tool": tool, "args": args}


def _apply_fix_edit(context, fix: dict, file_path: str, output_fn) -> bool:
    """生成された修正編集を適用する（バックアップ付き・再帰防止）。

    execute_tool ではなく execute_builtin_tool を直接呼び、verify フックを再び踏まない。
    結果が "Error" 始まりでなければ True。
    """
    tool_name = fix.get("tool", "")
    tool_args = dict(fix.get("args", {}))
    _backup_if_file_edit(tool_name, tool_args)
    fname = os.path.basename(file_path) if file_path else "(path?)"
    try:
        result = tools.execute_builtin_tool(tool_name, tool_args)
    except Exception as e:
        output_fn(f"[System] 修正適用エラー({tool_name}, {fname}): {e}\n", end="", flush=True)
        return False
    ok = not (str(result) or "").startswith("Error")
    if ok:
        output_fn(f"[System] 修正適用: {tool_name}({fname})\n", end="", flush=True)
    else:
        output_fn(f"[System] 修正適用失敗({tool_name}, {fname}): {(result or '')[:200]}\n",
                  end="", flush=True)
    return ok


def run_verify_fix_loop(context, file_path: str, tool_name: str, tool_args: dict,
                        goal: str, output_fn) -> str:
    """/verify: ファイル編集後に実行ベース検証 → 自動修正を最大 N 往復（observe-and-fix）。

    run_review_loop と同形（try/except 全面ラップ・wall-clock 予算・最大ラウンド・
    例外時は最終状態を維持）。編集は既に実行済み。本関数は検証→修正を繰り返し、
    最終状態の検証サマリを observation 付加用に返す。例外時/未収束時は最終エラーを返し、
    編集結果を絶対に壊さない。.py 以外は ""（検証対象外・何も付加しない）。
    """
    from config import VERIFY_BUDGET_SEC, VERIFY_MAX_ROUNDS
    if not file_path or not str(file_path).endswith(".py"):
        return ""  # .py 以外は検証対象外

    from paths import resolve_venv_python
    venv_py = resolve_venv_python(file_path)
    python_exe = venv_py or sys.executable
    py_label = ".venv" if venv_py else "system"
    deadline = time.monotonic() + VERIFY_BUDGET_SEC
    last_error = ""
    rounds_done = 0

    output_fn(
        f"\n[System] === Verify-Fix Loop 開始 "
        f"(max {VERIFY_MAX_ROUNDS}往復, python={os.path.basename(python_exe)} [{py_label}]) ===\n"
    )
    try:
        for i in range(1, VERIFY_MAX_ROUNDS + 1):
            if time.monotonic() > deadline:
                output_fn("[System] 予算時間超過で終了します。\n")
                break
            rounds_done = i

            # 1. 実行検証
            error = _run_execution_verification(file_path, python_exe)
            if not error:
                output_fn(f"[System] ラウンド{i}: 検証クリア（実行エラーなし）。\n")
                output_fn("\n[System] === Verify-Fix Loop 終了 ===\n")
                return f"[検証結果]\n実行検証: 成功 ({rounds_done}ラウンド)"

            last_error = error
            output_fn(f"--- Round {i}/{VERIFY_MAX_ROUNDS} ---\n[検出エラー]\n{error[:400]}\n")

            # 2. 修正編集生成
            current_blob = _read_file_for_verify(file_path)
            fix = _generate_fix_edit(context.llm, file_path, current_blob, error, goal)
            if not fix:
                output_fn("[System] 修正編集の生成に失敗しました。ループを終了します。\n")
                break

            # 3. 修正適用（バックアップ付き・再帰防止）
            if not _apply_fix_edit(context, fix, file_path, output_fn):
                output_fn("[System] 修正編集の適用に失敗しました。ループを終了します。\n")
                break
            output_fn(f"[System] ラウンド{i}: 修正適用済み。再検証します。\n")
    except Exception as e:
        output_fn(f"[System] Verify-Fix Loop 中に例外が発生: {e}。最終状態を維持します。\n")

    output_fn("\n[System] === Verify-Fix Loop 終了 ===\n")
    snippet = (last_error[:600] if last_error else "(エラー詳細なし)")
    return f"[検証結果]\n実行検証: 未解決 ({rounds_done}ラウンド)\n最終エラー:\n{snippet}"


# =====================================================
# run_python 系（Python サンドボックス実行 + input() 自動入力）
# =====================================================
# Python コードを一時ディレクトリで python -u 実行し、input() のプロンプトを検出すると
# LLM が入力を生成して stdin に自動送信する（インタラクティブ自動入力）。
# run_verify_fix_loop と同形（try/except 全面ラップ・wall-clock 予算・例外時安全網）。

def _count_input_calls(code: str) -> int:
    """コード内の input() 呼出数を AST で数える（誤検出防止: 0 なら入力生成をスキップ）。

    構文エラー時は 0（実行時に SyntaxError を出させる）。AST で Call ノードの関数名が
    "input" のものだけ数える（input はビルトインなので ast.Name で判定）。
    """
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    n = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "input"):
            n += 1
    return n


def _looks_like_prompt(terminal: str) -> bool:
    """stdout 末尾の未改行断片が input() プロンプトらしいか。

    python -u により input(prompt) の prompt 文字列は即座にフラッシュされるので、
    末尾が : > ? ：？ のいずれか（＋末尾空白）で改行なしで終わる行なら入力待ちと推定。
    """
    from config import RUNPY_PROMPT_TAIL_RE
    if not terminal:
        return False
    line = terminal.rsplit("\n", 1)[-1]  # 末尾行（改行以降の未完了断片）
    return bool(re.search(RUNPY_PROMPT_TAIL_RE, line))


def _build_sandbox_env() -> dict:
    """サンドボックス実行用の環境変数を構築。

    os.environ のコピーから機密値(APIキー等)を削除する。ただし Windows で Python 起動に
    必要な PATH/SYSTEMROOT/TEMP/COMSPEC 等は残す（削ると起動自体が壊れる）。
    """
    drop_suffix = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    drop_prefix = ("OPENAI_", "ANTHROPIC_", "LM_")
    env = {}
    for k, v in os.environ.items():
        u = k.upper()
        if u.endswith(drop_suffix) or u.startswith(drop_prefix) or u == "API_KEY":
            continue
        env[k] = v
    return env


def _ensure_killed(proc) -> None:
    """子プロセスを確実に終了させる（stdin/stdout close → terminate → wait → kill）。

    mcp_client.stop / kill_process と同じ『確実に終わらせる』方針。
    Windows では terminate()==TerminateProcess。すでに終了済みなら何もしない。
    """
    if proc.poll() is not None:
        return
    for stream in (proc.stdin, proc.stdout):
        try:
            if stream is not None:
                stream.close()
        except Exception:
            pass
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass


def _generate_runpython_input(llm, code: str, prompt_text: str, input_history: list):
    """現在の input() プロンプトに対し、LLM が入力すべき1行を生成する。

    run_text_subquery / _generate_fix_edit と同じ try/except + SuppressStderr +
    _collect_subquery_response 構造。失敗時は None（ドライバが安全中断）。
    """
    from config import RUNPY_INPUT_MAX_TOKENS, RUNPY_INPUT_SYSTEM_PROMPT, RUNPY_INPUT_TEMPERATURE

    def _snip(text, limit=1500):
        text = (text or "").rstrip()
        return text[:limit] + "\n…(省略)" if len(text) > limit else text

    hist_block = ""
    if input_history:
        lines = [f"  プロンプト {h['prompt']!r} → 入力 {h['value']!r}"
                 for h in input_history[-5:]]
        hist_block = "【これまでの入力履歴(直近5件)】\n" + "\n".join(lines) + "\n\n"

    user_msg = (
        f"【実行中のPythonコード】\n```python\n{_snip(code)}\n```\n\n"
        f"{hist_block}"
        f"【現在のプロンプト(input() が表示した文字列。空文字ならプロンプトなしの input())】\n"
        f"{prompt_text!r}\n\n"
        f"このプロンプトに対してプログラムが期待する入力値を1行で出力してください。"
    )
    try:
        with SuppressStderr():
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": RUNPY_INPUT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=RUNPY_INPUT_MAX_TOKENS,
                temperature=RUNPY_INPUT_TEMPERATURE,
                stream=False,
            )
        raw = _collect_subquery_response(response)
    except Exception:
        return None
    raw = (raw or "").strip()
    if not raw:
        return None
    first = raw.splitlines()[0].strip()
    return first.strip("\"'")


def _runpy_driver_loop(proc, llm, code, stdin_seed, max_inputs, n_inputs, deadline, output_fn) -> str:
    """子プロセスをドライブし、input() プロンプトを検出して自動入力を送る。

    reader スレッドが stdout を read1(4096)（改行を待たず利用可能分を読む）で queue に流し、
    メインスレッドが queue.get(timeout=IDLE) で消費。末尾がプロンプトパターンで止まれば
    入力待ちと判定し LLM 生成値（または stdin_seed）を stdin に送る。プロセス終了 /
    総タイムアウト / max_inputs 到達で停止。

    read1 を使う理由: readline() は改行または EOF までブロックするため、input("名前: ") の
    ような「改行なしプロンプト」を検出できない。read1 は利用可能分だけ返すので即座に検出可能。

    戻り値: stdout（中央省略付き）＋ footer（停止理由・入力履歴・終了コード）。
    """
    import queue
    import threading

    from config import (
        RUNPY_IDLE_TIMEOUT_SEC,
        RUNPY_OUTPUT_MAX_CHARS,
        RUNPY_PROMPT_FALSEPOS_GRACE_SEC,
    )

    out_q = queue.Queue()
    SENTINEL = object()

    def _reader():
        # read1 で利用可能分だけ読む（改行を待たない）。EOF(空) で SENTINEL。
        # バイナリで読み、UTF-8 でデコード（text=False のため自前）。
        try:
            while True:
                chunk = proc.stdout.read1(4096)
                if not chunk:
                    break
                out_q.put(chunk.decode("utf-8", "replace"))
        except Exception:
            pass
        finally:
            out_q.put(SENTINEL)

    reader_th = threading.Thread(target=_reader, daemon=True)
    reader_th.start()

    captured = []
    input_history = []
    inputs_sent = 0
    seed_used = False
    last_terminal = ""        # 改行以降の未完了断片（プロンプト判定用）
    got_sentinel = False
    stop_reason = "完了"

    def _terminal_after(prev: str, chunk: str) -> str:
        """直前の未完了断片に chunk を結合し、最後の改行以降を返す。"""
        combined = prev + chunk
        if "\n" in combined:
            return combined.rsplit("\n", 1)[-1]
        return combined

    def _send_input(prompt_text):
        """現在の入力待ちに対し値を生成して送信。
        戻り値: True=送信成功 / False=max_inputs 到達 / None=生成失敗で中断推奨。"""
        nonlocal inputs_sent, seed_used, last_terminal
        if inputs_sent >= max_inputs:
            return False
        if stdin_seed is not None and not seed_used:
            value = stdin_seed
            seed_used = True
        else:
            value = _generate_runpython_input(llm, code, prompt_text, input_history)
            if value is None:
                return None
        # 1行に正規化（改行/復帰を空白に）
        value = value.replace("\n", " ").replace("\r", " ").strip()
        try:
            proc.stdin.write((value + "\n").encode("utf-8"))
            proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            return None
        inputs_sent += 1
        input_history.append({"prompt": prompt_text, "value": value})
        last_terminal = ""  # プロンプト解消
        output_fn(f"  [自動入力: {value}]\n", end="", flush=True)
        return True

    while True:
        if got_sentinel:
            break
        if time.monotonic() > deadline:
            stop_reason = "総タイムアウト"
            break

        remain = max(0.1, deadline - time.monotonic())
        wait = min(RUNPY_IDLE_TIMEOUT_SEC, remain)
        try:
            item = out_q.get(timeout=wait)
        except queue.Empty:
            # idle タイムアウト: プロセス生存中に出力がない → 引数なし input() の入力待ちの可能性
            if proc.poll() is None and n_inputs > 0:
                res = _send_input("(入力待ち・プロンプトなし)")
                if res is False:
                    stop_reason = f"max_inputs({max_inputs})到達"
                    break
                if res is None:
                    stop_reason = "入力生成失敗"
                    break
            continue

        if item is SENTINEL:
            got_sentinel = True
            continue

        captured.append(item)
        output_fn(item, end="", flush=True)
        last_terminal = _terminal_after(last_terminal, item)

        # プロンプト検出（末尾が : > ? 等で改行なし）
        if (n_inputs > 0 and inputs_sent < max_inputs
                and _looks_like_prompt(last_terminal)):
            # 短い猶予で追加出力を待つ → 来れば通常出力としてキャンセル（誤検出緩和）
            cancelled = False
            grace_end = time.monotonic() + RUNPY_PROMPT_FALSEPOS_GRACE_SEC
            while time.monotonic() < grace_end:
                try:
                    extra = out_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if extra is SENTINEL:
                    got_sentinel = True
                    break
                captured.append(extra)
                output_fn(extra, end="", flush=True)
                last_terminal = _terminal_after(last_terminal, extra)
                cancelled = True
                if not _looks_like_prompt(last_terminal):
                    break
            if not cancelled and not got_sentinel:
                res = _send_input(last_terminal or "(プロンプト)")
                if res is False:
                    stop_reason = f"max_inputs({max_inputs})到達"
                    break
                if res is None:
                    stop_reason = "入力生成失敗"
                    break

    # 残り出力を吸う
    while True:
        try:
            extra = out_q.get_nowait()
        except queue.Empty:
            break
        if extra is not SENTINEL:
            captured.append(extra)

    reader_th.join(timeout=2)

    full = "".join(captured)
    if len(full) > RUNPY_OUTPUT_MAX_CHARS:
        head = RUNPY_OUTPUT_MAX_CHARS // 2
        full = full[:head] + "\n…(省略)…\n" + full[-head:]

    footer = [f"[停止理由: {stop_reason}]", f"[自動入力: {inputs_sent}/{max_inputs} 回]"]
    if input_history:
        footer.append("[入力履歴]")
        for h in input_history:
            footer.append(f"  {h['prompt']!r} -> {h['value']!r}")
    rc = proc.returncode
    if rc is not None and rc != 0:
        footer.append(f"[終了コード: {rc}]")
    if stop_reason in ("総タイムアウト", f"max_inputs({max_inputs})到達"):
        footer.append(
            "[入力待ちで停止した可能性があります。stdin_seed で事前入力するか、"
            "max_inputs / timeout を増やして再実行してください。]"
        )
    return full + "\n" + "\n".join(footer)


def _execute_run_python(context, tool_args: dict, output_fn) -> str:
    """run_python のインターセプト処理（engine.execute_tool から呼ばれる）。

    コードを一時ファイルに書き出し python -u で Popen。stdout を read1 で監視し、
    input() プロンプトを検出したら LLM が入力を生成して stdin に送る。プロセス終了 /
    max_inputs 到達 / 総タイムアウトで終了。例外時も observation を壊さずエラー文字列を返す。
    """
    import tempfile

    from config import RUNPY_MAX_INPUTS, RUNPY_TOTAL_TIMEOUT_SEC
    from paths import resolve_venv_python

    code = str(tool_args.get("code", "") or "")
    if not code.strip():
        return "Error: code は必須です。"
    stdin_seed = tool_args.get("stdin_seed")
    if stdin_seed is not None:
        stdin_seed = str(stdin_seed)
    max_inputs = tool_args.get("max_inputs") or RUNPY_MAX_INPUTS
    timeout = tool_args.get("timeout") or RUNPY_TOTAL_TIMEOUT_SEC

    llm = getattr(context, "llm", None)
    if llm is None:
        return "Error: run_python の実行には LLM が必要です（CLI の LLM 未ロード状態では使えません）。"

    n_inputs = _count_input_calls(code)

    # python_exe: カレントディレクトリ起点で .venv を探索（なければ sys.executable）
    python_exe = sys.executable
    try:
        venv = resolve_venv_python(os.getcwd())
        if venv:
            python_exe = venv
    except Exception:
        pass

    deadline = time.monotonic() + timeout
    output_fn(
        f"\n[System] run_python 開始 "
        f"(python={os.path.basename(python_exe)}, input()検出数={n_inputs}, "
        f"max_inputs={max_inputs}, timeout={timeout}s)\n"
    )

    proc = None
    tmpdir_ctx = None
    return_value = None
    try:
        # TemporaryDirectory は with でなく手動管理: return 時の __exit__ で子プロセスが
        # まだディレクトリを掴んでいると cleanup が WinError 32 になるため、finally で
        # 必ず _ensure_killed（子終了）を先に行ってから cleanup する。
        tmpdir_ctx = tempfile.TemporaryDirectory(prefix="pixie_runpy_")
        tmpdir = tmpdir_ctx.name
        script_path = os.path.join(tmpdir, "snippet.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        env = _build_sandbox_env()
        # text=False（バイナリ）で扱い、reader は read1（改行を待たない）+ 自前 decode。
        # text=True の TextIOWrapper.read1 は期待通り動かないため。stderr は STDOUT に
        # マージ（stderr 用スレッド不要・デッドロック回避）。bufsize はデフォルト
        # （-1・BufferedReader）にし、read1 が「利用可能分だけ読む」仕様で動くようにする
        # （bufsize=0 の FileIO では read1 が即 EOF 判定になる）。
        proc = subprocess.Popen(
            [python_exe, "-u", script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=tmpdir,
            env=env,
        )
        result = _runpy_driver_loop(
            proc, llm, code, stdin_seed, max_inputs, n_inputs, deadline, output_fn
        )
        return_value = f"[run_python 実行結果]\n{result}"
    except Exception as e:
        return_value = f"Error: run_python の実行に失敗しました: {e}"
    finally:
        # 子プロセスを先に確実終了させてから一時ディレクトリを削除（WinError 32 回避）
        if proc is not None:
            _ensure_killed(proc)
        if tmpdir_ctx is not None:
            try:
                tmpdir_ctx.cleanup()
            except Exception:
                pass  # 削除失敗は無視（一時ディレクトリ・OS が後で消す）
    return return_value


# =====================================================
# delegate / analyze 系（独立サブエージェント調査・ファイル解析）
# =====================================================

def _get_file_hash(file_path: str) -> str:
    """ファイルのMD5ハッシュを計算する（キャッシュ検証用）。"""
    hasher = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


def _execute_analyze_file(context, tool_args: dict, output_fn) -> str:
    """analyze_file のインターセプト処理。

    ファイルハッシュベースのMarkdownキャッシュチェック → サブクエリ解析 → キャッシュ保存
    の一連の処理を行う。（RAGIndexerは使用しない）
    """
    file_path = str(tool_args.get("path", ""))
    analysis_prompt = tool_args.get("analysis_prompt")

    try:
        cache_dir = get_data_path(".pixie_notes")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "analysis_cache.md")

        # ファイルハッシュベースのキャッシュチェック
        current_hash = _get_file_hash(file_path)
        cached_entry = _lookup_analysis_cache(cache_path, file_path)

        if cached_entry and cached_entry.get("hash") == current_hash:
            output_fn(f"[System] Cache Hit: {os.path.basename(file_path)} は変更されていません。\n", end="", flush=True)
            # ステートボードにも記録
            try:
                sb = registry._state_board
                if sb:
                    sb.add_file_summary(file_path, cached_entry["result"][:500])
            except Exception:
                pass
            return cached_entry["result"]
        elif cached_entry:
            output_fn(f"[System] ファイル更新検出: {os.path.basename(file_path)} を再解析します...\n", end="", flush=True)

        # 実際の解析処理
        output_fn(f"[System] ファイルを裏で要約中({os.path.basename(file_path)})...\n", end="", flush=True)
        try:
            with open(file_path, encoding="utf-8") as f:
                file_content = f.read()
        except UnicodeDecodeError:
            with open(file_path, encoding="cp932") as f:
                file_content = f.read()
        # メイン/サブのラウンドロビン選択（delegate_research と共用カウンタ。
        # analyze_file も READONLY で並列実行されるため、複数同時にメイン/サブへ分散）
        global _delegate_server_counter
        delegate_llm = getattr(context, "delegate_llm", None)
        if delegate_llm is not None:
            with _delegate_server_lock:
                _delegate_server_counter += 1
                use_sub = (_delegate_server_counter % 2 == 0)
            llm_to_use = delegate_llm if use_sub else context.llm
        else:
            llm_to_use = context.llm
        result = run_text_subquery(llm_to_use, file_path, file_content, prompt=analysis_prompt)

        # 解析結果をMarkdownキャッシュに保存
        try:
            _save_analysis_cache(cache_path, file_path, current_hash, result)
            output_fn("[System] 解析結果をキャッシュに保存しました。\n", end="", flush=True)
        except Exception as cache_err:
            output_fn(f"[System] キャッシュ保存失敗: {cache_err}\n", end="", flush=True)

        # ステートボードにも記録
        try:
            sb = registry._state_board
            if sb:
                sb.add_file_summary(file_path, result[:500])
        except Exception:
            pass

        return result
    except Exception as e:
        return f"ファイルの読み込みまたは解析に失敗しました: {e}"


# 委譲サブエージェントのメイン/サブサーバー ラウンドロビン選択用（スレッドセーフ）。
# delegate_research は READONLY_TOOLS で並列実行されるため、複数スレッドから同時に
# _execute_delegate_research が呼ばれる。メイン/サブ2サーバーへ均等分散する。
_delegate_server_lock = threading.Lock()
_delegate_server_counter = 0


def _execute_delegate_research(context, tool_args: dict, output_fn) -> str:
    """delegate_research のインターセプト処理。

    独立コンテキストの調査サブエージェントを起動し、結論だけを返す。
    メインの state.chat_history には一切触れない（run_agent_subquery が保証）。
    """
    global _delegate_server_counter
    question = str(tool_args.get("question", "")).strip()
    if not question:
        return "Error: question は必須です。"
    file_hints = tool_args.get("file_hints")
    focus = tool_args.get("focus")
    max_steps = tool_args.get("max_steps")

    # メイン/サブのラウンドロビン選択（並列実行時の2サーバー分散）
    delegate_llm = getattr(context, "delegate_llm", None)
    if delegate_llm is not None:
        with _delegate_server_lock:
            _delegate_server_counter += 1
            use_sub = (_delegate_server_counter % 2 == 0)
        llm_to_use = delegate_llm if use_sub else context.llm
        # NOTE: supports_tool_role はアプリ全域フラグ。サブサーバーが別モデルで FC 非対応の
        # 場合、サブパスで不正確になるが、run_agent_subquery のネイティブ <tool_call> フォールバックが安全網。
        server_label = "(サブ鯖)" if use_sub else "(メイン鯖)"
    else:
        llm_to_use = context.llm
        server_label = ""

    output_fn(f"[System] 委譲サブエージェント起動{server_label}: 独立コンテキストで調査中...\n", end="", flush=True)
    try:
        conclusion = run_agent_subquery(
            llm_to_use,
            question=question,
            file_hints=file_hints,
            focus=focus,
            max_steps=max_steps,
            supports_tool_role=getattr(context, "supports_tool_role", False),
        )
    except Exception as e:
        return f"Error: サブエージェント実行中の例外: {e}"
    output_fn("[System] 委譲サブエージェント完了。\n", end="", flush=True)
    # [委譲調査の結論] ヘッダで、メインエージェントが生ツール出力と区別できるようにする
    return f"[委譲調査の結論]\n{conclusion}"


def _lookup_analysis_cache(cache_path: str, file_path: str) -> dict | None:
    """Markdownキャッシュからファイルの解析結果を検索する。

    キャッシュエントリの形式:
    ## filename.ext
    パス: `/path/to/file`
    ハッシュ: `md5hex`
    内容: 解析結果テキスト

    Returns:
        {"hash": "...", "result": "..."} or None
    """
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return None

    # エントリを分割して検索
    file_name = os.path.basename(file_path)
    # パターン: ## filename\nパス: `path`\nハッシュ: `hash`\n\nresult\n\n---
    entry_pattern = re.compile(
        rf'## {re.escape(file_name)}\s*\n'
        rf'パス: `{re.escape(file_path)}`\s*\n'
        rf'ハッシュ: `([a-f0-9]+)`\s*\n\n'
        rf'(.*?)\n\n---',
        re.DOTALL,
    )
    match = entry_pattern.search(content)
    if match:
        return {
            "hash": match.group(1),
            "result": match.group(2).strip(),
        }

    # 古い形式（ハッシュなし）のフォールバック
    old_pattern = re.compile(
        rf'## {re.escape(file_name)}\s*\n'
        rf'パス: `{re.escape(file_path)}`\s*\n\n'
        rf'(.*?)\n\n---',
        re.DOTALL,
    )
    old_match = old_pattern.search(content)
    if old_match:
        return {
            "hash": "",
            "result": old_match.group(1).strip(),
        }

    return None


def _save_analysis_cache(cache_path: str, file_path: str, file_hash: str, result: str):
    """解析結果をMarkdownキャッシュに保存（既存エントリがあれば置換）。"""
    file_name = os.path.basename(file_path)

    # 既存のエントリを削除
    if os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                content = f.read()
        except Exception:
            content = ""

        entry_pattern = re.compile(
            rf'\n*## {re.escape(file_name)}\s*\n.*?\n\n---\n*',
            re.DOTALL,
        )
        content = entry_pattern.sub('\n', content).strip()
    else:
        content = ""

    # 新しいエントリを追加
    new_entry = (
        f"\n## {file_name}\n"
        f"パス: `{file_path}`\n"
        f"ハッシュ: `{file_hash}`\n\n"
        f"{result}\n\n---\n"
    )
    content = content + "\n" + new_entry if content else new_entry

    with open(cache_path, "w", encoding="utf-8") as cf:
        cf.write(content)
