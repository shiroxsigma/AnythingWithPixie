"""
AnythingPixie — LLM接続モジュール

llama-cpp-python (GGUF) / LM Studio (OpenAI互換API) の初期化と接続を管理する。
依存: config.py (N_CTX, MAX_TOKENS), 標準ライブラリ
"""

import contextlib
import ctypes
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
import warnings

from .config import MAX_TOKENS, N_CTX, is_bonsai2_27b
from .turn_control import ControlledResponse, active_control

# =====================================================
# SuppressStderr — llama.cpp ログ抑制
# =====================================================

def _dummy_log_callback(level, text, user_data):
    pass

try:
    _log_callback_ctypes = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)(_dummy_log_callback)
    import llama_cpp
    _has_llama_cpp = True
except (ImportError, OSError):
    # ImportError: llama_cpp 未インストール
    # OSError: ctypes/ABI 不整合 (Windows で ImportError にならないことがある)
    # 広すぎる except Exception は、本来見えるべき ABI エラーを握り潰すため限定する。
    _has_llama_cpp = False
    warnings.warn(
        "llama_cpp を利用できません（未インストールまたはABI不整合）。"
        "LM Studio バックエンドのみ利用可能です。",
        ImportWarning,
        stacklevel=2,
    )


class _DummyWriter:
    def write(self, *args, **kwargs): pass
    def flush(self, *args, **kwargs): pass


class SuppressStderr(contextlib.AbstractContextManager):
    """llama.cppのCレベルのログ出力コールバックを上書きして完全に消去する。"""

    def __enter__(self):
        if _has_llama_cpp:
            llama_cpp.llama_log_set(_log_callback_ctypes, ctypes.c_void_p())
        self._old_stderr = sys.stderr
        sys.stderr = _DummyWriter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self._old_stderr


# =====================================================
# LM Studio バックエンド
# =====================================================

def _open_completion(request, timeout):
    control = active_control.get()
    if control is not None:
        return ControlledResponse(request, control)
    return urllib.request.urlopen(request, timeout=timeout)


def _error_body(error):
    try:
        return error.read().decode("utf-8")
    finally:
        error.close()


class LMStudioBackend:
    """LM StudioのOpenAI互換APIエンドポイントを利用するバックエンド（ストリーミング対応）。"""

    def __init__(self, base_url: str, api_key: str = "lm-studio", model: str = "local-model",
                 overall_timeout: float = 180.0, read_idle_timeout: float | None = None,
                 reasoning_effort: str | None = None):
        # Local Bonsai tool prompts took ~52s before responding in real tests.
        # Keep the overall deadline bounded; callers can override either limit.
        if read_idle_timeout is None:
            read_idle_timeout = 120.0 if is_bonsai2_27b(model) else 30.0
        for name, value in (("overall_timeout", overall_timeout), ("read_idle_timeout", read_idle_timeout)):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        # Bonsai defaults to xhigh on the server. Medium is its documented
        # balanced mode for shorter responses; it is not a hard time limit.
        self.reasoning_effort = reasoning_effort or ("medium" if is_bonsai2_27b(model) else None)
        self._n_ctx = self._fetch_n_ctx()
        # ストリーミング応答の全体タイムアウト（秒）。チャンク受信の有無にかかわらず
        # この時間を超えたら打ち切る。LM Studio が細切れに応答し続ける場合の無限待ち防止。
        self.overall_timeout = float(overall_timeout)
        # 個々のソケット受信のアイドルタイムアウト（秒）。完全無応答の検知に使用。
        self.read_idle_timeout = float(read_idle_timeout)
        # 直近の create_chat_completion 呼び出しで受け取った llama.cpp server の
        # timings フィールド（cache_n/prompt_n/prompt_ms 等）。診断用（prefix cache
        # ヒット率の可視化）。取得できなかった場合は None のまま（呼び出し元は
        # getattr で安全に参照すること）。
        self.last_timings = None
        # None=unknown/supported, False=the endpoint rejected llama.cpp's
        # request-level thinking budget extension once.
        self._thinking_budget_supported = None

    @classmethod
    def from_config(cls, server: dict):
        """Use the same connection settings at startup, switch, and embedding."""
        return cls(
            base_url=server.get("base_url", "http://localhost:1234/v1"),
            api_key=server.get("api_key", "lm-studio"),
            model=server.get("model", "local-model"),
            **{key: server[key] for key in ("overall_timeout", "read_idle_timeout", "reasoning_effort") if key in server},
        )

    def _fetch_n_ctx(self) -> int:
        """LM Studioの /v1/models から実際のコンテキスト長を取得する。
        取得できない場合はデフォルト32768を返す。
        """
        try:
            endpoint = f"{self.base_url}/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            req = urllib.request.Request(endpoint, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            # OpenAI互換フォーマット: data[] -> meta.n_ctx
            models = data.get("data", [])
            for m in models:
                if m.get("id") == self.model or self.model in m.get("id", ""):
                    n = m.get("meta", {}).get("n_ctx")
                    if n:
                        return int(n)
            # フォールバック: 最初のモデルの n_ctx を使う
            if models:
                n = models[0].get("meta", {}).get("n_ctx")
                if n:
                    return int(n)
        except Exception:
            pass
        return 32768

    def create_chat_completion(self, messages, *, max_tokens=MAX_TOKENS, temperature=0.7,
                               stream=True, tools=None, tool_choice="auto", response_format=None,
                               top_k=None, top_p=None, min_p=None, presence_penalty=None,
                               repeat_penalty=None,
                               thinking_budget_tokens=None, reasoning_effort=None, **kwargs):
        # 今回の呼び出し分の timings をリセット（前回呼び出しの値が誤って参照されないように）。
        self.last_timings = None
        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if tools:
            data["tools"] = tools
        # tool_choice: "auto"/"required"/"none" や {"type": "function", "function": {...}} を
        # そのまま透過する（llama-server の /v1/chat/completions が --jinja 時にネイティブ対応）。
        if tool_choice:
            data["tool_choice"] = tool_choice
        # response_format: JSON Schema 等による出力構造保証（未使用時は送らず互換性を保つ）。
        if response_format:
            data["response_format"] = response_format
        # モデル別サンプリングプロファイル（config.SAMPLING_PROFILES）由来の追加パラメータ。
        # top_k / repeat_penalty は OpenAI 標準にはないが、LM Studio / llama-server の
        # OpenAI互換 body 拡張フィールドとして両方とも同じキー名で受け付ける。
        # 指定時のみ body に含める（未指定時はサーバー既定値のまま・互換性維持）。
        if top_k is not None:
            data["top_k"] = top_k
        if top_p is not None:
            data["top_p"] = top_p
        if min_p is not None:
            data["min_p"] = min_p
        if presence_penalty is not None:
            data["presence_penalty"] = presence_penalty
        if repeat_penalty is not None:
            data["repeat_penalty"] = repeat_penalty
        effort = reasoning_effort if reasoning_effort is not None else getattr(self, "reasoning_effort", None)
        if effort is not None:
            data["reasoning_effort"] = effort
        if (thinking_budget_tokens is not None
                and getattr(self, "_thinking_budget_supported", None) is not False):
            data["thinking_budget_tokens"] = int(thinking_budget_tokens)

        req = urllib.request.Request(endpoint, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")

        def checked_choices(payload, *, streaming):
            if not isinstance(payload, dict):
                raise ValueError("Invalid completion response: expected an object")
            if "error" in payload:
                error = payload["error"]
                detail = error.get("message", error) if isinstance(error, dict) else error
                raise ValueError(f"Server error: {detail}")
            choices = payload.get("choices")
            if not isinstance(choices, list) or (not streaming and not choices):
                raise ValueError("Invalid completion response: missing choices")
            for choice in choices:
                if not isinstance(choice, dict):
                    raise ValueError("Invalid completion response: expected a choice object")
                if choice.get("finish_reason") == "error":
                    raise ValueError("Server ended completion with finish_reason=error")
                if choice.get("finish_reason") is not None and not isinstance(choice["finish_reason"], str):
                    raise ValueError("Invalid completion response: finish_reason must be text")
                field = "delta" if streaming else "message"
                message = choice.get(field, {} if streaming else None)
                if not isinstance(message, dict):
                    raise ValueError(f"Invalid completion response: expected a {field} object")
                for key in ("content", "reasoning_content"):
                    if message.get(key) is not None and not isinstance(message[key], str):
                        raise ValueError(f"Invalid completion response: {key} must be text")
                calls = message.get("tool_calls")
                if calls is not None and (not isinstance(calls, list)
                                          or any(not isinstance(call, dict) for call in calls)):
                    raise ValueError("Invalid completion response: tool_calls must be a list of objects")
                for call in calls or []:
                    if streaming and (type(call.get("index")) is not int or call["index"] < 0):
                        raise ValueError("Invalid completion response: tool call index must be a nonnegative integer")
                    function = call.get("function", {})
                    if not isinstance(function, dict):
                        raise ValueError("Invalid completion response: expected a tool function object")
                    for key in ("name", "arguments"):
                        if function.get(key) is not None and not isinstance(function[key], str):
                            raise ValueError(f"Invalid completion response: tool function {key} must be text")
            return choices

        # Keep all opening, retry and body-read failures in one terminal path.
        # TurnStopped and GeneratorExit inherit BaseException and must propagate.
        try:
            for attempt in range(2):
                try:
                    response = _open_completion(req, self.read_idle_timeout)
                    break
                except urllib.error.HTTPError as exc:
                    if attempt:
                        raise
                    if exc.code in {400, 422} and "thinking_budget_tokens" in data:
                        # Remember endpoints that reject the llama.cpp extension.
                        exc.close()
                        self._thinking_budget_supported = False
                        fallback_data = dict(data)
                        fallback_data.pop("thinking_budget_tokens", None)
                        req = urllib.request.Request(
                            endpoint, data=json.dumps(fallback_data).encode("utf-8"),
                            headers=headers, method="POST",
                        )
                    elif exc.code == 500:
                        print("\n[警告] LM Studio HTTP 500 エラー。2秒後にリトライします...")
                        exc.close()
                        control = active_control.get()
                        control.wait(2) if control else time.sleep(2)
                    else:
                        raise

            # The idle timeout alone cannot bound a continuously trickling stream.
            control = active_control.get()
            overall_deadline = time.monotonic() + (control.limits.stream_timeout if control else self.overall_timeout)
            # Also close the response if the caller stops consuming this generator.
            with response:
                if not stream:
                    result = json.loads(response.read().decode("utf-8"))
                    choice = checked_choices(result, streaming=False)[0]
                    message = choice["message"]
                    if isinstance(result.get("timings"), dict):
                        self.last_timings = result["timings"]
                    yield {"choices": [{"delta": {
                        "content": message.get("content"),
                        "reasoning_content": message.get("reasoning_content"),
                        "tool_calls": message.get("tool_calls"),
                        "role": message.get("role"),
                    }, "finish_reason": choice.get("finish_reason")}]}
                else:
                    completed = False
                    event_type = ""
                    for line in response:
                        if time.monotonic() > overall_deadline:
                            raise TimeoutError("Completion exceeded overall timeout")
                        line = line.decode("utf-8").strip()
                        if not line:
                            event_type = ""
                            continue
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                            continue
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if event_type == "error":
                            raise ValueError(f"Server error: {data_str[:300]}")
                        if data_str == "[DONE]":
                            completed = True
                            break
                        chunk = json.loads(data_str)
                        choices = checked_choices(chunk, streaming=True)
                        if isinstance(chunk.get("timings"), dict):
                            self.last_timings = chunk["timings"]
                        if choices:
                            if any(choice.get("finish_reason") for choice in choices):
                                completed = True
                            yield chunk
                    if not completed:
                        raise ValueError("Connection closed before completion")
        except Exception as exc:
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    detail = _error_body(exc)
                except Exception as read_error:
                    detail = f"Unable to read error body: {type(read_error).__name__}: {read_error}"
                detail = f"HTTP {exc.code}: {detail[:300]}"
            else:
                detail = f"{type(exc).__name__}: {exc}"
            print(f"\n[エラー] LM Studio APIエラー: {detail[:300]}")
            yield {"choices": [{"delta": {"content": f"\n(API Error: {detail[:300]})"},
                                  "finish_reason": "error"}],
                   "__llm_error__": detail}

    @property
    def n_ctx(self):
        return self._n_ctx

    def estimate_token_count(self, text: str) -> int:
        """LM Studio にはトークン化APIがないため文字数から概算する。"""
        return len(text) // 3


# =====================================================
# llama-cpp-python バックエンド
# =====================================================

class LlamaCppBackend:
    """llama-cpp-python GGUFバックエンドのラッパー。"""

    def __init__(self, model_path: str, n_ctx: int = N_CTX, n_gpu_layers: int = -1,
                 chat_handler=None):
        from llama_cpp import Llama
        with SuppressStderr():
            if chat_handler:
                self._llm = Llama(
                    model_path=model_path,
                    chat_handler=chat_handler,
                    n_ctx=n_ctx,
                    n_threads=4,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                )
            else:
                self._llm = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_threads=4,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                )

    def create_chat_completion(self, messages, *, max_tokens=MAX_TOKENS, temperature=0.7,
                               stream=True, tools=None, tool_choice="auto", response_format=None,
                               top_k=None, top_p=None, min_p=None, presence_penalty=None,
                               repeat_penalty=None, **kwargs):
        """llama-cpp-pythonのcreate_chat_completionに委譲する。"""
        call_kwargs = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        # response_format は llama-cpp-python 側が未使用時に None を渡すと非対応バージョンで
        # 例外になり得るため、指定時のみキーを追加する。
        if response_format:
            call_kwargs["response_format"] = response_format
        # モデル別サンプリングプロファイル（config.SAMPLING_PROFILES）由来の追加パラメータ。
        # llama-cpp-python の create_chat_completion はいずれもネイティブでキーワード対応。
        if top_k is not None:
            call_kwargs["top_k"] = top_k
        if top_p is not None:
            call_kwargs["top_p"] = top_p
        if min_p is not None:
            call_kwargs["min_p"] = min_p
        if presence_penalty is not None:
            call_kwargs["presence_penalty"] = presence_penalty
        if repeat_penalty is not None:
            call_kwargs["repeat_penalty"] = repeat_penalty
        return self._llm.create_chat_completion(**call_kwargs)

    @property
    def n_ctx(self):
        try:
            total = self._llm.n_ctx()
            return int(total) if total else N_CTX
        except Exception:
            return N_CTX

    @property
    def metadata(self):
        return self._llm.metadata

    def tokenize(self, text: str) -> list:
        try:
            return self._llm.tokenize(text.encode("utf-8"))
        except Exception:
            return [0] * (len(text) // 3)

    def estimate_token_count(self, text: str) -> int:
        """テキストのトークン数を正確に取得（フォールバックは文字数概算）。"""
        try:
            return len(self._llm.tokenize(text.encode("utf-8")))
        except Exception:
            return len(text) // 3


# =====================================================
# GGUF チャットテンプレート適用
# =====================================================

def _apply_chat_template_from_metadata(llm_backend: LlamaCppBackend) -> None:
    """GGUFメタデータからチャットテンプレートを抽出し、Jinja2ChatFormatter を設定する。"""
    template = llm_backend.metadata.get("tokenizer.chat_template", "")
    if not template:
        return
    try:
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter, chat_formatter_to_chat_completion_handler

        eos_token_id = int(llm_backend.metadata.get("tokenizer.ggml.eos_token_id", 2))
        # eos トークン文字列をメタデータから取得（非Qwenモデルで <|im_end|> 固定を避ける）。
        # ※ キー名は tokenizer.ggml.eos_token。存在しなければ ChatML 互換の <|im_end|> にフォールバック。
        eos_token = llm_backend.metadata.get("tokenizer.ggml.eos_token", "<|im_end|>")
        fmt = Jinja2ChatFormatter(
            template=template,
            eos_token=eos_token,
            bos_token="",
            stop_token_ids=[eos_token_id],
        )
        llm_backend._llm.chat_handler = chat_formatter_to_chat_completion_handler(fmt)
        print(f"[Chat Template] GGUF埋め込みテンプレートを適用しました ({len(template)}文字)")
    except Exception as e:
        print(f"[警告] チャットテンプレートの適用に失敗しました: {e}")


# =====================================================
# バックエンド初期化
# =====================================================

def initialize_backend(
    model_path: str,
    mmproj_path: str = "",
    lmstudio_config: dict = None,
    n_ctx: int = N_CTX,
    use_gpu: bool = True,
    use_vision_flag: str = None,
) -> tuple:
    """LLMバックエンドを初期化する。

    Args:
        model_path: GGUFモデルパス、または "LMSTUDIO"
        mmproj_path: マルチモーダルプロジェクターパス
        lmstudio_config: LM Studio接続設定
        n_ctx: コンテキストウィンドウサイズ
        use_gpu: GPU使用フラグ
        use_vision_flag: 'y'/'n'/None（対話プロンプト）

    Returns:
        (backend, use_vision, is_qwen35, is_lfm25, use_capture_suggestion) のタプル
    """
    is_qwen35 = False
    is_lfm25 = False  # [LFM専用] 不要時: この行 + 各 # [LFM専用] 行を削除
    use_vision = False
    use_capture_suggestion = False

    if model_path == "LMSTUDIO":
        print("\n=======================================================")
        print("LM Studio (ローカルAPI) 接続を開始します。")
        base_url = lmstudio_config.get("base_url", "http://localhost:1234/v1") if lmstudio_config else "http://localhost:1234/v1"
        print(f"ベースURL: {base_url}")

        backend = LMStudioBackend.from_config(lmstudio_config or {})
        print(f"コンテキスト長: {backend.n_ctx:,} トークン")
        is_lfm25 = "lfm" in (lmstudio_config or {}).get("model", "").lower()  # [LFM専用]

        if use_vision_flag == 'y':
            use_vision = True
            use_capture_suggestion = True
        elif use_vision_flag == 'n':
            pass
        else:
            ans = input("\n画像認識(Vision)機能を使用しますか？ (y/N): ").strip().lower()
            use_vision = (ans == 'y')
            if use_vision:
                use_capture_suggestion = True

    else:
        # mmprojの存在でVisionモデルかテキスト専用モデルかを判定
        use_vision = os.path.exists(mmproj_path)
        is_qwen35 = "qwen3.5" in model_path.lower()
        is_lfm25 = "lfm" in model_path.lower()  # [LFM専用]
        n_gpu_layers = -1 if use_gpu else 0

        if use_vision:
            if is_qwen35:
                from llama_cpp.llama_chat_format import Qwen35ChatHandler
                print("Qwen3.5-VL と画像推論モジュールを読み込んでいます（数秒〜数十秒かかります）...")
                with SuppressStderr():
                    chat_handler = Qwen35ChatHandler(clip_model_path=mmproj_path, verbose=False)
            else:
                from llama_cpp.llama_chat_format import Qwen3VLChatHandler
                print("Qwen3-VL と画像推論モジュールを読み込んでいます（数秒〜数十秒かかります）...")
                with SuppressStderr():
                    chat_handler = Qwen3VLChatHandler(clip_model_path=mmproj_path, verbose=False)

            backend = LlamaCppBackend(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                chat_handler=chat_handler,
            )
            print("\n=======================================================")
            print("画像認識ができる VLM（マルチモーダル）チャットを開始します。")
            use_capture_suggestion = True
        else:
            print("テキスト専用 LLM モデルを読み込んでいます（数秒かかる場合があります）...")
            backend = LlamaCppBackend(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
            )
            _apply_chat_template_from_metadata(backend)
            print("\n=======================================================")
            print("テキスト専用 LLM チャットを開始します。")

    return backend, use_vision, is_qwen35, is_lfm25, use_capture_suggestion  # [LFM専用] is_lfm25 追加
