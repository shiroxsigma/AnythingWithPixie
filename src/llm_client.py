"""
AnythingPixie — LLM接続モジュール

llama-cpp-python (GGUF) / LM Studio (OpenAI互換API) の初期化と接続を管理する。
依存: config.py (N_CTX, MAX_TOKENS), 標準ライブラリ
"""

import contextlib
import ctypes
import json
import os
import sys
import time
import urllib.error
import urllib.request
import warnings

from config import MAX_TOKENS, N_CTX

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

class LMStudioBackend:
    """LM StudioのOpenAI互換APIエンドポイントを利用するバックエンド（ストリーミング対応）。"""

    def __init__(self, base_url: str, api_key: str = "lm-studio", model: str = "local-model",
                 overall_timeout: float = 180.0, read_idle_timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._n_ctx = self._fetch_n_ctx()
        # ストリーミング応答の全体タイムアウト（秒）。チャンク受信の有無にかかわらず
        # この時間を超えたら打ち切る。LM Studio が細切れに応答し続ける場合の無限待ち防止。
        self.overall_timeout = overall_timeout
        # 個々のソケット受信のアイドルタイムアウト（秒）。完全無応答の検知に使用。
        self.read_idle_timeout = read_idle_timeout
        # 直近の create_chat_completion 呼び出しで受け取った llama.cpp server の
        # timings フィールド（cache_n/prompt_n/prompt_ms 等）。診断用（prefix cache
        # ヒット率の可視化）。取得できなかった場合は None のまま（呼び出し元は
        # getattr で安全に参照すること）。
        self.last_timings = None

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
                               top_k=None, top_p=None, repeat_penalty=None, **kwargs):
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
        if repeat_penalty is not None:
            data["repeat_penalty"] = repeat_penalty

        req = urllib.request.Request(endpoint, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")

        try:
            response = urllib.request.urlopen(req, timeout=self.read_idle_timeout)
        except urllib.error.HTTPError as e:
            if e.code == 500:
                print("\n[警告] LM Studio HTTP 500 エラー。2秒後にリトライします...")
                time.sleep(2)
                try:
                    response = urllib.request.urlopen(req, timeout=self.read_idle_timeout)
                except urllib.error.HTTPError as e2:
                    msg = e2.read().decode('utf-8') if hasattr(e2, 'read') else str(e2)
                    print(f"\n[エラー] LM Studio APIエラー (リトライ失敗): {msg[:200]}")
                    yield {"choices": [{"delta": {"content": f"\n(API Error: HTTP {e2.code})"}}]}
                    return
            else:
                msg = e.read().decode('utf-8') if hasattr(e, 'read') else str(e)
                print(f"\n[エラー] LM Studio APIエラー: {msg[:200]}")
                yield {"choices": [{"delta": {"content": f"\n(API Error: {e})"}}]}
                return
        except urllib.error.URLError as e:
            print(f"\n[エラー] LM Studio接続エラー: {e}")
            yield {"choices": [{"delta": {"content": f"\n(API Error: {e})"}}]}
            return

        # 全体タイムアウト: urlopen の timeout は「個々のソケット受信」のみをカバーするため、
        # LM Studio が細切れに応答し続ける（各チャンク受信は短時間）場合の無限待ちを防ぐ。
        overall_deadline = time.monotonic() + self.overall_timeout

        # with で response を確実にクローズ（ジェネレータ中断時の socket リークも防止）。
        with response:
            if not stream:
                result_bytes = response.read()
                result = json.loads(result_bytes.decode("utf-8"))
                choice = result["choices"][0]
                message = choice.get("message", {})
                # llama-server は非ストリーミング応答のトップレベルに timings
                # （cache_n/prompt_n等）を含めることがある（診断用、例外安全に取得）。
                if isinstance(result.get("timings"), dict):
                    self.last_timings = result["timings"]
                yield {"choices": [{"delta": {
                    "content": message.get("content"),
                    "tool_calls": message.get("tool_calls"),
                    "role": message.get("role"),
                }, "finish_reason": choice.get("finish_reason")}]}
            else:
                for line in response:
                    # チャンク受信のたびに全体タイムアウトを監視
                    if time.monotonic() > overall_deadline:
                        print(
                            f"\n[警告] LM Studio の応答が全体タイムアウト"
                            f"({self.overall_timeout:.0f}s)に達したため打ち切ります。"
                        )
                        yield {"choices": [{"delta": {"content": ""}, "finish_reason": "error"}]}
                        break
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue
                    if line == "data: [DONE]":
                        break
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            chunk = json.loads(data_str)
                            # llama-server はストリーミングの最終チャンク（finish_reason
                            # が設定される時）にのみ timings を含める（timings_per_token
                            # を明示的に要求しない限り）。診断用に受け取り次第保存する。
                            if isinstance(chunk.get("timings"), dict):
                                self.last_timings = chunk["timings"]
                            if "choices" in chunk and chunk["choices"]:
                                yield chunk
                            elif "error" in chunk:
                                # llama-server / LM Studio は生成失敗時（コンテキスト超過等）に
                                # choices を含まない SSE error イベント（event: error の直後の
                                # data: {"error": {...}}）を返すことがある。実測: LM Studio に
                                # ロード時の実効コンテキストがモデル一覧APIの申告値より小さい
                                # インスタンスへ、tools定義込みの長いプロンプトを送ると発生する
                                # （"n_keep >= n_ctx" エラー）。従来はここで黙って無視され、
                                # node_plan 側からは「空応答」としてしか見えず真の原因が失われて
                                # いたため、エラーメッセージを content として表面化させる。
                                err = chunk["error"]
                                err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                                print(f"\n[エラー] LM Studio ストリームエラー: {err_msg[:300]}")
                                yield {"choices": [{"delta": {"content": f"\n(API Error: {err_msg[:300]})"}, "finish_reason": "error"}]}
                                break
                        except json.JSONDecodeError:
                            pass

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
                               top_k=None, top_p=None, repeat_penalty=None, **kwargs):
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

        backend = LMStudioBackend(
            base_url=base_url,
            api_key=lmstudio_config.get("api_key", "lm-studio") if lmstudio_config else "lm-studio",
            model=lmstudio_config.get("model", "local-model") if lmstudio_config else "local-model",
        )
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
