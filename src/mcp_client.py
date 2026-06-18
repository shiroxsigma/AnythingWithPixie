import json
import queue
import subprocess
import threading
from typing import Any


class LightweightMCPClient:
    def __init__(self, command: list[str]):
        """
        MCPサーバーをサブプロセスとして起動し、通信を初期化する。
        例: command = ["npx", "-y", "@modelcontextprotocol/server-github"]
        """
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        self.response_queues: dict[int, queue.Queue] = {}
        self.request_id = 0
        self.lock = threading.Lock()

        # 1. stdout用のスレッド（メイン通信）
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.reader_thread.start()

        # 2. stderr用のスレッド（デッドロック回避＆ログ収集）
        self.stderr_thread = threading.Thread(target=self._read_stderr_loop, daemon=True)
        self.stderr_thread.start()

    def _read_loop(self):
        """サーバーの標準出力を常時監視し、メッセージをパースする。"""
        while self.process.poll() is None:
            line = self.process.stdout.readline()
            if not line:
                continue
            try:
                message = json.loads(line)
                if "id" in message:
                    msg_id = message["id"]
                    with self.lock:
                        if msg_id in self.response_queues:
                            self.response_queues[msg_id].put(message)
            except json.JSONDecodeError:
                pass

    def _read_stderr_loop(self):
        """
        stderr のバッファ溢れ（デッドロック）を防ぐための非同期リーダー。
        プロセスが終了するまで継続してストリームを読み捨てる。
        """
        for line in self.process.stderr:
            line_str = line.strip()
            if not line_str:
                continue
            # Pixieのコンソールにデバッグログとして流す
            print(f"[MCP Server Log] {line_str}")

    def _send_request(self, method: str, params: dict[str, Any] = None, timeout: int = 30) -> dict[str, Any]:
        """各種リクエストを送信し、結果を待機・取得する共通メソッド。"""
        with self.lock:
            self.request_id += 1
            current_id = self.request_id
            resp_queue = queue.Queue()
            self.response_queues[current_id] = resp_queue

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": current_id
        }
        if params is not None:
            payload["params"] = params

        # 送信
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()

        # 応答の待機
        try:
            response = resp_queue.get(timeout=timeout)
            return response
        except queue.Empty:
            return {"error": {"message": f"MCP timeout after {timeout}s"}}
        finally:
            with self.lock:
                if current_id in self.response_queues:
                    del self.response_queues[current_id]

    def initialize(self) -> dict[str, Any]:
        """サーバーとのハンドシェイクを行う。"""
        # initialize protocol as per JSON-RPC MCP
        # params contain protocol version and client capabilities
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {}
            },
            "clientInfo": {
                "name": "AnythingPixie",
                "version": "1.0.0"
            }
        }
        return self._send_request("initialize", params)

    def get_tool_list(self) -> list[dict[str, Any]]:
        """利用可能なツールの一覧を取得する。"""
        response = self._send_request("tools/list")
        if "error" in response:
            print(f"[MCP Error] tools/list: {response['error']}")
            return []

        # response["result"]["tools"] にツールのメタデータが入っている想定
        return response.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any], timeout: int = 30) -> str:
        """JSON-RPC形式でツール実行リクエストを送信し、応答を待機する。"""
        params = {
            "name": name,
            "arguments": arguments
        }
        response = self._send_request("tools/call", params, timeout=timeout)

        if "error" in response:
            return f"Error from MCP: {json.dumps(response['error'])}"

        # MCP結果の抽出 (content配列をテキストに結合)
        content = response.get("result", {}).get("content", [])
        return "\n".join([c.get("text", "") for c in content if c.get("type") == "text"])

    def stop(self):
        """サーバープロセスを終了する（terminate → wait → kill の段階的エスカレーション）。

        パイプを閉じ、待機中のリクエスト呼び出し側がタイムアウトまで無限ハング
        しないよう、保留中のキューに即時エラーを通知する。
        """
        if self.process.poll() is not None:
            return  # 既に終了済み
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        # パイプを閉じる（リソースリーク防止）
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            if pipe:
                try:
                    pipe.close()
                except Exception:
                    pass
        # 待機中の呼び出し側がタイムアウトまでハングしないよう即時エラー通知
        with self.lock:
            for q in self.response_queues.values():
                q.put({"error": {"message": "MCP server stopped"}})
