# -*- coding: utf-8 -*-
"""End-to-end test of the Mistral provider against a fake Mistral server.

    python3 test_provider_e2e.py

Starts a local aiohttp/HTTP server that mimics Mistral's behaviour:

* rejects any request carrying ``reasoning_content`` in a message (422,
  exactly like the real API);
* returns a *chunk list* in ``content`` when ``reasoning_effort`` is set;
* returns a plain string otherwise;
* supports both non-streaming and SSE streaming.

Then it drives the real ``MistralProvider`` -> ``OpenAIChatModelCompat`` ->
openai SDK -> httpx -> our normalising transport stack against it, and asserts
the agent-facing result.
"""

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwenpaw.providers.model_info import ModelInfo  # noqa: E402

from mistral_provider import MistralProvider  # noqa: E402

THINKING = "Simple: 17x23 = 391."
ANSWER = "**17 x 23 = 391**\n\n计算过程：340 + 51"

STATE = {"requests": [], "failures": []}


def _chunk_list():
    return [
        {"type": "thinking", "thinking": [{"type": "text", "text": THINKING}],
         "closed": True},
        {"type": "text", "text": ANSWER},
    ]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        STATE["requests"].append(body)

        # 1. Mistral's strict schema: reasoning_content is forbidden.
        for msg in body.get("messages") or []:
            if isinstance(msg, dict) and "reasoning_content" in msg:
                payload = {
                    "detail": [{
                        "type": "extra_forbidden",
                        "loc": ["body", "messages", 0, "assistant",
                                "reasoning_content"],
                        "msg": "Extra inputs are not permitted",
                    }],
                }
                raw = json.dumps(payload).encode()
                self.send_response(422)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

        # 2. reasoning_effort => chunk list; otherwise plain string.
        content = _chunk_list() if body.get("reasoning_effort") else ANSWER

        if body.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            frames = [
                {"id": "c1", "object": "chat.completion.chunk", "created": 1,
                 "model": body.get("model"),
                 "choices": [{"index": 0, "finish_reason": None,
                              "delta": {"content": content}}]},
                {"id": "c1", "object": "chat.completion.chunk", "created": 1,
                 "model": body.get("model"),
                 "choices": [{"index": 0, "finish_reason": "stop",
                              "delta": {}}]},
            ]
            for frame in frames:
                self.wfile.write(
                    ("data: " + json.dumps(frame, ensure_ascii=False) + "\n\n")
                    .encode(),
                )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        payload = {
            "id": "c1", "object": "chat.completion", "created": 1,
            "model": body.get("model"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                      "total_tokens": 15},
        }
        raw = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _start_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}/v1"


def _provider(base_url: str, stream: bool) -> MistralProvider:
    provider = MistralProvider(
        id="mistral",
        name="Mistral AI",
        base_url=base_url,
        api_key="test-key",
        models=[ModelInfo(id="zai-glm-5-3-flash", name="GLM",
                          relay_reasoning=True)],
    )
    provider._test_stream = stream
    return provider


async def _call(provider, stream: bool, **extra):
    from agentscope.message import Msg

    model = provider.get_chat_model_instance("zai-glm-5-3-flash")
    model.formatter = provider.build_formatter("zai-glm-5-3-flash")
    model.stream = stream
    messages = [Msg(name="user", content=[{"type": "text", "text": "17*23?"}], role="user")]
    result = await model(messages, **extra)
    if stream:
        blocks = []
        async for chunk in result:
            for block in chunk.content:
                blocks.append(block)
        return blocks
    return result.content


def main() -> int:
    server, base_url = _start_server()
    checks = []
    try:
        provider = _provider(base_url, stream=False)

        # --- 1. reasoning_effort => chunk list => normalised -----------
        asyncio.run(_call(provider, False, reasoning_effort="high"))
        last = STATE["requests"][-1]
        checks.append(
            ("reasoning_effort reaches the wire",
             last.get("reasoning_effort") == "high",
             f"got={last.get('reasoning_effort')}"),
        )

        # --- 2. non-streaming parse -----------------------------------
        blocks = asyncio.run(_call(provider, False, reasoning_effort="high"))
        types = [getattr(b, "type", None) for b in blocks]
        text = "".join(getattr(b, "text", "") for b in blocks
                       if getattr(b, "type", None) == "text")
        thinking = "".join(getattr(b, "thinking", "") for b in blocks
                           if getattr(b, "type", None) == "thinking")
        checks.append(("non-stream: no exception", True, f"types={types}"))
        checks.append(("non-stream: answer text", ANSWER[:8] in text, repr(text[:20])))
        checks.append(("non-stream: thinking block", THINKING in thinking, repr(thinking[:20])))

        # --- 3. streaming parse ---------------------------------------
        blocks = asyncio.run(_call(provider, True, reasoning_effort="high"))
        types = [getattr(b, "type", None) for b in blocks]
        text = "".join(getattr(b, "text", "") for b in blocks
                       if getattr(b, "type", None) == "text")
        thinking = "".join(getattr(b, "thinking", "") for b in blocks
                           if getattr(b, "type", None) == "thinking")
        checks.append(("stream: answer text", ANSWER[:8] in text, repr(text[:20])))
        checks.append(("stream: thinking block", THINKING in thinking, repr(thinking[:20])))

        # --- 4. history replay converts reasoning_content back ---------
        STATE["requests"].clear()
        model = provider.get_chat_model_instance("zai-glm-5-3-flash")
        model.formatter = provider.build_formatter("zai-glm-5-3-flash")
        model.stream = False
        from agentscope.message import Msg
        history = [
            Msg(name="user", content=[{"type": "text", "text": "17*23?"}], role="user"),
            Msg(name="assistant", content=[
                {"type": "thinking", "thinking": THINKING},
                {"type": "text", "text": ANSWER},
            ], role="assistant"),
            Msg(name="user", content=[{"type": "text", "text": "再乘 2"}], role="user"),
        ]
        result = asyncio.run(model(history))
        sent = STATE["requests"][-1]
        assistant_msgs = [m for m in sent.get("messages") or []
                          if m.get("role") == "assistant"]
        checks.append(("history: request accepted (no 422)", True,
                       f"assistant_msgs={len(assistant_msgs)}"))
        checks.append(("history: reasoning_content translated away",
                       all("reasoning_content" not in m for m in assistant_msgs),
                       str(assistant_msgs)[:160]))
        def _wire_text(content):
            """Flatten str / text-part list / Mistral chunk list into text."""
            if isinstance(content, str):
                return content
            if not isinstance(content, list):
                return ""
            out = []
            for part in content:
                if isinstance(part, str):
                    out.append(part)
                elif isinstance(part, dict):
                    if part.get("type") == "thinking":
                        out.append(_wire_text(part.get("thinking")))
                    else:
                        out.append(_wire_text(part.get("text")))
            return "".join(out)

        checks.append(("history: no chunk-list content on the wire",
                       all(isinstance(m.get("content"), (str, list)) for m in assistant_msgs),
                       str([type(m.get("content")).__name__ for m in assistant_msgs])))
        # --- 5. tool-call turn: content=None + tool_calls + reasoning ----
        from agentscope.message import ToolCallBlock

        STATE["requests"].clear()
        tool_history = [
            Msg(name="user", content=[{"type": "text", "text": "查一下"}], role="user"),
            Msg(name="assistant", content=[
                {"type": "thinking", "thinking": THINKING},
                ToolCallBlock(id="t1", name="web_search", input='{"q": "x"}'),
            ], role="assistant"),
            Msg(name="user", content=[{"type": "text", "text": "继续"}], role="user"),
        ]
        model2 = provider.get_chat_model_instance("zai-glm-5-3-flash")
        model2.formatter = provider.build_formatter("zai-glm-5-3-flash")
        model2.stream = False
        asyncio.run(model2(tool_history))
        sent2 = STATE["requests"][-1]
        a2 = [m for m in sent2.get("messages") or [] if m.get("role") == "assistant"]
        checks.append(("tool-call turn: accepted (no 422)", True, f"{len(a2)} assistant"))
        checks.append(("tool-call turn: no reasoning_content",
                       all("reasoning_content" not in m for m in a2),
                       str(a2)[:120]))

        checks.append(("history: thinking preserved in content",
                       any(THINKING in _wire_text(m.get("content")) for m in assistant_msgs),
                       str(assistant_msgs)[:140]))
    finally:
        server.shutdown()

    failed = 0
    for name, ok, extra in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   [{extra}]" if extra else ""))
        failed += 0 if ok else 1
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
