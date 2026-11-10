# -*- coding: utf-8 -*-
"""Offline tests for the Mistral response normaliser.

    python3 test_transport.py

No network, no API key: the transport is driven with a fake inner transport
that returns canned Mistral payloads (captured from a real response).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from mistral_transport import (  # noqa: E402
    MistralNormalizingTransport,
    rewrite_sse_event,
    split_chunks,
)

# Captured from a real Mistral response (zai-glm-5-3-flash, reasoning_effort=high)
REAL_CONTENT = [
    {
        "type": "thinking",
        "thinking": [{"type": "text", "text": "Simple: 17×23 = 391."}],
        "closed": True,
    },
    {"type": "text", "text": "**17 × 23 = 391**\n\n计算过程：…"},
]


class _FakeTransport(httpx.AsyncBaseTransport):
    def __init__(self, body: bytes, content_type: str):
        self.body = body
        self.content_type = content_type

    async def handle_async_request(self, request):
        return httpx.Response(
            status_code=200,
            headers={"content-type": self.content_type},
            content=self.body,
            request=request,
        )


async def _run_json_case() -> list[tuple[str, bool, str]]:
    payload = {
        "id": "abc",
        "object": "chat.completion",
        "created": 1,
        "model": "zai-glm-5-3-flash",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": REAL_CONTENT},
            }
        ],
    }
    transport = MistralNormalizingTransport(
        _FakeTransport(json.dumps(payload).encode(), "application/json"),
    )
    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    response = await transport.handle_async_request(request)
    body = json.loads(await response.aread())
    message = body["choices"][0]["message"]

    checks = [
        ("content is a string", isinstance(message["content"], str),
         f"type={type(message['content']).__name__}"),
        ("text chunk preserved", message["content"].startswith("**17 × 23 = 391**"),
         repr(message["content"][:20])),
        ("thinking moved to reasoning_content",
         message.get("reasoning_content") == "Simple: 17×23 = 391.",
         repr(message.get("reasoning_content"))),
        ("content-length header refreshed",
         response.headers.get("content-length") == str(len(json.dumps(body, ensure_ascii=False).encode())),
         response.headers.get("content-length")),
    ]
    # The rewritten body must be parseable by the openai SDK's own model.
    from openai.types.chat import ChatCompletion
    try:
        ChatCompletion.model_validate(body)
        checks.append(("openai SDK accepts rewritten body", True, ""))
    except Exception as exc:  # noqa: BLE001
        checks.append(("openai SDK accepts rewritten body", False, str(exc)[:120]))
    return checks


async def _run_sse_case() -> list[tuple[str, bool, str]]:
    events = [
        {"id": "1", "object": "chat.completion.chunk", "created": 1,
         "model": "m",
         "choices": [{"index": 0, "delta": {"content": [
             {"type": "thinking", "thinking": [{"type": "text", "text": "thi"}]}]},
             "finish_reason": None}]},
        {"id": "1", "object": "chat.completion.chunk", "created": 1,
         "model": "m",
         "choices": [{"index": 0, "delta": {"content": [
             {"type": "thinking", "thinking": [{"type": "text", "text": "nk"}],
              "closed": True},
             {"type": "text", "text": "391"}]}, "finish_reason": None}]},
        {"id": "1", "object": "chat.completion.chunk", "created": 1,
         "model": "m",
         "choices": [{"index": 0, "delta": {"content": "!"},
                      "finish_reason": "stop"}]},
    ]
    body = "".join(
        "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        for event in events
    ) + "data: [DONE]\n\n"

    transport = MistralNormalizingTransport(
        _FakeTransport(body.encode(), "text/event-stream"),
    )
    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    response = await transport.handle_async_request(request)

    raw = ""
    async for piece in response.aiter_bytes():
        raw += piece.decode()
    frames = [f for f in raw.split("\n\n") if f.strip()]
    parsed = []
    for frame in frames:
        data = frame[len("data:"):].strip()
        if data == "[DONE]":
            continue
        parsed.append(json.loads(data))

    thinking = "".join(
        d["choices"][0]["delta"].get("reasoning_content") or "" for d in parsed
    )
    text = "".join(d["choices"][0]["delta"].get("content") or "" for d in parsed)

    checks = [
        ("SSE frames preserved", len(parsed) == 3, f"{len(parsed)} frames"),
        ("thinking accumulated", thinking == "think", repr(thinking)),
        ("text accumulated", text == "391!", repr(text)),
        ("[DONE] kept", raw.rstrip().endswith("[DONE]"), repr(raw[-20:])),
    ]
    from openai.types.chat import ChatCompletionChunk
    try:
        for d in parsed:
            ChatCompletionChunk.model_validate(d)
        checks.append(("openai SDK accepts rewritten chunks", True, ""))
    except Exception as exc:  # noqa: BLE001
        checks.append(("openai SDK accepts rewritten chunks", False, str(exc)[:120]))
    return checks


def _unit_checks() -> list[tuple[str, bool, str]]:
    checks = []
    t, a = split_chunks(REAL_CONTENT)
    checks.append(("split_chunks thinking", t == "Simple: 17×23 = 391.", repr(t)))
    checks.append(("split_chunks answer", a.startswith("**17 × 23"), repr(a[:12])))

    t2, a2 = split_chunks("plain")
    checks.append(("plain str passes through", (t2, a2) == ("", "plain"), f"{t2!r},{a2!r}"))

    t3, a3 = split_chunks([{"type": "mystery", "payload": 1}, {"type": "text", "text": "ok"}])
    checks.append(("unknown chunk type dropped", (t3, a3) == ("", "ok"), f"{t3!r},{a3!r}"))

    # request direction: every shape of assistant turn must lose the field
    from mistral_transport import normalize_request_body

    tool_call_turn = {"messages": [{
        "role": "assistant", "content": None, "reasoning_content": "why",
        "tool_calls": [{"id": "t1", "type": "function",
                        "function": {"name": "f", "arguments": "{}"}}],
    }]}
    changed = normalize_request_body(tool_call_turn)
    msg = tool_call_turn["messages"][0]
    checks.append((
        "tool-call turn: reasoning converted, field gone",
        changed and "reasoning_content" not in msg
        and isinstance(msg["content"], list)
        and msg["content"][0]["type"] == "thinking",
        json.dumps(msg, ensure_ascii=False)[:120],
    ))

    for shape in (None, "ANS", [{"type": "text", "text": "ANS"}]):
        body = {"messages": [{"role": "assistant", "content": shape,
                              "reasoning_content": "r"}]}
        normalize_request_body(body)
        if "reasoning_content" in body["messages"][0]:
            checks.append((f"assistant content={type(shape).__name__} converted",
                           False, str(body["messages"][0])[:100]))
            break
    else:
        checks.append(("assistant content shapes all converted", True, "str/list/None"))

    stray = {"messages": [{"role": "user", "content": "q",
                           "reasoning_content": "r"}]}
    normalize_request_body(stray)
    checks.append(("stray reasoning on non-assistant dropped",
                   "reasoning_content" not in stray["messages"][0],
                   str(stray["messages"][0])))

    rewritten = rewrite_sse_event('data: {"choices":[]}\n\nevent: ping')
    checks.append(("non-data lines untouched", "event: ping" in rewritten, repr(rewritten)))
    return checks


def main() -> int:
    import asyncio

    checks = _unit_checks()
    checks += asyncio.run(_run_json_case())
    checks += asyncio.run(_run_sse_case())

    failed = 0
    for name, ok, extra in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   [{extra}]" if extra else ""))
        failed += 0 if ok else 1
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
