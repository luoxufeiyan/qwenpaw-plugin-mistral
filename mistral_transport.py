# -*- coding: utf-8 -*-
"""Transport-level normalisation of Mistral's chunk-list responses.

Why this exists
---------------
When a reasoning knob is set, Mistral answers with a *chunk list* instead of a
string::

    "content": [
        {"type": "thinking", "thinking": [{"type": "text", "text": "..."}],
         "closed": true},
        {"type": "text", "text": "391"},
    ]

The openai SDK (2.33.x) types ``content`` as ``str`` and rejects that payload
in pydantic validation, i.e. **before** QwenPaw's formatter ever sees it::

    ValidationError: choices.0.message.content
        Input should be a valid string [input_type=list]

This module rewrites the payload on the wire, so the SDK only ever sees the
DeepSeek-style shape QwenPaw already understands:

* chunk list -> ``content`` string (text chunks only)
* thinking chunks -> ``reasoning_content`` string

Both the non-streaming JSON body and the streaming SSE frames are handled.
Requests are untouched.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

#: Response headers whose bodies are rewritten.
_JSON_TYPE = "application/json"
_SSE_TYPE = "text/event-stream"


def _join_text(value: Any) -> str:
    """Flatten a ``str`` / list-of-chunks / list-of-text-chunks into text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _join_text(value.get("text") or value.get("content") or "")
    if isinstance(value, (list, tuple)):
        return "".join(_join_text(item) for item in value)
    return str(value)


def split_chunks(chunks: Any) -> tuple[str, str]:
    """Split a Mistral chunk list into ``(thinking_text, answer_text)``.

    Unknown chunk types are ignored rather than stringified, so a future
    chunk type cannot leak ``{'type': ...}`` into the model's answer.
    """
    if isinstance(chunks, str):
        return "", chunks
    if not isinstance(chunks, (list, tuple)):
        return "", _join_text(chunks)

    thinking: list[str] = []
    answer: list[str] = []
    for chunk in chunks:
        if isinstance(chunk, str):
            answer.append(chunk)
            continue
        if not isinstance(chunk, dict):
            answer.append(str(chunk))
            continue
        ctype = chunk.get("type")
        if ctype == "thinking":
            thinking.append(_join_text(chunk.get("thinking")))
        elif ctype in (None, "text"):
            answer.append(_join_text(chunk.get("text")))
        else:
            logger.debug("Mistral: ignoring unknown chunk type %r", ctype)
    return "".join(thinking), "".join(answer)


def _merge_reasoning(message: dict, thinking: str) -> None:
    """Append *thinking* to ``reasoning_content`` without losing existing."""
    if not thinking:
        return
    existing = message.get("reasoning_content")
    if isinstance(existing, str) and existing:
        message["reasoning_content"] = existing + thinking
    else:
        message["reasoning_content"] = thinking


def normalize_message(message: Any) -> bool:
    """Rewrite one ``message`` object in place. Returns True if changed."""
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if not isinstance(content, (list, tuple)):
        return False
    thinking, answer = split_chunks(content)
    message["content"] = answer
    _merge_reasoning(message, thinking)
    return True


# ----------------------------------------------------------------------
# request direction
# ----------------------------------------------------------------------
def _chunk_list_from_message(message: dict) -> list[dict] | None:
    """Convert a DeepSeek-style assistant message into Mistral's chunk list.

    Returns ``None`` when there is nothing to convert.  The result follows
    Mistral's documented shape: a ``thinking`` chunk (when reasoning exists)
    followed by a ``text`` chunk (when content exists).

    ``content`` may legitimately be ``None``: an assistant turn that only
    emits tool calls carries ``content: null`` plus ``tool_calls``.  Those
    turns still need their reasoning translated, otherwise the field is
    rejected with 422 and the whole tool-call chain breaks (this is exactly
    what happened with the ``content: null`` + ``tool_calls`` shape).
    """
    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning:
        return None
    content = message.get("content")
    if isinstance(content, list):
        content = _join_text(content)
    if content is None:
        content = ""
    if not isinstance(content, str):
        content = str(content)
    chunks: list[dict] = [
        {
            "type": "thinking",
            "thinking": [{"type": "text", "text": reasoning}],
            "closed": True,
        },
    ]
    if content:
        chunks.append({"type": "text", "text": content})
    return chunks


def normalize_request_body(payload: Any) -> bool:
    """Translate ``reasoning_content`` into Mistral's chunk list, in place.

    Mistral rejects a ``reasoning_content`` key outright (HTTP 422
    ``extra_forbidden``) and expects the trace inside ``content``.  This is the
    request-side counterpart of :func:`normalize_completion`.

    Any ``reasoning_content`` that cannot be converted is **dropped** rather
    than forwarded: losing a trace degrades quality, forwarding it kills the
    request.
    """
    if not isinstance(payload, dict):
        return False
    changed = False
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        if "reasoning_content" not in message:
            continue
        if message.get("role") != "assistant":
            # Reasoning only belongs on assistant turns.  Anything else is a
            # leftover from another provider and would 422 just the same.
            message.pop("reasoning_content", None)
            changed = True
            continue
        chunks = _chunk_list_from_message(message)
        if chunks is not None:
            message["content"] = chunks
        else:
            logger.debug(
                "Mistral: dropping unconvertible reasoning_content (%d chars)",
                len(str(message.get("reasoning_content") or "")),
            )
        message.pop("reasoning_content", None)
        changed = True
    return changed


def normalize_completion(payload: Any) -> bool:
    """Rewrite a non-streaming ``chat.completion`` body in place."""
    if not isinstance(payload, dict):
        return False
    changed = False
    for choice in payload.get("choices") or []:
        if isinstance(choice, dict) and normalize_message(choice.get("message")):
            changed = True
    return changed


def normalize_chunk(payload: Any) -> bool:
    """Rewrite one streaming ``chat.completion.chunk`` payload in place."""
    if not isinstance(payload, dict):
        return False
    changed = False
    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if not isinstance(content, (list, tuple)):
            continue
        thinking, answer = split_chunks(content)
        delta["content"] = answer
        _merge_reasoning(delta, thinking)
        changed = True
    return changed


def rewrite_sse_event(raw: str) -> str:
    """Rewrite one SSE event block. Non-data lines are preserved."""
    lines = raw.split("\n")
    out: list[str] = []
    for line in lines:
        if not line.startswith("data:"):
            out.append(line)
            continue
        data = line[len("data:"):].strip()
        if not data or data == "[DONE]":
            out.append(line)
            continue
        try:
            payload = json.loads(data)
        except (TypeError, ValueError):
            out.append(line)
            continue
        if normalize_chunk(payload):
            out.append("data: " + json.dumps(payload, ensure_ascii=False))
        else:
            out.append(line)
    return "\n".join(out)


class MistralNormalizingTransport(httpx.AsyncBaseTransport):
    """httpx transport that normalises Mistral responses on the way in."""

    def __init__(self, inner: httpx.AsyncBaseTransport | None = None) -> None:
        self._inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        request = self._rewrite_request(request)
        response = await self._inner.handle_async_request(request)
        content_type = response.headers.get("content-type", "")

        if _JSON_TYPE in content_type:
            body = await response.aread()
            try:
                payload = json.loads(body)
            except (TypeError, ValueError):
                payload = None
            if payload is not None and normalize_completion(payload):
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                headers = httpx.Headers(response.headers)
                headers["content-length"] = str(len(body))
                return httpx.Response(
                    status_code=response.status_code,
                    headers=headers,
                    content=body,
                    request=request,
                )
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                content=body,
                request=request,
            )

        if _SSE_TYPE in content_type:
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                stream=_NormalizingAsyncByteStream(response),
                request=request,
            )

        return response

    def _rewrite_request(self, request: httpx.Request) -> httpx.Request:
        """Rewrite ``reasoning_content`` in the outgoing JSON body."""
        body = request.content
        if not body:
            return request
        try:
            payload = json.loads(body)
        except (TypeError, ValueError):
            return request
        if not normalize_request_body(payload):
            return request
        new_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = httpx.Headers(request.headers)
        headers["content-length"] = str(len(new_body))
        return httpx.Request(
            method=request.method,
            url=request.url,
            headers=headers,
            content=new_body,
            extensions=request.extensions,
        )

    async def aclose(self) -> None:
        await self._inner.aclose()


class _NormalizingAsyncByteStream(httpx.AsyncByteStream):
    """Re-chunk an SSE body, rewriting each complete event."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._buffer = ""
        self._iterator: Any = None

    async def __aiter__(self):
        iterator = self._response.aiter_text()
        async for text in iterator:
            self._buffer += text
            # SSE events are separated by a blank line.
            while "\n\n" in self._buffer:
                event, self._buffer = self._buffer.split("\n\n", 1)
                if not event.strip():
                    continue
                yield (rewrite_sse_event(event) + "\n\n").encode("utf-8")
        tail = self._buffer.strip()
        if tail:
            yield (rewrite_sse_event(tail) + "\n\n").encode("utf-8")
        self._buffer = ""

    async def aclose(self) -> None:
        await self._response.aclose()
