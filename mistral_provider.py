# -*- coding: utf-8 -*-
"""Mistral AI provider for QwenPaw.

Why this provider exists
------------------------
Mistral's OpenAI-compatible endpoint validates the request body with a strict
schema (pydantic ``extra="forbid"``).  QwenPaw's OpenAI formatter injects a
DeepSeek-specific sibling field ``reasoning_content`` into assistant messages
whenever the conversation history contains thinking blocks, which makes
Mistral answer ``422 extra_forbidden``.

Two knobs are set here to stay on Mistral's documented wire format:

1. ``_get_relay_reasoning()`` always returns ``False``.  Mistral represents
   reasoning as ``ThinkChunk`` entries *inside* ``message.content`` and asks
   callers to replay the whole assistant message; it never accepts a
   ``reasoning_content`` key.  See
   https://docs.mistral.ai/inference/conversations/reasoning

2. ``cache_modes`` declares ``openai`` so ``prompt_cache_key`` survives
   ``providers/adapters/cache_policy.py`` validation (the base ``Provider``
   declares an empty frozenset, and any cache control is then rejected with
   ``Unsupported cache control``).  ``prepare_request`` derives a stable key
   from the conversation's leading messages, which is what Mistral's prompt
   caching wants: cached input tokens are billed at 10% of the input price.

Nothing else about the OpenAI wire format is changed: tools, streaming and
media promotion keep going through ``OpenAIProvider``.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Iterable

from qwenpaw.providers.model_info import ModelInfo
from qwenpaw.providers.openai_provider import OpenAIProvider
from qwenpaw.providers.thinking import ThinkingControl

try:  # package import (plugin loader)
    from .mistral_transport import MistralNormalizingTransport
except ImportError:  # flat import (selftest)
    from mistral_transport import MistralNormalizingTransport

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.mistral.ai/v1"

#: Prefix applied to derived cache keys so they are recognisable in logs.
CACHE_KEY_PREFIX = "qwenpaw-mistral-"

#: Roles that make up the reusable prompt prefix.
_PREFIX_ROLES = ("system", "developer")

#: Request keys that must never reach Mistral through this provider.
#:
#: Setting any reasoning knob makes Mistral answer with a *chunk list* in
#: ``choices[].message.content`` / ``delta.content``:
#:   [{"type": "thinking", "thinking": [{"type": "text", ...}], "closed": true},
#:    {"type": "text", "text": "..."}]
#: The openai SDK (2.33.x) types ``content`` as ``str`` and rejects that shape
#: with a ValidationError *before* QwenPaw can normalise anything.  The
#: transport installed by ``get_chat_model_instance`` rewrites those payloads
#: into ``content`` + ``reasoning_content``, so the knobs below are allowed
#: again — this tuple is kept as the single place to re-disable them if the
#: normaliser ever has to be turned off.
UNSUPPORTED_REASONING_KEYS: tuple[str, ...] = ()

#: Mistral documents ``reasoning_effort`` only for these model families.
#: Anything else (DeepSeek, Codestral) gets no thinking control, so the agent's
#: thinking level can never turn into an unsupported request parameter.
_EFFORT_MODELS: dict[str, ThinkingControl] = {
    # Mistral's docs do not list DeepSeek, but a live probe (2026-11-10)
    # shows it accepts low/high/max and returns progressively longer
    # thinking chunks (161 / 216 / 342 chars), so the control is enabled.
    "deepseek-v4-1-flash-1017": ThinkingControl(
        kind="effort",
        efforts=["low", "high", "max"],
        wire="native",
    ),
    "zai-glm-5-3-flash": ThinkingControl(
        kind="effort",
        efforts=["low", "high", "max"],
        wire="native",
    ),
    "zai-glm-5-3": ThinkingControl(
        kind="effort",
        efforts=["low", "high", "max"],
        wire="native",
    ),
    "mistral-large-4-0": ThinkingControl(
        kind="effort",
        efforts=["low", "medium", "high"],
        wire="native",
    ),
    "mistral-small-2603": ThinkingControl(
        kind="effort",
        efforts=["low", "medium", "high"],
        supports_off=True,
        wire="native",
    ),
}

def _block_text(block: Any) -> str:
    """Return the text of one content block, whatever shape it has."""
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        for key in ("text", "content"):
            value = block.get(key)
            if isinstance(value, str):
                return value
        return ""
    return str(block) if block is not None else ""


def _content_text(content: Any) -> str:
    """Flatten ``str`` or block-list content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Iterable):
        return "".join(_block_text(item) for item in content)
    return str(content)


#: Curated catalog.  Prices are intentionally left empty: pricing lives in
#: Mistral's own table and guessing keys here would misreport billing.  The
#: provider still discovers the full list through ``GET /v1/models``.
BUILTIN_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="deepseek-v4-1-flash-1017",
        name="DeepSeek V4.1 Flash 1017 (Mistral-hosted)",
        max_input_length=1_000_000,
        max_output_length=128_000,
        supports_tool_calling=True,
        supports_image=True,
        relay_reasoning=True,
        thinking_control=_EFFORT_MODELS["deepseek-v4-1-flash-1017"],
        is_recommended=True,
    ),
    ModelInfo(
        id="zai-glm-5-3-flash",
        name="GLM 5.3 Flash (Mistral-hosted)",
        max_input_length=1_000_000,
        supports_tool_calling=True,
        supports_image=True,
        relay_reasoning=True,
        thinking_control=_EFFORT_MODELS["zai-glm-5-3-flash"],
        is_recommended=True,
    ),
    ModelInfo(
        id="zai-glm-5-3",
        name="GLM 5.3 (Mistral-hosted)",
        max_input_length=1_000_000,
        supports_tool_calling=True,
        relay_reasoning=True,
        thinking_control=_EFFORT_MODELS["zai-glm-5-3"],
    ),
    ModelInfo(
        id="mistral-large-4-0",
        name="Mistral Large 4",
        supports_tool_calling=True,
        supports_image=True,
        relay_reasoning=True,
        thinking_control=_EFFORT_MODELS["mistral-large-4-0"],
    ),
    ModelInfo(
        id="mistral-small-2603",
        name="Mistral Small 4",
        supports_tool_calling=True,
        supports_image=True,
        relay_reasoning=True,
        thinking_control=_EFFORT_MODELS["mistral-small-2603"],
    ),
    ModelInfo(
        id="codestral-25-08",
        name="Codestral",
        supports_tool_calling=True,
        relay_reasoning=True,
    ),
]


class MistralProvider(OpenAIProvider):
    """OpenAI-compatible provider pinned to Mistral's wire format."""

    #: ``prompt_cache_key`` is only forwarded when a cache mode is declared.
    cache_modes = frozenset({"openai"})

    # ------------------------------------------------------------------
    # catalog
    # ------------------------------------------------------------------
    @classmethod
    def get_default_models(cls) -> list[ModelInfo]:
        """Seed the provider catalog at registration time.

        ``ProviderManager._prepare_plugin_registration`` calls this hook and
        puts the result into ``ProviderInfo.models``.  Without it the provider
        is registered with an empty model list, and the UI stays empty until a
        manual remote sync succeeds.
        """
        return [model.model_copy() for model in BUILTIN_MODELS]

    # ------------------------------------------------------------------
    # reasoning
    # ------------------------------------------------------------------
    def _get_relay_reasoning(self, model_id: str) -> bool:
        """Relay ``reasoning_content`` — but only because of the transport.

        Mistral's HTTP API rejects a ``reasoning_content`` key on the wire
        (HTTP 422 ``extra_forbidden``) and carries reasoning inside
        ``content`` as ``ThinkChunk`` entries instead.  The normalising
        transport installed in ``get_chat_model_instance`` is what makes the
        two representations interchangeable:

        * response: ``content`` chunk list -> ``content`` + ``reasoning_content``
        * request:  ``reasoning_content`` -> ``content`` chunk list

        With both directions translated, relaying is correct and gives the
        model its own reasoning trace back on the next turn, which Mistral's
        docs require for multi-turn coherence.
        """
        del model_id
        return True

    # ------------------------------------------------------------------
    # model construction
    # ------------------------------------------------------------------
    def get_chat_model_instance(self, model_id: str):
        """Build the chat model with the normalising transport attached."""
        model = super().get_chat_model_instance(model_id)
        # Swap in the Mistral formatter: the base OpenAI formatter drops
        # ThinkingBlock, which would leave nothing for the relay to carry.
        try:
            model.formatter = self.build_formatter(model_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Mistral: formatter install failed: %s", exc)
        # ``OpenAIChatModel.__init__`` already built an AsyncClient; swap its
        # transport for the normaliser so every request/response goes through
        # it (streaming included).
        client = getattr(model, "client", None)
        inner = getattr(client, "_client", None)
        if inner is None:
            logger.warning(
                "Mistral: could not locate the httpx client on %s; "
                "chunk-list responses will fail to parse.",
                type(model).__name__,
            )
            return model
        try:
            inner._transport = MistralNormalizingTransport(  # noqa: SLF001
                inner._transport,  # noqa: SLF001
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Mistral: transport install failed: %s", exc)
        return model

    # ------------------------------------------------------------------
    # formatter
    # ------------------------------------------------------------------
    def build_formatter(self, model_id: str):
        """Return a formatter that puts reasoning back into ``content``.

        AgentScope's ``OpenAIChatFormatter`` drops ``ThinkingBlock`` when it
        builds an assistant message ("OpenAI API does not accept
        reasoning/thinking content in conversation history"), so QwenPaw's
        relay step has nothing left to attach ``reasoning_content`` to.  For
        Mistral the trace belongs *inside* ``content`` as a ``ThinkChunk``,
        which is exactly what the transport emits for the request side, so
        the block has to survive formatting.

        The override is deliberately narrow: only the ``ThinkingBlock`` branch
        changes.  Every other behaviour (media promotion, file blocks, tool
        calls) still comes from QwenPaw's own formatter wrapper.
        """
        from agentscope.message import ThinkingBlock

        base = self.get_chat_model_instance(model_id).formatter
        base_class = type(base)

        class _MistralFormatter(base_class):  # type: ignore[misc,valid-type]
            async def format(self, msgs):  # noqa: D102
                messages = await super().format(msgs)
                # Re-attach thinking traces that the base formatter dropped,
                # in the same order the assistant messages were emitted.
                traces: list[str] = []
                for msg in msgs:
                    if getattr(msg, "role", None) != "assistant":
                        continue
                    content = getattr(msg, "content", None)
                    if not isinstance(content, list):
                        traces.append("")
                        continue
                    traces.append(
                        "".join(
                            getattr(block, "thinking", "") or ""
                            for block in content
                            if isinstance(block, ThinkingBlock)
                        ),
                    )

                assistant_wire = [
                    m for m in messages if m.get("role") == "assistant"
                ]
                if len(assistant_wire) == len(traces):
                    for wire_msg, trace in zip(assistant_wire, traces):
                        if trace:
                            wire_msg["reasoning_content"] = trace
                elif any(traces):
                    logger.warning(
                        "Mistral: assistant message count mismatch "
                        "(%d wire vs %d source); reasoning not re-attached.",
                        len(assistant_wire),
                        len(traces),
                    )
                return messages

        return _MistralFormatter(**base.model_dump())

    # ------------------------------------------------------------------
    # prompt caching
    # ------------------------------------------------------------------
    def _derive_cache_key(self, messages: list[dict]) -> str | None:
        """Build a stable key from the conversation's reusable prefix.

        The leading system/developer messages plus the first user message are
        stable for the whole conversation, so the same key is reused on every
        turn while a new conversation gets a different one.
        """
        parts: list[str] = []
        for message in messages:
            role = message.get("role")
            if role in _PREFIX_ROLES:
                parts.append(_content_text(message.get("content")))
                continue
            if role == "user":
                parts.append(_content_text(message.get("content")))
                break
        if not parts:
            return None
        joined = "\x1f".join(parts)
        digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]
        return f"{CACHE_KEY_PREFIX}{digest}"

    def prepare_request(
        self,
        model_id: str,
        protocol: str,
        kwargs: dict,
    ) -> dict:
        """Strip unsupported reasoning knobs, then add ``prompt_cache_key``."""
        payload = dict(kwargs)
        dropped = [
            key for key in UNSUPPORTED_REASONING_KEYS if key in payload
        ]
        if dropped:
            for key in dropped:
                payload.pop(key, None)
            logger.warning(
                "Mistral model '%s': dropped unsupported reasoning "
                "parameters %s (chunk-list content is rejected by the "
                "openai SDK before parsing).",
                model_id,
                dropped,
            )
        if protocol == "chat" and "prompt_cache_key" not in payload:
            messages = payload.get("messages")
            key = (
                self._derive_cache_key(messages)
                if isinstance(messages, list)
                else None
            )
            if key:
                payload["prompt_cache_key"] = key
        return super().prepare_request(model_id, protocol, payload)
